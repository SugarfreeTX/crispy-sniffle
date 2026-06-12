#!/usr/bin/env python3
"""
Shadow Grok Decision Replay Simulator

Replays the decisions recorded in shadow_grok_log.jsonl (Grok LLM actions vs.
the deterministic auto-hold / rule-based gates) against the actual prices and
suggested_position_sizes observed during those runs.

Produces hypothetical P&L, equity curves, trade lists, and summary metrics
for two policies on the *exact same* sequence of decision points:
- grok_policy: follow the logged grok_action
- det_policy: follow the logged deterministic_action (GROK_DECIDED treated
  as HOLD for a conservative "rules-only" baseline)

Usage (recommended):
  python equity_msft/shadow_replay.py \
    --shadow-log equity_msft/shadow_grok_log.jsonl \
    --output-dir equity_msft/shadow_replay_outputs \
    --dry-logs-dir equity_msft/test_runs \
    --initial-cash 99961.94

The script is fully offline. It extracts per-decision prices + suggested sizes
by parsing the accompanying dry-run trading logs (the lines that logged
"Successfully fetched MSFT data. Price: $..., Suggested shares: ...").

It re-uses the same sizing math, drawdown/streak multipliers, and sell
percentage logic that the live loop uses (via shared.risk_management and
adapted snippets from complete_daily_loop).

Outputs (modeled on backtest_outputs/):
  grok_equity.csv / det_equity.csv
  grok_trades.csv / det_trades.csv
  metrics_grok.json / metrics_det.json
  comparison_summary.json
  stats_grok.txt / stats_det.txt

Add --plot (if matplotlib available) for a simple equity curve image.

For future runs, if you enrich the shadow log (see plan), the simulator will
prefer the embedded "price" / "suggested_position_size" fields and will not
need the dry logs for matching.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make shared/ importable when run from repo root or equity_msft/
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from shared.risk_management import (
        get_drawdown_level,
        get_loss_streak_multiplier,
    )
except Exception:  # fallback if run in odd cwd
    get_drawdown_level = None
    get_loss_streak_multiplier = None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ShadowEntry:
    date: str
    deterministic_action: str
    deterministic_reason: str
    grok_action: Optional[str]
    grok_reason: str
    packet_summary: Dict[str, Any]
    # Optional enriched fields (future-proof)
    price: Optional[float] = None
    suggested_position_size: Optional[int] = None
    atr_14: Optional[float] = None
    trend_label: Optional[str] = None
    regime: Optional[str] = None
    portfolio_snapshot: Optional[Dict[str, Any]] = None


@dataclass
class ReplayState:
    cash: float
    shares: int
    cost_basis: float
    peak_equity: float
    trades: List[Dict[str, Any]] = field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    consecutive_loss_streak: int = 0


# ---------------------------------------------------------------------------
# Loading & matching
# ---------------------------------------------------------------------------

def load_shadow_log(path: Path) -> List[ShadowEntry]:
    entries: List[ShadowEntry] = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            entry = ShadowEntry(
                date=raw.get("date", ""),
                deterministic_action=raw.get("deterministic_action", "HOLD"),
                deterministic_reason=raw.get("deterministic_reason", ""),
                grok_action=raw.get("grok_action"),
                grok_reason=raw.get("grok_reason", ""),
                packet_summary=raw.get("packet_summary", {}),
                price=raw.get("price"),
                suggested_position_size=raw.get("suggested_position_size"),
                atr_14=raw.get("atr_14"),
                trend_label=raw.get("trend_label"),
                regime=raw.get("regime") or raw.get("packet_summary", {}).get("regime"),
                portfolio_snapshot=raw.get("portfolio_snapshot"),
            )
            entries.append(entry)
    return entries


def _parse_fetched_lines(log_path: Path) -> List[Dict[str, Any]]:
    """Extract price, rsi, suggested etc from a trading log (tolerant of slight formatting variations)."""
    results = []
    # Be tolerant: the line can span slight variations or the regex can be searched across the file.
    # We only really need Price + RSI + Suggested shares for matching.
    pat = re.compile(
        r"Successfully fetched MSFT data\..*?"
        r"Price:\s*\$?([\d.]+).*?"
        r"RSI:\s*([\d.]+).*?"
        r"Suggested shares:\s*(\d+)",
        re.I | re.S,
    )
    try:
        txt = open(log_path, "r", errors="ignore").read()
        for m in pat.finditer(txt):
            results.append({
                "price": float(m.group(1)),
                "rsi": float(m.group(2)),
                "suggested": int(m.group(3)),
                "source_file": log_path.name,
            })
    except Exception:
        pass
    return results


def build_price_and_size_lookup(
    dry_logs_dir: Optional[Path],
    shadow_entries: List[ShadowEntry],
) -> Dict[int, Dict[str, Any]]:
    """
    Return a lookup index -> {price, suggested, rsi, ...} for each shadow entry.

    Strategy:
    - If the shadow entry already has price + suggested (enriched log), use it.
    - Scan a broad set of trading logs (test_runs + main trading_log.txt + any other)
      for "Successfully fetched" lines and match by (rsi closeness + suggested) .
      These (rsi, suggested) pairs are very distinctive.
    """
    lookup: Dict[int, Dict[str, Any]] = {}

    # Pass 1: use any pre-embedded values from enriched logs
    for i, e in enumerate(shadow_entries):
        if e.price is not None and e.suggested_position_size is not None:
            lookup[i] = {
                "price": float(e.price),
                "suggested": int(e.suggested_position_size),
                "rsi": e.packet_summary.get("rsi"),
                "regime": e.regime or e.packet_summary.get("regime"),
                "source": "enriched_log",
            }

    if len(lookup) == len(shadow_entries):
        return lookup

    # Collect candidate log files broadly
    candidate_logs: List[Path] = []
    if dry_logs_dir and dry_logs_dir.exists():
        candidate_logs.extend(sorted(dry_logs_dir.glob("trading_log_dry_*.txt")))
    # Also the main trading_log (contains many fetches from shadow/dry runs)
    main_log = Path("equity_msft/trading_log.txt")
    if main_log.exists():
        candidate_logs.append(main_log)
    # Any other trading logs under equity_msft for good measure
    for p in Path("equity_msft").glob("**/trading_log*.txt"):
        if p not in candidate_logs:
            candidate_logs.append(p)

    # Parse all
    all_fetched: List[Dict[str, Any]] = []
    for p in candidate_logs:
        try:
            all_fetched.extend(_parse_fetched_lines(p))
        except Exception:
            pass

    if not all_fetched:
        print("[warn] No 'Successfully fetched' lines found in dry logs or trading_log.txt. Will rely on reason scraping.")
        # fall through to per-entry fallback below

    # Pass 2: best-effort match for remaining entries
    for i, e in enumerate(shadow_entries):
        if i in lookup:
            continue
        rsi = e.packet_summary.get("rsi")
        # Robust suggested extraction from reason text (various phrasings seen in the log)
        suggested = e.suggested_position_size
        if suggested is None:
            m = re.search(r"suggested_position_size[=:]\s*(\d+)", e.grok_reason or "")
            if not m:
                m = re.search(r"size[=:]\s*(\d+)", e.grok_reason or "")
            if not m:
                m = re.search(r"valid (?:position )?size[=: ]*(\d+)", e.grok_reason or "", re.I)
            if m:
                suggested = int(m.group(1))
            else:
                # very loose last resort
                m = re.search(r"(\d{2,3})\s*(?:shares|size|position)", e.grok_reason or "", re.I)
                if m:
                    suggested = int(m.group(1))

        best = None
        best_score = 1e9
        for obs in all_fetched:
            dr = abs((obs.get("rsi") or 999) - (rsi or 999)) if rsi is not None else 5.0
            ds = abs(obs.get("suggested", 999) - (suggested or 999)) if suggested else 50
            score = dr * 8 + ds * 1.0   # favor suggested match a bit more
            if score < best_score:
                best_score = score
                best = obs

        # Accept aggressively when suggested matches well (the distinctive signal);
        # many shadow entries only have partial context and the log corpus has the right sizes.
        accepted = False
        if best:
            ds_sug = abs(best.get("suggested", 999) - (suggested or 999)) if suggested else 999
            if suggested and ds_sug <= 5:           # suggested is the strongest key
                accepted = True
            elif rsi is not None and abs((best.get("rsi") or 999) - rsi) <= 2.0:
                accepted = True
            elif best_score < 35:
                accepted = True

        if accepted and best:
            lookup[i] = {
                "price": best["price"],
                "suggested": best["suggested"],
                "rsi": best.get("rsi"),
                "regime": best.get("regime"),
                "source": f"logs:{best.get('source_file', 'unknown')}",
            }
        else:
            # Fallback: at least carry suggested if we parsed it; price=None will cause qty=0 for that step
            m = re.search(r"(\d{2,3})", e.grok_reason or "")
            sug = suggested or (int(m.group(1)) if m else 0)
            lookup[i] = {
                "price": None,
                "suggested": sug,
                "rsi": rsi,
                "regime": e.regime or e.packet_summary.get("regime"),
                "source": "unmatched_fallback",
            }

    return lookup


# ---------------------------------------------------------------------------
# Core simulation (adapted from execute_trade + risk helpers)
# ---------------------------------------------------------------------------

def _final_qty(
    suggested: int,
    price: float,
    cash: float,
    shares: int,
    total_equity: float,
    max_position_pct: float = 0.20,
    streak_mult: float = 1.0,
) -> int:
    """Mirror the qty logic used in complete_daily_loop.execute_trade (dry-run path)."""
    if suggested <= 0 or price <= 0:
        return 0
    max_affordable = int(cash // price)
    max_position_value = total_equity * max_position_pct
    current_position_value = shares * price
    available = max(max_position_value - current_position_value, 0.0)
    max_by_limit = int(available // price) if price > 0 else 0

    qty = min(suggested, max_affordable, max_by_limit)
    qty = int(qty * streak_mult)

    # Probe floor (same as live)
    if qty <= 0 and suggested > 0 and max_affordable >= 1 and max_by_limit >= 1:
        qty = 1
    return max(qty, 0)


def _parse_sell_pct(reason: str, unrealized_pct: float) -> float:
    """Adapted from complete_daily_loop.parse_sell_percentage + prompt tiers."""
    m = re.search(r"(?:sell|exit|reduce)\s*(?:about|around|roughly)?\s*(\d+)%?", reason, re.I)
    if m:
        try:
            pct = float(m.group(1)) / 100.0
            if 0.01 <= pct <= 1.0:
                return pct
        except Exception:
            pass
    # tiers from prompt / code
    if unrealized_pct < 8.0:
        return 1.0
    elif unrealized_pct < 15.0:
        return 0.30
    elif unrealized_pct < 25.0:
        return 0.40
    else:
        return 0.60


class ReplaySim:
    """Lightweight single-policy replay engine."""

    def __init__(self, initial_cash: float = 100000.0, initial_shares: int = 0, initial_cost: float = 0.0):
        self.state = ReplayState(
            cash=initial_cash,
            shares=initial_shares,
            cost_basis=initial_cost,
            peak_equity=initial_cash,
        )
        self.initial_cash = initial_cash

    @property
    def equity(self) -> float:
        # Use last known price if curve not empty, else initial
        if self.state.equity_curve:
            return self.state.equity_curve[-1]["equity"]
        return self.state.cash + self.state.shares * 0.0  # neutral until first price

    def _record(self, step: int, date: str, action: str, price: float, suggested: int, reason: str = ""):
        eq = self.state.cash + self.state.shares * price
        self.state.peak_equity = max(self.state.peak_equity, eq)
        dd = ((self.state.peak_equity - eq) / self.state.peak_equity * 100.0) if self.state.peak_equity > 0 else 0.0
        self.state.equity_curve.append({
            "step": step,
            "date": date,
            "equity": round(eq, 2),
            "cash": round(self.state.cash, 2),
            "shares": self.state.shares,
            "price": price,
            "action": action,
            "suggested": suggested,
            "drawdown_pct": round(dd, 4),
        })

    def step(
        self,
        step: int,
        date: str,
        action: str,
        price: float,
        suggested: int,
        reason: str = "",
        max_position_pct: float = 0.20,
    ) -> Tuple[float, float]:
        if price is None or price <= 0:
            # Cannot trade or MTM reliably — just record current equity with last price if any
            last_price = self.state.equity_curve[-1]["price"] if self.state.equity_curve else 0.0
            self._record(step, date, "SKIP_NO_PRICE", last_price or 0.0, suggested, reason)
            return self.equity, 0.0

        total_equity = self.state.cash + self.state.shares * price

        # streak multiplier (best effort; we don't have full streak history, default 1.0)
        streak_mult = 1.0
        if get_loss_streak_multiplier is not None:
            try:
                streak_mult = get_loss_streak_multiplier(
                    {"consecutive_loss_streak": self.state.consecutive_loss_streak, "max_consecutive_losses": 5}
                )
            except Exception:
                streak_mult = 1.0

        realized = 0.0
        act = (action or "HOLD").upper()

        if act == "BUY":
            qty = _final_qty(
                suggested=suggested,
                price=price,
                cash=self.state.cash,
                shares=self.state.shares,
                total_equity=total_equity,
                max_position_pct=max_position_pct,
                streak_mult=streak_mult,
            )
            if qty > 0:
                execution_price = price
                cost = qty * execution_price
                self.state.cash -= cost
                new_shares = self.state.shares + qty
                if new_shares > 0:
                    old_cb = self.state.cost_basis if self.state.shares > 0 else 0.0
                    self.state.cost_basis = ((self.state.shares * old_cb) + (qty * execution_price)) / new_shares
                self.state.shares = new_shares
                self.state.trades.append({
                    "step": step,
                    "date": date,
                    "action": "BUY",
                    "qty": qty,
                    "price": execution_price,
                    "reason": reason[:160] if reason else "",
                    "realized_pnl": 0.0,
                    "equity_after": round(self.state.cash + self.state.shares * price, 2),
                })
            else:
                act = "HOLD"  # reduced to zero

        elif act == "SELL":
            if self.state.shares > 0:
                unreal_pct = 0.0
                if self.state.shares > 0 and self.state.cost_basis > 0:
                    unreal_pct = (price - self.state.cost_basis) / self.state.cost_basis * 100.0
                sell_frac = _parse_sell_pct(reason, unreal_pct)
                sell_qty = max(1, int(self.state.shares * sell_frac))
                sell_qty = min(sell_qty, self.state.shares)

                execution_price = price
                proceeds = sell_qty * execution_price
                # realized on the sold portion (average cost)
                realized = (execution_price - self.state.cost_basis) * sell_qty if self.state.cost_basis > 0 else 0.0

                self.state.cash += proceeds
                self.state.shares -= sell_qty
                if self.state.shares == 0:
                    self.state.cost_basis = 0.0
                    # simplistic streak update (win if realized > 0)
                    if realized > 0:
                        self.state.consecutive_loss_streak = 0
                    else:
                        self.state.consecutive_loss_streak += 1

                self.state.trades.append({
                    "step": step,
                    "date": date,
                    "action": "SELL",
                    "qty": -sell_qty,
                    "price": execution_price,
                    "reason": reason[:160] if reason else "",
                    "realized_pnl": round(realized, 2),
                    "equity_after": round(self.state.cash + self.state.shares * price, 2),
                })
            else:
                act = "HOLD"

        # Always mark to market / record curve
        self._record(step, date, act, price, suggested, reason)
        return self.equity, realized

    def finalize_metrics(self, policy: str) -> Dict[str, Any]:
        curve = self.state.equity_curve
        if not curve:
            final_eq = self.initial_cash
            peak = self.initial_cash
            mdd = 0.0
        else:
            final_eq = curve[-1]["equity"]
            peak = max(p["equity"] for p in curve)
            mdd = min(p["drawdown_pct"] for p in curve)

        trades = self.state.trades
        buy_trades = [t for t in trades if t["action"] == "BUY"]
        sell_trades = [t for t in trades if t["action"] == "SELL"]

        # Simple win rate on closed sells that have realized_pnl recorded
        closed_pnls = [t["realized_pnl"] for t in sell_trades if t.get("realized_pnl")]
        wins = sum(1 for p in closed_pnls if p > 0)
        losses = sum(1 for p in closed_pnls if p < 0)
        win_rate = (wins / len(closed_pnls) * 100.0) if closed_pnls else 0.0

        gross_win = sum(p for p in closed_pnls if p > 0)
        gross_loss = -sum(p for p in closed_pnls if p < 0)
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)

        # Naive daily-ish sharpe (using step returns)
        rets = []
        prev = self.initial_cash
        for p in curve:
            r = (p["equity"] - prev) / prev if prev > 0 else 0.0
            rets.append(r)
            prev = p["equity"]
        if rets:
            mean_r = sum(rets) / len(rets)
            std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rets) / len(rets)) or 1e-9
            sharpe = (mean_r / std_r) * math.sqrt(len(rets))  # crude annualization factor
        else:
            sharpe = 0.0

        return {
            "policy": policy,
            "initial_cash": round(self.initial_cash, 2),
            "final_equity": round(final_eq, 2),
            "peak_equity": round(peak, 2),
            "return_pct": round((final_eq - self.initial_cash) / self.initial_cash * 100.0, 4),
            "max_drawdown_pct": round(mdd, 4),
            "trade_count": len(trades),
            "buy_count": len(buy_trades),
            "sell_count": len(sell_trades),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else "inf",
            "sharpe_ratio": round(sharpe, 4),
            "closed_pnl_count": len(closed_pnls),
            "gross_wins": round(gross_win, 2),
            "gross_losses": round(gross_loss, 2),
        }


def replay_policy(
    entries: List[ShadowEntry],
    lookup: Dict[int, Dict[str, Any]],
    policy: str,  # "grok" or "det"
    initial_cash: float,
) -> Dict[str, Any]:
    sim = ReplaySim(initial_cash=initial_cash)
    for i, e in enumerate(entries):
        ctx = lookup.get(i, {})
        price = ctx.get("price")
        suggested = ctx.get("suggested") or 0

        if policy == "grok":
            raw_action = e.grok_action or "HOLD"
        else:
            raw_action = e.deterministic_action
            if raw_action == "GROK_DECIDED":
                raw_action = "HOLD"  # conservative baseline for pure-det comparison

        reason = e.grok_reason if policy == "grok" else e.deterministic_reason
        sim.step(
            step=i,
            date=e.date,
            action=raw_action,
            price=price if price is not None else 0.0,
            suggested=suggested,
            reason=reason,
        )

    metrics = sim.finalize_metrics(policy)
    return {
        "metrics": metrics,
        "curve": sim.state.equity_curve,
        "trades": sim.state.trades,
        "sim": sim,
    }


# ---------------------------------------------------------------------------
# Writers (style-compatible with backtest outputs)
# ---------------------------------------------------------------------------

def write_equity_csv(path: Path, curve: List[Dict[str, Any]]):
    if not curve:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "date", "equity", "cash", "shares", "price", "action", "suggested", "drawdown_pct"])
        w.writeheader()
        for row in curve:
            w.writerow({k: row.get(k) for k in w.fieldnames})


def write_trades_csv(path: Path, trades: List[Dict[str, Any]]):
    if not trades:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "date", "action", "qty", "price", "realized_pnl", "equity_after", "reason"])
        w.writeheader()
        for t in trades:
            w.writerow({
                "step": t.get("step"),
                "date": t.get("date"),
                "action": t.get("action"),
                "qty": t.get("qty"),
                "price": t.get("price"),
                "realized_pnl": t.get("realized_pnl", 0.0),
                "equity_after": t.get("equity_after"),
                "reason": t.get("reason", "")[:200],
            })


def write_metrics_json(path: Path, metrics: Dict[str, Any], extra: Optional[Dict] = None):
    data = dict(metrics)
    if extra:
        data.update(extra)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_stats_txt(path: Path, metrics: Dict[str, Any], title: str):
    lines = [
        f"=== {title} ===",
        f"Policy: {metrics.get('policy')}",
        f"Initial cash: ${metrics.get('initial_cash'):,.2f}",
        f"Final equity: ${metrics.get('final_equity'):,.2f}",
        f"Return: {metrics.get('return_pct'):.2f}%",
        f"Peak equity: ${metrics.get('peak_equity'):,.2f}",
        f"Max DD: {metrics.get('max_drawdown_pct'):.2f}%",
        f"Trades: {metrics.get('trade_count')} (BUY {metrics.get('buy_count')}, SELL {metrics.get('sell_count')})",
        f"Win rate (closed): {metrics.get('win_rate_pct'):.1f}%",
        f"Profit factor: {metrics.get('profit_factor')}",
        f"Sharpe (step-based): {metrics.get('sharpe_ratio'):.3f}",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_comparison(path: Path, m_grok: Dict, m_det: Dict):
    delta_return = m_grok["return_pct"] - m_det["return_pct"]
    delta_dd = m_grok["max_drawdown_pct"] - m_det["max_drawdown_pct"]  # less negative is better
    data = {
        "grok": m_grok,
        "det": m_det,
        "delta": {
            "return_pct_grok_minus_det": round(delta_return, 4),
            "max_dd_grok_minus_det": round(delta_dd, 4),
            "trade_count_diff": m_grok["trade_count"] - m_det["trade_count"],
            "notes": "Positive delta_return means Grok policy outperformed on total return in this replay window.",
        },
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay shadow_grok_log decisions (Grok vs deterministic) for hypothetical P&L analysis."
    )
    p.add_argument("--shadow-log", default="equity_msft/shadow_grok_log.jsonl", help="Path to the shadow jsonl")
    p.add_argument("--output-dir", default="equity_msft/shadow_replay_outputs", help="Where to write csv/json/txt artifacts")
    p.add_argument("--dry-logs-dir", default="equity_msft/test_runs", help="Directory containing trading_log_dry_*.txt (for price/sizing)")
    p.add_argument("--initial-cash", type=float, default=99961.94, help="Starting cash (matches observed dry-run state)")
    p.add_argument("--plot", action="store_true", help="Try to emit a simple equity curve plot (requires matplotlib)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    shadow_path = Path(args.shadow_log)
    out_dir = Path(args.output_dir)
    dry_dir = Path(args.dry_logs_dir) if args.dry_logs_dir else None
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[shadow-replay] Loading shadow log: {shadow_path}")
    entries = load_shadow_log(shadow_path)
    print(f"[shadow-replay] {len(entries)} entries, {len(set(e.date for e in entries))} unique dates")

    print("[shadow-replay] Building price/suggested lookup (dry logs + enriched fields)...")
    lookup = build_price_and_size_lookup(dry_dir, entries)

    unmatched = [i for i, e in enumerate(entries) if lookup.get(i, {}).get("price") is None]
    if unmatched:
        print(f"[warn] {len(unmatched)} entries had no reliable price match — they will contribute 0 qty trades (HOLD-like).")

    # Run both policies
    print("[shadow-replay] Replaying grok_policy ...")
    grok_res = replay_policy(entries, lookup, "grok", args.initial_cash)
    print("[shadow-replay] Replaying det_policy ...")
    det_res = replay_policy(entries, lookup, "det", args.initial_cash)

    # Write artifacts
    write_equity_csv(out_dir / "grok_equity.csv", grok_res["curve"])
    write_equity_csv(out_dir / "det_equity.csv", det_res["curve"])
    write_trades_csv(out_dir / "grok_trades.csv", grok_res["trades"])
    write_trades_csv(out_dir / "det_trades.csv", det_res["trades"])

    write_metrics_json(out_dir / "metrics_grok.json", grok_res["metrics"])
    write_metrics_json(out_dir / "metrics_det.json", det_res["metrics"])
    write_comparison(out_dir / "comparison_summary.json", grok_res["metrics"], det_res["metrics"])

    write_stats_txt(out_dir / "stats_grok.txt", grok_res["metrics"], "GROK POLICY (shadow log)")
    write_stats_txt(out_dir / "stats_det.txt", det_res["metrics"], "DETERMINISTIC POLICY (shadow log)")

    # Optional plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            g_curve = [p["equity"] for p in grok_res["curve"]]
            d_curve = [p["equity"] for p in det_res["curve"]]
            plt.figure(figsize=(10, 5))
            plt.plot(g_curve, label="Grok policy", linewidth=1.5)
            plt.plot(d_curve, label="Det policy", linewidth=1.5, alpha=0.8)
            plt.title("Hypothetical Equity (Shadow Replay)")
            plt.xlabel("Decision step")
            plt.ylabel("Equity ($)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "equity_curves.png", dpi=140)
            print(f"[shadow-replay] Wrote plot to {out_dir / 'equity_curves.png'}")
        except Exception as ex:
            print(f"[warn] Plot skipped (matplotlib issue): {ex}")

    # Console summary
    mg = grok_res["metrics"]
    md = det_res["metrics"]
    print("\n" + "=" * 64)
    print("SHADOW REPLAY COMPARISON (Grok vs Deterministic)")
    print("=" * 64)
    print(f"Entries processed: {len(entries)}")
    print(f"Initial cash:      ${args.initial_cash:,.2f}")
    print()
    print(f"{'Metric':<22} {'Grok':>14} {'Det':>14} {'Delta':>12}")
    print("-" * 64)
    print(f"{'Final equity':<22} ${mg['final_equity']:>13,.2f} ${md['final_equity']:>13,.2f} ${(mg['final_equity']-md['final_equity']):>11,.2f}")
    print(f"{'Return %':<22} {mg['return_pct']:>13.2f}% {md['return_pct']:>13.2f}% {(mg['return_pct']-md['return_pct']):>11.2f}%")
    print(f"{'Max DD %':<22} {mg['max_drawdown_pct']:>13.2f}% {md['max_drawdown_pct']:>13.2f}% {(mg['max_drawdown_pct']-md['max_drawdown_pct']):>11.2f}%")
    print(f"{'Trade count':<22} {mg['trade_count']:>14} {md['trade_count']:>14} {(mg['trade_count']-md['trade_count']):>12}")
    print(f"{'Win rate (closed)':<22} {mg['win_rate_pct']:>13.1f}% {md['win_rate_pct']:>13.1f}%")
    print(f"{'Profit factor':<22} {mg['profit_factor']:>14} {md['profit_factor']:>14}")
    print()
    print(f"Artifacts written to: {out_dir}")
    print("=" * 64)

    if unmatched:
        print(f"\n[note] {len(unmatched)} steps had no price match from dry logs. Their contribution was neutral (no position changes).")
        print("       Re-run with --dry-logs-dir pointing at all relevant test_runs, or enrich the shadow log for exact fidelity.")


if __name__ == "__main__":
    main()
