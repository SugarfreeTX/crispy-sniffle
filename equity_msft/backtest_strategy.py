from __future__ import annotations

from typing import Any, Dict, Tuple
import logging
from pathlib import Path
import sys

import pandas as pd
from backtesting import Strategy

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.indicators import calculate_atr, calculate_rsi, calculate_sma_trend
from shared.risk_management import (
    can_open_new_position,
    get_drawdown_level,
    get_loss_streak_multiplier,
)

logger = logging.getLogger(__name__)


def should_auto_hold(
    packet: Dict[str, Any],
    thresholds: Dict[str, float] | None = None,
) -> Tuple[bool, str]:
    """Replicate the local HOLD gate from the live loop for deterministic backtests."""
    md = packet.get("market_data", {})
    port = packet.get("portfolio", {})
    config = thresholds or {}

    neutral_rsi_low = float(config.get("neutral_rsi_low", 40.0))
    neutral_rsi_high = float(config.get("neutral_rsi_high", 60.0))
    neutral_rel_volume_max = float(config.get("neutral_rel_volume_max", 1.3))
    bullish_hold_rsi = float(config.get("bullish_hold_rsi", 62.0))
    bearish_hold_rsi = float(config.get("bearish_hold_rsi", 38.0))

    rsi = float(md.get("rsi_14", 50.0))
    regime = str(md.get("market_regime", "Normal"))
    rel_vol = float(md.get("relative_volume", 1.0))
    drawdown_pct = float(port.get("current_drawdown_pct", 0.0))
    trend_label = str(md.get("trend_label", "Neutral"))
    regime_changed = bool(md.get("regime_changed_today", False))

    if drawdown_pct >= 10.0:
        return True, f"Emergency drawdown ({drawdown_pct:.1f}%) - all new risk blocked"

    if drawdown_pct >= 8.0:
        return True, (
            f"Restricted drawdown ({drawdown_pct:.1f}%) - HOLD or exit only, no new BUY positions"
        )

    if drawdown_pct >= 5.0 and ("Bullish" not in trend_label or rsi > 35):
        return True, f"Caution drawdown ({drawdown_pct:.1f}%) - not extreme oversold setup"

    if (
        neutral_rsi_low <= rsi <= neutral_rsi_high
        and regime == "Normal"
        and rel_vol <= neutral_rel_volume_max
        and not regime_changed
    ):
        return True, "Neutral RSI, Normal regime, low volume, no regime change"

    if drawdown_pct >= 6.0:
        return True, f"Drawdown at {drawdown_pct:.2f}% - approaching max drawdown limit"

    if "Bullish" in trend_label and rsi >= bullish_hold_rsi:
        return True, f"Bullish trend but RSI {rsi:.1f} not low enough for entry"

    if "Bearish" in trend_label and rsi <= bearish_hold_rsi:
        return True, f"Bearish trend but RSI {rsi:.1f} not high enough for exit"

    return False, "No auto-hold rule triggered"


# ── Backtest participation improvements (2026-03-17) ─────────────────────────
# To reduce excessive cash drag in persistent adverse regimes while preserving
# drawdown control, three changes were made (backtest only; live code unchanged):
#   1. Regime size multipliers relaxed: High Vol 0.5→0.75, Elevated 0.75→1.0.
#      Soft 0.90× damper applies after regime_caution_days=18 of persistent
#      adverse regime instead of a hard HOLD block.
#   2. max_consecutive_losses=999 disables the streak size gate at class level;
#      get_loss_streak_multiplier() is bypassed via an explicit guard (it uses
#      independent DEFAULT_STREAK_THRESHOLDS, not max_consecutive_losses).
#      CLI --max-consecutive-losses default stays 5 for backward-compatible sweeps.
#      sell_overbought_rsi raised 74→88; take_profit_pnl_pct tightened 8→7%
#      so profit-target remains the primary exit trigger.
#   3. 1-share probe floor in _execute_buy(): when all multipliers round a valid
#      BUY signal to 0, trade exactly 1 share if cash and position limits allow.
# ─────────────────────────────────────────────────────────────────────────────
class MSFTDailyBacktestStrategy(Strategy):
    """Deterministic daily-bar strategy adapted from the live MSFT loop."""

    risk_per_trade_pct = 0.02
    max_position_size_pct = 0.20
    max_drawdown_pct = 0.10
    min_atr = 3.5
    max_atr = 18.0
    slippage_bps = 5.0
    max_consecutive_losses = 999  # disabled for backtest; CLI sweep default stays 5
    regime_caution_days = 18      # days before soft 0.90× damper in adverse regime
    history_window = 260

    # Tuned defaults for better short-window participation.
    neutral_rsi_low = 40.0
    neutral_rsi_high = 60.0
    neutral_rel_volume_max = 1.3
    bullish_hold_rsi = 62.0
    bearish_hold_rsi = 38.0
    buy_pullback_rsi = 46.0
    sell_bearish_rsi = 52.0
    sell_overbought_rsi = 88.0    # raised 74→88; take_profit_pnl_pct is the primary exit
    take_profit_pnl_pct = 7.0     # tightened 8→7% to pair with relaxed overbought RSI
    take_profit_rsi = 58.0
    extreme_setup_rsi = 30.0
    extreme_setup_rel_vol = 1.0

    def init(self) -> None:
        self.initial_capital = float(self.equity)
        self.peak_equity = float(self.equity)
        self.cost_basis = 0.0
        self.consecutive_loss_streak = 0
        self.last_regime = ""
        self.regime_days_in_state = 0
        self.last_reason = ""

    def next(self) -> None:
        packet = self._build_packet()
        if packet is None:
            return

        action, reason = self._determine_action(packet)
        self.last_reason = reason
        self._execute_action(action, packet)

    def _current_shares(self) -> int:
        return max(int(round(float(self.position.size))), 0)

    def _price_with_slippage(self, price: float, is_buy: bool) -> float:
        slippage_factor = float(self.slippage_bps) / 10000.0
        if is_buy:
            return price * (1.0 + slippage_factor)
        return price * (1.0 - slippage_factor)

    def _build_packet(self) -> Dict[str, Any] | None:
        if len(self.data.Close) < 30:
            return None

        window = min(int(self.history_window), len(self.data.Close))
        closes = [float(v) for v in self.data.Close[-window:]]
        highs = [float(v) for v in self.data.High[-window:]]
        lows = [float(v) for v in self.data.Low[-window:]]
        volumes = [int(v) for v in self.data.Volume[-window:]]

        current_price = float(closes[-1])

        avg_vol_20 = int(pd.Series(volumes).rolling(20).mean().iloc[-1]) if len(volumes) >= 20 else int(volumes[-1])
        rel_volume = round((volumes[-1] / avg_vol_20), 2) if avg_vol_20 > 0 else 1.0

        returns = pd.Series(closes).pct_change().dropna()
        volatility = round(float(returns.std()), 5) if not returns.empty else 0.0

        rsi_14 = calculate_rsi(closes)
        atr_14 = calculate_atr(highs, lows, closes)

        tr_df = pd.DataFrame({"High": highs, "Low": lows, "Close": closes})
        high_low = tr_df["High"] - tr_df["Low"]
        high_close_prev = (tr_df["High"] - tr_df["Close"].shift(1)).abs()
        low_close_prev = (tr_df["Low"] - tr_df["Close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        hist_atr = tr.rolling(window=14).mean()

        atr_rank = hist_atr.rank(pct=True)
        atr_rank_last = atr_rank.iloc[-1] if not atr_rank.empty else float("nan")
        atr_percentile = round(float(atr_rank_last * 100), 1) if pd.notna(atr_rank_last) else 50.0

        atr_14_day_avg = float(hist_atr.iloc[-14:].mean()) if hist_atr.iloc[-14:].notna().any() else atr_14
        atr_expansion_ratio = round(float(atr_14 / atr_14_day_avg), 2) if atr_14_day_avg > 0 else 1.0

        if atr_expansion_ratio >= 2.0:
            regime = "High Volatility Regime"
            regime_multiplier = 0.75  # relaxed: was 0.5
        elif atr_expansion_ratio >= 1.5:
            regime = "Elevated Volatility"
            regime_multiplier = 1.0   # relaxed: was 0.75
        elif atr_expansion_ratio <= 0.5:
            regime = "Low Volatility"
            regime_multiplier = 1.0
        else:
            regime = "Normal"
            regime_multiplier = 1.0

        if regime == self.last_regime:
            self.regime_days_in_state += 1
        else:
            self.last_regime = regime
            self.regime_days_in_state = 1

        # Soft 0.90× damper after persistent adverse regime; entries still allowed.
        # Hard HOLD only fires via should_auto_hold() at drawdown >= 8%.
        if (
            regime in ("High Volatility Regime", "Elevated Volatility")
            and self.regime_days_in_state >= int(self.regime_caution_days)
        ):
            regime_multiplier *= 0.90

        trend = calculate_sma_trend(
            closes,
            current_price=current_price,
            short_window=50,
            long_window=200,
            bullish_label="Bullish (above {long} SMA)",
            bearish_label="Bearish (below {short} SMA)",
            neutral_label="Neutral / Sideways",
            insufficient_label="Insufficient data (need {long} days)",
        )

        shares = self._current_shares()
        position_value = shares * current_price
        total_equity = float(self.equity)
        cash = max(total_equity - position_value, 0.0)

        self.peak_equity = max(self.peak_equity, total_equity)
        current_drawdown_pct = (
            ((self.peak_equity - total_equity) / self.peak_equity) * 100.0
            if self.peak_equity > 0
            else 0.0
        )

        unrealized_pnl = (current_price - self.cost_basis) * shares if shares > 0 else 0.0
        position_notional = shares * current_price
        unrealized_pnl_pct = round((unrealized_pnl / position_notional) * 100.0, 2) if position_notional > 0 else 0.0

        dd_level, dd_name, dd_size_multiplier = get_drawdown_level(current_drawdown_pct)

        streak_state = {
            "consecutive_loss_streak": self.consecutive_loss_streak,
            "max_consecutive_losses": int(self.max_consecutive_losses),
        }
        # Bypass when streak cap is disabled (max_consecutive_losses >= 999).
        # DEFAULT_STREAK_THRESHOLDS in risk_management is independent of
        # max_consecutive_losses, so must be explicitly bypassed here.
        if int(self.max_consecutive_losses) >= 999:
            streak_multiplier = 1.0
        else:
            streak_multiplier = get_loss_streak_multiplier(streak_state)  # type: ignore[arg-type]

        risk_amount = cash * float(self.risk_per_trade_pct)
        base_shares = int(risk_amount / (atr_14 * 2.0)) if atr_14 > 0 else 0
        suggested_shares = int(base_shares * regime_multiplier * dd_size_multiplier * streak_multiplier)

        packet = {
            "portfolio": {
                "cash": round(cash, 2),
                "shares": shares,
                "cost_basis": round(self.cost_basis, 4),
                "total_equity": round(total_equity, 2),
                "current_drawdown_pct": round(current_drawdown_pct, 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "drawdown_level": dd_level,
                "drawdown_name": dd_name,
                "drawdown_size_multiplier": dd_size_multiplier,
                "consecutive_loss_streak": self.consecutive_loss_streak,
                "max_consecutive_losses": int(self.max_consecutive_losses),
                "loss_streak_multiplier": streak_multiplier,
            },
            "market_data": {
                "price": round(current_price, 2),
                "history": [round(v, 2) for v in closes[-60:]],
                "volume": int(volumes[-1]),
                "volatility": volatility,
                "rsi_14": float(rsi_14),
                "atr_14": float(atr_14),
                "atr_percentile": float(atr_percentile),
                "atr_expansion_ratio": float(atr_expansion_ratio),
                "market_regime": regime,
                "regime_multiplier": regime_multiplier,
                "regime_days_in_state": self.regime_days_in_state,
                "regime_changed_today": self.regime_days_in_state == 1,
                "stop_loss_suggestion": round(current_price - (atr_14 * 2.0), 2),
                "take_profit_suggestion": round(current_price + (atr_14 * 3.0), 2),
                "suggested_position_size": max(suggested_shares, 0),
                "sma_50": trend.get("sma_short"),
                "sma_200": trend.get("sma_long"),
                "price_above_200_sma": bool(trend.get("price_above_long_sma", False)),
                "trend_label": str(trend.get("trend_label", "Neutral / Sideways")),
                "latest_volume": int(volumes[-1]),
                "avg_volume_20d": int(avg_vol_20),
                "relative_volume": rel_volume,
            },
            "constraints": {
                "max_position_size_pct": float(self.max_position_size_pct),
                "max_drawdown_pct": float(self.max_drawdown_pct),
                "risk_per_trade_pct": float(self.risk_per_trade_pct),
                "min_atr": float(self.min_atr),
                "max_atr": float(self.max_atr),
            },
        }

        return packet

    def _determine_action(self, packet: Dict[str, Any]) -> Tuple[str, str]:
        md = packet["market_data"]
        port = packet["portfolio"]

        hold_thresholds = {
            "neutral_rsi_low": float(self.neutral_rsi_low),
            "neutral_rsi_high": float(self.neutral_rsi_high),
            "neutral_rel_volume_max": float(self.neutral_rel_volume_max),
            "bullish_hold_rsi": float(self.bullish_hold_rsi),
            "bearish_hold_rsi": float(self.bearish_hold_rsi),
        }
        auto_hold, hold_reason = should_auto_hold(packet, thresholds=hold_thresholds)
        if auto_hold:
            return "HOLD", hold_reason

        trend_label = str(md["trend_label"])
        rsi = float(md["rsi_14"])
        rel_volume = float(md["relative_volume"])
        atr_14 = float(md["atr_14"])
        shares = int(port["shares"])
        drawdown_level = int(port["drawdown_level"])

        if "Insufficient data" in trend_label:
            return "HOLD", "Need 200 bars before trend-aware trading"

        if shares > 0:
            if "Bearish" in trend_label and rsi >= float(self.sell_bearish_rsi):
                return "SELL", "Bearish trend with elevated RSI"
            if rsi >= float(self.sell_overbought_rsi):
                return "SELL", "RSI overbought"
            if (
                float(port["unrealized_pnl_pct"]) >= float(self.take_profit_pnl_pct)
                and rsi >= float(self.take_profit_rsi)
            ):
                return "SELL", "Taking partial profits in strength"

        if drawdown_level >= 2:
            return "HOLD", "Drawdown protection active"

        if atr_14 < float(self.min_atr) or atr_14 > float(self.max_atr):
            return "HOLD", "ATR outside trade bounds"

        if float(port["loss_streak_multiplier"]) <= 0.0:
            return "HOLD", "Loss streak protection active"

        bullish = "Bullish" in trend_label or bool(md.get("price_above_200_sma", False))
        extreme_setup = (
            rsi < float(self.extreme_setup_rsi)
            and rel_volume > float(self.extreme_setup_rel_vol)
        )

        if bullish and rsi <= float(self.buy_pullback_rsi):
            return "BUY", "Bullish trend pullback setup"

        if extreme_setup and "Bearish" not in trend_label:
            return "BUY", "Extreme RSI pullback with conviction volume"

        return "HOLD", "No deterministic edge"

    def _execute_action(self, action: str, packet: Dict[str, Any]) -> None:
        md = packet["market_data"]
        constraints = packet["constraints"]
        port = packet["portfolio"]

        atr_14 = float(md["atr_14"])
        if action in {"BUY", "SELL"}:
            if not (float(constraints["min_atr"]) <= atr_14 <= float(constraints["max_atr"])):
                return
            max_dd_pct = float(constraints["max_drawdown_pct"]) * 100.0
            if float(port["current_drawdown_pct"]) > max_dd_pct:
                return

        if action == "BUY":
            self._execute_buy(packet)
        elif action == "SELL":
            self._execute_sell(packet)

    def _execute_buy(self, packet: Dict[str, Any]) -> None:
        md = packet["market_data"]
        port = packet["portfolio"]

        price = float(md["price"])
        suggested_shares = int(md["suggested_position_size"])
        cash = float(port["cash"])
        shares = int(port["shares"])
        total_equity = float(port["total_equity"])

        guard_state = {
            "consecutive_loss_streak": self.consecutive_loss_streak,
            "max_consecutive_losses": int(self.max_consecutive_losses),
        }
        allowed, _ = can_open_new_position(
            guard_state,  # type: ignore[arg-type]
            float(port["current_drawdown_pct"]),
            max_drawdown_pct=float(self.max_drawdown_pct) * 100.0,
        )
        if not allowed:
            return

        max_affordable = int(cash // price) if price > 0 else 0
        max_position_value = total_equity * float(self.max_position_size_pct)
        current_position_value = shares * price
        available_position_capacity = max(max_position_value - current_position_value, 0.0)
        max_by_position_limit = int(available_position_capacity // price) if price > 0 else 0

        qty = min(suggested_shares, max_affordable, max_by_position_limit)
        # Probe floor: multipliers may round a genuine BUY signal down to 0.
        # Enforce 1-share minimum when base signal was positive and capacity allows.
        if qty <= 0 and suggested_shares > 0 and max_affordable >= 1 and max_by_position_limit >= 1:
            qty = 1
        if qty <= 0:
            return

        self.buy(size=qty)

        fill_price = self._price_with_slippage(price, is_buy=True)
        new_shares = shares + qty
        if new_shares > 0:
            self.cost_basis = ((shares * self.cost_basis) + (qty * fill_price)) / new_shares

    def _execute_sell(self, packet: Dict[str, Any]) -> None:
        md = packet["market_data"]
        shares = self._current_shares()
        if shares <= 0:
            return

        price = float(md["price"])
        trend_label = str(md["trend_label"])

        position_value = shares * price
        unrealized_pnl = (price - self.cost_basis) * shares
        unrealized_pnl_pct = (unrealized_pnl / position_value) * 100.0 if position_value > 0 else 0.0

        if unrealized_pnl_pct < 8.0:
            sell_pct = 1.0
        elif unrealized_pnl_pct < 15.0:
            sell_pct = 0.30
        elif unrealized_pnl_pct < 25.0:
            sell_pct = 0.40
        else:
            sell_pct = 0.60

        if "Bearish" in trend_label:
            sell_pct = 1.0

        qty = int(shares * sell_pct)
        if qty < 1:
            qty = shares
        qty = min(qty, shares)
        if qty <= 0:
            return

        self.sell(size=qty)

        fill_price = self._price_with_slippage(price, is_buy=False)
        realized_pnl = (fill_price - self.cost_basis) * qty
        remaining_shares = shares - qty

        if remaining_shares <= 0:
            self.cost_basis = 0.0

        if realized_pnl > 0:
            self.consecutive_loss_streak = 0
        else:
            self.consecutive_loss_streak += 1
