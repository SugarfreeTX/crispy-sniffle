import yfinance as yf
import pandas as pd
import json
import argparse
from datetime import datetime, timedelta
import requests
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, cast
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# NOTE: No pandas_market_calendars — crypto (XRP-USD) trades 24/7; NYSE calendar checks do not apply.

# ── sys.path patch ─────────────────────────────────────────────────────────────
# When invoked directly (`python crypto_xrp/complete_daily_loop.py`), Python adds
# the script's directory to sys.path, not the repo root. Insert the repo root so
# `shared.*` imports resolve correctly regardless of invocation method.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── Shared infrastructure ──────────────────────────────────────────────────────
from shared.types import PortfolioDict, MarketDataDict, PacketDict, TradeRecordDict
from shared.portfolio_state import load_portfolio_state, save_portfolio_state
from shared.indicators import calculate_rsi, calculate_atr, calculate_sma_trend
from shared.risk_management import (
    get_drawdown_level,
    get_loss_streak_multiplier,
    can_open_new_position,
    apply_all_risk_multipliers,
)
from shared.logging import log_trade, send_email_summary
from shared.grok_decision import query_grok, parse_action

# NOTE: In this script, PortfolioDict["shares"] represents integer quantity of XRP tokens,
# not equity shares. The field name is inherited from shared/types.py.

"""
XRP Crypto — Daily Trading Loop
────────────────────────────────
How It Works:
1. Loads current portfolio from saved state file (shared/portfolio_state.json)
2. Fetches XRP-USD data from Yahoo Finance (1 year daily candles)
3. Computes indicators: RSI(14), ATR(14), ATR percentile, SMAs, regime, volume
4. Applies local risk filters (auto-hold, drawdown, loss streak)
5. Sends data packet to Grok for BUY / SELL / HOLD decision
6. Executes decision: updates portfolio state + logs trade
7. Sends daily email summary
"""

# ── Path setup ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent        # crypto_xrp/
REPO_ROOT = BASE_DIR.parent                       # repo root
LOG_FILE = BASE_DIR / "trading_log.txt"
PORTFOLIO_FILE = REPO_ROOT / "shared" / "portfolio_state.json"
TRADE_HISTORY_FILE = BASE_DIR / "trade_history.json"
ENV_FILE = REPO_ROOT / ".env"
FALLBACK_ENV_FILE = BASE_DIR / ".env"

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ── Environment loading ────────────────────────────────────────────────────────

def load_env_with_fallback() -> Optional[Path]:
    """
    Load environment variables from the repo root .env, falling back to crypto_xrp/.env.

    Returns:
        Path to the .env file that was loaded, or None if neither was found.
    """
    if ENV_FILE.exists():
        load_dotenv(dotenv_path=ENV_FILE)
        return ENV_FILE

    if FALLBACK_ENV_FILE.exists():
        load_dotenv(dotenv_path=FALLBACK_ENV_FILE)
        logger.warning(
            "Root .env not found at %s; using fallback %s",
            ENV_FILE,
            FALLBACK_ENV_FILE,
        )
        return FALLBACK_ENV_FILE

    load_dotenv()
    logger.warning(
        "No .env file found at %s or %s; relying on process environment",
        ENV_FILE,
        FALLBACK_ENV_FILE,
    )
    return None


# ── Signal handling ────────────────────────────────────────────────────────────

def _install_signal_logging() -> None:
    """
    Install handlers for SIGINT, SIGTERM, and SIGHUP that log the signal before
    allowing default termination behavior. Useful for diagnosing launchd/daemon exits.
    """

    def _handler(signum: int, _frame) -> None:
        try:
            sig_name = signal.Signals(signum).name
        except Exception:
            sig_name = str(signum)

        logger.warning(
            "Signal received: %s (%s) | pid=%s ppid=%s | TERM_PROGRAM=%s",
            signum,
            sig_name,
            os.getpid(),
            os.getppid(),
            os.getenv("TERM_PROGRAM"),
        )

        # SIGINT → raise KeyboardInterrupt so main() can catch it cleanly.
        # SIGTERM / SIGHUP → log and let the default action terminate the process.
        if signum == signal.SIGINT:
            raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        if sig is not None:
            try:
                signal.signal(sig, _handler)
            except Exception:
                continue


_install_signal_logging()


# ── Portfolio metrics ──────────────────────────────────────────────────────────

def calculate_portfolio_metrics(portfolio: PortfolioDict, current_price: float) -> Dict[str, Any]:
    """
    Calculate comprehensive portfolio metrics including unrealized P&L and drawdown.

    Note: For XRP, portfolio["shares"] represents integer XRP token quantity,
    not equity shares. The field name is inherited from shared/types.py.

    Args:
        portfolio: Current PortfolioDict state loaded from portfolio_state.json.
        current_price: Latest XRP-USD close price in USD.

    Returns:
        Dict with keys: total_equity, position_value, unrealized_pnl, total_pnl,
        total_return_pct, peak_value, current_drawdown_pct.
    """
    cash: float = portfolio["cash"]
    shares: int = portfolio["shares"]
    cost_basis: float = portfolio.get("cost_basis", 0.0)
    initial_capital: float = portfolio.get("initial_capital", 100000.0)
    peak_value: float = portfolio.get("peak_value", initial_capital)

    position_value = shares * current_price
    total_equity = cash + position_value
    unrealized_pnl = (current_price - cost_basis) * shares if shares > 0 else 0.0
    total_pnl = total_equity - initial_capital
    total_return_pct = ((total_equity - initial_capital) / initial_capital) * 100

    new_peak = max(peak_value, total_equity)
    current_drawdown = ((new_peak - total_equity) / new_peak) * 100 if new_peak > 0 else 0.0

    return {
        "total_equity": round(total_equity, 2),
        "position_value": round(position_value, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_return_pct, 2),
        "peak_value": round(new_peak, 2),
        "current_drawdown_pct": round(current_drawdown, 2),
    }


# ── Market data fetch ──────────────────────────────────────────────────────────

def fetch_xrp_daily() -> Optional[PacketDict]:
    """
    Fetch XRP-USD daily market data from Yahoo Finance and build a full trading
    packet with indicators, portfolio state, and constraints.

    Crypto-specific notes:
    - No NYSE calendar checks — XRP trades 24/7.
    - Data freshness check accepts today or yesterday (daily bars may settle slowly).
    - 'shares' in PortfolioDict = integer XRP token quantity.
    - ATR constraints use XRP-appropriate dollar thresholds (not equity-scale values).

    Returns:
        A PacketDict-compatible dict ready for query_grok(), or None on failure.
    """
    logger.info("Starting XRP data fetch...")

    try:
        portfolio = load_portfolio_state(PORTFOLIO_FILE, ENV_FILE)

        ticker = yf.Ticker("XRP-USD")
        logger.info("Fetching XRP-USD market data from Yahoo Finance (1y daily)...")
        hist = ticker.history(period="1y", interval="1d")

        if hist.empty:
            logger.error("No historical data returned from Yahoo Finance for XRP-USD.")
            return None

        # ── Data freshness check ───────────────────────────────────────────────
        # Crypto is 24/7; accept today or yesterday. Anything older signals a data issue.
        last_bar_date = hist.index[-1].date()
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)

        if last_bar_date < yesterday:
            logger.warning(
                f"Stale data detected: Last bar is {last_bar_date} "
                f"(expected {yesterday} or {today}). Aborting."
            )
            return None

        logger.info(f"Data fresh: Last bar {last_bar_date}")

        latest = hist.iloc[-1]
        current_price = round(float(latest["Close"]), 6)

        # ── Volume ────────────────────────────────────────────────────────────
        if len(hist) >= 20:
            avg_vol_20 = int(hist["Volume"].rolling(20).mean().iloc[-1])
            rel_volume = round(latest["Volume"] / avg_vol_20, 2) if avg_vol_20 > 0 else 1.0
        else:
            avg_vol_20 = int(latest["Volume"])
            rel_volume = 1.0

        # ── Close history (last 60 bars for Grok context) ─────────────────────
        n_days = min(60, len(hist))
        close_history = [round(p, 6) for p in hist["Close"].tolist()[-n_days:]]

        # ── Volatility (std dev of daily returns) ─────────────────────────────
        returns = hist["Close"].pct_change().dropna()
        volatility = round(float(returns.std()), 5)

        # ── RSI(14) ───────────────────────────────────────────────────────────
        rsi_14 = calculate_rsi(hist["Close"].tolist())

        # ── ATR(14) via shared.indicators (takes lists, not DataFrame) ─────────
        atr_14 = calculate_atr(
            hist["High"].tolist(),
            hist["Low"].tolist(),
            hist["Close"].tolist(),
        )

        if atr_14 == 0.0:
            logger.warning("ATR is 0.0 — position sizing will be 0. Check data quality.")

        # ── ATR percentile and regime (inline, full history) ──────────────────
        high_low = hist["High"] - hist["Low"]
        high_close_prev = abs(hist["High"] - hist["Close"].shift(1))
        low_close_prev = abs(hist["Low"] - hist["Close"].shift(1))
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        hist_atr = tr.rolling(window=14).mean()

        atr_percentile = round(float(hist_atr.rank(pct=True).iloc[-1] * 100), 1)
        atr_14_day_avg = hist_atr.iloc[-14:].mean()
        atr_expansion_ratio = round(atr_14 / atr_14_day_avg, 2) if atr_14_day_avg > 0 else 1.0

        # ── Market regime ──────────────────────────────────────────────────────
        if atr_expansion_ratio >= 2.0:
            regime = "High Volatility Regime"
            regime_multiplier = 0.5
        elif atr_expansion_ratio >= 1.5:
            regime = "Elevated Volatility"
            regime_multiplier = 0.75
        elif atr_expansion_ratio <= 0.5:
            regime = "Low Volatility"
            regime_multiplier = 1.0
        else:
            regime = "Normal"
            regime_multiplier = 1.0

        # ── Regime persistence ─────────────────────────────────────────────────
        last_regime = portfolio.get("last_regime", "Normal")
        regime_days = portfolio.get("regime_days_in_state", 1)

        if regime == last_regime:
            regime_days += 1
        else:
            regime_days = 1
            logger.info(f"Regime changed: {last_regime} → {regime}")

        # ── SMA trend (50/200-day) via shared.indicators ───────────────────────
        sma_result = calculate_sma_trend(
            hist["Close"].tolist(),
            current_price=current_price,
            short_window=50,
            long_window=200,
        )
        sma_50: Optional[float] = sma_result["sma_short"]
        sma_200: Optional[float] = sma_result["sma_long"]
        trend_label: str = sma_result["trend_label"]
        price_above_200: bool = sma_result["price_above_long_sma"]

        # ── ATR-based stop-loss / take-profit ──────────────────────────────────
        stop_loss = round(current_price - (atr_14 * 2), 6)
        take_profit = round(current_price + (atr_14 * 3), 6)

        # ── Position sizing (risk 2% of cash, 2x ATR stop) ────────────────────
        # 'shares' here = integer XRP token quantity
        risk_per_trade_pct = 0.02
        risk_amount = portfolio["cash"] * risk_per_trade_pct
        base_tokens = int(risk_amount / (atr_14 * 2)) if atr_14 > 0 else 0
        suggested_tokens = int(base_tokens * regime_multiplier)

        # ── Portfolio metrics and drawdown ─────────────────────────────────────
        metrics = calculate_portfolio_metrics(portfolio, current_price)
        drawdown_pct = metrics["current_drawdown_pct"]
        dd_level, dd_name, dd_size_multiplier = get_drawdown_level(drawdown_pct)

        unrealized_pnl_pct = (
            round(metrics["unrealized_pnl"] / (portfolio["shares"] * current_price) * 100, 2)
            if portfolio["shares"] > 0
            else 0.0
        )

        # ── Build trading packet ───────────────────────────────────────────────
        packet: PacketDict = {  # type: ignore[assignment]
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": "XRP-USD",
            "portfolio": {
                "cash": round(portfolio["cash"], 2),
                "shares": portfolio["shares"],          # XRP token quantity (int)
                "cost_basis": round(portfolio["cost_basis"], 6),
                "total_equity": metrics["total_equity"],
                "unrealized_pnl": metrics["unrealized_pnl"],
                "total_return_pct": metrics["total_return_pct"],
                "current_drawdown_pct": metrics["current_drawdown_pct"],
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "drawdown_level": dd_level,
                "drawdown_name": dd_name,
                "drawdown_size_multiplier": dd_size_multiplier,
                "consecutive_loss_streak": portfolio.get("consecutive_loss_streak", 0),
                "max_consecutive_losses": portfolio.get("max_consecutive_losses", 5),
                "loss_streak_multiplier": get_loss_streak_multiplier(portfolio),
            },
            "market_data": {
                "price": current_price,
                "history": close_history,
                "volume": int(latest["Volume"]),
                "volatility": volatility,
                "rsi_14": rsi_14,
                "atr_14": atr_14,
                "atr_percentile": atr_percentile,
                "atr_expansion_ratio": atr_expansion_ratio,
                "market_regime": regime,
                "regime_multiplier": regime_multiplier,
                "regime_days_in_state": regime_days,
                "regime_changed_today": (regime_days == 1),
                "stop_loss_suggestion": stop_loss,
                "take_profit_suggestion": take_profit,
                "suggested_position_size": suggested_tokens,
                "sma_50": sma_50,
                "sma_200": sma_200,
                "price_above_200_sma": price_above_200,
                "trend_label": trend_label,
                "latest_volume": int(latest["Volume"]),
                "avg_volume_20d": avg_vol_20,
                "relative_volume": rel_volume,
            },
            "constraints": {
                "max_position_size_pct": 0.20,
                "max_drawdown_pct": 0.10,
                "risk_per_trade_pct": 0.02,
                # ATR bounds are XRP-appropriate (price ~$0.50–$5.00 range)
                "min_atr": 0.005,
                "max_atr": 1.50,
            },
        }

        logger.info(
            f"Successfully fetched XRP-USD data. "
            f"Price: ${current_price:.4f}, RSI: {rsi_14}, ATR: {atr_14:.4f}, "
            f"Regime: {regime}, Suggested tokens: {suggested_tokens}, "
            f"Trend: {trend_label}, Rel Vol: {rel_volume:.2f} "
            f"(Latest Vol: {int(latest['Volume'])}, Avg 20d: {avg_vol_20})"
        )
        return packet

    except Exception as e:
        logger.error(f"Error fetching XRP-USD data: {e}")
        return None


# ── Local safety guard (pre-Grok) ─────────────────────────────────────────────

def should_auto_hold(packet: PacketDict) -> Tuple[bool, str]:
    """
    Quick local check before calling Grok. If conditions strongly favour HOLD,
    skip the API call entirely and return (True, reason).

    Crypto-specific thresholds mirror the MSFT version; RSI neutral band and
    regime rules are identical since the indicators are asset-agnostic.

    Args:
        packet: The trading packet built by fetch_xrp_daily().

    Returns:
        Tuple of (should_hold: bool, reason: str).
    """
    if packet is None:
        return False, "No packet provided"

    md = packet.get("market_data", {})
    port = packet.get("portfolio", {})

    rsi = md.get("rsi_14", 50.0)
    regime = md.get("market_regime", "Normal")
    rel_vol = md.get("relative_volume", 1.0)
    drawdown_pct = port.get("current_drawdown_pct", 0.0)
    trend_label = md.get("trend_label", "Neutral")
    regime_changed = md.get("regime_changed_today", False)

    # Drawdown-based rules (highest priority)
    if drawdown_pct >= 10.0:
        return True, f"Emergency drawdown ({drawdown_pct:.1f}%) — all new risk blocked"

    if drawdown_pct >= 8.0:
        return True, f"Restricted drawdown ({drawdown_pct:.1f}%) — HOLD or exit only, no new BUY"

    if drawdown_pct >= 5.0:
        if "Bullish" not in trend_label or rsi > 35:
            return True, f"Caution drawdown ({drawdown_pct:.1f}%) — not an extreme oversold setup"

    # Neutral zone — most common auto-hold case
    if (35 <= rsi <= 65 and
            regime == "Normal" and
            rel_vol <= 1.3 and
            not regime_changed):
        return True, "Neutral RSI, Normal regime, low relative volume, no regime change"

    # Approaching max drawdown
    if drawdown_pct >= 6.0:
        return True, f"Drawdown at {drawdown_pct:.2f}% — approaching max drawdown limit"

    # Bullish trend but not oversold enough for entry
    if "Bullish" in trend_label and rsi >= 58:
        return True, f"Bullish trend but RSI {rsi} not low enough for entry"

    # Bearish trend but not overbought enough for exit
    if "Bearish" in trend_label and rsi <= 42:
        return True, f"Bearish trend but RSI {rsi} not high enough for exit"

    return False, "No auto-hold rule triggered"


# ── Grok decision ──────────────────────────────────────────────────────────────

def get_grok_decision(packet: PacketDict) -> Tuple[str, str]:
    """
    Run the auto-hold guard, then call Grok if needed.

    Steps:
    1. Check should_auto_hold() — return HOLD immediately if triggered.
    2. Validate GROK_API_KEY is present before hitting the network.
    3. Call shared query_grok() → parse_action().
    4. Default to HOLD on any failure.

    Args:
        packet: Trading packet from fetch_xrp_daily().

    Returns:
        Tuple of (action: str, reason: str) where action is BUY / SELL / HOLD.
    """
    auto_hold, auto_reason = should_auto_hold(packet)
    if auto_hold:
        logger.info(f"Auto-HOLD triggered (skipping Grok): {auto_reason}")
        return "HOLD", auto_reason

    api_key = os.getenv("GROK_API_KEY")
    if not api_key:
        logger.error("GROK_API_KEY not set — defaulting to HOLD")
        return "HOLD", "GROK_API_KEY missing"

    response = query_grok(packet, api_key)
    action, reason = parse_action(response)
    logger.info(f"Grok decision: {action} | {reason}")
    return action, reason


# ── Daily trade idempotency guard ──────────────────────────────────────────────

def has_executed_trade_today() -> bool:
    """
    Return True if a live BUY or SELL was already recorded today in trade_history.json.

    Prevents duplicate execution if the script runs more than once in a calendar day
    (e.g. launchd retries, manual re-runs).

    Returns:
        True if a BUY/SELL_PARTIAL/SELL_FULL entry for today exists; False otherwise.
    """
    try:
        if not TRADE_HISTORY_FILE.exists():
            return False

        with open(TRADE_HISTORY_FILE, "r") as f:
            trade_history = json.load(f)

        today_str = datetime.now().strftime("%Y-%m-%d")
        executed_actions = {"BUY", "SELL_PARTIAL", "SELL_FULL"}

        for trade in reversed(trade_history):
            timestamp = str(trade.get("timestamp", ""))
            action = str(trade.get("action", "")).upper()
            if timestamp.startswith(today_str) and action in executed_actions:
                return True
        return False
    except Exception as e:
        logger.warning(f"Unable to verify daily trade idempotency: {e}")
        return False


# ── Alpaca order placement ─────────────────────────────────────────────────────

def place_alpaca_order(
    api_key: str,
    secret_key: str,
    symbol: str,
    qty: int,
    side: str,
) -> Optional[Dict[str, Any]]:
    """
    Submit a market order to the Alpaca paper trading API.

    Crypto note: symbol must use the slash format (e.g. "XRP/USD").
    Crypto orders require time_in_force="gtc" — "day" is not supported.

    Args:
        api_key: Alpaca API key ID.
        secret_key: Alpaca secret key.
        symbol: Alpaca crypto symbol (e.g. "XRP/USD").
        qty: Integer token quantity.
        side: "buy" or "sell" (case-insensitive).

    Returns:
        Alpaca order response dict, or None on failure.
    """
    url = "https://paper-api.alpaca.markets/v2/orders"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
        "Content-Type": "application/json",
    }
    order = {
        "symbol": symbol,
        "qty": str(qty),          # Alpaca crypto API expects qty as a string
        "side": side.lower(),
        "type": "market",
        "time_in_force": "gtc",   # Crypto requires "gtc", not "day"
    }
    logger.info(f"Placing {side.upper()} order for {qty} {symbol} via Alpaca")

    try:
        r = requests.post(url, json=order, headers=headers, timeout=15)
        r.raise_for_status()
        result = r.json()
        logger.info(f"Order placed successfully. Order ID: {result.get('id', 'Unknown')}")
        return result
    except requests.exceptions.Timeout:
        logger.error("Alpaca API request timed out")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Alpaca API request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Alpaca response as JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error placing Alpaca order: {e}")
        return None


# ── Trade execution ────────────────────────────────────────────────────────────

def execute_trade(
    action: str,
    reason: str,
    packet: PacketDict,
    api_key: Optional[str],
    secret_key: Optional[str],
    dry_run: bool = False,
) -> bool:
    """
    Execute a BUY, SELL (partial or full), or HOLD decision for XRP-USD.

    Guards applied in order:
    1. ATR guardrails (execution-time, not just in the Grok prompt).
    2. Max drawdown hard block.
    3. Daily idempotency — one live BUY/SELL per calendar day.
    4. Loss streak / can_open_new_position check (BUY only).

    Portfolio state is reloaded from disk before writing to preserve fields that
    are not present in the packet snapshot. On any critical failure the function
    logs and returns False without retrying.

    Crypto notes:
    - portfolio["shares"] = integer XRP token quantity (not equity shares).
    - cost_basis is price-per-token (6 decimal precision).
    - Alpaca symbol must be "XRP/USD" (slash format required for crypto orders).
    - Partial sell tiers mirror the MSFT script for consistency.

    Args:
        action: "BUY", "SELL", or "HOLD".
        reason: Short explanation from Grok or a local safety guard.
        packet: Full trading packet produced by fetch_xrp_daily().
        api_key: Alpaca API key (may be None in dry-run mode).
        secret_key: Alpaca secret key (may be None in dry-run mode).
        dry_run: If True, simulate execution without touching Alpaca or saving state.

    Returns:
        True if the intended action was completed (or dry-run simulated); False on error.
    """
    price: float = packet["market_data"]["price"]
    suggested_tokens: int = packet["market_data"]["suggested_position_size"]
    portfolio = packet["portfolio"]
    constraints = packet["constraints"]

    cash: float = portfolio["cash"]
    shares: int = portfolio["shares"]           # XRP token quantity
    cost_basis: float = portfolio.get("cost_basis", 0.0)
    current_drawdown_pct: float = portfolio.get("current_drawdown_pct", 0.0)
    total_equity: float = portfolio.get("total_equity", cash)

    logger.info(
        "Executing action: %s | portfolio: $%.2f cash, %d XRP tokens, "
        "equity: $%.2f, drawdown: %.2f%%",
        action, cash, shares, total_equity, current_drawdown_pct,
    )

    # ── ATR guardrails ─────────────────────────────────────────────────────────
    atr_14: float = packet.get("market_data", {}).get("atr_14", 0.0)
    min_atr: float = constraints.get("min_atr", 0.0)
    max_atr: float = constraints.get("max_atr", float("inf"))
    if action in {"BUY", "SELL"} and not (min_atr <= atr_14 <= max_atr):
        logger.warning(
            "TRADE BLOCKED: ATR (%.5f) outside allowed range [%.5f, %.5f]",
            atr_14, min_atr, max_atr,
        )
        if not dry_run:
            log_trade(
                "BLOCKED_ATR", 0, price,
                f"ATR out of bounds: {atr_14}",
                total_equity,
                {"atr_14": atr_14, "min_atr": min_atr, "max_atr": max_atr, "requested_action": action},
                trade_history_file=TRADE_HISTORY_FILE,
            )
        else:
            logger.info("DRY-RUN: Would record BLOCKED_ATR event")
        return False

    # ── Max drawdown hard block ────────────────────────────────────────────────
    max_drawdown_pct: float = constraints.get("max_drawdown_pct", 0.10) * 100
    if current_drawdown_pct > max_drawdown_pct:
        logger.warning(
            "TRADE BLOCKED: Drawdown (%.2f%%) exceeds max allowed (%.2f%%)",
            current_drawdown_pct, max_drawdown_pct,
        )
        if not dry_run:
            log_trade(
                "BLOCKED_DRAWDOWN", 0, price,
                f"Drawdown too high: {current_drawdown_pct:.2f}%",
                total_equity,
                trade_history_file=TRADE_HISTORY_FILE,
            )
        else:
            logger.info("DRY-RUN: Would record BLOCKED_DRAWDOWN event")
        return False

    # ── Daily idempotency guard ────────────────────────────────────────────────
    if action in {"BUY", "SELL"} and not dry_run and has_executed_trade_today():
        logger.warning("TRADE BLOCKED: A live BUY/SELL was already executed today")
        log_trade(
            "BLOCKED_DUPLICATE", 0, price,
            "Duplicate daily trade prevented",
            total_equity,
            {"requested_action": action},
            trade_history_file=TRADE_HISTORY_FILE,
        )
        return False

    # ── BUY branch ─────────────────────────────────────────────────────────────
    if action == "BUY":
        allowed, block_reason = can_open_new_position(cast(PortfolioDict, portfolio), current_drawdown_pct)
        if not allowed:
            logger.warning("BUY BLOCKED: %s", block_reason)
            if not dry_run:
                log_trade(
                    "BLOCKED_LOSS_STREAK", 0, price, block_reason, total_equity,
                    trade_history_file=TRADE_HISTORY_FILE,
                )
            else:
                logger.info("DRY-RUN: Would block BUY — %s", block_reason)
            return False

        max_affordable: int = int(cash // price)

        # 20% of total equity position cap
        max_position_value: float = total_equity * constraints.get("max_position_size_pct", 0.20)
        current_position_value: float = shares * price
        available_capacity: float = max_position_value - current_position_value
        max_by_position_limit: int = int(available_capacity / price) if available_capacity > 0 else 0

        qty: int = min(suggested_tokens, max_affordable, max_by_position_limit) if suggested_tokens > 0 else 0

        # Loss-streak size multiplier
        streak_multiplier: float = get_loss_streak_multiplier(cast(PortfolioDict, portfolio))
        if streak_multiplier < 1.0:
            logger.info(
                "Loss streak %d → applying size multiplier %.2f",
                portfolio.get("consecutive_loss_streak", 0), streak_multiplier,
            )
        qty = int(qty * streak_multiplier)

        if qty < 1 and streak_multiplier > 0:
            qty = 1   # Minimum probe position if multiplier still allows any size

        if qty <= 0:
            logger.warning("BUY reduced to 0 tokens due to constraints or loss streak")
            if not dry_run:
                log_trade(
                    "BLOCKED_BUY", 0, price,
                    "No capacity to buy (constraints / loss streak)",
                    total_equity,
                    trade_history_file=TRADE_HISTORY_FILE,
                )
            else:
                logger.info("DRY-RUN: Would record BLOCKED_BUY event")
            return False

        logger.info(
            "Position sizing: suggested=%d, affordable=%d, cap=%d, final=%d",
            suggested_tokens, max_affordable, max_by_position_limit, qty,
        )

        if dry_run:
            execution_price = float(price)
            new_cash = cash - (qty * execution_price)
            new_shares = shares + qty
            new_cost_basis = (
                ((shares * cost_basis) + (qty * execution_price)) / new_shares
                if new_shares > 0 else 0.0
            )
            new_equity = new_cash + (new_shares * execution_price)
            logger.info(
                "DRY-RUN BUY: Would buy %d XRP at $%.6f | "
                "cash: $%.2f → tokens: %d, cost_basis: $%.6f, equity: $%.2f",
                qty, execution_price, new_cash, new_shares, new_cost_basis, new_equity,
            )
            return True

        if not api_key or not secret_key:
            logger.error("Missing Alpaca credentials for live BUY execution")
            return False

        result = place_alpaca_order(api_key, secret_key, "XRP/USD", qty, "buy")
        if result:
            fill_price_raw = result.get("filled_avg_price")
            execution_price = float(fill_price_raw) if fill_price_raw is not None else float(price)

            new_cash = cash - (qty * execution_price)
            new_shares = shares + qty
            new_cost_basis = (
                ((shares * cost_basis) + (qty * execution_price)) / new_shares
                if new_shares > 0 else 0.0
            )

            current_portfolio = load_portfolio_state(PORTFOLIO_FILE, ENV_FILE)
            updated_portfolio: PortfolioDict = {
                **current_portfolio,             # type: ignore[misc]
                "cash": round(new_cash, 2),
                "shares": new_shares,
                "cost_basis": round(new_cost_basis, 6),
            }
            new_equity = new_cash + (new_shares * execution_price)
            updated_portfolio["peak_value"] = max(
                current_portfolio.get("peak_value", new_equity), new_equity
            )

            try:
                save_portfolio_state(updated_portfolio, PORTFOLIO_FILE)
                log_trade(
                    "BUY", qty, execution_price, reason, new_equity,
                    {
                        "rsi_14": packet["market_data"]["rsi_14"],
                        "atr_14": packet["market_data"]["atr_14"],
                        "regime": packet["market_data"]["market_regime"],
                    },
                    trade_history_file=TRADE_HISTORY_FILE,
                )
                logger.info(
                    "BUY executed: %d XRP at $%.6f | cash: $%.2f, tokens: %d",
                    qty, execution_price, new_cash, new_shares,
                )
                return True
            except Exception as e:
                logger.critical(
                    "CRITICAL: BUY executed but portfolio save failed: %s | "
                    "Manual check required: bought %d XRP at $%.6f",
                    e, qty, execution_price,
                )
                return False
        else:
            logger.error("BUY order failed — Alpaca returned no result")
            return False

    # ── SELL branch ────────────────────────────────────────────────────────────
    elif action == "SELL":
        if shares <= 0:
            logger.warning("No XRP tokens to sell")
            if not dry_run:
                log_trade(
                    "BLOCKED_SELL", 0, price, "No tokens to sell", total_equity,
                    trade_history_file=TRADE_HISTORY_FILE,
                )
            else:
                logger.info("DRY-RUN: Would record BLOCKED_SELL event")
            return False

        position_value = shares * price
        unrealized_pnl = (price - cost_basis) * shares
        unrealized_pnl_pct = round((unrealized_pnl / position_value) * 100, 2) if position_value > 0 else 0.0

        # Partial sell tiers (mirror MSFT for consistency)
        if unrealized_pnl_pct < 8.0:
            sell_pct = 1.0          # Full exit if gain is small or a loss
        elif unrealized_pnl_pct < 15.0:
            sell_pct = 0.30         # 30% partial
        elif unrealized_pnl_pct < 25.0:
            sell_pct = 0.40         # 40% partial
        else:
            sell_pct = 0.60         # 60% at high gains

        # Override to full exit in Bearish trend
        trend_label: str = packet["market_data"].get("trend_label", "Unknown")
        if "Bearish" in trend_label:
            sell_pct = 1.0
            reason += " (full exit — Bearish trend)"

        qty = int(shares * sell_pct)
        if qty < 1:
            qty = shares   # Minimum full sell if fractional rounding gives 0

        logger.info(
            "SELL decision: unrealized PnL %.2f%%, trend: %s → selling %.0f%% (%d tokens)",
            unrealized_pnl_pct, trend_label, sell_pct * 100, qty,
        )

        if dry_run:
            execution_price = float(price)
            new_cash = cash + (qty * execution_price)
            realized_pnl = (execution_price - cost_basis) * qty
            remaining_shares = max(shares - qty, 0)
            new_cost_basis = cost_basis if remaining_shares > 0 else 0.0
            new_equity = new_cash + (remaining_shares * execution_price)
            logger.info(
                "DRY-RUN SELL: Would sell %d XRP at $%.6f | "
                "cash: $%.2f, remaining tokens: %d, realized PnL: $%.2f, equity: $%.2f",
                qty, execution_price, new_cash, remaining_shares, realized_pnl, new_equity,
            )
            return True

        if not api_key or not secret_key:
            logger.error("Missing Alpaca credentials for live SELL execution")
            return False

        result = place_alpaca_order(api_key, secret_key, "XRP/USD", qty, "sell")
        if result:
            fill_price_raw = result.get("filled_avg_price")
            execution_price = float(fill_price_raw) if fill_price_raw is not None else float(price)

            new_cash = cash + (qty * execution_price)
            realized_pnl = (execution_price - cost_basis) * qty
            remaining_shares = shares - qty
            new_cost_basis = cost_basis if remaining_shares > 0 else 0.0

            current_portfolio = load_portfolio_state(PORTFOLIO_FILE, ENV_FILE)
            updated_portfolio: PortfolioDict = {
                **current_portfolio,             # type: ignore[misc]
                "cash": round(new_cash, 2),
                "shares": remaining_shares,
                "cost_basis": round(new_cost_basis, 6),
            }
            new_equity = new_cash + (remaining_shares * execution_price)
            updated_portfolio["peak_value"] = max(
                current_portfolio.get("peak_value", new_equity), new_equity
            )

            # Update loss streak
            was_win: bool = realized_pnl > 0
            current_streak: int = current_portfolio.get("consecutive_loss_streak", 0)
            new_streak: int = 0 if was_win else current_streak + 1
            updated_portfolio["consecutive_loss_streak"] = new_streak

            if was_win:
                logger.info("Realized WIN → loss streak RESET to 0 (PnL: $%.2f)", realized_pnl)
            else:
                logger.info("Realized LOSS → streak now %d (PnL: $%.2f)", new_streak, realized_pnl)

            try:
                save_portfolio_state(updated_portfolio, PORTFOLIO_FILE)
                log_trade(
                    "SELL_PARTIAL" if sell_pct < 1.0 else "SELL_FULL",
                    qty,
                    execution_price,
                    f"{reason} | PnL {unrealized_pnl_pct:.2f}% | Sold {sell_pct * 100:.0f}%",
                    new_equity,
                    {
                        "realized_pnl": round(realized_pnl, 2),
                        "remaining_shares": remaining_shares,
                        "rsi_14": packet["market_data"]["rsi_14"],
                        "atr_14": packet["market_data"]["atr_14"],
                        "regime": packet["market_data"]["market_regime"],
                        "loss_streak_after": new_streak,
                        "was_win": was_win,
                    },
                    trade_history_file=TRADE_HISTORY_FILE,
                )
                logger.info(
                    "SELL executed: %d XRP (%.0f%%) at $%.6f | "
                    "realized PnL: $%.2f | remaining tokens: %d",
                    qty, sell_pct * 100, execution_price, realized_pnl, remaining_shares,
                )
                return True
            except Exception as e:
                logger.critical(
                    "CRITICAL: SELL executed but portfolio save failed: %s | "
                    "Manual check required: sold %d XRP at $%.6f",
                    e, qty, execution_price,
                )
                return False
        else:
            logger.error("SELL order failed — Alpaca returned no result")
            return False

    # ── HOLD branch ────────────────────────────────────────────────────────────
    else:
        logger.info("Holding position: %s — %s", action, reason)
        if not dry_run:
            log_trade(
                "HOLD", 0, price, reason, total_equity,
                {
                    "rsi_14": packet["market_data"]["rsi_14"],
                    "atr_14": packet["market_data"]["atr_14"],
                    "regime": packet["market_data"]["market_regime"],
                },
                trade_history_file=TRADE_HISTORY_FILE,
            )
        else:
            logger.info("DRY-RUN HOLD: No trade, no portfolio mutation")
        return True


# ── Main entry point ───────────────────────────────────────────────────────────

def main(dry_run: bool = False) -> None:
    """
    Main XRP trading loop.

    Steps:
    1. Load environment and validate required API keys.
    2. Fetch XRP-USD market data and build the trading packet.
    3. Run the auto-hold guard + Grok decision via get_grok_decision().
    4. Execute the trade via execute_trade().
    5. Persist regime state to portfolio file.

    Crypto note: no market-open or NYSE calendar check — XRP trades 24/7.
    Email summary is intentionally omitted; this script runs every 4 hours and
    email volume would be excessive.

    Args:
        dry_run: If True, simulate execution without placing orders or mutating state.
    """
    packet = None
    action: Optional[str] = None
    reason: Optional[str] = None

    logger.info("=== Starting XRP Daily Trading Loop ===")
    if dry_run:
        logger.info("DRY-RUN MODE: No orders will be placed; state files will not be modified.")

    # ── Environment and API key validation ────────────────────────────────────
    load_env_with_fallback()

    GROK_API_KEY = os.getenv("GROK_API_KEY")
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
    ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

    if not GROK_API_KEY:
        logger.error("GROK_API_KEY not found in environment. Check your .env file. Aborting.")
        return

    if not dry_run:
        if not ALPACA_API_KEY:
            logger.error("ALPACA_API_KEY not found in environment. Check your .env file. Aborting.")
            return
        if not ALPACA_SECRET_KEY:
            logger.error("ALPACA_SECRET_KEY not found in environment. Check your .env file. Aborting.")
            return

    try:
        # ── Step 1: Fetch market data ──────────────────────────────────────────
        packet = fetch_xrp_daily()
        if packet is None:
            logger.error("Failed to fetch XRP-USD data. Aborting trading loop.")
            return

        # ── Step 2: Get decision (auto-hold guard + Grok) ─────────────────────
        action, reason = get_grok_decision(packet)

        # ── Step 3: Execute trade ──────────────────────────────────────────────
        success = execute_trade(
            action, reason, packet,
            ALPACA_API_KEY, ALPACA_SECRET_KEY,
            dry_run=dry_run,
        )

        # ── Step 4: Persist regime state ──────────────────────────────────────
        if not dry_run:
            updated_portfolio = load_portfolio_state(PORTFOLIO_FILE, ENV_FILE)
            updated_portfolio["last_regime"] = packet["market_data"]["market_regime"]
            updated_portfolio["regime_days_in_state"] = packet["market_data"]["regime_days_in_state"]
            save_portfolio_state(updated_portfolio, PORTFOLIO_FILE)
            logger.info(
                "Regime persistence updated: %s (%d day(s))",
                packet["market_data"]["market_regime"],
                packet["market_data"]["regime_days_in_state"],
            )

        # ── Run summary line ───────────────────────────────────────────────────
        dd_pct = packet["portfolio"].get("current_drawdown_pct", 0.0)
        regime = packet["market_data"].get("market_regime", "Unknown")
        days_in_regime = packet["market_data"].get("regime_days_in_state", 0)
        logger.info(
            "Run summary | Action: %s | Drawdown: %.1f%% | Regime: %s (%d days)",
            action, dd_pct, regime, days_in_regime,
        )

        if success:
            logger.info("=== XRP trading loop completed successfully ===")
        else:
            logger.warning("=== XRP trading loop completed with a non-fatal issue ===")

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt (SIGINT) received — exiting early.")
    except Exception:
        logger.exception("Unexpected error in XRP main trading loop")

    logger.info("=== XRP trading loop finished ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XRP-USD trading loop (every 4 hours)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate without placing orders or writing state files",
    )
    args = parser.parse_args()

    TEST_DIR = BASE_DIR / "test_runs"
    TEST_DIR.mkdir(exist_ok=True)

    if args.dry_run:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        LOG_FILE = TEST_DIR / f"trading_log_dry_{ts}.txt"
        TRADE_HISTORY_FILE = TEST_DIR / f"trade_history_dry_{ts}.json"
    else:
        LOG_FILE = BASE_DIR / "trading_log.txt"
        TRADE_HISTORY_FILE = BASE_DIR / "trade_history.json"

    # Swap the file handler to point at the correct log file
    # (logging.basicConfig already ran at module import time).
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            h.close()
            root_logger.removeHandler(h)
    root_logger.addHandler(logging.FileHandler(LOG_FILE))

    logger.info("'%s': Logging to %s", "DRY-RUN" if args.dry_run else "LIVE", LOG_FILE)

    main(dry_run=args.dry_run)
