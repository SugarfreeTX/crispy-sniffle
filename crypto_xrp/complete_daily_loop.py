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
from typing import Optional, Dict, Any, Tuple
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# NOTE: No pandas_market_calendars — crypto (XRP-USD) trades 24/7; NYSE calendar checks do not apply.

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
1. Loads current portfolio from saved state file (crypto_xrp/portfolio_state.json)
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
PORTFOLIO_FILE = BASE_DIR / "portfolio_state.json"
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
