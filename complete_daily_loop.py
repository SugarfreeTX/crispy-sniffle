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
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import pandas_market_calendars as mcal


"""
-How It Works-
1. Loads current portfolio from saved state file
2. Fetches MSFT data with error recovery
3. Sends to Grok for analysis with timeout protection
4. Executes trades with full error handling
5. Updates portfolio state automatically after successful trades
6. Logs everything for full audit trail
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def _install_signal_logging() -> None:
    """Log termination-style signals so we can diagnose unexpected exits.

    VS Code and other runners may send signals (SIGINT/SIGTERM) when stopping or
    restarting a run session.
    """

    def _handler(signum: int, _frame) -> None:
        try:
            sig_name = signal.Signals(signum).name
        except Exception:
            sig_name = str(signum)

        logger.warning(
            "Signal received: %s (%s) | pid=%s ppid=%s | TERM_PROGRAM=%s VSCODE_PID=%s",
            signum,
            sig_name,
            os.getpid(),
            os.getppid(),
            os.getenv("TERM_PROGRAM"),
            os.getenv("VSCODE_PID"),
        )

    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None), getattr(signal, "SIGHUP", None)):
        if sig is None:
            continue
        try:
            signal.signal(sig, _handler)
        except Exception:
            # Some signals can't be trapped in some contexts.
            continue


_install_signal_logging()

# Portfolio state file
PORTFOLIO_FILE = "portfolio_state.json"
TRADE_HISTORY_FILE = "trade_history.json"


def previous_trading_day(reference_date):
    """Return the previous weekday (Mon-Fri) for a given date."""
    prev_day = reference_date - timedelta(days=1)
    while prev_day.weekday() >= 5:  # Saturday/Sunday
        prev_day -= timedelta(days=1)
    return prev_day

def calculate_rsi(prices: list, period: int = 14) -> float:
    """Calculate RSI from price history, returns rounded value"""
    try:
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI if insufficient data
        
        price_series = pd.Series(prices)
        delta = price_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Avoid division by zero
        rs = gain / loss.replace(0, 0.0001)
        rsi = 100 - (100 / (1 + rs))
        
        return round(rsi.iloc[-1], 1)  # Return most recent RSI, rounded to 1 decimal
    
    except Exception as e:
        logger.warning(f"Error calculating RSI: {e}. Returning neutral RSI of 50.0")
        return 50.0  # Return neutral RSI if calculation fails

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range (ATR) from OHLC data"""
    try:
        if len(df) < period + 1:
            return 0.0  # Return 0 if insufficient data
        
        # Calculate the three potential ranges for True Range
        high_low = df['High'] - df['Low']
        high_close_prev = abs(df['High'] - df['Close'].shift(1))
        low_close_prev = abs(df['Low'] - df['Close'].shift(1))
        
        # True Range is the maximum of those three
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Calculate ATR (Simple Moving Average of True Range)
        atr = tr.rolling(window=period).mean()
        
        return round(atr.iloc[-1], 2)  # Return most recent ATR, rounded to 2 decimals
    
    except Exception as e:
        logger.warning(f"Error calculating ATR: {e}. Returning 0.0")
        return 0.0  # Return 0 if calculation fails

def load_portfolio_state() -> Dict[str, Any]:
    """Load portfolio state from file, return default if file doesn't exist"""
    default_portfolio = {
        "cash": 100000.00,
        "shares": 0,
        "cost_basis": 0.00,
        "initial_capital": 100000.00,
        "peak_value": 100000.00,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        if os.path.exists(PORTFOLIO_FILE):
            with open(PORTFOLIO_FILE, "r") as f:
                portfolio = json.load(f)
                logger.info(f"Loaded portfolio state: {portfolio}")
                return portfolio
        else:
            logger.info("No existing portfolio file found, using default portfolio")
            save_portfolio_state(default_portfolio)
            return default_portfolio
    except Exception as e:
        logger.error(f"Error loading portfolio state: {e}. Using default portfolio.")
        return default_portfolio

def save_portfolio_state(portfolio: Dict[str, Any]) -> None:
    """Save portfolio state to file with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            portfolio["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Write to temp file first, then rename (atomic operation)
            temp_file = f"{PORTFOLIO_FILE}.tmp"
            with open(temp_file, "w") as f:
                json.dump(portfolio, f, indent=2)
            os.replace(temp_file, PORTFOLIO_FILE)
            logger.info(f"Saved portfolio state: {portfolio}")
            return
        except Exception as e:
            logger.error(f"Error saving portfolio state (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                logger.critical("CRITICAL: Failed to save portfolio state after all retries!")
                raise  # Re-raise on final attempt

def log_trade(action: str, qty: int, price: float, reason: str, portfolio_value: float, metrics: Optional[Dict[str, Any]] = None) -> None:
    """Log trade to trade history file"""
    try:
        # Load existing trade history
        trade_history = []
        if os.path.exists(TRADE_HISTORY_FILE):
            with open(TRADE_HISTORY_FILE, "r") as f:
                trade_history = json.load(f)
        
        # Create trade record
        trade_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
            "qty": qty,
            "price": price,
            "reason": reason,
            "portfolio_value": round(portfolio_value, 2)
        }
        
        # Add optional metrics
        if metrics:
            trade_record.update(metrics)
        
        trade_history.append(trade_record)
        
        # Save updated history
        with open(TRADE_HISTORY_FILE, "w") as f:
            json.dump(trade_history, f, indent=2)
        
        logger.info(f"Logged trade: {action} {qty} shares at ${price:.2f}")
    except Exception as e:
        logger.error(f"Error logging trade: {e}")


def has_executed_trade_today() -> bool:
    """Return True if a live BUY/SELL was already recorded today in trade history."""
    try:
        if not os.path.exists(TRADE_HISTORY_FILE):
            return False

        with open(TRADE_HISTORY_FILE, "r") as f:
            trade_history = json.load(f)

        today_str = datetime.now().strftime("%Y-%m-%d")
        executed_actions = {"BUY", "SELL", "SELL_PARTIAL", "SELL_FULL"}

        for trade in reversed(trade_history):
            timestamp = str(trade.get("timestamp", ""))
            action = str(trade.get("action", "")).upper()
            if timestamp.startswith(today_str) and action in executed_actions:
                return True
        return False
    except Exception as e:
        logger.warning(f"Unable to verify daily trade idempotency: {e}")
        return False

def is_market_open() -> bool:
    """Check if NYSE is scheduled to trade today (year-safe weekend + holiday handling)."""
    now = datetime.now()

    # Fast path for weekends
    if now.weekday() >= 5:  # Sat (5) or Sun (6)
        logger.info("Market closed: Weekend")
        return False
    
    today_date = now.date()
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=today_date, end_date=today_date)

    if schedule.empty:
        logger.info(f"Market closed: NYSE has no session on {today_date.strftime('%Y-%m-%d')}")
        return False

    logger.info("Market expected open today (NYSE session found)")
    return True

def calculate_portfolio_metrics(portfolio: Dict[str, Any], current_price: float) -> Dict[str, Any]:
    """Calculate comprehensive portfolio metrics including P&L and drawdown"""
    cash = portfolio["cash"]
    shares = portfolio["shares"]
    cost_basis = portfolio.get("cost_basis", 0.00)
    initial_capital = portfolio.get("initial_capital", 100000.00)
    peak_value = portfolio.get("peak_value", initial_capital)
    
    # Calculate values
    position_value = shares * current_price
    total_equity = cash + position_value
    unrealized_pnl = (current_price - cost_basis) * shares if shares > 0 else 0.0
    total_pnl = total_equity - initial_capital
    total_return_pct = ((total_equity - initial_capital) / initial_capital) * 100
    
    # Update peak value for drawdown calculation
    new_peak = max(peak_value, total_equity)
    current_drawdown = ((new_peak - total_equity) / new_peak) * 100 if new_peak > 0 else 0.0
    
    return {
        "total_equity": round(total_equity, 2),
        "position_value": round(position_value, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_return_pct, 2),
        "peak_value": round(new_peak, 2),
        "current_drawdown_pct": round(current_drawdown, 2)
    }

def fetch_msft_daily() -> Optional[Dict[str, Any]]:
    """Fetch MSFT market data and build trading packet with current portfolio state"""
    logger.info("Starting MSFT data fetch...")
    
    try:
        # Load current portfolio state
        portfolio = load_portfolio_state()
        
        # Pull the last 1 year of MSFT daily candles
        ticker = yf.Ticker("MSFT")
        logger.info("Fetching MSFT market data from Yahoo Finance...")
        hist = ticker.history(period="1y") # Fetch 1 year to ensure we have enough data for indicators, but we'll use only recent data for history array and to make sma_200 compute sooner
        
        if hist.empty:
            logger.error("No historical data returned from Yahoo Finance")
            return None
        
        last_bar_date = hist.index[-1].date()
        today = datetime.now().date()
        prev_trade_day = previous_trading_day(today)

        # Accept today (if data includes current session) or the latest prior trading day.
        if last_bar_date not in [prev_trade_day, today]:
            logger.warning(f"Stale data detected: Last bar is {last_bar_date} — market likely closed or data issue. Aborting.")
            return None
        
        logger.info(f"Data fresh: Last bar {last_bar_date}")
        
        # Extract the most recent row (yesterday's close if run after market close)
        latest = hist.iloc[-1]

        # Volume enhancements
        if len(hist) >= 20:
            avg_vol_20 = int(hist['Volume'].rolling(20).mean().iloc[-1])
            rel_volume = round(latest["Volume"] / avg_vol_20, 2) if avg_vol_20 > 0 else 1.0
        else:
            avg_vol_20 = int(latest["Volume"])
            rel_volume = 1.0  # Neutral fallback

        # Build the history array (close prices only) - rounded and limited for token efficiency
        n_days = 60 if len(hist) >= 60 else len(hist)
        close_history = [round(price, 2) for price in hist["Close"].tolist()[-n_days:]]  # Last 60 days only

        # Compute simple volatility (std dev of returns)
        returns = hist["Close"].pct_change().dropna()
        volatility = round(float(returns.std()), 5)
        
        # Calculate RSI(14)
        rsi_14 = calculate_rsi(hist["Close"].tolist())
        
        # Calculate ATR(14) and ATR percentile
        atr_14 = calculate_atr(hist)
        
        # Warning for zero or very low ATR
        if atr_14 == 0.0:
            logger.warning("ATR is 0.0 - position sizing will be 0. Check data quality.")
        elif atr_14 < 1.0:
            logger.warning(f"ATR is very low ({atr_14}) - position sizing may be affected.")
        
        # Calculate ATR for all historical data to get percentile and regime
        high_low = hist['High'] - hist['Low']
        high_close_prev = abs(hist['High'] - hist['Close'].shift(1))
        low_close_prev = abs(hist['Low'] - hist['Close'].shift(1))
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        hist_atr = tr.rolling(window=14).mean()
        atr_percentile = round(float((hist_atr.rank(pct=True).iloc[-1]) * 100), 1)
        
        # Regime Filtering: Calculate ATR expansion ratio
        atr_14_day_avg = hist_atr.iloc[-14:].mean()  # Average of last 14 ATR values
        atr_expansion_ratio = round(atr_14 / atr_14_day_avg, 2) if atr_14_day_avg > 0 else 1.0
        
        # Determine market regime and position size multiplier
        if atr_expansion_ratio >= 2.0:
            regime = "High Volatility Regime"
            regime_multiplier = 0.5  # Reduce position to 50%
        elif atr_expansion_ratio >= 1.5:
            regime = "Elevated Volatility"
            regime_multiplier = 0.75  # Reduce position to 75%
        elif atr_expansion_ratio <= 0.5:
            regime = "Low Volatility"
            regime_multiplier = 1.0  # Keep normal position
        else:
            regime = "Normal"
            regime_multiplier = 1.0  # Normal position sizing
        
        # Get current price
        current_price = round(float(latest["Close"]), 2)

        # Calculate 50-day and 200-day SMAs for trend analysis
        sma_50 = None
        sma_200 = None
        if len(hist) >= 200:
            sma_200 = round(hist['Close'].rolling(200).mean().iloc[-1], 2)
        if len(hist) >= 50:
            sma_50 = round(hist['Close'].rolling(50).mean().iloc[-1], 2)
        # Determine trend label
        if sma_200 is None:
            trend_label = "Insufficient data (need 200 days)"
        elif current_price > sma_200:
            trend_label = "Bullish (above 200 SMA)"
        elif sma_50 is not None and current_price < sma_50:
            trend_label = "Bearish (below 50 SMA)"
        else:
            trend_label = "Neutral / Sideways"

        # Optional: also keep the boolean (convert to Python bool for JSON serialization)
        price_above_200 = bool(sma_200 is not None and current_price > sma_200)
        
        # Calculate ATR-based stop-loss and take-profit levels
        stop_loss = round(current_price - (atr_14 * 2), 2)  # 2x ATR below current price
        take_profit = round(current_price + (atr_14 * 3), 2)  # 3x ATR above current price
        
        # Calculate suggested position size based on ATR (risk 2% per trade)
        risk_per_trade_pct = 0.02
        risk_amount = portfolio["cash"] * risk_per_trade_pct
        base_shares = int(risk_amount / (atr_14 * 2)) if atr_14 > 0 else 0  # 2x ATR stop
        
        # Apply regime-based position sizing
        suggested_shares = int(base_shares * regime_multiplier)
        
        # Calculate portfolio metrics including drawdown
        metrics = calculate_portfolio_metrics(portfolio, current_price)

        # Add unrealized PnL percentage for easier use
        unrealized_pnl_pct = round(metrics["unrealized_pnl"] / (portfolio["shares"] * current_price) * 100, 2) if portfolio["shares"] > 0 else 0.0
        
        # Build the JSON packet with current portfolio state - all values rounded
        packet = {
            "timestamp": datetime.now().strftime("%Y-%m-%d"),
            "symbol": "MSFT",
            "portfolio": {
                "cash": round(portfolio["cash"], 2),
                "shares": portfolio["shares"],
                "cost_basis": round(portfolio["cost_basis"], 2),
                "total_equity": metrics["total_equity"],
                "unrealized_pnl": metrics["unrealized_pnl"],
                "total_return_pct": metrics["total_return_pct"],
                "current_drawdown_pct": metrics["current_drawdown_pct"],
                "unrealized_pnl_pct": unrealized_pnl_pct,
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
                "stop_loss_suggestion": stop_loss,
                "take_profit_suggestion": take_profit,
                "suggested_position_size": suggested_shares,
                "sma_50": sma_50,
                "sma_200": sma_200,
                "price_above_200_sma": price_above_200,
                "trend_label": trend_label,   # ← this is the string Grok will see
                "latest_volume": int(latest["Volume"]),
                "avg_volume_20d": avg_vol_20,
                "relative_volume": rel_volume,
            },
            "constraints": {
                "max_position_size_pct": 0.20,
                "max_drawdown_pct": 0.10,
                "risk_per_trade_pct": 0.02,
                "min_atr": 3.5,
                "max_atr": 18.0
            }
        }
        
        # logger.info(f"Successfully fetched MSFT data. Price: ${current_price:.2f}, Volatility: {volatility:.4f}, RSI: {rsi_14}, ATR: {atr_14} (percentile: {atr_percentile}%), Expansion: {atr_expansion_ratio}x, Regime: {regime}, Stop: ${stop_loss}, Target: ${take_profit}, Suggested shares: {suggested_shares}, Latest Volume: {int(latest['Volume'])}, Avg Volume 20d: {avg_vol_20}, Relative Volume: {rel_volume}")
        logger.info(
            f"Successfully fetched MSFT data. "
            f"Price: ${current_price:.2f}, RSI: {rsi_14}, ATR: {atr_14}, Regime: {regime}, "
            f"Suggested shares: {suggested_shares}, Rel Vol: {rel_volume:.2f} "
            f"(Latest Vol: {int(latest['Volume'])}, Avg 20d: {avg_vol_20})"
        )
        return packet
        
    except Exception as e:
        logger.error(f"Error fetching MSFT data: {e}")
        return None

# Grok API call functions
def query_grok(packet: Dict[str, Any], api_key: str) -> Optional[Dict[str, Any]]:
    """Query Grok AI for trading decision with error handling"""
    url = "https://api.x.ai/v1/chat/completions"
    
    logger.info("Sending data packet to Grok for analysis...")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prompt = f"""
You are an automated trading decision agent.
Your task is to analyze the provided JSON packet and choose exactly one action for the next trading day.

Allowed actions:
- BUY
- SELL
- HOLD

Decision rules:
- Use only the data inside the JSON packet.
- Evaluate price trend, volatility, volume, RSI(14), ATR(14), ATR percentile, market regime, and recent history.
- CRITICAL: Respect the market_regime and regime_multiplier for position sizing - suggested_position_size already accounts for regime-based adjustments.
- High Volatility Regime (ATR expansion > 200%): Position size automatically reduced to 50% to maintain consistent dollar risk.
- Elevated Volatility (ATR expansion 150-200%): Position size reduced to 75%.
- Use stop_loss_suggestion and take_profit_suggestion for risk management.
- Avoid trading if ATR is outside min_atr/max_atr range (extreme volatility conditions).
- Consider portfolio constraints: cash, shares, cost_basis, and risk limits.
- Do not assume future prices or external market conditions.
- Do not add commentary, disclaimers, or formatting.
- If deciding to SELL and unrealized_pnl_pct > 0:
  - In Bullish or Neutral trend: Consider partial exit if gains are meaningful.
  - In Bearish trend: Recommend full exit (100%).
- Output format remains the same, but you may include partial percentage in REASON if suggesting partial (e.g. "Partial sell recommended due to +18% unrealized gain in bullish trend").

Additional trend filter (apply after core rules):
- Strongly prefer BUY when trend_label indicates "Bullish" (or price_above_200_sma is true) and RSI is low/oversold.
- Be very cautious with SELL signals when trend_label is "Bullish" — only consider if RSI is extremely overbought (>80) and other signals align strongly.
- Exception: Consider BUY if RSI <25 AND relative_volume >1.2 (indicating strong oversold with above-average conviction/participation).
- Use sma_50 and sma_200 values to assess how far price is from key levels if needed.
- When signals conflict (e.g. oversold RSI but Bearish trend), prioritize trend_label over short-term RSI unless RSI is extreme (>80 or <20).
- If trend_label is "Insufficient data" or sma_200 is None/null, default to HOLD regardless of other indicators.

Output format (must match exactly):

ACTION: <BUY or SELL or HOLD>
REASON: <one short sentence>

No additional text. No markdown. No code blocks.

Data packet:
{json.dumps(packet)}
"""

    body = {
        "model": "grok-4-1-fast-reasoning-latest",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=body, timeout=30)
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response headers: {response.headers}")
        if response.status_code == 403:
            logger.error(f"403 Forbidden error. Response text: {response.text}")
        response.raise_for_status()
        
        result = response.json()
        logger.info("Successfully received response from Grok")
        return result
        
    except requests.exceptions.Timeout:
        logger.error("Grok API request timed out")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Grok API request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Grok response as JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error calling Grok API: {e}")
        return None


def parse_action(response: Dict[str, Any]) -> tuple[str, str]:
    """Parse Grok response to extract trading action and reason"""
    try:
        text = response["choices"][0]["message"]["content"]
        logger.info(f"Grok response: {text.strip()}")
        
        # Extract reason if present
        reason = "No reason provided"
        if "REASON:" in text:
            reason = text.split("REASON:")[1].strip().split("\n")[0]

        # Parse ACTION line explicitly to avoid false positives from free text.
        action = "HOLD"
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("ACTION:"):
                candidate = stripped.split(":", 1)[1].strip().upper()
                if candidate in {"BUY", "SELL", "HOLD"}:
                    action = candidate
                else:
                    logger.warning(f"Invalid ACTION value from Grok: {candidate}. Defaulting to HOLD.")
                break

        return action, reason
        
    except (KeyError, IndexError) as e:
        logger.error(f"Error parsing Grok response: {e}. Raw response: {response}")
        return "HOLD", "Error parsing response"  # Default to safe action
# Alpaca order functions

def place_alpaca_order(api_key: str, secret_key: str, symbol: str, qty: int, side: str) -> Optional[Dict[str, Any]]:
    """Place order with Alpaca API with error handling"""
    url = "https://paper-api.alpaca.markets/v2/orders"

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
        "Content-Type": "application/json"
    }

    order = {
        "symbol": symbol,
        "qty": qty,
        "side": side.lower(),   # "buy" or "sell"
        "type": "market",
        "time_in_force": "day"
    }
    
    logger.info(f"Placing {side.upper()} order for {qty} shares of {symbol}")

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
        logger.error(f"Unexpected error placing order: {e}")
        return None



# Convert Grok's action into an Alpaca trade
def execute_trade(
    action: str,
    reason: str,
    packet: Dict[str, Any],
    api_key: Optional[str],
    secret_key: Optional[str],
    dry_run: bool = False
) -> bool:
    """Execute trade based on Grok decision with comprehensive validation and portfolio update"""
    price = packet["market_data"]["price"]
    suggested_shares = packet["market_data"]["suggested_position_size"]
    portfolio = packet["portfolio"]
    constraints = packet["constraints"]
    
    cash = portfolio["cash"]
    shares = portfolio["shares"]
    cost_basis = portfolio.get("cost_basis", 0.0)
    current_drawdown_pct = portfolio.get("current_drawdown_pct", 0.0)
    total_equity = portfolio.get("total_equity", cash)
    
    logger.info(f"Executing action: {action} | Current portfolio: ${cash:.2f} cash, {shares} shares, Equity: ${total_equity:.2f}, Drawdown: {current_drawdown_pct:.2f}%")

    # Enforce ATR guardrails at execution time (not just in model prompt)
    atr_14 = packet.get("market_data", {}).get("atr_14", 0.0)
    min_atr = constraints.get("min_atr", 0.0)
    max_atr = constraints.get("max_atr", float("inf"))
    if action in {"BUY", "SELL"} and not (min_atr <= atr_14 <= max_atr):
        logger.warning(
            f"TRADE BLOCKED: ATR ({atr_14}) outside allowed range [{min_atr}, {max_atr}]"
        )
        if not dry_run:
            log_trade("BLOCKED_ATR", 0, price, f"ATR out of bounds: {atr_14}", total_equity, {
                "atr_14": atr_14,
                "min_atr": min_atr,
                "max_atr": max_atr,
                "requested_action": action
            })
        else:
            logger.info("DRY-RUN: Would record BLOCKED_ATR event")
        return False
    
    # Check drawdown protection
    max_drawdown_pct = constraints.get("max_drawdown_pct", 0.10) * 100  # Convert to percentage
    if current_drawdown_pct > max_drawdown_pct:
        logger.warning(f"TRADE BLOCKED: Current drawdown ({current_drawdown_pct:.2f}%) exceeds maximum allowed ({max_drawdown_pct:.2f}%)")
        if not dry_run:
            log_trade("BLOCKED_DRAWDOWN", 0, price, f"Drawdown too high: {current_drawdown_pct:.2f}%", total_equity)
        else:
            logger.info("DRY-RUN: Would record BLOCKED_DRAWDOWN event")
        return False

    # Idempotency guard: do not place more than one live trade per day.
    if action in {"BUY", "SELL"} and not dry_run and has_executed_trade_today():
        logger.warning("TRADE BLOCKED: A live BUY/SELL trade has already been executed today")
        log_trade("BLOCKED_DUPLICATE", 0, price, "Duplicate daily trade prevented", total_equity, {
            "requested_action": action
        })
        return False

    if action == "BUY":
        # Calculate maximum affordable shares with cash
        max_affordable = int(cash // price)
        
        # Apply max position size constraint (20% of total equity)
        max_position_value = total_equity * constraints.get("max_position_size_pct", 0.20)
        current_position_value = shares * price
        available_position_capacity = max_position_value - current_position_value
        max_by_position_limit = int(available_position_capacity / price) if available_position_capacity > 0 else 0
        
        # Use the minimum of: suggested shares, affordable shares, position limit
        qty = min(suggested_shares, max_affordable, max_by_position_limit) if suggested_shares > 0 else 0
        
        logger.info(f"Position sizing: Suggested={suggested_shares}, Affordable={max_affordable}, PositionLimit={max_by_position_limit}, Final={qty}")
        
        if qty <= 0:
            logger.warning("Cannot buy: quantity is 0 after applying constraints")
            if not dry_run:
                log_trade("BLOCKED_BUY", 0, price, "No capacity to buy (constraints)", total_equity)
            else:
                logger.info("DRY-RUN: Would record BLOCKED_BUY event")
            return False
            
        if qty > 0:
            if dry_run:
                execution_price = float(price)
                new_cash = cash - (qty * execution_price)
                new_shares = shares + qty
                old_cost_basis = cost_basis if shares > 0 else 0
                new_cost_basis = ((shares * old_cost_basis) + (qty * execution_price)) / new_shares if new_shares > 0 else 0
                new_equity = new_cash + (new_shares * execution_price)
                logger.info(
                    f"DRY-RUN BUY: Would buy {qty} MSFT at ${execution_price:.2f}. "
                    f"Simulated portfolio -> cash: ${new_cash:.2f}, shares: {new_shares}, cost_basis: ${new_cost_basis:.2f}, equity: ${new_equity:.2f}"
                )
                return True

            if not api_key or not secret_key:
                logger.error("Missing Alpaca credentials for live BUY execution")
                return False

            result = place_alpaca_order(api_key, secret_key, "MSFT", qty, "buy")
            if result:
                fill_price_raw = result.get("filled_avg_price")
                execution_price = float(fill_price_raw) if fill_price_raw is not None else float(price)
                # Calculate new portfolio state
                new_cash = cash - (qty * execution_price)
                new_shares = shares + qty
                old_cost_basis = cost_basis if shares > 0 else 0
                new_cost_basis = ((shares * old_cost_basis) + (qty * execution_price)) / new_shares if new_shares > 0 else 0
                
                # Load current portfolio to preserve other fields
                current_portfolio = load_portfolio_state()
                updated_portfolio = {
                    **current_portfolio,
                    "cash": new_cash,
                    "shares": new_shares,
                    "cost_basis": new_cost_basis
                }
                
                # Update peak value if needed
                new_equity = new_cash + (new_shares * execution_price)
                updated_portfolio["peak_value"] = max(current_portfolio.get("peak_value", new_equity), new_equity)
                
                try:
                    save_portfolio_state(updated_portfolio)
                    # Log trade after successful save
                    log_trade("BUY", qty, execution_price, reason, new_equity, {
                        "rsi_14": packet["market_data"]["rsi_14"],
                        "atr_14": packet["market_data"]["atr_14"],
                        "regime": packet["market_data"]["market_regime"]
                    })
                    logger.info(f"BUY executed: {qty} shares at ${execution_price:.2f}. New portfolio: ${new_cash:.2f} cash, {new_shares} shares")
                    return True
                except Exception as e:
                    logger.critical(f"CRITICAL ERROR: Trade executed but portfolio save failed: {e}")
                    logger.critical(f"Manual intervention required: BUY {qty} shares at ${execution_price:.2f} was executed")
                    return False
            else:
                logger.error("BUY order failed")
                return False
        else:
            logger.warning("Not enough cash to buy shares")
            return False

    elif action == "SELL":
        if shares <= 0:
            logger.warning("No shares to sell")
            if not dry_run:
                log_trade("BLOCKED_SELL", 0, price, "No shares to sell", total_equity)
            else:
                logger.info("DRY-RUN: Would record BLOCKED_SELL event")
            return False

        # Get current unrealized PnL % (re-calculate to be sure)
        position_value = shares * price
        unrealized_pnl = (price - cost_basis) * shares if shares > 0 else 0.0
        unrealized_pnl_pct = round((unrealized_pnl / position_value) * 100, 2) if position_value > 0 else 0.0

        # Determine sell percentage
        if unrealized_pnl_pct < 8.0:
            sell_pct = 1.0          # Full sell if gains are small or loss
        elif unrealized_pnl_pct < 15.0:
            sell_pct = 0.30         # 30%
        elif unrealized_pnl_pct < 25.0:
            sell_pct = 0.40         # 40%
        else:
            sell_pct = 0.60         # 60% at high gains

        # Override to full sell in Bearish trend
        trend_label = packet["market_data"].get("trend_label", "Unknown")
        if "Bearish" in trend_label:
            sell_pct = 1.0
            reason += " (full exit due to Bearish trend)"

        qty = int(shares * sell_pct)
        if qty < 1:
            qty = shares  # Minimum 1 share or full if fractional rounding down

        logger.info(f"SELL decision: Unrealized PnL {unrealized_pnl_pct:.2f}%, "
                    f"Trend: {trend_label}, Selling {sell_pct*100:.0f}% → {qty} shares")

        if dry_run:
            execution_price = float(price)
            new_cash = cash + (qty * execution_price)
            realized_pnl = (execution_price - cost_basis) * qty
            remaining_shares = max(shares - qty, 0)
            if remaining_shares > 0:
                new_cost_basis = cost_basis
            else:
                new_cost_basis = 0.0
            new_equity = new_cash + (remaining_shares * execution_price)
            logger.info(
                f"DRY-RUN SELL: Would sell {qty} MSFT at ${execution_price:.2f}. "
                f"Simulated portfolio -> cash: ${new_cash:.2f}, shares: {remaining_shares}, cost_basis: ${new_cost_basis:.2f}, "
                f"realized_pnl: ${realized_pnl:.2f}, equity: ${new_equity:.2f}"
            )
            return True

        if not api_key or not secret_key:
            logger.error("Missing Alpaca credentials for live SELL execution")
            return False

        result = place_alpaca_order(api_key, secret_key, "MSFT", qty, "sell")
        if result:
            fill_price_raw = result.get("filled_avg_price")
            execution_price = float(fill_price_raw) if fill_price_raw is not None else float(price)
            # Update portfolio
            new_cash = cash + (qty * execution_price)
            realized_pnl = (execution_price - cost_basis) * qty

            # Weighted average cost basis for remaining shares
            if shares - qty > 0:
                remaining_shares = shares - qty
                new_cost_basis = ((shares * cost_basis) - (qty * cost_basis)) / remaining_shares
            else:
                remaining_shares = 0
                new_cost_basis = 0.0

            current_portfolio = load_portfolio_state()
            updated_portfolio = {
                **current_portfolio,
                "cash": new_cash,
                "shares": remaining_shares,
                "cost_basis": round(new_cost_basis, 2)
            }

            # Update peak value
            new_equity = new_cash + (remaining_shares * execution_price)
            updated_portfolio["peak_value"] = max(
                current_portfolio.get("peak_value", new_equity), new_equity
            )

            try:
                save_portfolio_state(updated_portfolio)
                log_trade(
                    "SELL_PARTIAL" if sell_pct < 1.0 else "SELL_FULL",
                    qty,
                    execution_price,
                    f"{reason} | PnL {unrealized_pnl_pct:.2f}% | Sold {sell_pct*100:.0f}%",
                    new_equity,
                    {
                        "realized_pnl": round(realized_pnl, 2),
                        "remaining_shares": remaining_shares,
                        "rsi_14": packet["market_data"]["rsi_14"],
                        "atr_14": packet["market_data"]["atr_14"],
                        "regime": packet["market_data"]["market_regime"]
                    }
                )
                logger.info(f"SELL executed: {qty} shares ({sell_pct*100:.0f}%) at ${execution_price:.2f}. "
                            f"Realized P&L: ${realized_pnl:.2f}. Remaining shares: {remaining_shares}")
                return True
            except Exception as e:
                logger.critical(f"CRITICAL: SELL executed but save failed: {e}")
                logger.critical(f"Manual check required: Sold {qty} shares of MSFT")
                return False
        else:
            logger.error("SELL order failed")
            return False
    
    else:  # HOLD or any other action
        logger.info(f"Holding position: {action} - {reason}")
        if not dry_run:
            log_trade("HOLD", 0, price, reason, total_equity, {
                "rsi_14": packet["market_data"]["rsi_14"],
                "atr_14": packet["market_data"]["atr_14"],
                "regime": packet["market_data"]["market_regime"]
            })
        else:
            logger.info("DRY-RUN HOLD: No trade, no portfolio mutation")
        return True

def main(dry_run: bool = False, ignore_market_check: bool = False):
    """Main trading loop with comprehensive error handling and logging"""
    logger.info("=== Starting Daily Trading Loop ===")
    if dry_run:
        logger.info("DRY-RUN MODE ENABLED: No orders will be placed and no state files will be modified.")
    
    # Check if market is open (basic weekend check)
    if not ignore_market_check and not is_market_open():
        logger.warning("Market is closed (weekend/holiday). Exiting.")
        return
    if ignore_market_check:
        logger.warning("Market check ignored via flag; continuing execution.")
    
    # Load environment variables from .env file
    load_dotenv()
    
    # API Keys
    GROK_API_KEY = os.getenv("GROK_API_KEY")
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
    ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
    
    # Check required API keys
    if not GROK_API_KEY:
        logger.error("GROK_API_KEY not found in environment variables. Please check your .env file.")
        return
    if not dry_run:
        if not ALPACA_API_KEY:
            logger.error("ALPACA_API_KEY not found in environment variables. Please check your .env file.")
            return
        if not ALPACA_SECRET_KEY:
            logger.error("ALPACA_SECRET_KEY not found in environment variables. Please check your .env file.")
            return
    
    try:
        # Step 1: Fetch market data
        packet = fetch_msft_daily()
        if packet is None:
            logger.error("Failed to fetch market data. Aborting trading loop.")
            return
        
        # Step 2: Get AI decision
        response = query_grok(packet, GROK_API_KEY)
        if response is None:
            logger.error("Failed to get response from Grok. Aborting trading loop.")
            return
            
        action, reason = parse_action(response)
        logger.info(f"Grok decision: {action} - {reason}")
        
        # Step 3: Execute trade
        success = execute_trade(action, reason, packet, ALPACA_API_KEY, ALPACA_SECRET_KEY, dry_run=dry_run)
        
        if success:
            logger.info("=== Trading loop completed successfully ===")
        else:
            logger.error("=== Trading loop completed with errors ===")

    except KeyboardInterrupt:
        # This is almost always a SIGINT from the terminal/IDE (e.g., VS Code re-run/stop).
        # Log it explicitly so it doesn't look like a mysterious crash.
        logger.warning("KeyboardInterrupt (SIGINT) received; exiting early.")
        return
            
    except Exception as e:
        # Use logger.exception to include the full traceback in logs for post-mortems.
        logger.exception("Unexpected error in main trading loop")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run daily MSFT trading loop")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate decision and execution without placing orders or writing state files"
    )
    parser.add_argument(
        "--ignore-market-check",
        action="store_true",
        help="Bypass market open/holiday check (useful for weekend dry-runs)"
    )
    args = parser.parse_args()

    main(dry_run=args.dry_run, ignore_market_check=args.ignore_market_check)