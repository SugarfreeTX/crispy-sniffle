import json
import requests
import os
import logging
from typing import Dict, Any, Tuple, Optional

from dotenv import load_dotenv
from shared.types import PacketDict

logger = logging.getLogger(__name__)

GROK_API_URL = "https://api.x.ai/v1/chat/completions"


def build_grok_prompt(packet: PacketDict) -> str:
    """
    Build the full prompt string for Grok based on the data packet.
    This is where all decision rules live — easy to version or override per market.
    """
    return f"""
You are an automated trading decision agent.
Your task is to analyze the provided JSON packet and choose exactly one action for the next trading period.

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
- Consider portfolio constraints: cash, shares/position, cost_basis, and risk limits.
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

- If regime_days_in_state >= 10 and market_regime contains "Volatility Regime", strongly prefer HOLD (prolonged stress regime).
- If regime_changed_today is true, treat the regime as a fresh shift → be extra cautious, especially on new entries.
- In "High Volatility Regime" or "Elevated Volatility", strongly prefer HOLD or very small positions (respect the already-reduced suggested_position_size) unless RSI < 20 AND trend_label is "Bullish".

- consecutive_loss_streak shows current count of consecutive realized losses.
- loss_streak_multiplier is the allowed fraction of normal BUY size (1.0 = full, 0.5 = half, 0.25 = quarter, 0.0 = no new buys).
- Respect loss_streak_multiplier when suggesting BUY size; prefer HOLD/SELL if multiplier < 1.0 and conditions aren't exceptional.
- Do NOT suggest BUY when streak >= max_consecutive_losses (currently 5).

- drawdown_level (0=normal, 1=caution, 2=restricted, 3=emergency) indicates current drawdown severity.
- drawdown_name gives a human-readable name (e.g. "Caution (5-7.9%)", "Emergency (>10%)").
- drawdown_size_multiplier is the maximum allowed fraction of normal position size (e.g. 0.5 = half size).
- Drawdown rules (apply after regime rules):
    - If drawdown_level >= 2 (Restricted or Emergency), strongly prefer HOLD or partial/full exit -- DO NOT open new BUY positions.
    - If drawdown_level == 1 (Caution), be very conservative: favor HOLD, only consider small BUY if RSI extremely oversold (<20) in strong Bullish trend.
    - Always respect drawdown_size_multiplier when sizing any position.
    - Higher drawdown levels override other signals toward caution or defense.

Output format (must match exactly):

ACTION: <BUY or SELL or HOLD>
REASON: <one short sentence>

No additional text. No markdown. No code blocks.

Data packet:
{json.dumps(packet, indent=2)}
"""


def query_grok(packet: PacketDict, api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Send packet to Grok API and return the raw response.
    Returns None on failure (timeout, error, etc.).
    """
    if not api_key:
        load_dotenv()
        api_key = os.getenv("GROK_API_KEY")
        if not api_key:
            logger.error("GROK_API_KEY not found in environment or provided")
            return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prompt = build_grok_prompt(packet)

    body = {
        "model": "grok-4-1-fast-reasoning-latest",  # or your preferred model
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,  # low for consistent decisions
        "max_tokens": 300
    }

    logger.info("Sending data packet to Grok for analysis...")

    try:
        response = requests.post(GROK_API_URL, headers=headers, json=body, timeout=30)
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


def parse_action(response: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Parse Grok's chat completion response into (action, reason).
    Defaults to "HOLD" on any parsing failure.
    """
    if not response:
        return "HOLD", "No response from Grok"

    try:
        text = response["choices"][0]["message"]["content"].strip()
        logger.info(f"Grok raw response: {text}")

        action = "HOLD"
        reason = "No reason provided or parsing failed"

        lines = text.splitlines()
        for line in lines:
            stripped = line.strip()
            if stripped.upper().startswith("ACTION:"):
                candidate = stripped.split(":", 1)[1].strip().upper()
                if candidate in {"BUY", "SELL", "HOLD"}:
                    action = candidate
                else:
                    logger.warning(f"Invalid ACTION from Grok: {candidate}")
                break

        if "REASON:" in text:
            reason_part = text.split("REASON:", 1)[1].strip()
            reason = reason_part.split("\n")[0].strip()  # first line only

        return action, reason

    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Error parsing Grok response: {e}. Raw: {response}")
        return "HOLD", "Error parsing Grok response"