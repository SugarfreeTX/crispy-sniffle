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

Allowed actions: BUY, SELL, HOLD

HARD BLOCKS (override all else):
- drawdown_level >= 2 → HOLD or SELL only (never BUY)
- loss_streak_multiplier <= 0.0 or consecutive_loss_streak >= max_consecutive_losses → HOLD or SELL only
- drawdown_level == 1 → BUY only if RSI < 20 AND "Bullish" in trend_label AND suggested_position_size >= 1

Core rules:
- Respect suggested_position_size (already regime- & drawdown-adjusted) — never suggest larger.
- Only BUY if suggested_position_size >= 1.
- Prioritize trend_label over short-term RSI unless RSI <20 or >85.
- Strongly prefer BUY in "Bullish" (or price_above_200_sma=True) + low/oversold RSI.
- In "High Volatility Regime" or "Elevated Volatility" → favor HOLD unless extreme oversold + Bullish.
- If regime_days_in_state >= 10 and "Volatility" in market_regime → strongly prefer HOLD.
- If regime_changed_today → extra caution on new entries.
- For SELL with profit: approximate tiers (<8% full, 8-15% ~30%, 15-25% ~40%, >25% ~60%); full exit in Bearish.

Output format (exact, nothing else):
ACTION: <BUY or SELL or HOLD>
REASON: <one short sentence>

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