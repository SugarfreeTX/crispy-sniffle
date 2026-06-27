import csv
import os
import time
import logging
import json
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
# Kalshi SDK (install kalshi_python_sync or similar)
from kalshi_python_async import KalshiClient  # or your preferred client
from kalshi_python_async.api_client import ApiClient
from kalshi_python_async.configuration import Configuration
from kalshi_python_async.api.market_api import MarketApi
# For Polymarket: from py_clob_client_v2 import ClobClient, etc.
# Grok API calls (use xAI SDK or requests)

# Config
# 1. Configure credentials
# config = Configuration(
#     host="https://kalshi.co", # Demo environment
#     key_id="YOUR_KEY_ID",
#     private_key="YOUR_RSA_PRIVATE_KEY"
# )

# # 2. Initialize the client
# client = ApiClient(config)
# market_client = market_api.MarketApi(client)

# # 3. Use the SDK (e.g., get a market)
# response = market_client.get_market("FED-26DEC-T4.50")
# print(response)

BASE_DIR = Path(__file__).resolve().parent

KALSHI_API_KEY = os.getenv("KALSHI_API_KEY")
# ... other creds
GROK_API_KEY = os.getenv("GROK_API_KEY")  # or xAI client
GROK_MODEL = os.getenv("GROK_MODEL", "grok-4.20-multi-agent-0309")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logging.warning("Invalid %s=%r; using default %.4f", name, raw, default)
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


MIN_EDGE = _env_float("MIN_EDGE", 0.08)
MAX_POSITION_PCT = _env_float("MAX_POSITION_PCT", 0.02)
FRACTIONAL_KELLY = _env_float("FRACTIONAL_KELLY", 0.5)
TRADING_FEES = _env_float("TRADING_FEES", 0.01)
SLIPPAGE = _env_float("SLIPPAGE", 0.002)
MODEL_WEIGHT_BASE = _env_float("MODEL_WEIGHT_BASE", 0.4)
MODEL_WEIGHT_MAX = _env_float("MODEL_WEIGHT_MAX", 0.6)
MIN_MODEL_CONFIDENCE = _env_float("MIN_MODEL_CONFIDENCE", 0.55)
SHADOW_MODE = _env_bool("SHADOW_MODE", False)
SHADOW_LOG_JSONL = os.getenv("SHADOW_LOG_JSONL", str(BASE_DIR / "shadow_calibration_log.jsonl"))
SHADOW_LOG_CSV = os.getenv("SHADOW_LOG_CSV", str(BASE_DIR / "shadow_calibration_log.csv"))
LOG_FILE = os.getenv("LOG_FILE", "pred_market_bot.log")

SHADOW_CSV_FIELDS = [
    "timestamp",
    "ticker",
    "market_prob",
    "model_prob",
    "blended_prob",
    "confidence",
    "direction",
    "edge",
    "realized_outcome",
    "shadow_mode",
    "executed",
]

logging.basicConfig(filename=LOG_FILE, level=logging.INFO
                    , format='%(asctime)s - %(levelname)s - %(message)s')


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _extract_first_json_dict(text: str):
    """Extract and parse the first JSON object from a model response string."""
    if not text:
        return None

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _build_grok_prompt(event_description: str, strict: bool = False) -> str:
    if strict:
        return (
            "You are estimating a binary event probability for trading. "
            "Respond with exactly one JSON object and nothing else. "
            "No markdown, no code fences, no prose before or after the JSON. "
            'Required schema: {"prob": <float 0-1>, "confidence": <float 0-1>, "reasoning": "<string>"}. '
            f"Event: {event_description}"
        )
    return (
        "You are estimating a binary event probability for trading. "
        "Return JSON only with keys: prob, confidence, reasoning. "
        "Constraints: prob and confidence must be numbers in [0,1]. "
        f"Event: {event_description}"
    )


def _parse_grok_probability_response(content: str) -> Optional[dict]:
    parsed = _extract_first_json_dict(content)
    if not parsed:
        return None

    try:
        prob = _clamp(float(parsed.get("prob", -1)), 0.0, 1.0)
        confidence = _clamp(float(parsed.get("confidence", -1)), 0.0, 1.0)
    except (TypeError, ValueError):
        return None

    reasoning = str(parsed.get("reasoning", ""))
    if prob < 0 or confidence < 0:
        return None

    return {
        "prob": prob,
        "reasoning": reasoning,
        "confidence": confidence,
        "raw": content,
    }


def _call_grok_chat(event_description: str, strict: bool = False) -> Optional[str]:
    payload = {
        "model": GROK_MODEL,
        "messages": [{
            "role": "user",
            "content": _build_grok_prompt(event_description, strict=strict),
        }],
    }
    headers = {"Authorization": f"Bearer {GROK_API_KEY}"}
    resp = requests.post(
        "https://api.x.ai/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _ensure_shadow_csv_header(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHADOW_CSV_FIELDS)
        writer.writeheader()


def log_shadow_calibration(
    *,
    ticker: str,
    market_prob: float,
    model_prob: float,
    blended_prob: float,
    confidence: float,
    direction: Optional[str],
    edge: float,
    realized_outcome: Optional[bool] = None,
) -> None:
    """Append one calibration row to shadow JSONL and CSV logs."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "market_prob": round(market_prob, 6),
        "model_prob": round(model_prob, 6),
        "blended_prob": round(blended_prob, 6),
        "confidence": round(confidence, 6),
        "direction": direction,
        "edge": round(edge, 6),
        "realized_outcome": realized_outcome,
        "shadow_mode": True,
        "executed": False,
    }

    jsonl_path = Path(SHADOW_LOG_JSONL)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")

    csv_path = Path(SHADOW_LOG_CSV)
    _ensure_shadow_csv_header(csv_path)
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHADOW_CSV_FIELDS)
        writer.writerow(entry)

def calculate_position_size(edge: float, bankroll: float, grok_confidence: float = 0.7) -> int:
    """Fractional Kelly position sizing with safety caps. 
    edge: Estimated edge (probability advantage after fees)
    Returns number of contracts (assuming $1 per contract on Kalshi).
    """
    if edge <= 0 or bankroll <= 0:
        return 0
    
    # Kelly fraction = edge / odds (simplified for binary outcome)
    # For yes/no at price p, effective odds ~ p / (1-p) but we approximate with edge 
    kelly_fraction = edge  # Simplified for small edges

    # Apply fractional Kelly + confidence adjustment
    fraction = kelly_fraction * grok_confidence * FRACTIONAL_KELLY  # Fractional Kelly for safety
    fraction = min(fraction, MAX_POSITION_PCT)  # Cap max position size

    position = int(bankroll * fraction)  # Assuming $1 per contract
    position = max(position, 0)  # Ensure non-negative

    return position


def blend_probability(model_prob: float, market_prob: float, model_confidence: float) -> float:
    """Blend model and market probabilities with a confidence-capped model weight."""
    model_prob = _clamp(model_prob, 0.0, 1.0)
    market_prob = _clamp(market_prob, 0.0, 1.0)
    model_confidence = _clamp(model_confidence, 0.0, 1.0)

    model_weight = _clamp(MODEL_WEIGHT_BASE * model_confidence, 0.0, MODEL_WEIGHT_MAX)
    blended = model_weight * model_prob + (1.0 - model_weight) * market_prob
    return _clamp(blended, 0.0, 1.0)

def get_kalshi_markets(client, limit=100):
    """Fetch open markets with liquidity."""
    # Use client.get_markets(status="open", etc.) or raw API
    markets = client.get_markets(...)  # Adapt to SDK
    df = pd.DataFrame(markets)
    df = df[(df['volume'] > 10000) & (df['close_time'] > time.time() + 86400*2)]  # Filter
    return df

def grok_estimate_probability(event_description: str) -> Optional[dict]:
    """Use Grok API for calibrated prob + reasoning."""
    try:
        content = _call_grok_chat(event_description, strict=False)
        parsed = _parse_grok_probability_response(content or "")

        if not parsed:
            logging.warning("Grok response was not valid JSON; retrying once with strict prompt.")
            strict_content = _call_grok_chat(event_description, strict=True)
            parsed = _parse_grok_probability_response(strict_content or "")
            if not parsed:
                logging.warning("Strict Grok retry still returned invalid JSON; skipping event.")
                return None
            logging.info("Recovered valid Grok JSON via strict prompt retry.")

        return parsed
    except Exception as exc:
        logging.exception("Grok probability estimation failed: %s", exc)
        return None

def calculate_edge(market_prob_yes: float, blended_prob_yes: float, fees=TRADING_FEES, slippage=SLIPPAGE):
    """Cost-adjusted edge on YES/NO using blended model-market probability."""
    market_prob_yes = _clamp(market_prob_yes, 0.0, 1.0)
    blended_prob_yes = _clamp(blended_prob_yes, 0.0, 1.0)
    trading_costs = fees + slippage

    yes_edge = blended_prob_yes - market_prob_yes - trading_costs
    no_edge = (1.0 - blended_prob_yes) - (1.0 - market_prob_yes) - trading_costs

    if yes_edge > MIN_EDGE:
        return "YES", yes_edge
    if no_edge > MIN_EDGE:
        return "NO", no_edge
    return None, 0

def place_kalshi_order(client, ticker, side, count, price):
    """Execute with risk checks."""
    # client.place_order(...)
    logging.info(f"Placed {side} on {ticker} at {price}")

def main_loop():
    client = KalshiClient(...)  # Init with auth
    if SHADOW_MODE:
        logging.info(
            "SHADOW_MODE enabled — logging calibration rows to %s and %s (no execution).",
            SHADOW_LOG_JSONL,
            SHADOW_LOG_CSV,
        )
    while True:  # Or scheduled
        markets = get_kalshi_markets(client)
        for _, m in markets.iterrows():
            desc = m['title'] + " - " + m.get('description', '')
            ticker = m['ticker']
            market_prob = _clamp(float(m['yes_price']) / 100.0, 0.0, 1.0)
            grok_data = grok_estimate_probability(desc)
            if not grok_data:
                continue

            blended_prob = blend_probability(
                model_prob=grok_data['prob'],
                market_prob=market_prob,
                model_confidence=grok_data['confidence'],
            )
            direction, edge = calculate_edge(market_prob, blended_prob)

            if SHADOW_MODE:
                log_shadow_calibration(
                    ticker=ticker,
                    market_prob=market_prob,
                    model_prob=grok_data['prob'],
                    blended_prob=blended_prob,
                    confidence=grok_data['confidence'],
                    direction=direction,
                    edge=edge,
                )
                logging.info(
                    "SHADOW-MODE: %s | direction=%s edge=%.2f%% market=%.3f model=%.3f blend=%.3f conf=%.2f",
                    ticker,
                    direction or "NONE",
                    edge * 100.0,
                    market_prob,
                    grok_data['prob'],
                    blended_prob,
                    grok_data['confidence'],
                )
                continue

            if grok_data['confidence'] < MIN_MODEL_CONFIDENCE:
                logging.info(
                    "Skip low-confidence model output: %s | confidence=%.2f",
                    ticker,
                    grok_data['confidence'],
                )
                continue

            if direction and edge > 0:
                size = calculate_position_size(
                    edge,
                    bankroll=client.get_balance(),
                    grok_confidence=grok_data['confidence'],
                )
                if size > 0:
                    place_kalshi_order(client, ticker, direction, size, market_prob)
                    logging.info(
                        "Edge trade: %s | Edge: %.2f%% | market=%.3f model=%.3f blend=%.3f conf=%.2f",
                        desc,
                        edge * 100.0,
                        market_prob,
                        grok_data['prob'],
                        blended_prob,
                        grok_data['confidence'],
                    )
        
        time.sleep(300)  # Poll interval

if __name__ == "__main__":
    main_loop()