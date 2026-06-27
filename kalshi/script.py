import csv
import os
import time
import logging
import json
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
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
KALSHI_API_BASE = os.getenv("KALSHI_API_BASE", "https://api.elections.kalshi.com/trade-api/v2")
KALSHI_BACKFILL_SLEEP_SEC = _env_float("KALSHI_BACKFILL_SLEEP_SEC", 0.1)
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


def _normalize_realized_outcome(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "null", "none", "na", "n/a"}:
            return None
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def load_shadow_calibration_log(jsonl_path: Optional[str] = None) -> list[dict]:
    """Load shadow calibration rows from JSONL."""
    path = Path(jsonl_path or SHADOW_LOG_JSONL)
    if not path.exists():
        return []

    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                logging.warning("Skipping malformed JSONL row %s:%s (%s)", path, line_no, exc)
                continue
            row["realized_outcome"] = _normalize_realized_outcome(row.get("realized_outcome"))
            rows.append(row)
    return rows


def _write_shadow_calibration_jsonl(rows: list[dict], jsonl_path: Path) -> None:
    lines = [json.dumps(row, ensure_ascii=True) for row in rows]
    content = "\n".join(lines)
    if lines:
        content += "\n"

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(jsonl_path)


def _write_shadow_calibration_csv(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHADOW_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SHADOW_CSV_FIELDS})
    tmp_path.replace(csv_path)


def _parse_kalshi_market_result(market_payload: dict) -> Optional[bool]:
    result = str(market_payload.get("result", "")).strip().lower()
    if result == "yes":
        return True
    if result == "no":
        return False
    return None


def fetch_kalshi_market(ticker: str, client: Any = None) -> Optional[dict]:
    """Fetch one Kalshi market payload by ticker."""
    if client is not None:
        try:
            if hasattr(client, "get_market"):
                response = client.get_market(ticker)
            elif hasattr(client, "markets_api") and hasattr(client.markets_api, "get_market"):
                response = client.markets_api.get_market(ticker)
            else:
                response = None

            if response is not None:
                if hasattr(response, "market"):
                    market = response.market
                    if hasattr(market, "to_dict"):
                        return market.to_dict()
                    if hasattr(market, "model_dump"):
                        return market.model_dump()
                    if isinstance(market, dict):
                        return market
                if isinstance(response, dict):
                    return response.get("market", response)
        except Exception as exc:
            logging.warning("Kalshi client lookup failed for %s (%s); falling back to HTTP.", ticker, exc)

    url = f"{KALSHI_API_BASE.rstrip('/')}/markets/{ticker}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 404:
            logging.warning("Kalshi market not found: %s", ticker)
            return None
        resp.raise_for_status()
        payload = resp.json()
        return payload.get("market", payload)
    except Exception as exc:
        logging.warning("Kalshi HTTP lookup failed for %s: %s", ticker, exc)
        return None


def backfill_shadow_realized_outcomes(
    *,
    client: Any = None,
    jsonl_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """
    Backfill realized_outcome for shadow rows whose markets have settled on Kalshi.

    realized_outcome is True when the market settled YES, False when it settled NO,
    and remains null while the market is still open or undetermined.
    """
    jsonl_file = Path(jsonl_path or SHADOW_LOG_JSONL)
    csv_file = Path(csv_path or SHADOW_LOG_CSV)
    rows = load_shadow_calibration_log(str(jsonl_file))
    if not rows:
        logging.info("No shadow calibration rows found at %s", jsonl_file)
        return {
            "rows_total": 0,
            "rows_pending": 0,
            "rows_updated": 0,
            "tickers_checked": 0,
            "tickers_settled": 0,
            "tickers_unsettled": 0,
            "tickers_missing": 0,
        }

    pending_rows = [row for row in rows if row.get("realized_outcome") is None]
    pending_tickers = sorted({str(row.get("ticker", "")).strip() for row in pending_rows if row.get("ticker")})

    outcome_by_ticker: dict[str, Optional[bool]] = {}
    tickers_settled = 0
    tickers_unsettled = 0
    tickers_missing = 0

    for ticker in pending_tickers:
        market_payload = fetch_kalshi_market(ticker, client=client)
        if market_payload is None:
            outcome_by_ticker[ticker] = None
            tickers_missing += 1
            continue

        realized = _parse_kalshi_market_result(market_payload)
        outcome_by_ticker[ticker] = realized
        if realized is None:
            tickers_unsettled += 1
        else:
            tickers_settled += 1

        if KALSHI_BACKFILL_SLEEP_SEC > 0:
            time.sleep(KALSHI_BACKFILL_SLEEP_SEC)

    rows_updated = 0
    for row in rows:
        if row.get("realized_outcome") is not None:
            continue
        ticker = str(row.get("ticker", "")).strip()
        realized = outcome_by_ticker.get(ticker)
        if realized is None:
            continue
        row["realized_outcome"] = realized
        rows_updated += 1

    if rows_updated and not dry_run:
        _write_shadow_calibration_jsonl(rows, jsonl_file)
        _write_shadow_calibration_csv(rows, csv_file)
        logging.info(
            "Backfilled %s shadow rows across %s settled tickers (%s, %s).",
            rows_updated,
            tickers_settled,
            jsonl_file,
            csv_file,
        )
    elif rows_updated:
        logging.info(
            "Dry run: would backfill %s shadow rows across %s settled tickers.",
            rows_updated,
            tickers_settled,
        )
    else:
        logging.info("No settled outcomes available yet for pending shadow rows.")

    return {
        "rows_total": len(rows),
        "rows_pending": len(pending_rows),
        "rows_updated": rows_updated,
        "tickers_checked": len(pending_tickers),
        "tickers_settled": tickers_settled,
        "tickers_unsettled": tickers_unsettled,
        "tickers_missing": tickers_missing,
    }


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

def _parse_cli():
    import argparse

    parser = argparse.ArgumentParser(description="Kalshi prediction market bot")
    parser.add_argument(
        "--backfill-outcomes",
        action="store_true",
        help="Backfill realized_outcome in shadow calibration logs from settled Kalshi markets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --backfill-outcomes, report updates without writing log files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli()
    if args.backfill_outcomes:
        stats = backfill_shadow_realized_outcomes(dry_run=args.dry_run)
        print(json.dumps(stats, indent=2))
    else:
        main_loop()