import csv
import asyncio
import os
import time
import logging
import json
import sqlite3
import importlib
import inspect
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Coroutine, Optional, cast
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
# Kalshi SDK (install kalshi_python_sync or similar)
try:
    from kalshi_python_async import KalshiClient  # or your preferred client
    from kalshi_python_async.api_client import ApiClient
    from kalshi_python_async.configuration import Configuration
    from kalshi_python_async.api.market_api import MarketApi
except ImportError:
    KalshiClient = None  # type: ignore[assignment]
    ApiClient = None  # type: ignore[assignment]
    Configuration = None  # type: ignore[assignment]
    MarketApi = None  # type: ignore[assignment]
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
KALSHI_KEY_ID = os.getenv("KALSHI_KEY_ID")
KALSHI_PRIVATE_KEY = os.getenv("KALSHI_PRIVATE_KEY")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH")
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
SHADOW_DB_PATH = os.getenv("SHADOW_DB_PATH", str(BASE_DIR / "shadow_calibration.db"))
KALSHI_API_BASE = os.getenv("KALSHI_API_BASE", "https://api.elections.kalshi.com/trade-api/v2")
KALSHI_HOST = os.getenv("KALSHI_HOST", KALSHI_API_BASE)
KALSHI_MIN_VOLUME = _env_float("KALSHI_MIN_VOLUME", 10000)
KALSHI_MIN_DAYS_TO_CLOSE = _env_float("KALSHI_MIN_DAYS_TO_CLOSE", 2.0)
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


def _load_private_key_material() -> Optional[str]:
    if KALSHI_PRIVATE_KEY and KALSHI_PRIVATE_KEY.strip():
        return KALSHI_PRIVATE_KEY

    if KALSHI_PRIVATE_KEY_PATH and KALSHI_PRIVATE_KEY_PATH.strip():
        key_path = Path(KALSHI_PRIVATE_KEY_PATH).expanduser()
        if key_path.exists():
            return key_path.read_text(encoding="utf-8")
        logging.warning("KALSHI_PRIVATE_KEY_PATH does not exist: %s", key_path)

    return None


def _object_to_dict(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return {}


def _extract_collection(payload: Any, keys: tuple[str, ...]) -> list[Any]:
    if isinstance(payload, list):
        return payload

    as_dict = _object_to_dict(payload)
    for key in keys:
        if key in as_dict and isinstance(as_dict[key], list):
            return as_dict[key]

    for key in keys:
        if hasattr(payload, key):
            candidate = getattr(payload, key)
            if isinstance(candidate, list):
                return candidate

    return []


def _coerce_close_time_epoch(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric > 1e12:
            return numeric / 1000.0
        return numeric
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            pass
        normalized = text.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized).timestamp()
        except ValueError:
            return None
    return None


def _normalize_yes_price(raw: dict) -> Optional[float]:
    for key in ("yes_price", "yes_ask", "yes_bid", "yes", "yesPrice"):
        if key not in raw or raw[key] is None:
            continue
        try:
            value = float(raw[key])
        except (TypeError, ValueError):
            continue
        if 0.0 <= value <= 1.0:
            return value * 100.0
        return value
    return None


def _normalize_market_record(raw_record: Any) -> Optional[dict]:
    raw = _object_to_dict(raw_record)
    if not raw:
        return None

    ticker = raw.get("ticker") or raw.get("market_ticker") or raw.get("event_ticker")
    if not ticker:
        return None

    yes_price = _normalize_yes_price(raw)
    if yes_price is None:
        return None

    volume = raw.get("volume")
    if volume is None:
        volume = raw.get("open_interest", raw.get("liquidity", 0))
    try:
        volume = float(volume)
    except (TypeError, ValueError):
        volume = 0.0

    close_time = _coerce_close_time_epoch(
        raw.get("close_time")
        or raw.get("closeTime")
        or raw.get("expiration_time")
        or raw.get("expirationTime")
    )

    return {
        "ticker": str(ticker),
        "title": str(raw.get("title") or raw.get("subtitle") or raw.get("event_title") or ticker),
        "description": str(raw.get("description") or raw.get("rules_primary") or ""),
        "yes_price": float(_clamp(yes_price, 0.0, 100.0)),
        "volume": volume,
        "close_time": close_time if close_time is not None else 0.0,
    }


def _kalshi_auth_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if KALSHI_API_KEY:
        headers["Authorization"] = f"Bearer {KALSHI_API_KEY}"
    return headers


def _resolve_maybe_awaitable(result: Any) -> Any:
    if inspect.isawaitable(result):
        coroutine_result = cast(Coroutine[Any, Any, Any], result)
        try:
            return asyncio.run(coroutine_result)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coroutine_result)
            finally:
                loop.close()
    return result


def _sdk_call_with_fallback(api: Any, method_names: tuple[str, ...], kwargs: dict[str, Any]) -> Any:
    if api is None:
        return None

    for method_name in method_names:
        if not hasattr(api, method_name):
            continue
        method = getattr(api, method_name)
        try:
            return _resolve_maybe_awaitable(method(**kwargs))
        except TypeError:
            try:
                return _resolve_maybe_awaitable(method(kwargs))
            except Exception:
                continue
        except Exception:
            continue
    return None


def init_kalshi_client() -> dict[str, Any]:
    """Initialize Kalshi SDK clients when credentials are available."""
    if Configuration is None or ApiClient is None or MarketApi is None:
        logging.warning("Kalshi SDK not installed; falling back to HTTP-only mode.")
        return {
            "client": None,
            "api_client": None,
            "market_api": None,
            "wallet_api": None,
            "order_api": None,
        }

    private_key_material = _load_private_key_material()

    sdk_client = None
    api_client = None
    market_api = None
    wallet_api = None
    order_api = None

    if KALSHI_KEY_ID and private_key_material:
        try:
            config = Configuration()
            if hasattr(config, "host"):
                config.host = KALSHI_HOST
            if hasattr(config, "key_id"):
                config.key_id = KALSHI_KEY_ID
            if hasattr(config, "private_key"):
                config.private_key = private_key_material

            api_client = ApiClient(config)
            market_api = MarketApi(api_client)

            try:
                wallet_module = importlib.import_module("kalshi_python_async.api.wallet_api")
                wallet_api = wallet_module.WalletApi(api_client)
            except Exception:
                wallet_api = None

            try:
                order_module = importlib.import_module("kalshi_python_async.api.order_api")
                order_api = order_module.OrderApi(api_client)
            except Exception:
                order_api = None

            if KalshiClient is not None:
                try:
                    sdk_client = KalshiClient(config)
                except TypeError:
                    try:
                        sdk_client = KalshiClient(api_client)
                    except Exception:
                        sdk_client = None

            logging.info("Initialized Kalshi SDK client with key_id auth.")
        except Exception as exc:
            logging.warning("Kalshi SDK initialization failed: %s", exc)
    else:
        logging.warning(
            "Kalshi SDK credentials missing. Set KALSHI_KEY_ID and KALSHI_PRIVATE_KEY (or KALSHI_PRIVATE_KEY_PATH)."
        )

    return {
        "client": sdk_client,
        "api_client": api_client,
        "market_api": market_api,
        "wallet_api": wallet_api,
        "order_api": order_api,
    }


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


def _connect_shadow_db(db_path: Optional[str] = None) -> sqlite3.Connection:
    db_file = Path(db_path or SHADOW_DB_PATH)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_file)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA synchronous=NORMAL;")
    return connection


def init_shadow_db(db_path: Optional[str] = None) -> None:
    with _connect_shadow_db(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS shadow_calibration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                ticker TEXT NOT NULL,
                market_prob REAL NOT NULL,
                model_prob REAL NOT NULL,
                blended_prob REAL NOT NULL,
                confidence REAL NOT NULL,
                direction TEXT,
                edge REAL NOT NULL,
                realized_outcome INTEGER,
                shadow_mode INTEGER NOT NULL DEFAULT 1,
                executed INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_shadow_calibration_ticker ON shadow_calibration (ticker)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_shadow_calibration_outcome ON shadow_calibration (realized_outcome)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_shadow_calibration_timestamp ON shadow_calibration (timestamp)"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS executed_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                calibration_id INTEGER,
                timestamp TEXT NOT NULL,
                ticker TEXT NOT NULL,
                side TEXT NOT NULL,
                count INTEGER NOT NULL,
                yes_price INTEGER NOT NULL,
                status TEXT NOT NULL,
                response_json TEXT,
                error TEXT,
                FOREIGN KEY (calibration_id) REFERENCES shadow_calibration(id)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_executed_orders_calibration_id ON executed_orders (calibration_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_executed_orders_ticker ON executed_orders (ticker)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_executed_orders_timestamp ON executed_orders (timestamp)"
        )


def _bool_to_db(value: Optional[bool]) -> Optional[int]:
    if value is None:
        return None
    return 1 if value else 0


def _db_to_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return _normalize_realized_outcome(value)


def _insert_shadow_calibration_db(entry: dict, db_path: Optional[str] = None) -> int:
    init_shadow_db(db_path)
    with _connect_shadow_db(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO shadow_calibration (
                timestamp,
                ticker,
                market_prob,
                model_prob,
                blended_prob,
                confidence,
                direction,
                edge,
                realized_outcome,
                shadow_mode,
                executed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.get("timestamp"),
                entry.get("ticker"),
                entry.get("market_prob"),
                entry.get("model_prob"),
                entry.get("blended_prob"),
                entry.get("confidence"),
                entry.get("direction"),
                entry.get("edge"),
                _bool_to_db(_normalize_realized_outcome(entry.get("realized_outcome"))),
                _bool_to_db(_normalize_realized_outcome(entry.get("shadow_mode"))) or 0,
                _bool_to_db(_normalize_realized_outcome(entry.get("executed"))) or 0,
            ),
        )
    if cursor.lastrowid is None:
        raise RuntimeError("Failed to insert shadow_calibration row.")
    return int(cursor.lastrowid)


def _update_shadow_calibration_executed(
    calibration_id: int,
    executed: bool,
    db_path: Optional[str] = None,
) -> None:
    init_shadow_db(db_path)
    with _connect_shadow_db(db_path) as connection:
        connection.execute(
            "UPDATE shadow_calibration SET executed = ? WHERE id = ?",
            (_bool_to_db(executed) or 0, calibration_id),
        )


def log_executed_order(
    *,
    calibration_id: Optional[int],
    ticker: str,
    side: str,
    count: int,
    yes_price: int,
    status: str,
    response: Optional[Any] = None,
    error: Optional[str] = None,
    db_path: Optional[str] = None,
) -> None:
    init_shadow_db(db_path)
    response_json: Optional[str] = None
    if response is not None:
        try:
            response_json = json.dumps(response, default=str, ensure_ascii=True)
        except TypeError:
            response_json = json.dumps(_object_to_dict(response), default=str, ensure_ascii=True)

    with _connect_shadow_db(db_path) as connection:
        connection.execute(
            """
            INSERT INTO executed_orders (
                calibration_id,
                timestamp,
                ticker,
                side,
                count,
                yes_price,
                status,
                response_json,
                error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                calibration_id,
                datetime.now(timezone.utc).isoformat(),
                ticker,
                str(side).upper(),
                int(count),
                int(yes_price),
                status,
                response_json,
                error,
            ),
        )


def _load_shadow_calibration_from_db(db_path: Optional[str] = None) -> list[dict]:
    db_file = Path(db_path or SHADOW_DB_PATH)
    if not db_file.exists():
        return []

    init_shadow_db(str(db_file))
    with _connect_shadow_db(str(db_file)) as connection:
        rows = connection.execute(
            """
            SELECT
                timestamp,
                ticker,
                market_prob,
                model_prob,
                blended_prob,
                confidence,
                direction,
                edge,
                realized_outcome,
                shadow_mode,
                executed
            FROM shadow_calibration
            ORDER BY id ASC
            """
        ).fetchall()

    result: list[dict] = []
    for row in rows:
        realized = _db_to_bool(row["realized_outcome"])
        shadow_mode = _db_to_bool(row["shadow_mode"])
        executed = _db_to_bool(row["executed"])
        result.append(
            {
                "timestamp": row["timestamp"],
                "ticker": row["ticker"],
                "market_prob": float(row["market_prob"]),
                "model_prob": float(row["model_prob"]),
                "blended_prob": float(row["blended_prob"]),
                "confidence": float(row["confidence"]),
                "direction": row["direction"],
                "edge": float(row["edge"]),
                "realized_outcome": realized,
                "shadow_mode": True if shadow_mode is None else shadow_mode,
                "executed": False if executed is None else executed,
            }
        )
    return result


def _shadow_db_row_count(db_path: Optional[str] = None) -> int:
    db_file = Path(db_path or SHADOW_DB_PATH)
    if not db_file.exists():
        return 0
    init_shadow_db(str(db_file))
    with _connect_shadow_db(str(db_file)) as connection:
        row = connection.execute("SELECT COUNT(1) AS row_count FROM shadow_calibration").fetchone()
    if row is None:
        return 0
    return int(row["row_count"])


def _format_calibration_bucket(low: float, high: float) -> str:
    return f"[{low:.1f},{high:.1f})"


def get_shadow_summary(db_path: Optional[str] = None) -> dict[str, Any]:
    """Return lightweight summary stats for shadow calibration performance."""
    init_shadow_db(db_path)
    with _connect_shadow_db(db_path) as connection:
        totals_row = connection.execute(
            """
            SELECT
                COUNT(1) AS rows_total,
                SUM(CASE WHEN realized_outcome IS NULL THEN 1 ELSE 0 END) AS rows_pending,
                SUM(CASE WHEN realized_outcome IS NOT NULL THEN 1 ELSE 0 END) AS rows_resolved,
                AVG(edge) AS avg_edge,
                AVG(confidence) AS avg_confidence
            FROM shadow_calibration
            """
        ).fetchone()

        direction_row = connection.execute(
            """
            SELECT
                SUM(CASE WHEN direction = 'YES' THEN 1 ELSE 0 END) AS yes_signals,
                SUM(CASE WHEN direction = 'NO' THEN 1 ELSE 0 END) AS no_signals,
                SUM(CASE WHEN direction IS NULL OR direction = '' THEN 1 ELSE 0 END) AS no_trade_signals
            FROM shadow_calibration
            """
        ).fetchone()

        brier_row = connection.execute(
            """
            SELECT AVG((blended_prob - realized_outcome) * (blended_prob - realized_outcome)) AS brier_score
            FROM shadow_calibration
            WHERE realized_outcome IS NOT NULL
            """
        ).fetchone()

        calibration_rows = connection.execute(
            """
            SELECT
                CAST(blended_prob * 10 AS INTEGER) AS prob_bin,
                COUNT(1) AS samples,
                AVG(blended_prob) AS avg_pred,
                AVG(realized_outcome) AS realized_rate
            FROM shadow_calibration
            WHERE realized_outcome IS NOT NULL
            GROUP BY prob_bin
            ORDER BY prob_bin ASC
            """
        ).fetchall()

        order_rows = connection.execute(
            """
            SELECT
                COUNT(1) AS orders_total,
                SUM(CASE WHEN status = 'submitted' THEN 1 ELSE 0 END) AS orders_submitted,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS orders_failed
            FROM executed_orders
            """
        ).fetchone()

    rows_total = int(totals_row["rows_total"] or 0)
    rows_pending = int(totals_row["rows_pending"] or 0)
    rows_resolved = int(totals_row["rows_resolved"] or 0)

    summary: dict[str, Any] = {
        "db_path": str(Path(db_path or SHADOW_DB_PATH)),
        "rows_total": rows_total,
        "rows_pending": rows_pending,
        "rows_resolved": rows_resolved,
        "avg_edge": float(totals_row["avg_edge"]) if totals_row and totals_row["avg_edge"] is not None else None,
        "avg_confidence": float(totals_row["avg_confidence"]) if totals_row and totals_row["avg_confidence"] is not None else None,
        "yes_signals": int(direction_row["yes_signals"] or 0),
        "no_signals": int(direction_row["no_signals"] or 0),
        "no_trade_signals": int(direction_row["no_trade_signals"] or 0),
        "brier_score": float(brier_row["brier_score"]) if brier_row and brier_row["brier_score"] is not None else None,
        "orders_total": int(order_rows["orders_total"] or 0),
        "orders_submitted": int(order_rows["orders_submitted"] or 0),
        "orders_failed": int(order_rows["orders_failed"] or 0),
        "calibration_bins": [],
    }

    bins: list[dict[str, Any]] = []
    for row in calibration_rows:
        prob_bin = int(row["prob_bin"])
        low = max(0.0, min(0.9, prob_bin / 10.0))
        high = min(1.0, low + 0.1)
        bins.append(
            {
                "bucket": _format_calibration_bucket(low, high),
                "samples": int(row["samples"] or 0),
                "avg_pred": float(row["avg_pred"]),
                "realized_rate": float(row["realized_rate"]),
                "gap": float(row["avg_pred"] - row["realized_rate"]),
            }
        )
    summary["calibration_bins"] = bins
    return summary


def get_recent_executed_orders(limit: int = 20, db_path: Optional[str] = None) -> dict[str, Any]:
    """Return recent executed orders joined with calibration context."""
    init_shadow_db(db_path)
    query_limit = max(1, int(limit))
    with _connect_shadow_db(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                eo.id,
                eo.timestamp,
                eo.ticker,
                eo.side,
                eo.count,
                eo.yes_price,
                eo.status,
                eo.error,
                sc.edge,
                sc.confidence,
                sc.market_prob,
                sc.model_prob,
                sc.blended_prob,
                sc.direction,
                sc.executed
            FROM executed_orders eo
            LEFT JOIN shadow_calibration sc
                ON sc.id = eo.calibration_id
            ORDER BY eo.id DESC
            LIMIT ?
            """,
            (query_limit,),
        ).fetchall()

    items: list[dict[str, Any]] = []
    for row in rows:
        items.append(
            {
                "id": int(row["id"]),
                "timestamp": row["timestamp"],
                "ticker": row["ticker"],
                "side": row["side"],
                "count": int(row["count"]),
                "yes_price": int(row["yes_price"]),
                "status": row["status"],
                "error": row["error"],
                "edge": float(row["edge"]) if row["edge"] is not None else None,
                "confidence": float(row["confidence"]) if row["confidence"] is not None else None,
                "market_prob": float(row["market_prob"]) if row["market_prob"] is not None else None,
                "model_prob": float(row["model_prob"]) if row["model_prob"] is not None else None,
                "blended_prob": float(row["blended_prob"]) if row["blended_prob"] is not None else None,
                "direction": row["direction"],
                "executed": _db_to_bool(row["executed"]),
            }
        )

    return {
        "db_path": str(Path(db_path or SHADOW_DB_PATH)),
        "limit": query_limit,
        "count": len(items),
        "orders": items,
    }


def get_failed_orders_report(limit: int = 20, db_path: Optional[str] = None) -> dict[str, Any]:
    """Return recent failed orders plus grouped error summaries."""
    init_shadow_db(db_path)
    query_limit = max(1, int(limit))
    with _connect_shadow_db(db_path) as connection:
        failed_rows = connection.execute(
            """
            SELECT
                eo.id,
                eo.timestamp,
                eo.ticker,
                eo.side,
                eo.count,
                eo.yes_price,
                eo.status,
                eo.error,
                sc.edge,
                sc.confidence,
                sc.market_prob,
                sc.model_prob,
                sc.blended_prob,
                sc.direction,
                sc.executed
            FROM executed_orders eo
            LEFT JOIN shadow_calibration sc
                ON sc.id = eo.calibration_id
            WHERE eo.status = 'failed'
            ORDER BY eo.id DESC
            LIMIT ?
            """,
            (query_limit,),
        ).fetchall()

        grouped_rows = connection.execute(
            """
            SELECT
                COALESCE(NULLIF(TRIM(error), ''), '<empty>') AS error_text,
                COUNT(1) AS failures
            FROM executed_orders
            WHERE status = 'failed'
            GROUP BY COALESCE(NULLIF(TRIM(error), ''), '<empty>')
            ORDER BY failures DESC, error_text ASC
            """
        ).fetchall()

        totals_row = connection.execute(
            """
            SELECT
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failures_total,
                SUM(CASE WHEN status = 'submitted' THEN 1 ELSE 0 END) AS submitted_total
            FROM executed_orders
            """
        ).fetchone()

    failed_orders: list[dict[str, Any]] = []
    for row in failed_rows:
        failed_orders.append(
            {
                "id": int(row["id"]),
                "timestamp": row["timestamp"],
                "ticker": row["ticker"],
                "side": row["side"],
                "count": int(row["count"]),
                "yes_price": int(row["yes_price"]),
                "status": row["status"],
                "error": row["error"],
                "edge": float(row["edge"]) if row["edge"] is not None else None,
                "confidence": float(row["confidence"]) if row["confidence"] is not None else None,
                "market_prob": float(row["market_prob"]) if row["market_prob"] is not None else None,
                "model_prob": float(row["model_prob"]) if row["model_prob"] is not None else None,
                "blended_prob": float(row["blended_prob"]) if row["blended_prob"] is not None else None,
                "direction": row["direction"],
                "executed": _db_to_bool(row["executed"]),
            }
        )

    grouped_errors: list[dict[str, Any]] = []
    for row in grouped_rows:
        grouped_errors.append(
            {
                "error": row["error_text"],
                "failures": int(row["failures"]),
            }
        )

    failures_total = int(totals_row["failures_total"] or 0) if totals_row else 0
    submitted_total = int(totals_row["submitted_total"] or 0) if totals_row else 0
    attempted_total = failures_total + submitted_total
    failure_rate = (failures_total / attempted_total) if attempted_total > 0 else None

    return {
        "db_path": str(Path(db_path or SHADOW_DB_PATH)),
        "limit": query_limit,
        "failures_total": failures_total,
        "submitted_total": submitted_total,
        "attempted_total": attempted_total,
        "failure_rate": failure_rate,
        "grouped_errors": grouped_errors,
        "failed_orders": failed_orders,
    }


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

    _insert_shadow_calibration_db(entry)

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


def load_shadow_calibration_log(jsonl_path: Optional[str] = None, db_path: Optional[str] = None) -> list[dict]:
    """Load shadow calibration rows from JSONL."""
    db_rows = _load_shadow_calibration_from_db(db_path)
    if db_rows:
        return db_rows

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

    # One-time bootstrap: if DB is empty, import existing JSONL history.
    if rows and _shadow_db_row_count(db_path) == 0:
        for row in rows:
            _insert_shadow_calibration_db(row, db_path=db_path)

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
    sdk_client = client.get("client") if isinstance(client, dict) else client
    market_api = client.get("market_api") if isinstance(client, dict) else None

    if sdk_client is not None:
        try:
            if hasattr(sdk_client, "get_market"):
                response = sdk_client.get_market(ticker=ticker)
            elif hasattr(sdk_client, "markets_api") and hasattr(sdk_client.markets_api, "get_market"):
                response = sdk_client.markets_api.get_market(ticker=ticker)
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

    if market_api is not None:
        response = _sdk_call_with_fallback(
            market_api,
            ("get_market", "market_get"),
            {"ticker": ticker},
        )
        if response is not None:
            payload = _object_to_dict(response)
            return payload.get("market", payload)

    url = f"{KALSHI_API_BASE.rstrip('/')}/markets/{ticker}"
    try:
        resp = requests.get(url, headers=_kalshi_auth_headers(), timeout=15)
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
    db_file = Path(SHADOW_DB_PATH)
    rows = load_shadow_calibration_log(str(jsonl_file), db_path=str(db_file))
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
        with _connect_shadow_db(str(db_file)) as connection:
            for ticker, realized in outcome_by_ticker.items():
                if realized is None:
                    continue
                connection.execute(
                    """
                    UPDATE shadow_calibration
                    SET realized_outcome = ?
                    WHERE ticker = ?
                      AND realized_outcome IS NULL
                    """,
                    (_bool_to_db(realized), ticker),
                )

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
    sdk_client = client.get("client") if isinstance(client, dict) else client
    market_api = client.get("market_api") if isinstance(client, dict) else None

    raw_markets: list[Any] = []

    if sdk_client is not None:
        response = _sdk_call_with_fallback(
            sdk_client,
            ("get_markets", "list_markets"),
            {"status": "open", "limit": limit},
        )
        if response is not None:
            raw_markets = _extract_collection(response, ("markets", "data", "results"))

    if not raw_markets and market_api is not None:
        response = _sdk_call_with_fallback(
            market_api,
            ("get_markets", "list_markets"),
            {"status": "open", "limit": limit},
        )
        if response is not None:
            raw_markets = _extract_collection(response, ("markets", "data", "results"))

    if not raw_markets:
        url = f"{KALSHI_API_BASE.rstrip('/')}/markets"
        params = {"status": "open", "limit": limit}
        resp = requests.get(url, params=params, headers=_kalshi_auth_headers(), timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        raw_markets = payload.get("markets", payload.get("data", payload if isinstance(payload, list) else []))

    normalized_markets = []
    for raw_market in raw_markets:
        record = _normalize_market_record(raw_market)
        if record is not None:
            normalized_markets.append(record)

    df = pd.DataFrame(normalized_markets)
    if df.empty:
        return df

    min_close_time = time.time() + 86400 * KALSHI_MIN_DAYS_TO_CLOSE
    df = df[(df['volume'] > KALSHI_MIN_VOLUME) & (df['close_time'] > min_close_time)]
    return df


def get_kalshi_balance(client: Any) -> float:
    """Get available cash/balance from SDK if possible, else HTTP fallback."""
    sdk_client = client.get("client") if isinstance(client, dict) else client
    wallet_api = client.get("wallet_api") if isinstance(client, dict) else None

    if sdk_client is not None:
        for method_name in ("get_balance", "get_cash_balance", "get_portfolio_balance"):
            if not hasattr(sdk_client, method_name):
                continue
            method = getattr(sdk_client, method_name)
            try:
                result = _resolve_maybe_awaitable(method())
                if isinstance(result, (int, float)):
                    return float(result)
                as_dict = _object_to_dict(result)
                for key in ("balance", "cash", "available_balance", "available_cash"):
                    if key in as_dict:
                        return float(as_dict[key])
            except Exception:
                continue

    if wallet_api is not None:
        response = _sdk_call_with_fallback(
            wallet_api,
            ("get_balance", "get_portfolio_balance", "get_wallet_balance"),
            {},
        )
        if response is not None:
            as_dict = _object_to_dict(response)
            for key in ("balance", "cash", "available_balance", "available_cash"):
                if key in as_dict:
                    return float(as_dict[key])

    url = f"{KALSHI_API_BASE.rstrip('/')}/portfolio/balance"
    resp = requests.get(url, headers=_kalshi_auth_headers(), timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    balance = payload.get("balance", payload.get("cash", payload.get("available_balance")))
    if balance is None:
        raise RuntimeError("Unable to determine Kalshi balance from API response.")
    return float(balance)

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
    if count <= 0:
        logging.info("Skip order with non-positive size: %s", count)
        return None

    sdk_client = client.get("client") if isinstance(client, dict) else client
    order_api = client.get("order_api") if isinstance(client, dict) else None

    side_value = str(side).upper()
    yes_price = int(round(_clamp(float(price), 0.0, 1.0) * 100))
    order_payload = {
        "ticker": ticker,
        "side": side_value,
        "count": int(count),
        "yes_price": yes_price,
        "order_type": "market",
    }

    if sdk_client is not None:
        response = _sdk_call_with_fallback(
            sdk_client,
            ("place_order", "create_order", "post_order"),
            order_payload,
        )
        if response is not None:
            logging.info("Placed %s order via SDK client: %s x%s @ %s", side_value, ticker, count, yes_price)
            return response

    if order_api is not None:
        response = _sdk_call_with_fallback(
            order_api,
            ("create_order", "place_order", "post_order"),
            order_payload,
        )
        if response is not None:
            logging.info("Placed %s order via SDK order API: %s x%s @ %s", side_value, ticker, count, yes_price)
            return response

    url = f"{KALSHI_API_BASE.rstrip('/')}/portfolio/orders"
    resp = requests.post(url, json=order_payload, headers=_kalshi_auth_headers(), timeout=20)
    resp.raise_for_status()
    response_json = resp.json()
    logging.info("Placed %s order via HTTP: %s x%s @ %s", side_value, ticker, count, yes_price)
    return response_json

def main_loop():
    client = init_kalshi_client()
    init_shadow_db()
    if SHADOW_MODE:
        logging.info(
            "SHADOW_MODE enabled — logging calibration rows to %s, %s, and %s (no execution).",
            SHADOW_DB_PATH,
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
                calibration_entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "ticker": ticker,
                    "market_prob": round(market_prob, 6),
                    "model_prob": round(grok_data['prob'], 6),
                    "blended_prob": round(blended_prob, 6),
                    "confidence": round(grok_data['confidence'], 6),
                    "direction": direction,
                    "edge": round(edge, 6),
                    "realized_outcome": None,
                    "shadow_mode": False,
                    "executed": False,
                }
                calibration_id = _insert_shadow_calibration_db(calibration_entry)

                size = calculate_position_size(
                    edge,
                    bankroll=get_kalshi_balance(client),
                    grok_confidence=grok_data['confidence'],
                )
                if size > 0:
                    yes_price = int(round(_clamp(float(market_prob), 0.0, 1.0) * 100))
                    try:
                        response = place_kalshi_order(client, ticker, direction, size, market_prob)
                        _update_shadow_calibration_executed(calibration_id, executed=response is not None)
                        log_executed_order(
                            calibration_id=calibration_id,
                            ticker=ticker,
                            side=direction,
                            count=size,
                            yes_price=yes_price,
                            status="submitted",
                            response=response,
                        )
                        logging.info(
                            "Edge trade: %s | Edge: %.2f%% | market=%.3f model=%.3f blend=%.3f conf=%.2f",
                            desc,
                            edge * 100.0,
                            market_prob,
                            grok_data['prob'],
                            blended_prob,
                            grok_data['confidence'],
                        )
                    except Exception as exc:
                        _update_shadow_calibration_executed(calibration_id, executed=False)
                        log_executed_order(
                            calibration_id=calibration_id,
                            ticker=ticker,
                            side=direction,
                            count=size,
                            yes_price=yes_price,
                            status="failed",
                            error=str(exc),
                        )
                        logging.exception("Order placement failed for %s: %s", ticker, exc)
        
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
    parser.add_argument(
        "--shadow-summary",
        action="store_true",
        help="Print SQLite shadow calibration summary metrics as JSON.",
    )
    parser.add_argument(
        "--orders-recent",
        nargs="?",
        const=20,
        type=int,
        metavar="N",
        help="Print the most recent N executed orders joined with calibration context (default 20).",
    )
    parser.add_argument(
        "--orders-failures",
        nargs="?",
        const=20,
        type=int,
        metavar="N",
        help="Print recent failed orders plus grouped error summaries (default 20).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli()
    if args.backfill_outcomes:
        stats = backfill_shadow_realized_outcomes(dry_run=args.dry_run)
        print(json.dumps(stats, indent=2))
    elif args.shadow_summary:
        stats = get_shadow_summary()
        print(json.dumps(stats, indent=2))
    elif args.orders_recent is not None:
        stats = get_recent_executed_orders(limit=args.orders_recent)
        print(json.dumps(stats, indent=2))
    elif args.orders_failures is not None:
        stats = get_failed_orders_report(limit=args.orders_failures)
        print(json.dumps(stats, indent=2))
    else:
        main_loop()