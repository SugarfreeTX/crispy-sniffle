from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EQUITY_DIR = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EQUITY_DIR) not in sys.path:
    sys.path.insert(0, str(EQUITY_DIR))

try:
    from equity_msft.backtest import (
        DEFAULT_DATA_PATH,
        load_or_refresh_data,
        run_backtest,
        safe_float,
    )
    from equity_msft.research_loop.pipeline import Config, config_to_dict
except ModuleNotFoundError:
    from backtest import (  # type: ignore
        DEFAULT_DATA_PATH,
        load_or_refresh_data,
        run_backtest,
        safe_float,
    )
    from pipeline import Config, config_to_dict  # type: ignore


MIN_BARS_FOR_SMA = 220
TRADING_DAYS_PER_YEAR = 252


def resolve_csv_path(csv_path: str | Path | None = None) -> Path:
    if csv_path:
        return Path(csv_path).expanduser().resolve()
    return Path(DEFAULT_DATA_PATH).expanduser().resolve()


def load_data(
    csv_path: str | Path | None = None,
    *,
    start: str = "2018-01-01",
    end: str | None = None,
    refresh_data: bool = False,
    symbol: str = "MSFT",
) -> pd.DataFrame:
    resolved = resolve_csv_path(csv_path)
    end_date = end or datetime.now().strftime("%Y-%m-%d")
    return load_or_refresh_data(
        symbol=symbol,
        csv_path=resolved,
        start=start,
        end=end_date,
        refresh_data=refresh_data,
    )


def config_to_args(cfg: Config) -> SimpleNamespace:
    return SimpleNamespace(**config_to_dict(cfg))


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def _pct_to_frac(value: Any) -> float:
    return _finite(value, 0.0) / 100.0


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def split_search_holdout(
    data: pd.DataFrame,
    holdout_start: str | None,
    holdout_end: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not holdout_start:
        return data, data.iloc[0:0].copy()

    start_ts = pd.Timestamp(holdout_start)
    search = data[data.index < start_ts].copy()
    holdout = data[data.index >= start_ts].copy()
    if holdout_end:
        holdout = holdout[holdout.index <= pd.Timestamp(holdout_end)].copy()
    return search, holdout


def with_warmup(history: pd.DataFrame, window: pd.DataFrame, warmup_bars: int) -> pd.DataFrame:
    """Prepend prior bars so 200-SMA / ATR / RSI are live on the first test bar."""
    if window.empty:
        return window
    if warmup_bars <= 0:
        return window.copy()

    first_ts = window.index[0]
    prior = history[history.index < first_ts].iloc[-warmup_bars:]
    combined = pd.concat([prior, window])
    return combined[~combined.index.duplicated(keep="last")].sort_index()


def _equity_frame(stats: pd.Series) -> pd.DataFrame | None:
    equity = stats.get("_equity_curve")
    if not isinstance(equity, pd.DataFrame) or equity.empty:
        return None
    frame = equity.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        if "Date" in frame.columns:
            frame = frame.set_index("Date")
        else:
            frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame.sort_index()
    return frame


def _window_slice(frame: pd.DataFrame, start: Any | None, end: Any | None) -> pd.DataFrame:
    sliced = frame
    if start is not None:
        sliced = sliced[sliced.index >= pd.Timestamp(start)]
    if end is not None:
        sliced = sliced[sliced.index <= pd.Timestamp(end)]
    return sliced


def _sharpe_from_equity(equity: pd.Series) -> float:
    returns = equity.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    std = float(returns.std(ddof=0))
    if std <= 0:
        return 0.0
    return float((returns.mean() / std) * math.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown_from_equity(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return abs(float(dd.min())) if len(dd) else 0.0


def _count_trades(stats: pd.Series, start: Any | None, end: Any | None) -> int:
    trades = stats.get("_trades")
    if not isinstance(trades, pd.DataFrame) or trades.empty:
        raw = stats.get("# Trades", 0)
        return int(_finite(raw, 0.0))

    if start is None and end is None:
        return int(len(trades))

    start_ts = pd.Timestamp(start) if start is not None else None
    end_ts = pd.Timestamp(end) if end is not None else None

    if "EntryTime" in trades.columns:
        times = pd.to_datetime(trades["EntryTime"])
        mask = pd.Series(True, index=trades.index)
        if start_ts is not None:
            mask &= times >= start_ts
        if end_ts is not None:
            mask &= times <= end_ts
        return int(mask.sum())

    return int(len(trades))


def extract_metrics(
    stats: pd.Series,
    *,
    score_start: Any | None = None,
    score_end: Any | None = None,
) -> dict[str, Any]:
    """Normalize backtesting.py stats to the research-loop metric schema.

    Returns are fractions (0.23 = 23%). Drawdown is an absolute fraction.
    When score_start/score_end are set, return/drawdown/sharpe/trades are
    computed on that window so walk-forward warmup bars do not dilute the score.
    """
    equity = _equity_frame(stats)
    use_window = score_start is not None or score_end is not None

    if use_window and equity is not None and "Equity" in equity.columns:
        window = _window_slice(equity, score_start, score_end)
        if len(window) >= 2:
            eq = window["Equity"].astype(float)
            start_eq = float(eq.iloc[0])
            end_eq = float(eq.iloc[-1])
            total_return = (end_eq / start_eq) - 1.0 if start_eq > 0 else 0.0
            n_days = max(len(eq) - 1, 1)
            annualized = (1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0
            metrics = {
                "total_return": _finite(total_return),
                "annualized_return": _finite(annualized),
                "cagr": _finite(annualized),
                "sharpe": _finite(_sharpe_from_equity(eq)),
                "max_drawdown": _finite(_max_drawdown_from_equity(eq)),
                "num_trades": _count_trades(stats, score_start, score_end),
                "equity_final": _finite(end_eq),
                "equity_peak": _finite(float(eq.max())),
                "start": str(window.index[0]),
                "end": str(window.index[-1]),
                "scored_window": True,
            }
            metrics["buy_hold_return"] = _pct_to_frac(safe_float(stats, "Buy & Hold Return [%]"))
            metrics["win_rate"] = _pct_to_frac(safe_float(stats, "Win Rate [%]"))
            metrics["profit_factor"] = _finite(safe_float(stats, "Profit Factor"))
            return _json_safe(metrics)

    return _json_safe(
        {
            "total_return": _pct_to_frac(safe_float(stats, "Return [%]")),
            "annualized_return": _pct_to_frac(safe_float(stats, "Return (Ann.) [%]")),
            "cagr": _pct_to_frac(safe_float(stats, "Return (Ann.) [%]")),
            "buy_hold_return": _pct_to_frac(safe_float(stats, "Buy & Hold Return [%]")),
            "sharpe": _finite(safe_float(stats, "Sharpe Ratio")),
            "max_drawdown": abs(_pct_to_frac(safe_float(stats, "Max. Drawdown [%]"))),
            "win_rate": _pct_to_frac(safe_float(stats, "Win Rate [%]")),
            "profit_factor": _finite(safe_float(stats, "Profit Factor")),
            "num_trades": int(_finite(stats.get("# Trades", 0), 0.0)),
            "equity_final": _finite(safe_float(stats, "Equity Final [$]")),
            "equity_peak": _finite(safe_float(stats, "Equity Peak [$]")),
            "start": str(stats.get("Start")),
            "end": str(stats.get("End")),
            "scored_window": False,
        }
    )


def run_backtest_and_extract_metrics(
    data: pd.DataFrame,
    cfg: Config,
    *,
    score_start: Any | None = None,
    score_end: Any | None = None,
) -> dict[str, Any]:
    if len(data) < MIN_BARS_FOR_SMA:
        raise ValueError(
            f"Need at least {MIN_BARS_FOR_SMA} bars for 200-SMA warmup, got {len(data)}"
        )
    stats = run_backtest(data, config_to_args(cfg))
    return extract_metrics(stats, score_start=score_start, score_end=score_end)
