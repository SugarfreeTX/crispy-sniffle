from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
import vectorbt as vbt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_RAW_CSV = DATA_DIR / "xrp_4h.csv"
DEFAULT_CLEAN_CSV = DATA_DIR / "xrp_4h_clean.csv"


@dataclass(frozen=True)
class Config:
    """Strategy and execution parameters for XRP 4H backtest."""

    initial_cash: float = 100000.0
    commission_bps: float = 5.0
    slippage_bps: float = 5.0

    # Fixed percent allocation per trade (independent of ATR)
    position_size_pct: float = 0.12

    # Trend and momentum filters on 4H closes
    sma_fast: int = 12
    sma_slow: int = 42
    sma_long: int = 90
    sell_rsi: float = 78.0

    # ATR used only as a volatility filter (not for sizing)
    atr_window: int = 14
    atr_filter_min_pct: float = 0.003
    atr_filter_max_pct: float = 0.06

    # Stops are fractional distances, e.g. 0.012 = 1.2%
    sl_stop: float = 0.012
    tp_stop: float = 0.05

    # Evaluate conditions at close, execute at next bar open
    freq: str = "4h"


def parse_args() -> argparse.Namespace:
    """Parse CLI options for data input and run mode."""

    parser = argparse.ArgumentParser(description="XRP 4H vectorbt strategy (v2).")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV path override. Defaults to cleaned file if present, else raw.",
    )
    parser.add_argument(
        "--save-stats",
        type=str,
        default="",
        help="Optional path to save stats as CSV.",
    )
    return parser.parse_args()


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load OHLCV data and enforce basic quality constraints.

    Args:
        csv_path: Input CSV path with Date/Open/High/Low/Close/Volume.

    Returns:
        DataFrame indexed by UTC Date with OHLCV columns.

    Raises:
        ValueError: If required columns are missing or no rows remain.
    """

    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = pd.read_csv(csv_path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_cols].copy()
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

    # Keep conservative OHLC validity guards.
    valid = (
        (df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
        & (df["High"] >= df[["Open", "Close"]].max(axis=1))
        & (df["Low"] <= df[["Open", "Close"]].min(axis=1))
        & (df["High"] >= df["Low"])
    )
    df = df.loc[valid].copy()

    if df.empty:
        raise ValueError("No valid rows available after data quality filtering")

    return df.set_index("Date")


def create_signals(data: pd.DataFrame, cfg: Config) -> Tuple[pd.Series, pd.Series]:
    """Create entry and exit signals from close-based indicators.

    Signals are generated from close-bar information only and are shifted by one
    bar so orders execute on the next candle open.

    Args:
        data: OHLCV DataFrame indexed by timestamp.
        cfg: Strategy configuration.

    Returns:
        Tuple of (entries_exec, exits_exec) boolean Series aligned to execution bar.
    """

    close = data["Close"]
    high = data["High"]
    low = data["Low"]

    rsi = vbt.RSI.run(close, window=14).rsi # type: ignore
    atr = vbt.ATR.run(high, low, close, window=cfg.atr_window).atr # type: ignore
    sma_fast = vbt.MA.run(close, window=cfg.sma_fast).ma # type: ignore
    sma_slow = vbt.MA.run(close, window=cfg.sma_slow).ma # type: ignore
    sma_long = vbt.MA.run(close, window=cfg.sma_long).ma # type: ignore 

    atr_pct = atr / close
    atr_ok = (atr_pct >= cfg.atr_filter_min_pct) & (atr_pct <= cfg.atr_filter_max_pct)

    long_term_uptrend = close > sma_long
    short_term_momentum = close > sma_fast

    entries_close = long_term_uptrend & short_term_momentum & (close > sma_slow) & atr_ok
    exits_close = (close < sma_fast) | (rsi >= cfg.sell_rsi)

    # Shift to enforce next-bar execution at open.
    entries_exec = entries_close.shift(1, fill_value=False)
    exits_exec = exits_close.shift(1, fill_value=False)

    return entries_exec.astype(bool), exits_exec.astype(bool)


def run_backtest(data: pd.DataFrame, cfg: Config) -> vbt.Portfolio:
    """Run vectorbt portfolio simulation with open-price execution.

    Args:
        data: Valid OHLCV data.
        cfg: Strategy configuration.

    Returns:
        vectorbt Portfolio instance.
    """

    entries, exits = create_signals(data, cfg)

    pf = vbt.Portfolio.from_signals(
        close=data["Open"],
        entries=entries,
        exits=exits,
        size=cfg.position_size_pct,
        size_type="percent",
        init_cash=cfg.initial_cash,
        fees=cfg.commission_bps / 10000.0,
        slippage=cfg.slippage_bps / 10000.0,
        direction="longonly",
        freq=cfg.freq,
        sl_stop=cfg.sl_stop,
        tp_stop=cfg.tp_stop,
    )

    return pf


def resolve_csv_path(cli_path: str | None) -> Path:
    """Resolve input data source path.

    Priority:
    1. Explicit --csv path
    2. cleaned CSV if present
    3. raw CSV fallback
    """

    if cli_path:
        return Path(cli_path)
    if DEFAULT_CLEAN_CSV.exists():
        return DEFAULT_CLEAN_CSV
    return DEFAULT_RAW_CSV


def main() -> None:
    """CLI entrypoint for running strategy v2 and printing key stats."""

    args = parse_args()
    csv_path = resolve_csv_path(args.csv)

    cfg = Config()
    data = load_data(csv_path)
    pf = run_backtest(data, cfg)

    stats = pf.stats()
    if stats is None:
        raise ValueError("Portfolio stats could not be generated. Ensure the portfolio contains valid data.")

    print(f"Using data: {csv_path}")
    print(stats)

    if args.save_stats:
        save_path = Path(args.save_stats)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(save_path)
        print(f"Saved stats: {save_path}")


if __name__ == "__main__":
    main()
