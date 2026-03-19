from __future__ import annotations

import argparse
from itertools import product
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, List

import pandas as pd
import yfinance as yf
from backtesting import Backtest

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from equity_msft.backtest_strategy import MSFTDailyBacktestStrategy
except ModuleNotFoundError:
    from backtest_strategy import MSFTDailyBacktestStrategy

DEFAULT_DATA_PATH = BASE_DIR / "data" / "msft_daily.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "backtest_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic MSFT backtest using backtesting.py")
    parser.add_argument("--symbol", default="MSFT", help="Ticker symbol for data refresh mode")
    parser.add_argument("--start", default="2020-01-01", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"), help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--csv", default=str(DEFAULT_DATA_PATH), help="Path to cached OHLCV CSV")
    parser.add_argument("--refresh-data", action="store_true", help="Refresh cached CSV from Yahoo Finance")
    parser.add_argument("--initial-cash", type=float, default=100000.0, help="Initial portfolio cash")
    parser.add_argument("--commission-bps", type=float, default=5.0, help="Commission in basis points")
    parser.add_argument("--slippage-bps", type=float, default=5.0, help="Execution slippage in basis points")
    parser.add_argument("--risk-per-trade-pct", type=float, default=0.02, help="Risk budget per trade")
    parser.add_argument("--max-position-size-pct", type=float, default=0.20, help="Max position size as pct of equity")
    parser.add_argument("--max-drawdown-pct", type=float, default=0.10, help="Hard drawdown stop as decimal")
    parser.add_argument("--min-atr", type=float, default=3.5, help="Minimum ATR required to trade")
    parser.add_argument("--max-atr", type=float, default=18.0, help="Maximum ATR allowed to trade")
    parser.add_argument("--max-consecutive-losses", type=int, default=5, help="Loss streak entry block threshold (strategy class default=999; kept at 5 here for sweep backward compatibility)")
    parser.add_argument("--regime-caution-days", type=int, default=18, help="Days in adverse regime before soft 0.90x size damper applies")

    # Threshold tuning knobs (single run or sweep baseline)
    parser.add_argument("--neutral-rsi-low", type=float, default=40.0, help="Neutral-zone lower RSI bound for auto-HOLD")
    parser.add_argument("--neutral-rsi-high", type=float, default=60.0, help="Neutral-zone upper RSI bound for auto-HOLD")
    parser.add_argument("--neutral-rel-volume-max", type=float, default=1.3, help="Max relative volume for neutral auto-HOLD")
    parser.add_argument("--bullish-hold-rsi", type=float, default=62.0, help="Auto-HOLD in bullish trend above this RSI")
    parser.add_argument("--bearish-hold-rsi", type=float, default=38.0, help="Auto-HOLD in bearish trend below this RSI")
    parser.add_argument("--buy-pullback-rsi", type=float, default=46.0, help="BUY pullback trigger RSI cap in bullish trend")
    parser.add_argument("--sell-bearish-rsi", type=float, default=52.0, help="SELL trigger RSI floor in bearish trend")
    parser.add_argument("--sell-overbought-rsi", type=float, default=88.0, help="SELL trigger RSI floor for overbought exits")
    parser.add_argument("--take-profit-pnl-pct", type=float, default=7.0, help="Take-profit SELL trigger unrealized PnL percent")
    parser.add_argument("--take-profit-rsi", type=float, default=58.0, help="Take-profit SELL trigger RSI floor")
    parser.add_argument("--extreme-setup-rsi", type=float, default=30.0, help="Extreme setup RSI threshold")
    parser.add_argument("--extreme-setup-rel-vol", type=float, default=1.0, help="Extreme setup relative-volume threshold")

    # Parameter sweep mode
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep instead of a single backtest")
    parser.add_argument("--sweep-buy-pullback-rsi", default="42,46,50", help="Comma-separated BUY pullback RSI values")
    parser.add_argument("--sweep-sell-overbought-rsi", default="82,85,88", help="Comma-separated SELL overbought RSI values")
    parser.add_argument("--sweep-min-atr", default="0.5,1.5,2.5,3.5", help="Comma-separated minimum ATR values")
    parser.add_argument("--sweep-extreme-setup-rsi", default="25,30,35", help="Comma-separated extreme setup RSI values")
    parser.add_argument("--sweep-extreme-setup-rel-vol", default="0.9,1.0,1.1", help="Comma-separated extreme setup relative-volume values")
    parser.add_argument("--sweep-max-runs", type=int, default=200, help="Cap on number of sweep combinations to evaluate")
    parser.add_argument("--sweep-top-n", type=int, default=10, help="How many top combinations to save")
    parser.add_argument("--sweep-min-trades", type=int, default=1, help="Minimum trades required to be ranking-eligible")
    parser.add_argument(
        "--sweep-sort-by",
        choices=["return_pct", "sharpe_ratio", "profit_factor", "trade_count", "max_drawdown_abs"],
        default="return_pct",
        help="Ranking key for sweep results",
    )

    parser.add_argument("--plot", action="store_true", help="Show backtesting.py chart")
    return parser.parse_args()


def parse_csv_floats(raw: str, name: str) -> List[float]:
    values: List[float] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value '{token}' in {name}") from exc

    if not values:
        raise ValueError(f"No valid values provided for {name}")
    return values


def validate_args(args: argparse.Namespace) -> None:
    if args.min_atr > args.max_atr:
        raise ValueError("--min-atr cannot be greater than --max-atr")
    if args.neutral_rsi_low >= args.neutral_rsi_high:
        raise ValueError("--neutral-rsi-low must be less than --neutral-rsi-high")
    if args.sweep_max_runs < 1:
        raise ValueError("--sweep-max-runs must be at least 1")
    if args.sweep_top_n < 1:
        raise ValueError("--sweep-top-n must be at least 1")
    if args.sweep_min_trades < 0:
        raise ValueError("--sweep-min-trades cannot be negative")


def build_strategy_params(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "risk_per_trade_pct": args.risk_per_trade_pct,
        "max_position_size_pct": args.max_position_size_pct,
        "max_drawdown_pct": args.max_drawdown_pct,
        "min_atr": args.min_atr,
        "max_atr": args.max_atr,
        "max_consecutive_losses": args.max_consecutive_losses,
        "regime_caution_days": args.regime_caution_days,
        "slippage_bps": args.slippage_bps,
        "neutral_rsi_low": args.neutral_rsi_low,
        "neutral_rsi_high": args.neutral_rsi_high,
        "neutral_rel_volume_max": args.neutral_rel_volume_max,
        "bullish_hold_rsi": args.bullish_hold_rsi,
        "bearish_hold_rsi": args.bearish_hold_rsi,
        "buy_pullback_rsi": args.buy_pullback_rsi,
        "sell_bearish_rsi": args.sell_bearish_rsi,
        "sell_overbought_rsi": args.sell_overbought_rsi,
        "take_profit_pnl_pct": args.take_profit_pnl_pct,
        "take_profit_rsi": args.take_profit_rsi,
        "extreme_setup_rsi": args.extreme_setup_rsi,
        "extreme_setup_rel_vol": args.extreme_setup_rel_vol,
    }


def _download_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data downloaded for {symbol} ({start} to {end})")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]) for col in df.columns]

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Downloaded data missing columns: {missing}")

    df = df[required_cols].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.sort_index(inplace=True)
    return df


def load_or_refresh_data(
    symbol: str,
    csv_path: Path,
    start: str,
    end: str,
    refresh_data: bool,
) -> pd.DataFrame:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if refresh_data or not csv_path.exists():
        data = _download_data(symbol, start, end)
        data.to_csv(csv_path, index_label="Date")
    else:
        data = pd.read_csv(csv_path, parse_dates=["Date"])  # type: ignore[arg-type]
        data.set_index("Date", inplace=True)
        data.sort_index(inplace=True)

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    data = data[required_cols].copy()

    for col in required_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data.dropna(inplace=True)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    data = data[(data.index >= start_ts) & (data.index <= end_ts)]

    if len(data) < 220:
        raise ValueError(
            "Filtered dataset is too short for 200-SMA warmup. "
            f"Rows available: {len(data)}"
        )

    return data


def safe_float(stats: pd.Series, key: str) -> float | None:
    value = stats.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_backtest(
    data: pd.DataFrame,
    args: argparse.Namespace,
    strategy_overrides: Dict[str, Any] | None = None,
) -> pd.Series:
    total_execution_cost_rate = (args.commission_bps + args.slippage_bps) / 10000.0

    bt = Backtest(
        data,
        MSFTDailyBacktestStrategy,
        cash=args.initial_cash,
        commission=total_execution_cost_rate,
        trade_on_close=True,
        hedging=False,
        exclusive_orders=True,
        finalize_trades=True,
    )

    strategy_params = build_strategy_params(args)
    if strategy_overrides:
        strategy_params.update(strategy_overrides)

    return bt.run(**strategy_params)


def plot_backtest(data: pd.DataFrame, args: argparse.Namespace) -> None:
    """Run a single pass and render chart output."""
    total_execution_cost_rate = (args.commission_bps + args.slippage_bps) / 10000.0
    bt = Backtest(
        data,
        MSFTDailyBacktestStrategy,
        cash=args.initial_cash,
        commission=total_execution_cost_rate,
        trade_on_close=True,
        hedging=False,
        exclusive_orders=True,
        finalize_trades=True,
    )
    bt.run(**build_strategy_params(args))
    bt.plot()


def sort_sweep_results(results_df: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    if results_df.empty:
        return results_df

    ranked = results_df.copy()
    numeric_cols = [
        "return_pct",
        "max_drawdown_pct",
        "max_drawdown_abs",
        "win_rate_pct",
        "profit_factor",
        "sharpe_ratio",
        "trade_count",
    ]
    for col in numeric_cols:
        if col in ranked.columns:
            ranked[col] = pd.to_numeric(ranked[col], errors="coerce")

    ascending = sort_by == "max_drawdown_abs"
    sort_fill = float("inf") if ascending else float("-inf")
    ranked[sort_by] = ranked[sort_by].fillna(sort_fill)

    return ranked.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)


def run_parameter_sweep(
    data: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    buy_pullback_values = parse_csv_floats(args.sweep_buy_pullback_rsi, "--sweep-buy-pullback-rsi")
    sell_overbought_values = parse_csv_floats(args.sweep_sell_overbought_rsi, "--sweep-sell-overbought-rsi")
    min_atr_values = parse_csv_floats(args.sweep_min_atr, "--sweep-min-atr")
    extreme_rsi_values = parse_csv_floats(args.sweep_extreme_setup_rsi, "--sweep-extreme-setup-rsi")
    extreme_rel_vol_values = parse_csv_floats(args.sweep_extreme_setup_rel_vol, "--sweep-extreme-setup-rel-vol")

    all_combos = list(
        product(
            buy_pullback_values,
            sell_overbought_values,
            min_atr_values,
            extreme_rsi_values,
            extreme_rel_vol_values,
        )
    )

    combos = all_combos[: args.sweep_max_runs]
    rows: List[Dict[str, Any]] = []

    for idx, combo in enumerate(combos, start=1):
        (
            buy_pullback_rsi,
            sell_overbought_rsi,
            min_atr,
            extreme_setup_rsi,
            extreme_setup_rel_vol,
        ) = combo

        stats = run_backtest(
            data,
            args,
            strategy_overrides={
                "buy_pullback_rsi": buy_pullback_rsi,
                "sell_overbought_rsi": sell_overbought_rsi,
                "min_atr": min_atr,
                "extreme_setup_rsi": extreme_setup_rsi,
                "extreme_setup_rel_vol": extreme_setup_rel_vol,
            },
        )

        max_drawdown_pct = safe_float(stats, "Max. Drawdown [%]")
        rows.append(
            {
                "combo_index": idx,
                "buy_pullback_rsi": buy_pullback_rsi,
                "sell_overbought_rsi": sell_overbought_rsi,
                "min_atr": min_atr,
                "extreme_setup_rsi": extreme_setup_rsi,
                "extreme_setup_rel_vol": extreme_setup_rel_vol,
                "return_pct": safe_float(stats, "Return [%]"),
                "max_drawdown_pct": max_drawdown_pct,
                "max_drawdown_abs": abs(max_drawdown_pct) if max_drawdown_pct is not None else None,
                "win_rate_pct": safe_float(stats, "Win Rate [%]"),
                "profit_factor": safe_float(stats, "Profit Factor"),
                "sharpe_ratio": safe_float(stats, "Sharpe Ratio"),
                "trade_count": int(stats.get("# Trades", 0)),
            }
        )

    if not rows:
        raise ValueError("Sweep produced no runs. Check --sweep-* parameters.")

    all_results_df = pd.DataFrame(rows)
    eligible_df = all_results_df[all_results_df["trade_count"] >= args.sweep_min_trades].copy()
    if eligible_df.empty:
        eligible_df = all_results_df.copy()

    ranked_df = sort_sweep_results(eligible_df, args.sweep_sort_by)
    top_n = min(max(1, args.sweep_top_n), len(ranked_df))
    top_df = ranked_df.head(top_n).copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    sweep_results_path = output_dir / f"sweep_results_{args.symbol.lower()}_{run_id}.csv"
    sweep_top_path = output_dir / f"sweep_top_{args.symbol.lower()}_{run_id}.csv"
    best_config_path = output_dir / f"sweep_best_{args.symbol.lower()}_{run_id}.json"

    all_results_df.to_csv(sweep_results_path, index=False)
    top_df.to_csv(sweep_top_path, index=False)

    best_row = top_df.iloc[0].to_dict()
    summary = {
        "run_id": run_id,
        "symbol": args.symbol,
        "start": args.start,
        "end": args.end,
        "data_rows": int(len(data)),
        "total_combinations": len(all_combos),
        "executed_runs": len(combos),
        "ranking_pool": int(len(eligible_df)),
        "ranking_filter_min_trades": int(args.sweep_min_trades),
        "sort_by": args.sweep_sort_by,
        "best": best_row,
        "paths": {
            "all_results_csv": str(sweep_results_path),
            "top_results_csv": str(sweep_top_path),
        },
    }

    with open(best_config_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summary["paths"]["best_json"] = str(best_config_path)
    return summary


def write_outputs(
    output_dir: Path,
    symbol: str,
    stats: pd.Series,
    data_rows: int,
    config: Dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    trades = stats.get("_trades")
    if isinstance(trades, pd.DataFrame) and not trades.empty:
        trades.to_csv(output_dir / f"trades_{symbol.lower()}_{run_id}.csv", index=False)

    equity_curve = stats.get("_equity_curve")
    if isinstance(equity_curve, pd.DataFrame) and not equity_curve.empty:
        equity_curve.to_csv(output_dir / f"equity_{symbol.lower()}_{run_id}.csv")

    summary = {
        "run_id": run_id,
        "symbol": symbol,
        "start": str(stats.get("Start")),
        "end": str(stats.get("End")),
        "data_rows": int(data_rows),
        "equity_final": safe_float(stats, "Equity Final [$]"),
        "equity_peak": safe_float(stats, "Equity Peak [$]"),
        "return_pct": safe_float(stats, "Return [%]"),
        "buy_hold_return_pct": safe_float(stats, "Buy & Hold Return [%]"),
        "cagr_pct": safe_float(stats, "Return (Ann.) [%]"),
        "max_drawdown_pct": safe_float(stats, "Max. Drawdown [%]"),
        "win_rate_pct": safe_float(stats, "Win Rate [%]"),
        "profit_factor": safe_float(stats, "Profit Factor"),
        "sharpe_ratio": safe_float(stats, "Sharpe Ratio"),
        "trade_count": int(stats.get("# Trades", 0)),
        "config": config,
    }

    with open(output_dir / f"metrics_{symbol.lower()}_{run_id}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / f"stats_{symbol.lower()}_{run_id}.txt", "w", encoding="utf-8") as f:
        f.write(str(stats))


def main() -> None:
    args = parse_args()
    validate_args(args)

    csv_path = Path(args.csv).expanduser().resolve()
    output_dir = DEFAULT_OUTPUT_DIR

    data = load_or_refresh_data(
        symbol=args.symbol,
        csv_path=csv_path,
        start=args.start,
        end=args.end,
        refresh_data=args.refresh_data,
    )

    if args.sweep:
        if args.plot:
            print("Plot is ignored in --sweep mode.")

        sweep_summary = run_parameter_sweep(data, args, output_dir)
        best = sweep_summary["best"]

        print("Sweep complete")
        print(
            f"Executed runs: {sweep_summary['executed_runs']} / "
            f"{sweep_summary['total_combinations']} combinations"
        )
        print(f"Ranking key: {sweep_summary['sort_by']}")
        print(f"Best return [%]: {best.get('return_pct')}")
        print(f"Best max drawdown [%]: {best.get('max_drawdown_pct')}")
        print(f"Best trades: {best.get('trade_count')}")
        print(f"Best params: buy_pullback_rsi={best.get('buy_pullback_rsi')}, "
              f"sell_overbought_rsi={best.get('sell_overbought_rsi')}, "
              f"min_atr={best.get('min_atr')}, "
              f"extreme_setup_rsi={best.get('extreme_setup_rsi')}, "
              f"extreme_setup_rel_vol={best.get('extreme_setup_rel_vol')}")
        print(f"Outputs: {sweep_summary['paths']}")
        return

    stats = run_backtest(data, args)

    config = {
        "csv": str(csv_path),
        "start": args.start,
        "end": args.end,
        "initial_cash": args.initial_cash,
        "commission_bps": args.commission_bps,
        "slippage_bps": args.slippage_bps,
        "strategy_params": build_strategy_params(args),
    }

    write_outputs(output_dir, args.symbol, stats, len(data), config)

    print("Backtest complete")
    print(f"Data rows: {len(data)}")
    print(f"Return [%]: {stats.get('Return [%]')}")
    print(f"Max Drawdown [%]: {stats.get('Max. Drawdown [%]')}")
    print(f"Win Rate [%]: {stats.get('Win Rate [%]')}")
    print(f"Trades: {stats.get('# Trades')}")
    print(f"Outputs: {output_dir}")

    if args.plot:
        plot_backtest(data, args)


if __name__ == "__main__":
    main()
