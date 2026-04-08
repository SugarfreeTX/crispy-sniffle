from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
STRATEGY_FILE = BASE_DIR / "xrp_vectorbt_best_v2.py"
DATA_CLEAN = BASE_DIR / "data" / "xrp_4h_clean.csv"
DATA_RAW = BASE_DIR / "data" / "xrp_4h.csv"


def load_strategy_module():
    """Load the v2 strategy module from file."""

    spec = importlib.util.spec_from_file_location("xrp_v2", STRATEGY_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load xrp_vectorbt_best_v2.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["xrp_v2"] = module
    spec.loader.exec_module(module)
    return module


def resolve_data_path(cli_csv: str) -> Path:
    """Resolve input data path from CLI override or defaults."""

    if cli_csv:
        return Path(cli_csv)
    if DATA_CLEAN.exists():
        return DATA_CLEAN
    return DATA_RAW


def run_walkforward_windows(module, data: pd.DataFrame, cfg, train_days: int, test_days: int, step_days: int) -> pd.DataFrame:
    """Run walk-forward test windows and return per-window metrics."""

    bars_per_day = 6
    train_bars = train_days * bars_per_day
    test_bars = test_days * bars_per_day
    step_bars = step_days * bars_per_day

    rows: List[Dict[str, float | int | str]] = []
    start = 0
    n = len(data)

    while start + train_bars + test_bars <= n:
        test_start = start + train_bars
        test_end = test_start + test_bars
        test_slice = data.iloc[test_start:test_end]

        pf = module.run_backtest(test_slice, cfg)
        st = pf.stats()

        rows.append(
            {
                "test_start": str(test_slice.index[0]),
                "test_end": str(test_slice.index[-1]),
                "bars": int(len(test_slice)),
                "Total Return [%]": float(st.get("Total Return [%]", 0.0)),
                "Benchmark Return [%]": float(st.get("Benchmark Return [%]", 0.0)),
                "Max Drawdown [%]": float(st.get("Max Drawdown [%]", 0.0)),
                "Total Trades": int(st.get("Total Trades", 0)),
                "Win Rate [%]": float(st.get("Win Rate [%]", 0.0)),
                "Profit Factor": float(st.get("Profit Factor", 0.0)),
                "Sharpe Ratio": float(st.get("Sharpe Ratio", 0.0)),
                "Calmar Ratio": float(st.get("Calmar Ratio", 0.0)),
                "Expectancy": float(st.get("Expectancy", 0.0)),
            }
        )

        start += step_bars

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> Dict[str, float | int]:
    """Compute summary stats from a walk-forward result table."""

    if df.empty:
        return {
            "windows": 0,
            "positive_windows": 0,
            "negative_windows": 0,
            "avg_oos_return_pct": 0.0,
            "median_oos_return_pct": 0.0,
            "avg_oos_drawdown_pct": 0.0,
            "avg_oos_sharpe": 0.0,
            "avg_oos_profit_factor": 0.0,
            "avg_oos_trades": 0.0,
        }

    returns = df["Total Return [%]"]
    drawdowns = df["Max Drawdown [%]"]
    sharpes = df["Sharpe Ratio"]
    pfs = df["Profit Factor"]
    trades = df["Total Trades"]

    return {
        "windows": int(len(df)),
        "positive_windows": int((returns > 0).sum()),
        "negative_windows": int((returns <= 0).sum()),
        "avg_oos_return_pct": float(returns.mean()),
        "median_oos_return_pct": float(returns.median()),
        "avg_oos_drawdown_pct": float(drawdowns.mean()),
        "avg_oos_sharpe": float(sharpes.mean()),
        "avg_oos_profit_factor": float(pfs.mean()),
        "avg_oos_trades": float(trades.mean()),
    }


def generate_base_walkforward(module, data: pd.DataFrame) -> Path:
    """Generate base walk-forward result for current defaults."""

    cfg = module.Config()
    df = run_walkforward_windows(module, data, cfg, train_days=365, test_days=90, step_days=30)
    out = BASE_DIR / "xrp_walkforward_v2_results.csv"
    df.to_csv(out, index=False)
    return out


def generate_sma_atr_grid(module, data: pd.DataFrame) -> Path:
    """Generate parameter robustness grid around SMA/ATR filter values."""

    sma_fast_vals = [10, 12, 14]
    sma_slow_vals = [30, 36, 42]
    sma_long_vals = [80, 90, 100]
    atr_min_vals = [0.002, 0.003]
    atr_max_vals = [0.05, 0.06]

    rows: List[Dict[str, float | int]] = []
    for sma_fast in sma_fast_vals:
        for sma_slow in sma_slow_vals:
            for sma_long in sma_long_vals:
                if not (sma_fast < sma_slow < sma_long):
                    continue
                for atr_min in atr_min_vals:
                    for atr_max in atr_max_vals:
                        if atr_min >= atr_max:
                            continue

                        cfg = module.Config(
                            sma_fast=sma_fast,
                            sma_slow=sma_slow,
                            sma_long=sma_long,
                            atr_filter_min_pct=atr_min,
                            atr_filter_max_pct=atr_max,
                        )

                        wf = run_walkforward_windows(module, data, cfg, train_days=365, test_days=90, step_days=30)
                        stats = summarize(wf)
                        robustness = float(stats["avg_oos_return_pct"]) * (
                            float(stats["positive_windows"]) / max(float(stats["windows"]), 1.0)
                        ) - 0.25 * float(stats["avg_oos_drawdown_pct"])

                        rows.append(
                            {
                                "sma_fast": sma_fast,
                                "sma_slow": sma_slow,
                                "sma_long": sma_long,
                                "atr_filter_min_pct": atr_min,
                                "atr_filter_max_pct": atr_max,
                                **stats,
                                "robustness_score": robustness,
                            }
                        )

    out_df = pd.DataFrame(rows).sort_values(["robustness_score", "avg_oos_return_pct"], ascending=False)
    out = BASE_DIR / "xrp_walkforward_v2_grid_results.csv"
    out_df.to_csv(out, index=False)
    return out


def generate_stoptp_grid(module, data: pd.DataFrame) -> Path:
    """Generate focused stop-loss/take-profit grid around the best SMA cluster."""

    sl_vals = [0.012, 0.015, 0.018, 0.022, 0.026]
    tp_vals = [0.05, 0.06, 0.08, 0.10, 0.12]

    rows: List[Dict[str, float | int]] = []
    for sl in sl_vals:
        for tp in tp_vals:
            if tp <= sl:
                continue

            cfg = module.Config(
                sma_fast=12,
                sma_slow=42,
                sma_long=90,
                atr_filter_min_pct=0.003,
                atr_filter_max_pct=0.06,
                sl_stop=sl,
                tp_stop=tp,
            )

            wf = run_walkforward_windows(module, data, cfg, train_days=365, test_days=90, step_days=30)
            stats = summarize(wf)
            robustness = float(stats["avg_oos_return_pct"]) * (
                float(stats["positive_windows"]) / max(float(stats["windows"]), 1.0)
            ) - 0.25 * float(stats["avg_oos_drawdown_pct"])

            rows.append(
                {
                    "sl_stop": sl,
                    "tp_stop": tp,
                    **stats,
                    "robustness_score": robustness,
                }
            )

    out_df = pd.DataFrame(rows).sort_values(["robustness_score", "avg_oos_return_pct"], ascending=False)
    out = BASE_DIR / "xrp_walkforward_v2_stoptp_grid_results.csv"
    out_df.to_csv(out, index=False)
    return out


def generate_confirmation_report(module, data: pd.DataFrame) -> Tuple[Path, Path]:
    """Generate alternate-window confirmation comparison and text summary."""

    configs = {
        "prior_defaults_12_36_90_sl1.8_tp8": module.Config(
            sma_fast=12,
            sma_slow=36,
            sma_long=90,
            sl_stop=0.018,
            tp_stop=0.08,
        ),
        "updated_defaults_12_42_90_sl1.2_tp5": module.Config(
            sma_fast=12,
            sma_slow=42,
            sma_long=90,
            sl_stop=0.012,
            tp_stop=0.05,
        ),
    }

    rows: List[Dict[str, float | int | str]] = []
    for label, cfg in configs.items():
        wf = run_walkforward_windows(module, data, cfg, train_days=540, test_days=120, step_days=60)
        stats = summarize(wf)
        rows.append(
            {
                "config": label,
                "train_days": 540,
                "test_days": 120,
                "step_days": 60,
                **stats,
            }
        )

    comp = pd.DataFrame(rows).sort_values("avg_oos_return_pct", ascending=False)
    out_csv = BASE_DIR / "xrp_v2_confirmation_comparison.csv"
    comp.to_csv(out_csv, index=False)

    out_txt = BASE_DIR / "xrp_v2_confirmation_summary.txt"
    with out_txt.open("w", encoding="utf-8") as f:
        f.write("XRP v2 Confirmation Walk-Forward Summary\n")
        f.write("Scheme: train=540d, test=120d, step=60d\n\n")
        if comp.empty:
            f.write("No windows available for this scheme.\n")
        else:
            for _, r in comp.iterrows():
                f.write(f"Config: {r['config']}\n")
                f.write(
                    f"  Windows: {int(r['windows'])} | Positive: {int(r['positive_windows'])} | Negative: {int(r['negative_windows'])}\n"
                )
                f.write(f"  Avg OOS Return: {r['avg_oos_return_pct']:.4f}%\n")
                f.write(f"  Median OOS Return: {r['median_oos_return_pct']:.4f}%\n")
                f.write(f"  Avg OOS Drawdown: {r['avg_oos_drawdown_pct']:.4f}%\n")
                f.write(f"  Avg OOS Sharpe: {r['avg_oos_sharpe']:.4f}\n")
                f.write(f"  Avg OOS Profit Factor: {r['avg_oos_profit_factor']:.4f}\n")
                f.write(f"  Avg OOS Trades: {r['avg_oos_trades']:.2f}\n\n")

    return out_csv, out_txt


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for report generation."""

    parser = argparse.ArgumentParser(description="Run all XRP v2 walk-forward report generations.")
    parser.add_argument("--csv", type=str, default="", help="Optional input CSV path override")
    return parser.parse_args()


def main() -> None:
    """Run all walk-forward analyses and refresh report artifacts."""

    args = parse_args()
    module = load_strategy_module()
    csv_path = resolve_data_path(args.csv)
    data = module.load_data(csv_path)

    outputs: List[Path] = []
    outputs.append(generate_base_walkforward(module, data))
    outputs.append(generate_sma_atr_grid(module, data))
    outputs.append(generate_stoptp_grid(module, data))
    conf_csv, conf_txt = generate_confirmation_report(module, data)
    outputs.extend([conf_csv, conf_txt])

    print(f"Using data: {csv_path}")
    print("Refreshed files:")
    for out in outputs:
        print(f"- {out}")


if __name__ == "__main__":
    main()
