from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
EQUITY_DIR = ROOT / "equity_msft"
BACKTEST_PATH = EQUITY_DIR / "backtest.py"
BACKTEST_OUTPUTS = EQUITY_DIR / "backtest_outputs"
DATA_PATH = EQUITY_DIR / "data" / "msft_daily.csv"
NOTEBOOK_PATH = EQUITY_DIR / "reviewer_pack_analysis.ipynb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full MSFT reviewer-pack pipeline and generate a handoff zip."
    )
    parser.add_argument("--start", default="2018-01-01", help="Primary run start date")
    parser.add_argument("--end", default="2025-12-31", help="Primary run end date")
    parser.add_argument("--python-executable", default=sys.executable, help="Python executable to invoke")
    parser.add_argument("--pack-tag", default="", help="Optional pack tag; default is timestamp")
    parser.add_argument("--sweep-max-runs", type=int, default=324, help="Max sweep combinations")
    parser.add_argument("--sweep-top-n", type=int, default=50, help="Top sweep rows to keep")
    parser.add_argument("--sweep-min-trades", type=int, default=1, help="Min trades for sweep ranking")
    parser.add_argument(
        "--notebook-path",
        default=str(NOTEBOOK_PATH),
        help="Path to handoff notebook to include in the reviewer pack",
    )
    parser.add_argument(
        "--allow-missing-notebook",
        action="store_true",
        help="Allow pack creation without notebook (default enforces notebook inclusion)",
    )
    parser.add_argument(
        "--sweep-sort-by",
        choices=["return_pct", "sharpe_ratio", "profit_factor", "trade_count", "max_drawdown_abs"],
        default="return_pct",
        help="Sweep ranking key",
    )
    return parser.parse_args()


def _validate_notebook(path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Notebook is not valid JSON: {path}") from exc

    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise RuntimeError(f"Notebook missing 'cells' list: {path}")


def _resolve_notebook_for_pack(args: argparse.Namespace) -> Path | None:
    requested = Path(str(args.notebook_path)).expanduser()
    if not requested.is_absolute():
        requested = (ROOT / requested).resolve()

    if requested.exists():
        _validate_notebook(requested)
        return requested

    if args.allow_missing_notebook:
        return None

    raise RuntimeError(
        "Notebook required for handoff but not found. "
        f"Expected at: {requested}. "
        "Use --notebook-path to point to a notebook or --allow-missing-notebook to bypass."
    )


def _run_and_log(command: List[str], log_file: Path) -> None:
    with open(log_file, "a", encoding="utf-8") as log:
        joined = " ".join(command)
        log.write(f"\n>>> {joined}\n")
        log.flush()
        print(f"\n>>> {joined}")

        process = subprocess.Popen(  # noqa: S603
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)

        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"Command failed ({return_code}): {joined}")


def _scenario_commands(args: argparse.Namespace) -> List[List[str]]:
    py = str(Path(args.python_executable).expanduser())
    bt = str(BACKTEST_PATH)

    primary = ["--start", args.start, "--end", args.end]

    return [
        [py, bt, "--refresh-data", *primary],
        [
            py,
            bt,
            *primary,
            "--sweep",
            "--sweep-max-runs",
            str(args.sweep_max_runs),
            "--sweep-top-n",
            str(args.sweep_top_n),
            "--sweep-min-trades",
            str(args.sweep_min_trades),
            "--sweep-sort-by",
            args.sweep_sort_by,
        ],
        [py, bt, *primary, "--commission-bps", "5", "--slippage-bps", "20"],
        [py, bt, *primary, "--max-consecutive-losses", "999"],
        [py, bt, *primary, "--min-atr", "0.0"],
        [py, bt, *primary, "--sell-overbought-rsi", "100"],
        [py, bt, *primary, "--take-profit-pnl-pct", "100", "--take-profit-rsi", "100"],
        [py, bt, "--start", "2021-01-01", "--end", "2022-12-31"],
        [py, bt, "--start", "2020-01-01", "--end", "2021-12-31"],
        [py, bt, "--start", "2022-01-01", "--end", "2023-12-31"],
        [py, bt, "--start", "2023-01-01", "--end", "2024-12-31"],
        [py, bt, "--start", "2024-01-01", "--end", "2025-12-31"],
    ]


def _classify_scenario(row: pd.Series) -> str:
    start = str(row.get("start", ""))
    end = str(row.get("end", ""))

    if start == "2018-01-01" and end == "2025-12-31":
        slippage = float(row.get("slippage_bps", 0) or 0)
        max_losses = int(row.get("max_consecutive_losses", 0) or 0)
        min_atr = float(row.get("min_atr", 0) or 0)
        sell_overbought = float(row.get("sell_overbought_rsi", 0) or 0)
        tp_pct = float(row.get("take_profit_pnl_pct", 0) or 0)
        tp_rsi = float(row.get("take_profit_rsi", 0) or 0)

        if slippage == 20.0:
            return "stress_slippage_20bps"
        if max_losses == 999:
            return "ablation_no_streak_cap"
        if min_atr == 0.0:
            return "ablation_min_atr_0"
        if sell_overbought == 100.0:
            return "ablation_disable_overbought_sell"
        if tp_pct == 100.0 and tp_rsi == 100.0:
            return "ablation_disable_take_profit"
        return "baseline"

    if start == "2021-01-01" and end == "2022-12-31":
        return "edge_2022_focus"
    if start == "2020-01-01" and end == "2021-12-31":
        return "walk_forward_2020_2021"
    if start == "2022-01-01" and end == "2023-12-31":
        return "walk_forward_2022_2023"
    if start == "2023-01-01" and end == "2024-12-31":
        return "walk_forward_2023_2024"
    if start == "2024-01-01" and end == "2025-12-31":
        return "walk_forward_2024_2025"
    return "other"


def _read_metrics_rows(metrics_files: Iterable[Path]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in sorted(metrics_files):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        sp = cfg.get("strategy_params", {})
        rows.append(
            {
                "run_id": payload.get("run_id"),
                "start": str(cfg.get("start")),
                "end": str(cfg.get("end")),
                "return_pct": payload.get("return_pct"),
                "max_drawdown_pct": payload.get("max_drawdown_pct"),
                "win_rate_pct": payload.get("win_rate_pct"),
                "sharpe_ratio": payload.get("sharpe_ratio"),
                "profit_factor": payload.get("profit_factor"),
                "trade_count": payload.get("trade_count"),
                "commission_bps": cfg.get("commission_bps"),
                "slippage_bps": cfg.get("slippage_bps"),
                "min_atr": sp.get("min_atr"),
                "max_consecutive_losses": sp.get("max_consecutive_losses"),
                "sell_overbought_rsi": sp.get("sell_overbought_rsi"),
                "take_profit_pnl_pct": sp.get("take_profit_pnl_pct"),
                "take_profit_rsi": sp.get("take_profit_rsi"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No metrics files found for pack summary generation")
    return df.sort_values("run_id").reset_index(drop=True)


def _build_pack(args: argparse.Namespace) -> Dict[str, Path]:
    pack_tag = args.pack_tag.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    pack_dir = EQUITY_DIR / f"reviewer_pack_{pack_tag}"
    zip_path = EQUITY_DIR / f"reviewer_pack_{pack_tag}.zip"
    marker_path = EQUITY_DIR / f".reviewer_pack_marker_{pack_tag}"
    current_pack_tag_path = EQUITY_DIR / ".current_pack_tag"
    latest_pack_path = EQUITY_DIR / ".latest_reviewer_pack.txt"
    notebook_source = _resolve_notebook_for_pack(args)

    if pack_dir.exists():
        raise RuntimeError(f"Pack directory already exists: {pack_dir}")

    (pack_dir / "backtest_outputs").mkdir(parents=True, exist_ok=False)
    (pack_dir / "data").mkdir(parents=True, exist_ok=False)
    run_log = pack_dir / "run_log.txt"

    marker_path.touch()
    current_pack_tag_path.write_text(pack_tag + "\n", encoding="utf-8")

    commands = _scenario_commands(args)
    for idx, command in enumerate(commands, start=1):
        with open(run_log, "a", encoding="utf-8") as log:
            log.write(f"\n### Scenario {idx}/{len(commands)}\n")
        _run_and_log(command, run_log)

    marker_mtime = marker_path.stat().st_mtime
    new_files = sorted(
        [p for p in BACKTEST_OUTPUTS.glob("*") if p.is_file() and p.stat().st_mtime > marker_mtime]
    )

    new_files_txt = pack_dir / "new_files.txt"
    new_files_txt.write_text("\n".join(str(p) for p in new_files) + "\n", encoding="utf-8")

    for path in new_files:
        shutil.copy2(path, pack_dir / "backtest_outputs" / path.name)

    if DATA_PATH.exists():
        shutil.copy2(DATA_PATH, pack_dir / "data" / DATA_PATH.name)
    if notebook_source is not None:
        shutil.copy2(notebook_source, pack_dir / "reviewer_pack_analysis.ipynb")

    metrics_df = _read_metrics_rows((pack_dir / "backtest_outputs").glob("metrics_msft_*.json"))
    metrics_df["scenario"] = metrics_df.apply(_classify_scenario, axis=1)
    metrics_df.to_csv(pack_dir / "scenario_summary.csv", index=False)

    latest_df = (
        metrics_df.sort_values("run_id")
        .groupby("scenario", as_index=False)
        .tail(1)
        .sort_values("scenario")
        .reset_index(drop=True)
    )
    latest_df.to_csv(pack_dir / "scenario_summary_latest.csv", index=False)

    sweep_best_files = sorted((pack_dir / "backtest_outputs").glob("sweep_best_msft_*.json"))
    sweep_best_summary: Dict[str, Any] = {}
    if sweep_best_files:
        sweep_best_summary = json.loads(sweep_best_files[-1].read_text(encoding="utf-8"))
    (pack_dir / "sweep_best_summary.json").write_text(
        json.dumps(sweep_best_summary, indent=2), encoding="utf-8"
    )

    counts = {
        "metrics": len(list((pack_dir / "backtest_outputs").glob("metrics_msft_*.json"))),
        "equity": len(list((pack_dir / "backtest_outputs").glob("equity_msft_*.csv"))),
        "trades": len(list((pack_dir / "backtest_outputs").glob("trades_msft_*.csv"))),
        "stats": len(list((pack_dir / "backtest_outputs").glob("stats_msft_*.txt"))),
        "sweep": len(list((pack_dir / "backtest_outputs").glob("sweep_*_msft_*.csv")))
        + len(list((pack_dir / "backtest_outputs").glob("sweep_best_msft_*.json"))),
    }

    manifest = pack_dir / "reviewer_pack_manifest.txt"
    notebook_manifest_line = (
        "- reviewer_pack_analysis.ipynb"
        if notebook_source is not None
        else "- reviewer_pack_analysis.ipynb (not included; --allow-missing-notebook set)"
    )
    manifest.write_text(
        "\n".join(
            [
                "MSFT Reviewer Pack",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Pack tag: {pack_tag}",
                "",
                "Counts from this rerun:",
                f"- metrics: {counts['metrics']}",
                f"- equity: {counts['equity']}",
                f"- trades: {counts['trades']}",
                f"- stats: {counts['stats']}",
                f"- sweep csv/json: {counts['sweep']}",
                "",
                "Included files:",
                "- scenario_summary.csv (all metrics rows)",
                "- scenario_summary_latest.csv (latest per scenario)",
                "- sweep_best_summary.json",
                notebook_manifest_line,
                "- data/msft_daily.csv (if present)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(pack_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(ROOT))

    latest_pack_path.write_text(
        "\n".join(
            [
                f"PACK_TAG={pack_tag}",
                f"PACK_DIR={pack_dir}",
                f"ZIP_PATH={zip_path}",
                f"MANIFEST={manifest}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "pack_dir": pack_dir,
        "zip_path": zip_path,
        "manifest": manifest,
        "latest_file": latest_pack_path,
    }


def main() -> None:
    args = parse_args()
    results = _build_pack(args)

    print("\nReviewer pack complete")
    print(f"Pack dir: {results['pack_dir']}")
    print(f"Zip: {results['zip_path']}")
    print(f"Manifest: {results['manifest']}")
    print(f"Latest pointer: {results['latest_file']}")


if __name__ == "__main__":
    main()