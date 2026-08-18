from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from equity_msft.research_loop._bootstrap import ensure_import_paths

ensure_import_paths()

from equity_msft.research_loop.backtest_runner import load_data, resolve_csv_path
from equity_msft.research_loop.pipeline import config_to_dict, load_config_from_json
from equity_msft.research_loop.supervisor_state import (
    load_supervisor_config,
    run_supervisor_loop,
    supervisor_config_to_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the MSFT research supervisor: backtest -> Grok/Optuna propose -> "
            "accept/reject with hard gates and an optional holdout year."
        )
    )
    parser.add_argument("--csv", type=str, default=None, help="OHLCV CSV. Defaults to data/msft_daily.csv")
    parser.add_argument("--start", type=str, default="2018-01-01", help="Data start date")
    parser.add_argument("--end", type=str, default=None, help="Data end date (default: today)")
    parser.add_argument("--config-json", type=str, default=None, help="Strategy Config overrides")
    parser.add_argument("--supervisor-config-json", type=str, default=None, help="SupervisorConfig overrides")
    parser.add_argument(
        "--proposal-source",
        type=str,
        default=None,
        help="Override proposal engine: grok, codex, optuna, or noop",
    )
    parser.add_argument("--output-json", type=str, default=None, help="Write final summary JSON here")
    parser.add_argument("--grok-preview-chars", type=int, default=500)
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Refresh the cached MSFT CSV from Yahoo Finance before running",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = resolve_csv_path(args.csv)
    data = load_data(csv_path, start=args.start, end=args.end, refresh_data=args.refresh_data)

    initial_cfg = load_config_from_json(args.config_json)
    sup_cfg = load_supervisor_config(args.supervisor_config_json)
    if args.proposal_source:
        sup_cfg.proposal_source = args.proposal_source

    mem = run_supervisor_loop(data, sup_cfg, initial_cfg=initial_cfg)

    snapshot_keys = ["total_return", "annualized_return", "sharpe", "max_drawdown", "num_trades", "buy_hold_return"]
    best_metrics_snapshot: dict[str, Any] = {k: mem.best_metrics.get(k) for k in snapshot_keys}
    holdout_snapshot: dict[str, Any] = {k: mem.holdout_metrics.get(k) for k in snapshot_keys}

    print(f"csv_path: {csv_path}")
    print(f"rows: {len(data)}")
    print(f"search_rows: {mem.search_rows}")
    print(f"holdout_rows: {mem.holdout_rows}")
    print(f"iterations_completed: {mem.iteration}")
    print(f"no_improve_streak: {mem.no_improve_streak}")
    print(f"best_score: {mem.best_score}")
    print(f"best_metrics_snapshot: {best_metrics_snapshot}")
    print(f"holdout_score: {mem.holdout_score}")
    print(f"holdout_metrics_snapshot: {holdout_snapshot}")
    print(
        "last_grok_eval_preview:",
        mem.grok_eval[: max(args.grok_preview_chars, 0)].replace("\n", " "),
    )
    print(f"best_params: {config_to_dict(mem.best_cfg)}")
    print(f"proposal_source: {sup_cfg.proposal_source}")
    print(f"log_jsonl_path: {sup_cfg.log_jsonl_path}")
    print(f"best_json_path: {sup_cfg.best_json_path}")

    if args.output_json:
        out_path = Path(args.output_json)
        if not out_path.is_absolute():
            from equity_msft.research_loop._bootstrap import REPO_ROOT

            out_path = REPO_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "csv_path": str(csv_path),
            "rows": len(data),
            "search_rows": mem.search_rows,
            "holdout_rows": mem.holdout_rows,
            "iterations_completed": mem.iteration,
            "no_improve_streak": mem.no_improve_streak,
            "best_score": mem.best_score,
            "best_metrics": mem.best_metrics,
            "best_params": config_to_dict(mem.best_cfg),
            "holdout_score": mem.holdout_score,
            "holdout_metrics": mem.holdout_metrics,
            "last_grok_eval": mem.grok_eval,
            "proposal_metadata": mem.proposal_metadata,
            "supervisor_config": supervisor_config_to_dict(sup_cfg),
        }
        out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"saved_output_json: {out_path}")


if __name__ == "__main__":
    main()
