from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# Ensure repository root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crypto_xrp.research_loop.pipeline import Config, config_from_dict, config_to_dict
from crypto_xrp.research_loop.supervisor_state import SupervisorConfig, run_supervisor_loop
from crypto_xrp.xrp_vectorbt_best_v2 import load_data, resolve_csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run XRP supervisor loop (backtest -> Grok -> Codex with accept/reject logic)."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV input path. Defaults to cleaned CSV if present.",
    )
    parser.add_argument(
        "--config-json",
        type=str,
        default=None,
        help="Optional JSON file with Config key/value overrides.",
    )
    parser.add_argument(
        "--supervisor-config-json",
        type=str,
        default=None,
        help="Optional JSON file with SupervisorConfig key/value overrides.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional output path to save final supervisor summary payload.",
    )
    parser.add_argument(
        "--grok-preview-chars",
        type=int,
        default=500,
        help="How many Grok response characters to print in preview.",
    )
    return parser.parse_args()


def _load_json_object(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return raw


def _load_strategy_config(path_str: str | None) -> Config:
    if not path_str:
        return Config()
    raw = _load_json_object(path_str)
    return config_from_dict(raw, base_cfg=Config())


def _load_supervisor_config(path_str: str | None) -> SupervisorConfig:
    if not path_str:
        return SupervisorConfig()
    raw = _load_json_object(path_str)

    allowed_keys = set(SupervisorConfig.__annotations__.keys())
    filtered = {k: v for k, v in raw.items() if k in allowed_keys}
    return SupervisorConfig(**filtered)


def main() -> None:
    args = parse_args()

    csv_path = resolve_csv_path(args.csv)
    data = load_data(csv_path)

    initial_cfg = _load_strategy_config(args.config_json)
    sup_cfg = _load_supervisor_config(args.supervisor_config_json)

    mem = run_supervisor_loop(data, sup_cfg, initial_cfg=initial_cfg)

    snapshot_keys = [
        "total_return",
        "annualized_return",
        "sharpe",
        "max_drawdown",
        "num_trades",
    ]
    best_metrics_snapshot: dict[str, Any] = {k: mem.best_metrics.get(k) for k in snapshot_keys}

    print(f"csv_path: {csv_path}")
    print(f"rows: {len(data)}")
    print(f"iterations_completed: {mem.iteration}")
    print(f"no_improve_streak: {mem.no_improve_streak}")
    print(f"best_score: {mem.best_score}")
    print(f"best_metrics_snapshot: {best_metrics_snapshot}")
    print(
        "last_grok_eval_preview:",
        mem.grok_eval[: max(args.grok_preview_chars, 0)].replace("\n", " "),
    )
    print(f"best_params: {config_to_dict(mem.best_cfg)}")
    print(f"log_jsonl_path: {sup_cfg.log_jsonl_path}")
    print(f"best_json_path: {sup_cfg.best_json_path}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "csv_path": str(csv_path),
            "rows": len(data),
            "iterations_completed": mem.iteration,
            "no_improve_streak": mem.no_improve_streak,
            "best_score": mem.best_score,
            "best_metrics": mem.best_metrics,
            "best_params": config_to_dict(mem.best_cfg),
            "last_grok_eval": mem.grok_eval,
            "supervisor_config": {
                "max_iterations": sup_cfg.max_iterations,
                "patience": sup_cfg.patience,
                "min_improvement": sup_cfg.min_improvement,
                "max_drawdown_limit": sup_cfg.max_drawdown_limit,
                "min_trades": sup_cfg.min_trades,
                "dd_penalty_weight": sup_cfg.dd_penalty_weight,
                "log_jsonl_path": sup_cfg.log_jsonl_path,
                "best_json_path": sup_cfg.best_json_path,
            },
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"saved_output_json: {out_path}")


if __name__ == "__main__":
    main()
