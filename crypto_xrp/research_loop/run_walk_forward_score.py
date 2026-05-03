from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Ensure repository root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crypto_xrp.research_loop.pipeline import Config, config_from_dict
from crypto_xrp.research_loop.walk_forward_scorer import (
    WalkForwardScoreSpec,
    WalkForwardSpec,
    evaluate_walk_forward,
    walk_forward_result_to_dict,
)
from crypto_xrp.xrp_vectorbt_best_v2 import load_data, resolve_csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run walk-forward objective scoring for the XRP strategy config."
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
        "--wf-spec-json",
        type=str,
        default=None,
        help="Optional JSON overrides for WalkForwardSpec.",
    )
    parser.add_argument(
        "--wf-score-json",
        type=str,
        default=None,
        help="Optional JSON overrides for WalkForwardScoreSpec.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional output path to save full walk-forward result payload.",
    )
    return parser.parse_args()


def _load_json_obj(path_str: str | None) -> dict:
    if not path_str:
        return {}
    p = Path(path_str)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{p} must contain a JSON object")
    return raw


def _build_dataclass_kwargs(raw: dict, cls) -> dict:
    allowed = set(cls.__annotations__.keys())
    return {k: v for k, v in raw.items() if k in allowed}


def main() -> None:
    args = parse_args()

    csv_path = resolve_csv_path(args.csv)
    data = load_data(csv_path)

    cfg_raw = _load_json_obj(args.config_json)
    cfg = config_from_dict(cfg_raw, base_cfg=Config()) if cfg_raw else Config()

    wf_spec_raw = _load_json_obj(args.wf_spec_json)
    wf_spec = WalkForwardSpec(**_build_dataclass_kwargs(wf_spec_raw, WalkForwardSpec))

    wf_score_raw = _load_json_obj(args.wf_score_json)
    wf_score = WalkForwardScoreSpec(**_build_dataclass_kwargs(wf_score_raw, WalkForwardScoreSpec))

    result = evaluate_walk_forward(data=data, cfg=cfg, wf_spec=wf_spec, score_spec=wf_score)

    print(f"csv_path: {csv_path}")
    print(f"rows: {len(data)}")
    print(f"walk_forward_score: {result.aggregate_score}")
    print(f"windows_total: {result.windows_total}")
    print(f"windows_valid: {result.windows_valid}")
    print(f"windows_rejected: {result.windows_rejected}")
    print(f"mean_sharpe: {result.mean_sharpe}")
    print(f"sharpe_std: {result.sharpe_std}")
    print(f"worst_max_drawdown: {result.worst_max_drawdown}")
    print(f"avg_num_trades: {result.avg_num_trades}")
    print(f"objective_components: {result.objective_components}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(walk_forward_result_to_dict(result), indent=2),
            encoding="utf-8",
        )
        print(f"saved_output_json: {out_path}")


if __name__ == "__main__":
    main()
