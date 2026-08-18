from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from equity_msft.research_loop.backtest_runner import load_data, resolve_csv_path
from equity_msft.research_loop.pipeline import Config, config_from_dict, load_json_object
from equity_msft.research_loop.walk_forward_scorer import (
    WalkForwardScoreSpec,
    WalkForwardSpec,
    evaluate_walk_forward,
    walk_forward_result_to_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score one MSFT config with walk-forward windows (warmup-aware)."
    )
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--config-json", type=str, default=None)
    parser.add_argument("--wf-spec-json", type=str, default=None)
    parser.add_argument("--wf-score-json", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--refresh-data", action="store_true")
    return parser.parse_args()


def _build_dataclass_kwargs(raw: dict, cls) -> dict:
    allowed = set(cls.__annotations__.keys())
    return {k: v for k, v in raw.items() if k in allowed}


def main() -> None:
    args = parse_args()
    csv_path = resolve_csv_path(args.csv)
    data = load_data(csv_path, start=args.start, end=args.end, refresh_data=args.refresh_data)

    cfg = Config()
    if args.config_json:
        cfg = config_from_dict(load_json_object(args.config_json), base_cfg=Config())

    wf_spec = WalkForwardSpec()
    if args.wf_spec_json:
        wf_spec = WalkForwardSpec(**_build_dataclass_kwargs(load_json_object(args.wf_spec_json), WalkForwardSpec))

    wf_score = WalkForwardScoreSpec()
    if args.wf_score_json:
        wf_score = WalkForwardScoreSpec(
            **_build_dataclass_kwargs(load_json_object(args.wf_score_json), WalkForwardScoreSpec)
        )

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
        if not out_path.is_absolute():
            out_path = _REPO_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(walk_forward_result_to_dict(result), indent=2, default=str),
            encoding="utf-8",
        )
        print(f"saved_output_json: {out_path}")


if __name__ == "__main__":
    main()
