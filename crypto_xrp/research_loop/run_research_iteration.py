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

from crypto_xrp.research_loop.pipeline import Config, config_from_dict, research_iteration
from crypto_xrp.xrp_vectorbt_best_v2 import load_data, resolve_csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one end-to-end XRP research iteration (backtest -> Grok -> Codex)."
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
        "--output-json",
        type=str,
        default=None,
        help="Optional output path to save new_params/metrics/grok_eval.",
    )
    parser.add_argument(
        "--grok-preview-chars",
        type=int,
        default=500,
        help="How many Grok response characters to print in preview.",
    )
    return parser.parse_args()


def _load_config(config_json_path: str | None) -> Config:
    if not config_json_path:
        return Config()

    cfg_path = Path(config_json_path)
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("--config-json must contain a JSON object")
    return config_from_dict(raw, base_cfg=Config())


def main() -> None:
    args = parse_args()

    csv_path = resolve_csv_path(args.csv)
    data = load_data(csv_path)
    cfg = _load_config(args.config_json)

    new_params, metrics, grok_eval = research_iteration(data, cfg)

    snapshot_keys = [
        "total_return",
        "annualized_return",
        "sharpe",
        "max_drawdown",
        "num_trades",
    ]
    metrics_snapshot: dict[str, Any] = {k: metrics.get(k) for k in snapshot_keys}

    print(f"csv_path: {csv_path}")
    print(f"rows: {len(data)}")
    print(f"metrics_snapshot: {metrics_snapshot}")
    print(f"grok_eval_len: {len(grok_eval)}")
    print(
        "grok_eval_preview:",
        grok_eval[: max(args.grok_preview_chars, 0)].replace("\n", " "),
    )
    print(f"new_params: {new_params}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "csv_path": str(csv_path),
            "rows": len(data),
            "metrics": metrics,
            "grok_eval": grok_eval,
            "new_params": new_params,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"saved_output_json: {out_path}")


if __name__ == "__main__":
    main()
