from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from equity_msft.research_loop.backtest_runner import load_data, resolve_csv_path
from equity_msft.research_loop.pipeline import load_config_from_json, research_iteration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one MSFT research iteration (backtest -> Grok review -> Grok/Codex params).",
    )
    parser.add_argument("--csv", type=str, default=None, help="OHLCV CSV. Defaults to data/msft_daily.csv")
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--config-json", type=str, default=None, help="Strategy Config overrides")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--grok-preview-chars", type=int, default=500)
    parser.add_argument("--refresh-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = resolve_csv_path(args.csv)
    data = load_data(csv_path, start=args.start, end=args.end, refresh_data=args.refresh_data)
    cfg = load_config_from_json(args.config_json)

    new_params, metrics, grok_eval = research_iteration(data, cfg)

    snapshot_keys = ["total_return", "annualized_return", "sharpe", "max_drawdown", "num_trades", "buy_hold_return"]
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
        if not out_path.is_absolute():
            out_path = _REPO_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "csv_path": str(csv_path),
            "rows": len(data),
            "metrics": metrics,
            "grok_eval": grok_eval,
            "new_params": new_params,
        }
        out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"saved_output_json: {out_path}")


if __name__ == "__main__":
    main()
