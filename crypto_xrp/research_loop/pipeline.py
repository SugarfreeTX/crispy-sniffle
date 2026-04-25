from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class Config:
    # Keep this schema aligned with xrp_vectorbt_best_v2.Config.
    initial_cash: float = 100000.0
    commission_bps: float = 5.0
    slippage_bps: float = 5.0
    position_size_pct: float = 0.12
    sma_fast: int = 12
    sma_slow: int = 42
    sma_long: int = 90
    sell_rsi: float = 78.0
    atr_window: int = 14
    atr_filter_min_pct: float = 0.003
    atr_filter_max_pct: float = 0.06
    sl_stop: float = 0.012
    tp_stop: float = 0.05
    freq: str = "4h"


def config_to_dict(cfg: Config) -> dict[str, Any]:
    return asdict(cfg)


def config_from_dict(params: dict[str, Any], base_cfg: Config | None = None) -> Config:
    # Keep unknown keys out so one unexpected suggestion does not break the run.
    allowed = {f.name for f in fields(Config)}
    merged = config_to_dict(base_cfg) if base_cfg is not None else {}
    merged.update({k: v for k, v in params.items() if k in allowed})
    return Config(**merged)


def research_iteration(data, cfg: Config):
    if __package__:
        from .backtest_runner import run_backtest_and_extract_metrics
        from .codex_client import refine_strategy_with_codex
        from .grok_client import evaluate_with_grok
    else:
        from backtest_runner import run_backtest_and_extract_metrics
        from codex_client import refine_strategy_with_codex
        from grok_client import evaluate_with_grok

    if not isinstance(cfg, Config):
        if isinstance(cfg, dict):
            cfg = config_from_dict(cfg)
        else:
            raise TypeError(f"cfg must be Config or dict, got {type(cfg)!r}")

    metrics = run_backtest_and_extract_metrics(data, cfg)
    grok_eval = evaluate_with_grok(metrics)

    proposed_params = refine_strategy_with_codex(config_to_dict(cfg), grok_eval)
    if not isinstance(proposed_params, dict):
        raise TypeError(
            "refine_strategy_with_codex must return a dict of updated parameters"
        )

    new_params = {
        k: v for k, v in proposed_params.items() if k in {f.name for f in fields(Config)}
    }
    return new_params, metrics, grok_eval
