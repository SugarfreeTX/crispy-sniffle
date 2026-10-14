from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

from equity_msft.research_loop._bootstrap import ensure_import_paths

ensure_import_paths()

from equity_msft.research_loop.pipeline import (
    SEARCH_PARAM_KEYS,
    Config,
    config_from_dict,
    config_to_dict,
)
from equity_msft.research_loop.walk_forward_scorer import (
    WalkForwardScoreSpec,
    WalkForwardSpec,
    evaluate_walk_forward,
)


# Plausible ranges for Optuna suggestions. Keep aligned with PARAM_STEP_RANGES
# in pipeline.py for knobs that both engines touch.
_OPTUNA_RANGES: dict[str, tuple[float, float]] = {
    "buy_pullback_rsi": (38.0, 65.0),
    "buy_grind_rsi_max": (55.0, 75.0),
    "sell_overbought_rsi": (74.0, 92.0),
    "min_atr": (0.5, 5.0),
    "extreme_setup_rsi": (20.0, 40.0),
    "extreme_setup_rel_vol": (0.8, 1.5),
    "bearish_entry_rsi": (18.0, 35.0),
    "bearish_exit_rsi": (40.0, 60.0),
    "bullish_hold_rsi": (52.0, 70.0),
    "take_profit_pnl_pct": (5.0, 10.0),
    "take_profit_rsi": (52.0, 70.0),
    "sell_bearish_rsi": (45.0, 62.0),
    "neutral_rsi_low": (30.0, 50.0),
    "neutral_rsi_high": (50.0, 70.0),
    "neutral_rel_volume_max": (1.0, 2.0),
}

# Default Optuna search (pre-focus). Neutral band knobs are available when an
# LLM seed names them, but are not swept on every plain optuna iteration.
_DEFAULT_OPTUNA_KEYS: tuple[str, ...] = (
    "buy_pullback_rsi",
    "buy_grind_rsi_max",
    "sell_overbought_rsi",
    "min_atr",
    "extreme_setup_rsi",
    "extreme_setup_rel_vol",
    "bearish_entry_rsi",
    "bearish_exit_rsi",
    "bullish_hold_rsi",
    "take_profit_pnl_pct",
    "take_profit_rsi",
    "sell_bearish_rsi",
)


def _resolve_focus_keys(focus_keys: Sequence[str] | None) -> list[str]:
    if not focus_keys:
        return list(_DEFAULT_OPTUNA_KEYS)
    allowed = set(SEARCH_PARAM_KEYS)
    focused = [k for k in focus_keys if k in allowed and k in _OPTUNA_RANGES]
    return focused or list(_DEFAULT_OPTUNA_KEYS)


def _build_candidate_from_trial(
    trial,
    base_cfg: Config,
    *,
    focus_keys: Sequence[str] | None = None,
) -> Config:
    """Search focused knobs (or the full SEARCH_PARAM_KEYS set) via Optuna."""
    base = config_to_dict(base_cfg)
    keys = _resolve_focus_keys(focus_keys)
    updated = dict(base)

    for key in keys:
        lo, hi = _OPTUNA_RANGES[key]
        if key == "bearish_exit_rsi":
            entry = float(updated.get("bearish_entry_rsi", base["bearish_entry_rsi"]))
            lo = max(entry + 5.0, lo)
            if lo >= hi:
                lo = min(hi - 1.0, hi)
        if key == "buy_grind_rsi_max":
            pullback = float(updated.get("buy_pullback_rsi", base["buy_pullback_rsi"]))
            lo = max(pullback, lo)
            if lo >= hi:
                lo = min(hi - 1.0, hi)
        updated[key] = trial.suggest_float(key, lo, hi)

    return config_from_dict(updated, base_cfg=base_cfg)


def propose_params_with_optuna(
    *,
    data,
    base_cfg: Config,
    trials_per_iteration: int,
    timeout_seconds: int | None,
    seed: int,
    wf_spec: WalkForwardSpec,
    wf_score_spec: WalkForwardScoreSpec,
    focus_keys: Sequence[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna is not installed. Add optuna to your environment to use proposal_source='optuna'."
        ) from exc

    resolved_focus = _resolve_focus_keys(focus_keys)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    trial_payloads: dict[int, dict[str, Any]] = {}

    def objective(trial):
        candidate_cfg = _build_candidate_from_trial(trial, base_cfg, focus_keys=resolved_focus)
        wf_result = evaluate_walk_forward(
            data=data,
            cfg=candidate_cfg,
            wf_spec=wf_spec,
            score_spec=wf_score_spec,
        )
        trial_payloads[trial.number] = {
            "aggregate_score": wf_result.aggregate_score,
            "windows_total": wf_result.windows_total,
            "windows_valid": wf_result.windows_valid,
            "windows_rejected": wf_result.windows_rejected,
            "mean_sharpe": wf_result.mean_sharpe,
            "sharpe_std": wf_result.sharpe_std,
            "worst_max_drawdown": wf_result.worst_max_drawdown,
            "avg_num_trades": wf_result.avg_num_trades,
            "objective_components": wf_result.objective_components,
            "candidate_params": {k: config_to_dict(candidate_cfg)[k] for k in SEARCH_PARAM_KEYS},
        }
        return wf_result.aggregate_score

    study.optimize(objective, n_trials=trials_per_iteration, timeout=timeout_seconds)

    if study.best_trial is None:
        raise RuntimeError("Optuna did not produce a best trial.")

    best_trial = study.best_trial
    best_payload = trial_payloads.get(best_trial.number, {})
    best_cfg_full = config_to_dict(
        config_from_dict(best_payload.get("candidate_params", {}), base_cfg=base_cfg)
    )
    base_dict = config_to_dict(base_cfg)
    best_overrides = {k: v for k, v in best_cfg_full.items() if base_dict.get(k) != v}

    metadata = {
        "engine": "optuna",
        "best_trial_number": best_trial.number,
        "best_value": None if best_trial.value is None else float(best_trial.value),
        "best_params": dict(best_trial.params),
        "trial_count": len(study.trials),
        "focus_keys": list(resolved_focus),
        "walk_forward_spec": asdict(wf_spec),
        "walk_forward_score_spec": asdict(wf_score_spec),
        "best_trial_payload": best_payload,
    }
    return best_overrides, metadata
