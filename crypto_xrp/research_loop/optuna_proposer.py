from __future__ import annotations

from dataclasses import asdict
from typing import Any

from crypto_xrp.research_loop.pipeline import Config, config_from_dict, config_to_dict
from crypto_xrp.research_loop.walk_forward_scorer import (
    WalkForwardScoreSpec,
    WalkForwardSpec,
    evaluate_walk_forward,
)


def _build_candidate_from_trial(trial, base_cfg: Config) -> Config:
    base = config_to_dict(base_cfg)

    sma_fast = trial.suggest_int("sma_fast", 8, 24)
    sma_slow = trial.suggest_int("sma_slow", max(sma_fast + 4, 20), 72)
    sma_long = trial.suggest_int("sma_long", max(sma_slow + 8, 80), 220)

    atr_filter_min_pct = trial.suggest_float("atr_filter_min_pct", 0.0015, 0.01)
    atr_filter_max_pct = trial.suggest_float(
        "atr_filter_max_pct", max(atr_filter_min_pct + 0.002, 0.01), 0.08
    )

    sl_stop = trial.suggest_float("sl_stop", 0.006, 0.04)
    tp_stop = trial.suggest_float("tp_stop", max(sl_stop + 0.01, 0.02), 0.16)

    position_size_pct = trial.suggest_float("position_size_pct", 0.04, 0.25)
    sell_rsi = trial.suggest_float("sell_rsi", 65.0, 90.0)

    updated = {
        **base,
        "sma_fast": sma_fast,
        "sma_slow": sma_slow,
        "sma_long": sma_long,
        "atr_filter_min_pct": atr_filter_min_pct,
        "atr_filter_max_pct": atr_filter_max_pct,
        "sl_stop": sl_stop,
        "tp_stop": tp_stop,
        "position_size_pct": position_size_pct,
        "sell_rsi": sell_rsi,
    }

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
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna is not installed. Add optuna to your environment to use proposal_source='optuna'."
        ) from exc

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    trial_payloads: dict[int, dict[str, Any]] = {}

    def objective(trial):
        candidate_cfg = _build_candidate_from_trial(trial, base_cfg)
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
            "candidate_params": config_to_dict(candidate_cfg),
        }

        return wf_result.aggregate_score

    study.optimize(objective, n_trials=trials_per_iteration, timeout=timeout_seconds)

    if study.best_trial is None:
        raise RuntimeError("Optuna did not produce a best trial.")

    best_trial = study.best_trial
    best_payload = trial_payloads.get(best_trial.number, {})
    best_cfg_full = best_payload.get("candidate_params", config_to_dict(base_cfg))

    # Return only key/value overrides relative to base config.
    base_dict = config_to_dict(base_cfg)
    best_overrides = {k: v for k, v in best_cfg_full.items() if base_dict.get(k) != v}

    metadata = {
        "engine": "optuna",
        "best_trial_number": best_trial.number,
        "best_value": float(best_trial.value),
        "best_params": dict(best_trial.params),
        "trial_count": len(study.trials),
        "walk_forward_spec": asdict(wf_spec),
        "walk_forward_score_spec": asdict(wf_score_spec),
        "best_trial_payload": best_payload,
    }

    return best_overrides, metadata
