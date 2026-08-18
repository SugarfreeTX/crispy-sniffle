from __future__ import annotations

from dataclasses import asdict
from typing import Any

from equity_msft.research_loop._bootstrap import ensure_import_paths

ensure_import_paths()

from equity_msft.research_loop.pipeline import Config, SEARCH_PARAM_KEYS, config_from_dict, config_to_dict
from equity_msft.research_loop.walk_forward_scorer import (
    WalkForwardScoreSpec,
    WalkForwardSpec,
    evaluate_walk_forward,
)


def _build_candidate_from_trial(trial, base_cfg: Config) -> Config:
    """Search the same knobs as backtest.py --sweep, plus a few exit/entry gates."""
    base = config_to_dict(base_cfg)

    buy_pullback_rsi = trial.suggest_float("buy_pullback_rsi", 38.0, 54.0)
    sell_overbought_rsi = trial.suggest_float("sell_overbought_rsi", 74.0, 92.0)
    min_atr = trial.suggest_float("min_atr", 0.5, 5.0)
    extreme_setup_rsi = trial.suggest_float("extreme_setup_rsi", 20.0, 40.0)
    extreme_setup_rel_vol = trial.suggest_float("extreme_setup_rel_vol", 0.8, 1.5)
    bearish_entry_rsi = trial.suggest_float("bearish_entry_rsi", 18.0, 35.0)
    bearish_exit_rsi = trial.suggest_float(
        "bearish_exit_rsi", max(bearish_entry_rsi + 5.0, 40.0), 60.0
    )
    bullish_hold_rsi = trial.suggest_float("bullish_hold_rsi", 52.0, 70.0)
    take_profit_pnl_pct = trial.suggest_float("take_profit_pnl_pct", 5.0, 10.0)
    take_profit_rsi = trial.suggest_float("take_profit_rsi", 52.0, 70.0)
    sell_bearish_rsi = trial.suggest_float("sell_bearish_rsi", 45.0, 62.0)

    updated = {
        **base,
        "buy_pullback_rsi": buy_pullback_rsi,
        "sell_overbought_rsi": sell_overbought_rsi,
        "min_atr": min_atr,
        "extreme_setup_rsi": extreme_setup_rsi,
        "extreme_setup_rel_vol": extreme_setup_rel_vol,
        "bearish_entry_rsi": bearish_entry_rsi,
        "bearish_exit_rsi": bearish_exit_rsi,
        "bullish_hold_rsi": bullish_hold_rsi,
        "take_profit_pnl_pct": take_profit_pnl_pct,
        "take_profit_rsi": take_profit_rsi,
        "sell_bearish_rsi": sell_bearish_rsi,
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

    optuna.logging.set_verbosity(optuna.logging.WARNING)
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
        "walk_forward_spec": asdict(wf_spec),
        "walk_forward_score_spec": asdict(wf_score_spec),
        "best_trial_payload": best_payload,
    }
    return best_overrides, metadata
