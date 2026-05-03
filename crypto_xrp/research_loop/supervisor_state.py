import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any

from crypto_xrp.research_loop.pipeline import Config, config_from_dict, config_to_dict, research_iteration
from crypto_xrp.research_loop.backtest_runner import run_backtest_and_extract_metrics
from crypto_xrp.research_loop.walk_forward_scorer import WalkForwardScoreSpec, WalkForwardSpec
from crypto_xrp.research_loop.optuna_proposer import propose_params_with_optuna


class LoopState(Enum):
    INIT = auto()
    EVALUATE_INCUMBENT = auto()
    PROPOSE_CANDIDATE = auto()
    EVALUATE_CANDIDATE = auto()
    DECIDE_ACCEPT = auto()
    UPDATE_BEST = auto()
    LOG_ITERATION = auto()
    CHECK_STOP = auto()
    DONE = auto()


@dataclass
class SupervisorConfig:
    max_iterations: int = 20
    patience: int = 5
    min_improvement: float = 0.01
    max_drawdown_limit: float = 0.35
    min_trades: int = 8
    dd_penalty_weight: float = 1.0
    proposal_source: str = "codex"

    # Optuna proposal settings.
    optuna_trials_per_iteration: int = 25
    optuna_timeout_seconds: int | None = None
    optuna_seed: int = 42

    # Walk-forward objective settings used by Optuna.
    wf_train_days: int = 365
    wf_test_days: int = 90
    wf_step_days: int = 30
    wf_bars_per_day: int = 6
    wf_min_test_bars: int = 200
    wf_min_windows_required: int = 3

    wf_dd_penalty_weight: float = 1.0
    wf_sharpe_stability_penalty_weight: float = 0.5
    wf_gate_max_drawdown_limit: float | None = 0.35
    wf_gate_min_trades: int | None = 8

    log_jsonl_path: str = "crypto_xrp/research_loop/supervisor_runs.jsonl"
    best_json_path: str = "crypto_xrp/research_loop/best_config.json"


@dataclass
class CandidateEval:
    cfg: Config
    metrics: dict[str, Any]
    score: float
    passed_hard_gate: bool


@dataclass
class SupervisorMemory:
    iteration: int = 0
    no_improve_streak: int = 0
    state: LoopState = LoopState.INIT

    incumbent_cfg: Config = field(default_factory=Config)
    incumbent_metrics: dict[str, Any] = field(default_factory=dict)
    incumbent_score: float = float("-inf")

    best_cfg: Config = field(default_factory=Config)
    best_metrics: dict[str, Any] = field(default_factory=dict)
    best_score: float = float("-inf")

    proposed_params: dict[str, Any] = field(default_factory=dict)
    grok_eval: str = ""
    proposal_metadata: dict[str, Any] = field(default_factory=dict)
    candidate_eval: CandidateEval | None = None


def _build_optuna_wf_specs(cfg: SupervisorConfig) -> tuple[WalkForwardSpec, WalkForwardScoreSpec]:
    wf_spec = WalkForwardSpec(
        train_days=cfg.wf_train_days,
        test_days=cfg.wf_test_days,
        step_days=cfg.wf_step_days,
        bars_per_day=cfg.wf_bars_per_day,
        min_test_bars=cfg.wf_min_test_bars,
        min_windows_required=cfg.wf_min_windows_required,
    )
    score_spec = WalkForwardScoreSpec(
        dd_penalty_weight=cfg.wf_dd_penalty_weight,
        sharpe_stability_penalty_weight=cfg.wf_sharpe_stability_penalty_weight,
        max_drawdown_limit=cfg.wf_gate_max_drawdown_limit,
        min_trades=cfg.wf_gate_min_trades,
    )
    return wf_spec, score_spec


def objective_score(metrics: dict[str, Any], dd_penalty_weight: float) -> float:
    sharpe = float(metrics.get("sharpe", 0.0))
    max_dd = abs(float(metrics.get("max_drawdown", 1.0)))
    return sharpe - dd_penalty_weight * max_dd


def passes_hard_gate(metrics: dict[str, Any], cfg: SupervisorConfig) -> bool:
    max_dd = abs(float(metrics.get("max_drawdown", 1.0)))
    num_trades = int(metrics.get("num_trades", 0))
    return (max_dd <= cfg.max_drawdown_limit) and (num_trades >= cfg.min_trades)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def save_best(path: Path, mem: SupervisorMemory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "best_score": mem.best_score,
        "best_metrics": mem.best_metrics,
        "best_cfg": config_to_dict(mem.best_cfg),
        "iteration": mem.iteration,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_supervisor_loop(data, sup_cfg: SupervisorConfig, initial_cfg: Config | None = None) -> SupervisorMemory:
    mem = SupervisorMemory()
    if initial_cfg is not None:
        mem.incumbent_cfg = initial_cfg
        mem.best_cfg = initial_cfg

    log_path = Path(sup_cfg.log_jsonl_path)
    best_path = Path(sup_cfg.best_json_path)

    while mem.state != LoopState.DONE:
        if mem.state == LoopState.INIT:
            mem.state = LoopState.EVALUATE_INCUMBENT

        elif mem.state == LoopState.EVALUATE_INCUMBENT:
            mem.incumbent_metrics = run_backtest_and_extract_metrics(data, mem.incumbent_cfg)
            mem.incumbent_score = objective_score(mem.incumbent_metrics, sup_cfg.dd_penalty_weight)

            if mem.best_score == float("-inf"):
                mem.best_score = mem.incumbent_score
                mem.best_cfg = mem.incumbent_cfg
                mem.best_metrics = mem.incumbent_metrics

            mem.state = LoopState.PROPOSE_CANDIDATE

        elif mem.state == LoopState.PROPOSE_CANDIDATE:
            proposal_source = sup_cfg.proposal_source.lower().strip()
            if proposal_source == "optuna":
                wf_spec, score_spec = _build_optuna_wf_specs(sup_cfg)
                new_params, proposal_metadata = propose_params_with_optuna(
                    data=data,
                    base_cfg=mem.incumbent_cfg,
                    trials_per_iteration=sup_cfg.optuna_trials_per_iteration,
                    timeout_seconds=sup_cfg.optuna_timeout_seconds,
                    seed=sup_cfg.optuna_seed + mem.iteration,
                    wf_spec=wf_spec,
                    wf_score_spec=score_spec,
                )
                mem.proposed_params = new_params
                mem.proposal_metadata = proposal_metadata
                mem.grok_eval = (
                    "optuna proposal | "
                    f"trial={proposal_metadata.get('best_trial_number')} "
                    f"value={proposal_metadata.get('best_value')} "
                    f"trials={proposal_metadata.get('trial_count')}"
                )
            elif proposal_source == "codex":
                # Existing pipeline: backtest -> Grok -> Codex proposal
                new_params, _, grok_eval = research_iteration(data, mem.incumbent_cfg)
                mem.proposed_params = new_params
                mem.proposal_metadata = {"engine": "codex"}
                mem.grok_eval = grok_eval
            else:
                raise ValueError(
                    f"Unsupported proposal_source={sup_cfg.proposal_source!r}. Use 'codex' or 'optuna'."
                )
            mem.state = LoopState.EVALUATE_CANDIDATE

        elif mem.state == LoopState.EVALUATE_CANDIDATE:
            candidate_cfg = config_from_dict(mem.proposed_params, base_cfg=mem.incumbent_cfg)
            candidate_metrics = run_backtest_and_extract_metrics(data, candidate_cfg)
            candidate_score = objective_score(candidate_metrics, sup_cfg.dd_penalty_weight)
            gate_ok = passes_hard_gate(candidate_metrics, sup_cfg)
            mem.candidate_eval = CandidateEval(candidate_cfg, candidate_metrics, candidate_score, gate_ok)
            mem.state = LoopState.DECIDE_ACCEPT

        elif mem.state == LoopState.DECIDE_ACCEPT:
            assert mem.candidate_eval is not None
            improve = mem.candidate_eval.score - mem.incumbent_score
            accept = mem.candidate_eval.passed_hard_gate and (improve >= sup_cfg.min_improvement)

            if accept:
                mem.incumbent_cfg = mem.candidate_eval.cfg
                mem.incumbent_metrics = mem.candidate_eval.metrics
                mem.incumbent_score = mem.candidate_eval.score
                mem.no_improve_streak = 0
            else:
                mem.no_improve_streak += 1

            mem.state = LoopState.UPDATE_BEST

        elif mem.state == LoopState.UPDATE_BEST:
            if mem.incumbent_score > mem.best_score:
                mem.best_score = mem.incumbent_score
                mem.best_cfg = mem.incumbent_cfg
                mem.best_metrics = mem.incumbent_metrics
                save_best(best_path, mem)
            mem.state = LoopState.LOG_ITERATION

        elif mem.state == LoopState.LOG_ITERATION:
            assert mem.candidate_eval is not None
            row = {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "iteration": mem.iteration,
                "incumbent_score": mem.incumbent_score,
                "best_score": mem.best_score,
                "no_improve_streak": mem.no_improve_streak,
                "proposed_params": mem.proposed_params,
                "proposal_metadata": mem.proposal_metadata,
                "candidate_score": mem.candidate_eval.score,
                "candidate_metrics": mem.candidate_eval.metrics,
                "candidate_passed_hard_gate": mem.candidate_eval.passed_hard_gate,
                "grok_eval_preview": mem.grok_eval[:500],
            }
            append_jsonl(log_path, row)
            mem.state = LoopState.CHECK_STOP

        elif mem.state == LoopState.CHECK_STOP:
            mem.iteration += 1
            stop = (
                mem.iteration >= sup_cfg.max_iterations
                or mem.no_improve_streak >= sup_cfg.patience
            )
            mem.state = LoopState.DONE if stop else LoopState.EVALUATE_INCUMBENT

    return mem