from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean, pstdev
from typing import Any

from crypto_xrp.research_loop.backtest_runner import run_backtest_and_extract_metrics
from crypto_xrp.research_loop.pipeline import Config


@dataclass(frozen=True)
class WalkForwardSpec:
    # 4h data has ~6 bars per day.
    train_days: int = 365
    test_days: int = 90
    step_days: int = 30
    bars_per_day: int = 6

    min_test_bars: int = 200
    min_windows_required: int = 3


@dataclass(frozen=True)
class WalkForwardScoreSpec:
    dd_penalty_weight: float = 1.0
    sharpe_stability_penalty_weight: float = 0.5

    # Optional hard gates for a window to count as valid.
    max_drawdown_limit: float | None = None
    min_trades: int | None = None


@dataclass(frozen=True)
class WindowResult:
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    metrics: dict[str, Any]
    passed_gate: bool


@dataclass(frozen=True)
class WalkForwardResult:
    aggregate_score: float
    windows_total: int
    windows_valid: int
    windows_rejected: int
    mean_sharpe: float
    sharpe_std: float
    worst_max_drawdown: float
    avg_num_trades: float
    objective_components: dict[str, float]
    windows: list[WindowResult]


@dataclass(frozen=True)
class _WindowIndex:
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def _build_window_indexes(n_bars: int, wf_spec: WalkForwardSpec) -> list[_WindowIndex]:
    train_bars = wf_spec.train_days * wf_spec.bars_per_day
    test_bars = wf_spec.test_days * wf_spec.bars_per_day
    step_bars = wf_spec.step_days * wf_spec.bars_per_day

    indexes: list[_WindowIndex] = []
    start = 0
    while start + train_bars + test_bars <= n_bars:
        train_end = start + train_bars
        test_start = train_end
        test_end = test_start + test_bars
        indexes.append(
            _WindowIndex(
                train_start=start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        start += step_bars

    return indexes


def _passes_window_gate(metrics: dict[str, Any], score_spec: WalkForwardScoreSpec) -> bool:
    if score_spec.max_drawdown_limit is not None:
        max_dd = abs(float(metrics.get("max_drawdown", 1.0)))
        if max_dd > score_spec.max_drawdown_limit:
            return False

    if score_spec.min_trades is not None:
        num_trades = int(metrics.get("num_trades", 0))
        if num_trades < score_spec.min_trades:
            return False

    return True


def evaluate_walk_forward(
    data,
    cfg: Config,
    wf_spec: WalkForwardSpec | None = None,
    score_spec: WalkForwardScoreSpec | None = None,
) -> WalkForwardResult:
    wf_spec = wf_spec or WalkForwardSpec()
    score_spec = score_spec or WalkForwardScoreSpec()

    n_bars = len(data)
    indexes = _build_window_indexes(n_bars, wf_spec)

    windows: list[WindowResult] = []
    for ix in indexes:
        test_slice = data.iloc[ix.test_start : ix.test_end]
        if len(test_slice) < wf_spec.min_test_bars:
            continue

        metrics = run_backtest_and_extract_metrics(test_slice, cfg)
        passed_gate = _passes_window_gate(metrics, score_spec)

        windows.append(
            WindowResult(
                train_start=ix.train_start,
                train_end=ix.train_end,
                test_start=ix.test_start,
                test_end=ix.test_end,
                metrics=metrics,
                passed_gate=passed_gate,
            )
        )

    valid_windows = [w for w in windows if w.passed_gate]

    if len(valid_windows) < wf_spec.min_windows_required:
        return WalkForwardResult(
            aggregate_score=float("-inf"),
            windows_total=len(windows),
            windows_valid=len(valid_windows),
            windows_rejected=len(windows) - len(valid_windows),
            mean_sharpe=0.0,
            sharpe_std=0.0,
            worst_max_drawdown=1.0,
            avg_num_trades=0.0,
            objective_components={
                "mean_sharpe": 0.0,
                "drawdown_penalty": score_spec.dd_penalty_weight,
                "stability_penalty": 0.0,
            },
            windows=windows,
        )

    sharpes = [float(w.metrics.get("sharpe", 0.0)) for w in valid_windows]
    max_dds = [abs(float(w.metrics.get("max_drawdown", 1.0))) for w in valid_windows]
    trades = [float(w.metrics.get("num_trades", 0.0)) for w in valid_windows]

    mean_sharpe = mean(sharpes)
    sharpe_std = pstdev(sharpes) if len(sharpes) > 1 else 0.0
    worst_max_drawdown = max(max_dds)
    avg_num_trades = mean(trades)

    drawdown_penalty = score_spec.dd_penalty_weight * worst_max_drawdown
    stability_penalty = score_spec.sharpe_stability_penalty_weight * sharpe_std
    aggregate_score = mean_sharpe - drawdown_penalty - stability_penalty

    return WalkForwardResult(
        aggregate_score=aggregate_score,
        windows_total=len(windows),
        windows_valid=len(valid_windows),
        windows_rejected=len(windows) - len(valid_windows),
        mean_sharpe=mean_sharpe,
        sharpe_std=sharpe_std,
        worst_max_drawdown=worst_max_drawdown,
        avg_num_trades=avg_num_trades,
        objective_components={
            "mean_sharpe": mean_sharpe,
            "drawdown_penalty": drawdown_penalty,
            "stability_penalty": stability_penalty,
        },
        windows=windows,
    )


def walk_forward_result_to_dict(result: WalkForwardResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["windows"] = [
        {
            **asdict(w),
            "metrics": w.metrics,
        }
        for w in result.windows
    ]
    return payload
