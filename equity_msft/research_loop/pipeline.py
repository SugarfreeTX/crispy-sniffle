from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


# Tunable knobs the loop is allowed to move. Risk-policy fields stay in Config
# so a run can override them, but Optuna / Grok / Codex are steered at these first.
SEARCH_PARAM_KEYS = (
    "buy_pullback_rsi",
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
    "neutral_rsi_low",
    "neutral_rsi_high",
    "neutral_rel_volume_max",
)


@dataclass(frozen=True)
class Config:
    """Parameter set passed into equity_msft.backtest.MSFTDailyBacktestStrategy.

    Defaults match backtest.py CLI defaults (not the strategy-class 999 streak
    disable) so a research run is comparable to the existing reviewer-pack
    single backtests.
    """

    initial_cash: float = 100000.0
    commission_bps: float = 5.0
    slippage_bps: float = 5.0
    risk_per_trade_pct: float = 0.02
    max_position_size_pct: float = 0.20
    max_drawdown_pct: float = 0.10
    min_atr: float = 3.5
    max_atr: float = 18.0
    max_consecutive_losses: int = 5
    regime_caution_days: int = 18
    neutral_rsi_low: float = 40.0
    neutral_rsi_high: float = 60.0
    neutral_rel_volume_max: float = 1.3
    bullish_hold_rsi: float = 62.0
    bearish_entry_rsi: float = 28.0
    bearish_exit_rsi: float = 52.0
    buy_pullback_rsi: float = 46.0
    sell_bearish_rsi: float = 52.0
    sell_overbought_rsi: float = 88.0
    take_profit_pnl_pct: float = 7.0
    take_profit_rsi: float = 58.0
    extreme_setup_rsi: float = 30.0
    extreme_setup_rel_vol: float = 1.0


def config_to_dict(cfg: Config) -> dict[str, Any]:
    return asdict(cfg)


def config_from_dict(params: dict[str, Any], base_cfg: Config | None = None) -> Config:
    allowed = {f.name for f in fields(Config)}
    merged = config_to_dict(base_cfg) if base_cfg is not None else config_to_dict(Config())
    merged.update({k: v for k, v in params.items() if k in allowed})
    return sanitize_config(Config(**merged))


def sanitize_config(cfg: Config) -> Config:
    """Keep LLM / Optuna proposals inside strategy invariants."""
    values = config_to_dict(cfg)

    if values["min_atr"] > values["max_atr"]:
        values["min_atr"], values["max_atr"] = values["max_atr"], values["min_atr"]

    if values["neutral_rsi_low"] >= values["neutral_rsi_high"]:
        lo = min(values["neutral_rsi_low"], values["neutral_rsi_high"])
        hi = max(values["neutral_rsi_low"], values["neutral_rsi_high"])
        if lo == hi:
            hi = lo + 1.0
        values["neutral_rsi_low"] = lo
        values["neutral_rsi_high"] = hi

    if values["bearish_entry_rsi"] > values["bearish_exit_rsi"]:
        values["bearish_entry_rsi"], values["bearish_exit_rsi"] = (
            values["bearish_exit_rsi"],
            values["bearish_entry_rsi"],
        )

    values["max_position_size_pct"] = min(max(float(values["max_position_size_pct"]), 0.01), 1.0)
    values["risk_per_trade_pct"] = min(max(float(values["risk_per_trade_pct"]), 0.001), 0.10)
    values["max_drawdown_pct"] = min(max(float(values["max_drawdown_pct"]), 0.01), 0.50)
    values["max_consecutive_losses"] = max(int(values["max_consecutive_losses"]), 1)
    values["regime_caution_days"] = max(int(values["regime_caution_days"]), 1)
    values["initial_cash"] = max(float(values["initial_cash"]), 1000.0)

    return Config(**values)


def load_json_object(path: str | Path) -> dict[str, Any]:
    raw = json_loads_object(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return raw


def json_loads_object(text: str) -> Any:
    import json

    return json.loads(text)


def load_config_from_json(path: str | Path | None, base_cfg: Config | None = None) -> Config:
    if not path:
        return base_cfg or Config()
    return config_from_dict(load_json_object(path), base_cfg=base_cfg or Config())


def research_iteration(data, cfg: Config, proposer: str = "grok"):
    """One human-style cycle: backtest -> Grok review -> parameter proposal."""
    if __package__:
        from .backtest_runner import run_backtest_and_extract_metrics
        from .codex_client import refine_strategy_with_codex
        from .grok_client import evaluate_with_grok, refine_strategy_with_grok
    else:
        from backtest_runner import run_backtest_and_extract_metrics
        from codex_client import refine_strategy_with_codex
        from grok_client import evaluate_with_grok, refine_strategy_with_grok

    if not isinstance(cfg, Config):
        if isinstance(cfg, dict):
            cfg = config_from_dict(cfg)
        else:
            raise TypeError(f"cfg must be Config or dict, got {type(cfg)!r}")

    proposer_name = proposer.lower().strip()
    if proposer_name == "grok":
        refine = refine_strategy_with_grok
    elif proposer_name == "codex":
        refine = refine_strategy_with_codex
    else:
        raise ValueError(f"Unsupported proposer={proposer!r}. Use 'grok' or 'codex'.")

    metrics = run_backtest_and_extract_metrics(data, cfg)
    grok_eval = evaluate_with_grok(metrics)
    proposed_params = refine(config_to_dict(cfg), grok_eval)
    if not isinstance(proposed_params, dict):
        raise TypeError(f"{refine.__name__} must return a dict of updated parameters")

    allowed = {f.name for f in fields(Config)}
    new_params = {k: v for k, v in proposed_params.items() if k in allowed}
    return new_params, metrics, grok_eval
