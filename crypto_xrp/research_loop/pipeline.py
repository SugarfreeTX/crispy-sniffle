from codex_client import refine_strategy_with_codex
from backtest_runner import run_backtest_and_extract_metrics
from grok_client import evaluate_with_grok

def research_iteration(data, cfg):
    # 1. Run backtest
    metrics = run_backtest_and_extract_metrics(data, cfg)

    # 2. Evaluate with Grok
    grok_eval = evaluate_with_grok(metrics)

    # 3. Ask Codex to refine parameters
    new_params = refine_strategy_with_codex(cfg.to_dict(), grok_eval)

    return new_params, metrics, grok_eval
