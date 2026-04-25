from xrp_vectorbt_best_v2 import run_backtest

def run_backtest_and_extract_metrics(data, cfg):
    pf = run_backtest(data, cfg)

    metrics = {
        "total_return": float(pf.total_return()),
        "annualized_return": float(pf.returns_acc.annualized().iloc[0]),
        "sharpe": float(pf.returns_acc.sharpe_ratio().iloc[0]),
        "sortino": float(pf.returns_acc.sortino_ratio().iloc[0]),
        "max_drawdown": float(pf.returns_acc.max_drawdown().iloc[0]),
        "win_rate": float(pf.trades.win_rate()),
        "profit_factor": float(pf.trades.profit_factor()),
        "expectancy": float(pf.trades.expectancy()),
        "num_trades": int(pf.trades.count()),
        "avg_trade_duration": float(pf.trades.duration.mean()),
        "best_trade": float(pf.trades.pnl.max()),
        "worst_trade": float(pf.trades.pnl.min()),
    }

    return metrics
