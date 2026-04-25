try:
    from ..xrp_vectorbt_best_v2 import run_backtest
except ImportError:
    from xrp_vectorbt_best_v2 import run_backtest


def _to_scalar(value):
    if hasattr(value, "iloc"):
        try:
            return value.iloc[0]
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value

def run_backtest_and_extract_metrics(data, cfg):
    pf = run_backtest(data, cfg)

    metrics = {
        "total_return": float(_to_scalar(pf.total_return())),
        "annualized_return": float(_to_scalar(pf.returns_acc.annualized())),
        "sharpe": float(_to_scalar(pf.returns_acc.sharpe_ratio())),
        "sortino": float(_to_scalar(pf.returns_acc.sortino_ratio())),
        "max_drawdown": float(_to_scalar(pf.returns_acc.max_drawdown())),
        "win_rate": float(_to_scalar(pf.trades.win_rate())),
        "profit_factor": float(_to_scalar(pf.trades.profit_factor())),
        "expectancy": float(_to_scalar(pf.trades.expectancy())),
        "num_trades": int(_to_scalar(pf.trades.count())),
        "avg_trade_duration": float(_to_scalar(pf.trades.duration.mean())),
        "best_trade": float(_to_scalar(pf.trades.pnl.max())),
        "worst_trade": float(_to_scalar(pf.trades.pnl.min())),
    }

    return metrics
