# xrp_vectorbt_walkforward_regime.py
import vectorbt as vbt
import pandas as pd
import numpy as np
from pathlib import Path

class Config:
    initial_cash = 100000.0
    commission_bps = 5.0
    slippage_bps = 5.0
    risk_per_trade_pct = 0.008
    max_position_size_pct = 0.12

    sma_fast = 15
    sma_slow = 50
    sma_long = 150
    sell_rsi = 80
    
    sl_stop = 1.8
    tp_stop = 8.0

    # Walk-forward settings
    train_days = 730
    test_days = 365
    step_days = 180


def load_data(csv_path="data/xrp_daily.csv"):
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


def create_signals(data, cfg):
    close = data["Close"]
    high = data["High"]
    low = data["Low"]

    rsi = vbt.RSI.run(close, window=14).rsi # type: ignore
    atr = vbt.ATR.run(high, low, close, window=14).atr # type: ignore
    sma_fast = vbt.MA.run(close, window=cfg.sma_fast).ma # type: ignore
    sma_slow = vbt.MA.run(close, window=cfg.sma_slow).ma # type: ignore
    sma_long = vbt.MA.run(close, window=cfg.sma_long).ma # type: ignore

    long_term_uptrend = close > sma_long
    short_term_momentum = close > sma_fast

    entries = long_term_uptrend & short_term_momentum & (close > sma_slow)
    exits = (close < sma_fast) | (rsi >= cfg.sell_rsi)

    equity_proxy = cfg.initial_cash * (close / close.iloc[0])
    risk_dollars = cfg.risk_per_trade_pct * equity_proxy
    stop_distance = atr * 2.0
    size = (risk_dollars / stop_distance).fillna(0).clip(upper=cfg.max_position_size_pct)

    return entries, exits, size


def run_backtest_on_window(data_window, cfg):
    entries, exits, size = create_signals(data_window, cfg)

    pf = vbt.Portfolio.from_signals(
        close=data_window["Close"],
        entries=entries,
        exits=exits,
        size=size,
        size_type="percent",
        init_cash=cfg.initial_cash,
        fees=cfg.commission_bps / 10000.0,
        slippage=cfg.slippage_bps / 10000.0,
        direction="longonly",
        freq="1D",
        sl_stop=cfg.sl_stop,
        tp_stop=cfg.tp_stop,
    )
    return pf.stats()


if __name__ == "__main__":
    data = load_data()
    cfg = Config()
    n = len(data)
    results = []

    start = 0
    while start + cfg.train_days + cfg.test_days <= n:
        test_start = start + cfg.train_days
        test_end = test_start + cfg.test_days

        test_window = data.iloc[test_start:test_end]

        try:
            stats = run_backtest_on_window(test_window, cfg)
            results.append({
                'test_start': test_window.index[0].date(),
                'test_end': test_window.index[-1].date(),
                'Return [%]': round(stats['Total Return [%]'] if 'Total Return [%]' in stats.index else 0, 3),  # type: ignore
                'Max Drawdown [%]': round(stats['Max Drawdown [%]'] if 'Max Drawdown [%]' in stats.index else 0, 3),  # type: ignore
                'Profit Factor': round(stats['Profit Factor'] if 'Profit Factor' in stats.index else 0, 3),  # type: ignore
                'Sharpe Ratio': round(stats['Sharpe Ratio'] if 'Sharpe Ratio' in stats.index else 0, 3),  # type: ignore
                'Calmar Ratio': round(stats['Calmar Ratio'] if 'Calmar Ratio' in stats.index else 0, 3),  # type: ignore
                'Total Trades': int(stats['Total Trades'] if 'Total Trades' in stats.index else 0),  # type: ignore
            })
        except Exception as e:
            print(f"Error in window starting {test_start}: {e}")

        start += cfg.step_days

    df_wf = pd.DataFrame(results)
    print("\n=== Walk-Forward Results (Regime-Filtered) ===")
    print(df_wf.to_string(index=False))

    print("\nSummary:")
    print(f"Average OOS Return     : {df_wf['Return [%]'].mean():.2f}%")
    print(f"Median OOS Return      : {df_wf['Return [%]'].median():.2f}%")
    print(f"Average OOS Calmar     : {df_wf['Calmar Ratio'].mean():.3f}")
    print(f"Average OOS Drawdown   : {df_wf['Max Drawdown [%]'].mean():.2f}%")
    print(f"Number of test windows : {len(df_wf)}")

    df_wf.to_csv("xrp_walkforward_regime_results.csv", index=False)
    print("\nResults saved to xrp_walkforward_regime_results.csv")