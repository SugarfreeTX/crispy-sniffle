# xrp_vectorbt_4h_final.py
import vectorbt as vbt
import pandas as pd
import numpy as np
from pathlib import Path


class Config:
    initial_cash = 100000.0
    commission_bps = 5.0          # 0.05%
    slippage_bps = 5.0            # 0.05%
    risk_per_trade_pct = 0.008    # keep same for now
    max_position_size_pct = 0.12

    # Adjusted for 4H timeframe
    sma_fast = 12                 # ~2 days
    sma_slow = 36                 # ~6 days
    sma_long = 90                 # ~15 days (regime filter)
    sell_rsi = 78
    
    sl_stop = 1.8                 # fixed % stop-loss
    tp_stop = 8.0                 # fixed % take-profit (main exit)

    # Partial profit taking - lock in gains on big winners
    partial_levels = [0.15, 0.40]   # +15% and +40% unrealized PnL
    partial_sizes  = [0.40, 0.30]   # sell 40%, then 30%


def load_data(csv_path="data/xrp_4h.csv"):
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

    # Bull regime filter
    long_term_uptrend = close > sma_long
    short_term_momentum = close > sma_fast

    entries = long_term_uptrend & short_term_momentum & (close > sma_slow)
    exits = (close < sma_fast) | (rsi >= cfg.sell_rsi)

    equity_proxy = cfg.initial_cash * (close / close.iloc[0])
    risk_dollars = cfg.risk_per_trade_pct * equity_proxy
    stop_distance = atr * 2.0
    size = (risk_dollars / stop_distance).fillna(0).clip(upper=cfg.max_position_size_pct)

    return entries, exits, size


def run_backtest(data, cfg=None):
    if cfg is None:
        cfg = Config()

    entries, exits, size = create_signals(data, cfg)

    pf = vbt.Portfolio.from_signals(
        close=data["Close"],
        entries=entries,
        exits=exits,
        size=size,
        size_type="percent",
        init_cash=cfg.initial_cash,
        fees=cfg.commission_bps / 10000.0,
        slippage=cfg.slippage_bps / 10000.0,
        direction="longonly",
        freq="4h",                    # ← 4-hour timeframe
        sl_stop=cfg.sl_stop,
        tp_stop=cfg.tp_stop,
    )

    return pf


if __name__ == "__main__":
    data = load_data()
    pf = run_backtest(data)
    print(pf.stats())