# xrp_vectorbt_sweep.py
import vectorbt as vbt
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product

class Config:
    initial_cash = 100000.0
    commission_bps = 5.0
    slippage_bps = 5.0
    min_vol_ratio = 0.004   # ATR/Close
    max_vol_ratio = 0.080   # generous upper bound for XRP
    
    sma_fast = 15
    sma_slow = 50
    sma_long = 150
    sell_rsi = 80
    
    # === Walk-Forward Settings ===
    train_days = 730      # ~2 years train
    test_days  = 365      # 1-year test
    step_days  = 180      # re-optimize every 6 months
    min_oos_trades = 25
    min_oos_calmar = 0.8
    max_oos_dd     = 35.0


def load_data(csv_path="data/xrp_daily.csv"):
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


def run_single_backtest(data, risk_pct, max_pos_pct, sl_mult, tp_mult, sma_fast, sma_slow, sma_long, cfg):
    close = data["Close"]
    high = data["High"]
    low = data["Low"]

    rsi = vbt.RSI.run(close, window=14).rsi # type: ignore
    atr = vbt.ATR.run(high, low, close, window=14).atr # type: ignore
    sma_f = vbt.MA.run(close, window=sma_fast).ma # type: ignore
    sma_s = vbt.MA.run(close, window=sma_slow).ma # type: ignore
    sma_l = vbt.MA.run(close, window=sma_long).ma # type: ignore
    
    vol_ratio = atr / close
    vol_ok = (vol_ratio >= cfg.min_vol_ratio) & (vol_ratio <= cfg.max_vol_ratio)
    
    long_term_uptrend = close > sma_l
    entries = long_term_uptrend & (close > sma_f) & (close > sma_s) & vol_ok
    exits = (close < sma_f) | (rsi >= cfg.sell_rsi)

    # === NEW: ATR-based percentage stops ===
    sl_stop = sl_mult * (atr / close)
    tp_stop = tp_mult * (atr / close)

    equity_proxy = 100000 * (close / close.iloc[0])
    risk_dollars = risk_pct * equity_proxy
    stop_distance = atr * sl_mult if 'sl_mult' in locals() else atr * cfg.sl_mult
    size = (risk_dollars / stop_distance).fillna(0).clip(upper=max_pos_pct)
    pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            size=size,
            size_type="percent",
            init_cash=100000.0,
            fees=0.0005,
            slippage=0.0005,
            direction="longonly",
            freq="1D",
            sl_stop=sl_stop,   # <-- now dynamic %
            tp_stop=tp_stop,   # <-- now dynamic %
        )

    stats = pf.stats()
    return {
        'risk_pct': risk_pct,
        'max_pos_pct': max_pos_pct,
        'sl_mult': sl_mult,
        'tp_mult': tp_mult,
        'sma_fast': sma_fast,
        'sma_slow': sma_slow,
        'sma_long': sma_long,
        'Return [%]': stats['Total Return [%]'] if 'Total Return [%]' in stats.index else 0,  # type: ignore
        'Max Drawdown [%]': stats['Max Drawdown [%]'] if 'Max Drawdown [%]' in stats.index else 0,  # type: ignore
        'Sharpe Ratio': stats['Sharpe Ratio'] if 'Sharpe Ratio' in stats.index else 0,  # type: ignore
        'Profit Factor': stats['Profit Factor'] if 'Profit Factor' in stats.index else 0,  # type: ignore
        'Total Trades': stats['Total Trades'] if 'Total Trades' in stats.index else 0,  # type: ignore
        'Calmar Ratio': stats['Calmar Ratio'] if 'Calmar Ratio' in stats.index else 0,  # type: ignore
    }

def run_walk_forward(data, risk_pct, max_pos_pct, sl_mult, tp_mult,
                     sma_fast, sma_slow, sma_long, cfg):
    close = data["Close"]
    n = len(close)
    results = []

    start = 0
    while start + cfg.train_days + cfg.test_days <= n:
        train_end = start + cfg.train_days
        test_end  = train_end + cfg.test_days

        # train_data = data.iloc[start:train_end]
        test_data  = data.iloc[train_end:test_end]

        # Only run on test (OOS) window for walk-forward validation
        try:
            # row_train = run_single_backtest(train_data, risk_pct, max_pos_pct,
            #                                      sl_mult, tp_mult, sma_fast, sma_slow, sma_long, cfg)
            row_test  = run_single_backtest(test_data,  risk_pct, max_pos_pct,
                                                 sl_mult, tp_mult, sma_fast, sma_slow, sma_long, cfg)
        except Exception as e:
            # handled in #4 below
            raise

        if (row_test['Total Trades'] >= cfg.min_oos_trades and
            row_test['Calmar Ratio'] >= cfg.min_oos_calmar and
            row_test['Max Drawdown [%]'] <= cfg.max_oos_dd):
            results.append(row_test)   # keep only stable OOS windows

        start += cfg.step_days

    if not results:
        return None

    df_test = pd.DataFrame(results)
    
    agg = {
        'risk_pct': risk_pct,
        'max_pos_pct': max_pos_pct,
        'sl_mult': sl_mult,
        'tp_mult': tp_mult,
        'sma_fast': sma_fast,
        'sma_slow': sma_slow,
        'sma_long': sma_long,
        
        'OOS Return [%]': df_test['Return [%]'].mean(),
        'OOS Calmar': df_test['Calmar Ratio'].mean(),
        'OOS Sharpe': df_test['Sharpe Ratio'].mean(),
        'OOS Profit Factor': df_test['Profit Factor'].mean(),
        
        'Avg Trades': df_test['Total Trades'].mean(),
        'Calmar Std': df_test['Calmar Ratio'].std(),          # stability
        
        # === Improved Drawdown Handling ===
        'Avg Max Drawdown [%]': df_test['Max Drawdown [%]'].mean(),   # typical DD
        'Worst Max Drawdown [%]': df_test['Max Drawdown [%]'].max(),  # worst-case DD (conservative)
        
        'Windows Passed': len(df_test)
    }
    return agg

def compute_composite_score(df):
    # Hard constraints — use worst-case DD for safety
    mask = (
        (df['Avg Trades'] >= 25) &
        (df['OOS Calmar'] >= 0.8) &
        (df['OOS Return [%]'] > 0) &
        (df.get('Worst Max Drawdown [%]', pd.Series([100])).fillna(100) <= 35.0)
    )
    df = df[mask].copy()
    if len(df) == 0:
        return pd.DataFrame()

    # Normalize metrics...
    df['calmar_norm'] = (df['OOS Calmar'] - df['OOS Calmar'].min()) / (df['OOS Calmar'].max() - df['OOS Calmar'].min() + 1e-9)
    df['sharpe_norm'] = (df['OOS Sharpe'] - df['OOS Sharpe'].min()) / (df['OOS Sharpe'].max() - df['OOS Sharpe'].min() + 1e-9)
    df['return_norm'] = (df['OOS Return [%]'] - df['OOS Return [%]'].min()) / (df['OOS Return [%]'].max() - df['OOS Return [%]'].min() + 1e-9)
    df['pf_norm']     = (df['OOS Profit Factor'] - 1) / (df['OOS Profit Factor'].max() - 1 + 1e-9)

    # Composite score (you can tweak weights)
    df['Composite Score'] = (
        0.45 * df['calmar_norm'] +
        0.25 * df['sharpe_norm'] +
        0.15 * df['return_norm'] +
        0.10 * df['pf_norm'] +
        0.05 * (1 - df['Calmar Std'] / df['Calmar Std'].max())
    )

    return df.sort_values('Composite Score', ascending=False)

if __name__ == "__main__":
    data = load_data()
    cfg = Config()   # now contains WFO + vol_ratio settings

    # Parameter grid (keep your focused ranges)
    risk_pcts   = [0.005, 0.006, 0.007]
    max_pos_pcts= [0.08, 0.10, 0.12]
    sl_mults    = [1.5, 1.8, 2.0]
    tp_mults    = [6.0, 8.0, 10.0]
    sma_fasts   = [10, 15, 20]
    sma_slows   = [40, 50, 60]
    sma_longs   = [120, 150, 180]

    wfo_results = []
    failed = []

    from itertools import product
    for params in product(risk_pcts, max_pos_pcts, sl_mults, tp_mults,
                          sma_fasts, sma_slows, sma_longs):
        risk_pct, max_pos, sl, tp, sf, ss, slong = params
        try:
            agg = run_walk_forward(data, risk_pct, max_pos, sl, tp, sf, ss, slong, cfg)
            if agg is not None:
                wfo_results.append(agg)
        except Exception as e:
            failed.append({
                'risk_pct': risk_pct, 'max_pos_pct': max_pos,
                'sl_mult': sl, 'tp_mult': tp,
                'sma_fast': sf, 'sma_slow': ss, 'sma_long': slong,
                'error': str(e)
            })

    # Final ranking
    df_wfo = pd.DataFrame(wfo_results)
    df_ranked = compute_composite_score(df_wfo)

    print("\n=== TOP 10 STABLE PARAMETER SETS (Walk-Forward) ===")
    print(df_ranked.head(10)[['risk_pct','max_pos_pct','sl_mult','tp_mult',
                              'sma_fast','sma_slow','sma_long',
                              'OOS Calmar','OOS Sharpe','Avg Trades',
                              'Avg Max Drawdown [%]', 'Worst Max Drawdown [%]',   # ← add these
                              'Composite Score']])

    df_ranked.to_csv("xrp_wfo_ranked.csv", index=False)
    print(f"\nFull WFO results saved → xrp_wfo_ranked.csv ({len(df_ranked)} stable combos)")

    if failed:
        pd.DataFrame(failed).to_csv("xrp_failed_combos.csv", index=False)
        print(f"⚠️  {len(failed)} parameter combos failed and were logged to xrp_failed_combos.csv")