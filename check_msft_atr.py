import yfinance as yf
import pandas as pd

def calculate_atr(df, period=14):
    """Calculate ATR from OHLC data"""
    high_low = df['High'] - df['Low']
    high_close_prev = abs(df['High'] - df['Close'].shift(1))
    low_close_prev = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

# Fetch MSFT data
print("Fetching MSFT data for ATR analysis...")
ticker = yf.Ticker("MSFT")
hist = ticker.history(period="6mo")  # 6 months for better analysis

if not hist.empty:
    # Calculate ATR
    hist['ATR'] = calculate_atr(hist)
    
    # Get current price
    current_price = hist['Close'].iloc[-1]
    
    # Get ATR statistics
    atr_current = hist['ATR'].iloc[-1]
    atr_mean = hist['ATR'].dropna().mean()
    atr_median = hist['ATR'].dropna().median()
    atr_min = hist['ATR'].dropna().min()
    atr_max = hist['ATR'].dropna().max()
    atr_std = hist['ATR'].dropna().std()
    
    # Calculate percentiles
    atr_25th = hist['ATR'].dropna().quantile(0.25)
    atr_75th = hist['ATR'].dropna().quantile(0.75)
    atr_95th = hist['ATR'].dropna().quantile(0.95)
    
    # Calculate as percentage of price
    atr_current_pct = (atr_current / current_price) * 100
    atr_mean_pct = (atr_mean / current_price) * 100
    
    print("\n" + "="*60)
    print("MSFT ATR ANALYSIS (Last 6 Months)")
    print("="*60)
    print(f"\nCurrent MSFT Price: ${current_price:.2f}")
    print(f"\nCurrent ATR(14): ${atr_current:.2f} ({atr_current_pct:.2f}% of price)")
    print(f"\nATR Statistics (last 6 months):")
    print(f"  Mean:      ${atr_mean:.2f} ({atr_mean_pct:.2f}%)")
    print(f"  Median:    ${atr_median:.2f}")
    print(f"  Min:       ${atr_min:.2f}")
    print(f"  Max:       ${atr_max:.2f}")
    print(f"  Std Dev:   ${atr_std:.2f}")
    print(f"\nPercentiles:")
    print(f"  25th:      ${atr_25th:.2f}")
    print(f"  75th:      ${atr_75th:.2f}")
    print(f"  95th:      ${atr_95th:.2f}")
    
    print("\n" + "="*60)
    print("THRESHOLD RECOMMENDATIONS")
    print("="*60)
    
    # Calculate suggested thresholds
    suggested_min = round(atr_25th * 0.5, 1)  # 50% below 25th percentile
    suggested_max = round(atr_95th * 1.2, 1)  # 20% above 95th percentile
    
    print(f"\nCurrent thresholds in your code:")
    print(f"  min_atr: 2.0")
    print(f"  max_atr: 15.0")
    
    print(f"\nSuggested thresholds based on data:")
    print(f"  min_atr: {suggested_min}  (allows very calm markets)")
    print(f"  max_atr: {suggested_max}  (blocks extreme volatility spikes)")
    
    print(f"\nConservative approach:")
    print(f"  min_atr: {round(atr_25th, 1)}  (25th percentile - blocks only very low volatility)")
    print(f"  max_atr: {round(atr_75th * 1.5, 1)}  (1.5x the 75th percentile - blocks high volatility)")
    
    print(f"\nPermissive approach:")
    print(f"  min_atr: 1.0  (allow almost any calm market)")
    print(f"  max_atr: {round(atr_max * 1.1, 1)}  (just above historical max)")
    
    print("\n" + "="*60)
    
    # Show how often current thresholds would block trades
    blocked_low = (hist['ATR'] < 2.0).sum()
    blocked_high = (hist['ATR'] > 15.0).sum()
    total_days = len(hist['ATR'].dropna())
    
    print(f"\nWith CURRENT thresholds (min=2.0, max=15.0):")
    print(f"  Days blocked (too low):  {blocked_low}/{total_days} ({blocked_low/total_days*100:.1f}%)")
    print(f"  Days blocked (too high): {blocked_high}/{total_days} ({blocked_high/total_days*100:.1f}%)")
    print(f"  Total trading days:      {total_days - blocked_low - blocked_high}/{total_days} ({(total_days-blocked_low-blocked_high)/total_days*100:.1f}%)")
    
    print("\n" + "="*60)
    
else:
    print("ERROR: Could not fetch MSFT data")
