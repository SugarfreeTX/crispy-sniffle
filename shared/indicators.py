from typing import Any, Dict, List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """
    Calculate RSI (Relative Strength Index) from a list of closing prices.
    
    Args:
        prices: List of closing prices (oldest to newest)
        period: RSI lookback period (default 14)
    
    Returns:
        RSI value rounded to 1 decimal place (50.0 if insufficient data or error)
    """
    try:
        if len(prices) < period + 1:
            return 50.0  # Neutral if not enough data
        
        price_series = pd.Series(prices)
        delta = price_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Avoid division by zero
        rs = gain / loss.replace(0, 0.0001)
        rsi = 100 - (100 / (1 + rs))
        
        return round(rsi.iloc[-1], 1)
    
    except Exception as e:
        logger.warning(f"Error calculating RSI: {e}. Returning neutral 50.0")
        return 50.0


def calculate_atr(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int = 14
) -> float:
    """
    Calculate Average True Range (ATR) from OHLC data.
    
    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices (for previous close comparison)
        period: ATR lookback period (default 14)
    
    Returns:
        ATR value rounded to 2 decimal places (0.0 if insufficient data or error)
    """
    try:
        if len(highs) < period + 1 or len(lows) != len(highs) or len(closes) != len(highs):
            return 0.0
        
        df = pd.DataFrame({
            'High': highs,
            'Low': lows,
            'Close': closes
        })
        
        high_low = df['High'] - df['Low']
        high_close_prev = abs(df['High'] - df['Close'].shift(1))
        low_close_prev = abs(df['Low'] - df['Close'].shift(1))
        
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return round(atr.iloc[-1], 2)
    
    except Exception as e:
        logger.warning(f"Error calculating ATR: {e}. Returning 0.0")
        return 0.0
    
def calculate_sma_trend(
        closes: List[float],
        current_price: Optional[float] = None,
        short_window: int = 50,
        long_window: int = 200,
        bullish_label: str = "Bullish (above {long} SMA)",
        bearish_label: str = "Bearish (below {short} SMA)",
        neutral_label: str = "Neutral / Sideways",
        insufficient_label: str = "Insufficient data (need {long} periods)"
        ) -> Dict[str, Any]:
    """
    Calculate short & long SMA, trend label, and whether price is above the long SMA.
    
    Designed to be flexible for different markets/timeframes:
    - Stocks: typically 50/200 daily
    - Crypto: might use 20/50, 50/100, or even 7/30 on daily/4h bars
    
    Args:
        closes: List of closing prices (oldest → newest)
        current_price: Optional current/last price (defaults to closes[-1])
        short_window, long_window: Periods for short/long SMA
        bullish_label, bearish_label, neutral_label, insufficient_label:
            Customizable label templates. Use {short} and {long} placeholders.
    
    Returns:
        Dict with:
        - sma_short: float or None
        - sma_long: float or None
        - trend_label: str
        - price_above_long_sma: bool
    """
    import pandas as pd
    
    if len(closes) < long_window:
        return {
            "sma_short": None,
            "sma_long": None,
            "trend_label": insufficient_label.format(long=long_window),
            "price_above_long_sma": False
        }
    
    series = pd.Series(closes)
    
    sma_short = None
    if len(closes) >= short_window:
        sma_short = round(series.rolling(short_window).mean().iloc[-1], 2)
    
    sma_long = round(series.rolling(long_window).mean().iloc[-1], 2)
    
    price = current_price if current_price is not None else closes[-1]
    
    if price > sma_long:
        trend_label = bullish_label.format(short=short_window, long=long_window)
    elif sma_short is not None and price < sma_short:
        trend_label = bearish_label.format(short=short_window, long=long_window)
    else:
        trend_label = neutral_label
    
    return {
        "sma_short": sma_short,
        "sma_long": sma_long,
        "trend_label": trend_label,
        "price_above_long_sma": bool(price > sma_long)
    }