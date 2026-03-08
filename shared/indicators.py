from typing import List, Optional
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