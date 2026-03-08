from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime


class PortfolioDict(TypedDict):
    """
    Core portfolio state (saved/loaded from portfolio_state.json)
    """
    cash: float
    shares: int                  # For stocks; in crypto this could be qty/coins
    cost_basis: float
    initial_capital: float
    peak_value: float
    last_updated: str            # ISO-like string: "YYYY-MM-DD HH:MM:SS"
    last_regime: str
    regime_days_in_state: int
    consecutive_loss_streak: int
    max_consecutive_losses: int


class MarketDataDict(TypedDict):
    """
    Market/indicator data sent to Grok
    """
    price: float
    history: List[float]         # Last N close prices
    volume: int
    volatility: float
    rsi_14: float
    atr_14: float
    atr_percentile: float
    atr_expansion_ratio: float
    market_regime: str
    regime_multiplier: float
    regime_days_in_state: int
    regime_changed_today: bool
    stop_loss_suggestion: float
    take_profit_suggestion: float
    suggested_position_size: int
    sma_50: Optional[float]
    sma_200: Optional[float]
    price_above_200_sma: bool
    trend_label: str
    latest_volume: int
    avg_volume_20d: int
    relative_volume: float


class PortfolioMetricsDict(TypedDict):
    """
    Calculated metrics (not stored, computed on the fly)
    """
    total_equity: float
    position_value: float
    unrealized_pnl: float
    total_pnl: float
    total_return_pct: float
    peak_value: float
    current_drawdown_pct: float


class PacketDict(TypedDict):
    """
    Full data packet sent to Grok (combines portfolio + market + constraints)
    """
    timestamp: str
    symbol: str
    portfolio: Dict[str, Any]          # Includes PortfolioDict + extras like drawdown_level
    market_data: MarketDataDict
    constraints: Dict[str, Any]


# Optional: Add more as needed later
# e.g. TradeRecordDict for trade_history.json
class TradeRecordDict(TypedDict, total=False):
    timestamp: str
    action: str
    qty: int
    price: float
    reason: str
    portfolio_value: float
    # Optional/known metrics
    realized_pnl: Optional[float]
    loss_streak_after: Optional[int]
    was_win: Optional[bool]
    rsi_14: Optional[float]
    atr_14: Optional[float]
    regime: Optional[str]
    metrics: Optional[Dict[str, Any]]