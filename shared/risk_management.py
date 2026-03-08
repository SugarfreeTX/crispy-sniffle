from typing import Dict, Any, Tuple, Optional
from shared.types import PortfolioDict

DEFAULT_STREAK_THRESHOLDS: Dict[int, float] = {
    3: 0.5,
    4: 0.25,
    5: 0.0
}

# ────────────────────────────────────────────────────────────────
# Drawdown-based risk controls
# ────────────────────────────────────────────────────────────────

def get_drawdown_level(drawdown_pct: float) -> Tuple[int, str, float]:
    """
    Return (level: int, name: str, size_multiplier: float)
    size_multiplier is applied to suggested shares/qty/position risk
    """
    if drawdown_pct >= 10.0:
        return 3, "Emergency (>10%)", 0.0
    elif drawdown_pct >= 8.0:
        return 2, "Restricted (8-9.9%)", 0.25
    elif drawdown_pct >= 5.0:
        return 1, "Caution (5-7.9%)", 0.5
    else:
        return 0, "Normal (<5%)", 1.0


# ────────────────────────────────────────────────────────────────
# Consecutive realized loss streak protection
# ────────────────────────────────────────────────────────────────

def get_loss_streak_multiplier(
    portfolio: PortfolioDict,
    thresholds: Optional[Dict[int, float]] = None
) -> float:
    """
    Returns size multiplier based on current consecutive loss streak.
    Uses DEFAULT_STREAK_THRESHOLDS if none provided.
    """
    thresholds = thresholds or DEFAULT_STREAK_THRESHOLDS
    streak = portfolio.get("consecutive_loss_streak", 0)
    
    for level, mult in sorted(thresholds.items(), reverse=True):
        if streak >= level:
            return mult
    return 1.0


def can_open_new_position(
    portfolio: PortfolioDict,
    drawdown_pct: float,
    max_drawdown_pct: float = 10.0
) -> Tuple[bool, str]:
    """
    Holistic check: combines drawdown + loss streak.
    Returns (allowed: bool, reason: str)
    """
    streak = portfolio.get("consecutive_loss_streak", 0)
    max_streak = portfolio.get("max_consecutive_losses", 5)
    
    if streak >= max_streak:
        return False, f"Loss streak protection active ({streak}/{max_streak}) — new entries blocked"
    
    if drawdown_pct >= max_drawdown_pct:
        return False, f"Drawdown {drawdown_pct:.1f}% exceeds max allowed {max_drawdown_pct}%"
    
    return True, "OK"


# ────────────────────────────────────────────────────────────────
# Position sizing helpers
# ────────────────────────────────────────────────────────────────

def apply_all_risk_multipliers(
    base_qty: int,
    portfolio: PortfolioDict,
    current_drawdown_pct: float,
    regime_multiplier: float = 1.0,
    custom_streak_thresholds: Optional[Dict[int, float]] = None
) -> int:
    """
    Applies layered risk reductions in order:
    1. Regime/volatility multiplier
    2. Drawdown multiplier
    3. Loss streak multiplier
    
    Returns final adjusted quantity (≥ 0)
    """
    if base_qty <= 0:
        return 0
    
    qty = base_qty
    
    # 1. Regime/vol
    qty = int(qty * regime_multiplier)
    
    # 2. Drawdown
    _, _, dd_mult = get_drawdown_level(current_drawdown_pct)
    qty = int(qty * dd_mult)
    
    # 3. Loss streak
    streak_mult = get_loss_streak_multiplier(portfolio, custom_streak_thresholds)
    qty = int(qty * streak_mult)
    
    return max(qty, 0)