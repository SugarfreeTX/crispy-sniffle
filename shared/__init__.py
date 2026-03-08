# shared/__init__.py
# Empty or just a docstring is also fine at first
"""Shared module for reusable trading logic."""
from .types import PortfolioDict
from .indicators import calculate_rsi, calculate_atr
from .portfolio_state import (
    load_portfolio_state,
    save_portfolio_state,
)
from .logging import log_trade, send_email_summary
from .risk_management import (
    get_drawdown_level,
    get_loss_streak_multiplier,
    can_open_new_position,
    apply_all_risk_multipliers
)
from .grok_decision import query_grok, parse_action