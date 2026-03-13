from typing import Dict, Any
import json
import os
from datetime import datetime
from pathlib import Path
import requests
import logging
from dotenv import load_dotenv
from typing import cast

from shared.types import PortfolioDict

logger = logging.getLogger(__name__)

# Default portfolio state (used on first run or if file missing/corrupt)
DEFAULT_PORTFOLIO: PortfolioDict = {
    "cash": 100000.00,
    "shares": 0,
    "cost_basis": 0.00,
    "initial_capital": 100000.00,
    "peak_value": 100000.00,
    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "last_regime": "Normal",
    "regime_days_in_state": 1,
    "consecutive_loss_streak": 0,
    "max_consecutive_losses": 5,
    # Future crypto note: could add "position_size": 0.0, "margin_used": 0.0, etc.
}

def fetch_alpaca_cash_balance(api_key: str, secret_key: str) -> float | None:
    """Fetch actual cash balance from Alpaca (paper or live). Returns None on failure."""
    url = "https://paper-api.alpaca.markets/v2/account"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        cash = float(data.get("cash", 0.0))
        logger.info(f"Fetched Alpaca cash balance: ${cash:.2f}")
        return cash
    except Exception as e:
        logger.warning(f"Failed to fetch Alpaca cash balance: {e}. Using local state.")
        return None


def load_portfolio_state(
    portfolio_file: Path = Path(__file__).resolve().parent.parent / "portfolio_state.json",
    env_file: Path = Path(__file__).resolve().parent.parent / ".env"
) -> PortfolioDict:
    """
    Load portfolio from JSON file, with defaults + optional Alpaca sync.
    
    Args:
        portfolio_file: Path to portfolio_state.json
        env_file: Path to .env (for Alpaca creds)
    
    Returns:
        PortfolioDict with all expected keys
    """
    portfolio: PortfolioDict = DEFAULT_PORTFOLIO.copy()

    if portfolio_file.exists():
        try:
            with open(portfolio_file, "r") as f:
                loaded = json.load(f)
                # Merge loaded data over defaults (preserves new fields in DEFAULT)
                portfolio.update(loaded)
                logger.info("Loaded existing portfolio state")
        except Exception as e:
            logger.error(f"Error loading portfolio state: {e}. Using defaults.")
    else:
        logger.info("No portfolio file found — using default state")

    # Defensive: ensure no keys went missing (type-ignore if needed)
    for key, value in DEFAULT_PORTFOLIO.items():
        portfolio.setdefault(key, value)  # type: ignore[attr-defined]

    # Sync cash from Alpaca if credentials available
    load_dotenv(dotenv_path=env_file)
    alpaca_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
    if alpaca_key and alpaca_secret:
        alpaca_cash = fetch_alpaca_cash_balance(alpaca_key, alpaca_secret)
        if alpaca_cash is not None:
            local_cash = portfolio["cash"]
            diff = local_cash - alpaca_cash
            if abs(diff) > 0.01:
                logger.warning(
                    f"Cash sync: local=${local_cash:.2f} | Alpaca=${alpaca_cash:.2f} | "
                    f"diff=${diff:.2f} — updating to Alpaca"
                )
            portfolio["cash"] = alpaca_cash

    portfolio["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return portfolio


def save_portfolio_state(
    portfolio: PortfolioDict,
    portfolio_file: Path = Path(__file__).resolve().parent.parent / "portfolio_state.json"
) -> None:
    """
    Save portfolio state to JSON with retry + atomic write.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            portfolio["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Atomic save: write to temp, then rename
            temp_file = portfolio_file.with_suffix(portfolio_file.suffix + ".tmp")
            with open(temp_file, "w") as f:
                json.dump(portfolio, f, indent=2)
            os.replace(temp_file, portfolio_file)
            
            logger.info(f"Saved portfolio state: {portfolio}")
            return
        except Exception as e:
            logger.error(f"Save failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                logger.critical("CRITICAL: Failed to save portfolio after all retries!")
                raise