from typing import Dict, Any, Optional
import json
import os
from datetime import datetime
from pathlib import Path
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

from shared.types import TradeRecordDict  # Assuming you add this to types.py

logger = logging.getLogger(__name__)

# Default paths (adjust if your structure differs)
DEFAULT_TRADE_HISTORY_FILE = Path(__file__).resolve().parent.parent / "trade_history.json"
DEFAULT_LOG_FILE = Path(__file__).resolve().parent.parent / "trading_log.txt"


def log_trade(
    action: str,
    qty: int,
    price: float,
    reason: str,
    portfolio_value: float,
    metrics: Optional[Dict[str, Any]] = None,
    trade_history_file: Path = DEFAULT_TRADE_HISTORY_FILE
) -> None:
    """
    Append a trade record to trade_history.json.
    
    Args:
        action: e.g. "BUY", "SELL_PARTIAL", "SELL_FULL", "HOLD", "BLOCKED_..."
        qty: Quantity (shares/coins)
        price: Execution or current price
        reason: Short explanation from Grok or local filter
        portfolio_value: Current total equity
        metrics: Optional dict with RSI, ATR, regime, streak, etc.
        trade_history_file: Path to JSON history file
    """
    try:
        trade_record: TradeRecordDict = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
            "qty": qty,
            "price": round(price, 2),
            "reason": reason,
            "portfolio_value": round(portfolio_value, 2)
        }
        
        if metrics:
            trade_record.update(metrics)  # type: ignore[arg-type]
        
        # Load existing history or start fresh
        trade_history = []
        if trade_history_file.exists():
            with open(trade_history_file, "r") as f:
                trade_history = json.load(f)
        
        trade_history.append(trade_record)
        
        # Save back
        with open(trade_history_file, "w") as f:
            json.dump(trade_history, f, indent=2)
        
        logger.info(f"Logged trade: {action} {qty} at ${price:.2f} | Reason: {reason}")
    
    except Exception as e:
        logger.error(f"Error logging trade to {trade_history_file}: {e}")


def send_email_summary(
    packet: Optional[Dict[str, Any]] = None,
    action: str = "SKIPPED",
    reason: str = "",
    dry_run: bool = False,
    log_path: Path = DEFAULT_LOG_FILE,
    env_file: Path = Path(__file__).resolve().parent.parent / ".env"
) -> None:
    """
    Send daily summary email with key metrics and today's log entries.
    Uses .env for SMTP credentials.
    """
    load_dotenv(dotenv_path=env_file)
    
    sender = os.getenv("EMAIL_SENDER") or ""
    password = os.getenv("EMAIL_PASSWORD") or ""
    recipient = os.getenv("EMAIL_RECIPIENT")
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    
    if not all([sender, password, recipient]):
        logger.warning("Email credentials missing in .env — skipping email summary")
        return
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_prefix = today_str + " "
    
    # Collect today's log lines
    log_lines_today = []
    try:
        if log_path.exists():
            with open(log_path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith(today_prefix):
                        log_lines_today.append(line.rstrip())
        else:
            log_lines_today.append("(log file not found)")
    except Exception as e:
        log_lines_today = [f"Could not read log: {type(e).__name__}: {e}"]
    
    # Truncate if too long
    if len(log_lines_today) > 400:
        log_lines_today = log_lines_today[:350] + ["… (truncated) …"] + log_lines_today[-50:]
    
    # Build email body
    body_parts = [
        f"MSFT Trading Bot — Daily Run Summary",
        f"Date:          {today_str}",
        f"Mode:          {'DRY-RUN' if dry_run else 'LIVE'}",
        f"Action:        {action}",
        f"Reason:        {reason or '—'}",
        "-" * 65
    ]
    
    if packet:
        md = packet.get("market_data", {})
        port = packet.get("portfolio", {})
        body_parts.extend([
            f"Price:         ${md.get('price', '—'):.2f}",
            f"RSI(14):       {md.get('rsi_14', '—')}",
            f"ATR(14):       {md.get('atr_14', '—')}",
            f"Regime:        {md.get('market_regime', '—')} ({md.get('regime_days_in_state', '—')} days)",
            f"Drawdown:      {port.get('current_drawdown_pct', '—'):.2f}%",
            f"Total Equity:  ${port.get('total_equity', port.get('cash', '—')):.2f}",
            "-" * 65
        ])
    
    body_parts.append("Today's log entries:")
    body_parts.append("")
    body_parts.extend(log_lines_today or ["(No entries found for today)"])
    
    body = "\n".join(body_parts)
    
    # Build and send MIME message
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = recipient # type: ignore[assignment]
    msg["Subject"] = f"MSFT Bot Daily Run • {today_str} • {action} ({'DRY' if dry_run else 'LIVE'})"
    
    msg.attach(MIMEText(body, "plain", _charset="utf-8"))
    
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            if sender and password:
                server.login(sender, password)
            server.send_message(msg)
        logger.info(f"Daily summary email sent to {recipient}")
    except Exception as e:
        logger.error(f"Failed to send summary email: {e}")