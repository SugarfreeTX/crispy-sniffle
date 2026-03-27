# python -c '
import json
with open("shadow_grok_log.jsonl") as f:
    for line in f:
        if not line.strip(): continue
        entry = json.loads(line.strip())
        det = entry.get("deterministic_action", "HOLD")
        grok = entry.get("grok_action")
        if det != grok:
            print("=== OVERRIDE DETECTED ===")
            print(f"Date: {entry.get("date")}")
            print(f"Deterministic: {det} → {entry.get("deterministic_reason")}")
            print(f"Grok: {grok} → {entry.get("grok_reason")}")
            print(f"Packet summary:")
            print(f"  RSI: {entry.get("packet_summary", {}).get("rsi")}")
            print(f"  Regime: {entry.get("packet_summary", {}).get("regime")}")
            print(f"  Drawdown %: {entry.get("packet_summary", {}).get("drawdown_pct")}")
            print(f"  Unrealized PnL %: {entry.get("packet_summary", {}).get("unrealized_pct")}")
            print(f"  Trend: {entry.get("packet_summary", {}).get("trend_label", "N/A")}")
            print("-" * 60)
