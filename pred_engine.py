import json
import datetime
import requests
import os
from dotenv import load_dotenv

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
load_dotenv()  # Load environment variables from .env file
GROK_API_KEY = os.getenv("GROK_API_KEY")
GROK_ENDPOINT = "https://api.x.ai/v1/chat/completions"

# ------------------------------------------------------------
# USER INPUT SECTION (MANUAL FOR WEEK 1)
# ------------------------------------------------------------

contract_title = input("Contract title: ")
contract_rules = input("Contract rules summary: ")
expiration_date = input("Expiration date (YYYY-MM-DD): ")

yes_price = float(input("YES price (0.00): "))
no_price = float(input("NO price (0.00): "))

print("\nEnter relevant data (3-5 bullet points). Type 'done' when finished:")
relevant_data = []
while True:
    item = input("- ")
    if item.lower() == "done":
        break
    relevant_data.append(item)

# ------------------------------------------------------------
# BUILD PACKET FOR GROK
# ------------------------------------------------------------

packet = {
    "contract": contract_title,
    "rules": contract_rules,
    "expiration": expiration_date,
    "market_prices": {
        "yes": yes_price,
        "no": no_price
    },
    "relevant_data": relevant_data
}

# ------------------------------------------------------------
# BUILD PROMPT FOR GROK
# ------------------------------------------------------------

prompt = f"""
You are a probability estimation agent.
Your job is to estimate the probability of a prediction-market event occurring.

CRITICAL RULES:
- You must reason ONLY from the data provided in this prompt.
- Do NOT use outside knowledge, memory, assumptions, or world models.
- If the provided data is insufficient, state that clearly and still give your best estimate with low confidence.
- Do NOT hallucinate facts, numbers, or events.
- Keep your reasoning grounded strictly in the inputs below.
- Estimate your probability INDEPENDENTLY before considering market prices.

------------------------------------------------------------
EVENT / CONTRACT
{contract_title}

RULES SUMMARY
{contract_rules}

TODAY'S DATE
{datetime.date.today()}

EXPIRATION DATE
{expiration_date}

RELEVANT DATA
{json.dumps(relevant_data, indent=2)}

------------------------------------------------------------
STEP 1: Using ONLY the event details and relevant data above, estimate the
probability that this event resolves YES. Think step-by-step before giving
a number.

STEP 2: Now consider the current market prices:
YES price: {yes_price}
NO price: {no_price}
Assess whether your independent estimate diverges from the market and why.

------------------------------------------------------------
OUTPUT FORMAT (follow EXACTLY)

REASONING:
<2-4 sentences of step-by-step analysis>

PROBABILITY: <0.00 to 1.00>

KEY_DRIVERS:
- <driver 1>
- <driver 2>
- <driver 3>
(add more if needed)

MARKET_DIVERGENCE: <brief note on how your estimate compares to the YES price>

CONFIDENCE: <low | medium | high>
  low = insufficient data or highly uncertain outcome
  medium = reasonable data but significant unknowns remain
  high = strong data clearly pointing in one direction
"""

# ------------------------------------------------------------
# SEND TO GROK
# ------------------------------------------------------------

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {GROK_API_KEY}"
}

payload = {
    "model": "grok-beta",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.0
}

response = requests.post(GROK_ENDPOINT, headers=headers, json=payload)
grok_output = response.json()["choices"][0]["message"]["content"]

print("\n--- RAW GROK OUTPUT ---")
print(grok_output)

# ------------------------------------------------------------
# PARSE GROK OUTPUT
# ------------------------------------------------------------

def parse_grok_output(text):
    """Parse structured fields from Grok's response."""
    result: dict = {"probability": None, "confidence": None, "reasoning": None, "market_divergence": None}
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("PROBABILITY:"):
            try:
                result["probability"] = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("CONFIDENCE:"):
            result["confidence"] = line.split(":", 1)[1].strip().lower()
        elif line.startswith("MARKET_DIVERGENCE:"):
            result["market_divergence"] = line.split(":", 1)[1].strip()
        elif line.startswith("REASONING:"):
            # Collect lines until the next section
            reasoning_lines = []
            for j in range(i + 1, len(lines)):
                if lines[j].strip() == "" or lines[j].startswith(("PROBABILITY:", "KEY_DRIVERS:", "MARKET_DIVERGENCE:", "CONFIDENCE:")):
                    break
                reasoning_lines.append(lines[j].strip())
            result["reasoning"] = " ".join(reasoning_lines)
    return result

parsed = parse_grok_output(grok_output)
grok_prob = parsed["probability"]

if grok_prob is None:
    print("\nERROR: Could not parse probability from Grok output.")
    exit()

# ------------------------------------------------------------
# CALCULATE EDGE
# ------------------------------------------------------------

edge = grok_prob - yes_price

print("\n--- SUMMARY ---")
print(f"Grok probability: {grok_prob:.2f}")
print(f"Market YES price: {yes_price:.2f}")
print(f"Edge: {edge:+.2f}")
if parsed["confidence"]:
    print(f"Confidence: {parsed['confidence']}")
if parsed["reasoning"]:
    print(f"Reasoning: {parsed['reasoning']}")
if parsed["market_divergence"]:
    print(f"Market divergence: {parsed['market_divergence']}")

if edge > 0:
    print("Suggested action: BUY YES")
elif edge < 0:
    print("Suggested action: BUY NO")
else:
    print("Suggested action: HOLD")

# ------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------

timestamp = datetime.datetime.now().isoformat()

log_entry = {
    "timestamp": timestamp,
    "contract": contract_title,
    "yes_price": yes_price,
    "no_price": no_price,
    "grok_probability": grok_prob,
    "edge": edge,
    "confidence": parsed["confidence"],
    "reasoning": parsed["reasoning"],
    "market_divergence": parsed["market_divergence"],
    "raw_output": grok_output
}

with open("prediction_log.jsonl", "a") as f:
    f.write(json.dumps(log_entry) + "\n")

print("\nLogged to prediction_log.jsonl")