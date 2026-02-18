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

print("\nEnter relevant data (3–5 bullet points). Type 'done' when finished:")
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
Your job is to estimate the probability of a prediction‑market event occurring.

CRITICAL RULES:
- You must reason ONLY from the data provided in this prompt.
- Do NOT use outside knowledge, memory, assumptions, or world models.
- If the provided data is insufficient, state that clearly.
- Do NOT hallucinate facts, numbers, or events.
- Keep your reasoning grounded strictly in the inputs below.

------------------------------------------------------------
EVENT / CONTRACT
{contract_title}

RULES SUMMARY
{contract_rules}

EXPIRATION DATE
{expiration_date}

MARKET PRICES
YES price: {yes_price}
NO price: {no_price}

RELEVANT DATA
{json.dumps(relevant_data, indent=2)}

------------------------------------------------------------
TASK
Using ONLY the information above:
1. Estimate the probability that the event resolves as YES.
2. Provide a short explanation of the key drivers.
3. State your confidence level (low, medium, high).

------------------------------------------------------------
OUTPUT FORMAT (follow EXACTLY)

PROBABILITY: <0.00 to 1.00>
KEY_DRIVERS:
- <bullet 1>
- <bullet 2>
- <bullet 3>
CONFIDENCE: <low/medium/high>

Do NOT output anything else.
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

def extract_probability(text):
    for line in text.splitlines():
        if line.startswith("PROBABILITY:"):
            try:
                return float(line.split(":")[1].strip())
            except:
                return None
    return None

grok_prob = extract_probability(grok_output)

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
    "raw_output": grok_output
}

with open("prediction_log.jsonl", "a") as f:
    f.write(json.dumps(log_entry) + "\n")

print("\nLogged to prediction_log.jsonl")