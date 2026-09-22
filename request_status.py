import os
import requests
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GROK_API_KEY")
assert key, "GROK_API_KEY was not loaded from .env"

response = requests.get(
    "https://api.x.ai/v1/models",
    headers={"Authorization": f"Bearer {key}"},
    timeout=30,
)
print("status:", response.status_code)
print("body:", response.text[:300])
print("Python script executed successfully.")