import yfinance as yf
import pandas as pd
from pathlib import Path

# Download 4H XRP-USD data (as far back as yfinance allows)
ticker = yf.Ticker("XRP-USD")
df = ticker.history(interval="4h", period="730d")   # max ~2 years for 4h; adjust if needed

# Or for maximum history (yfinance sometimes allows more with start/end)
# df = ticker.history(interval="4h", start="2018-01-01", end="2025-12-30")

df = df[["Open", "High", "Low", "Close", "Volume"]]
df.index.name = "Date"

# Save to your data folder
Path("data").mkdir(exist_ok=True)
df.to_csv("data/xrp_4h.csv")

print(f"Downloaded {len(df)} 4H bars")
print(df.tail())