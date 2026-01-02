# ==================================================
# PredictorX Web App – Trading Edition (ULTRA ROBUST)
# Fixes Streamlit Cloud numpy array formatting crash
# ==================================================

import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="PredictorX", layout="wide")

st.title("🔮 PredictorX Trading AI")
st.write("Real market data • AI forecast • Risk management (paper trading only)")

symbol = st.text_input("Enter ticker (e.g. AAPL, TSLA, BTC-USD):", value="TSLA").upper()
period = st.selectbox("Select time period:", ["1mo", "3mo", "6mo", "1y", "2y"])
future_steps = st.number_input("Future steps to predict:", min_value=1, max_value=30, value=5)

st.text("Fetching market data...")
try:
    data = yf.download(symbol, period=period, progress=False)
    if data is None or data.empty:
        st.error("❌ No data found. Check ticker or period.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error fetching data: {e}")
    st.stop()

# --- ULTRA robust Close extraction ---
# Handles: Series, DataFrame, MultiIndex columns, numpy arrays, weird shapes
close = data["Close"]

# If Close is a DataFrame (can happen with some yfinance outputs), take first column
if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]

# Convert to numpy, flatten hard to 1D, and ensure floats
prices = np.asarray(close.to_numpy() if hasattr(close, "to_numpy") else close, dtype=float)
prices = prices.reshape(-1)  # <-- forces 1D no matter what
prices = prices[np.isfinite(prices)]  # drop NaNs/inf

if prices.size < 10:
    st.error("❌ Not enough valid data. Try a longer period (6mo or 1y).")
    st.stop()

st.success("Market data loaded!")

# ===== AI MODEL =====
X = np.arange(len(prices)).reshape(-1, 1)
y = prices

model = LinearRegression()
model.fit(X, y)

confidence = float(r2_score(y, model.predict(X)) * 100)
confidence = max(0.0, min(confidence, 95.0))
confidence = round(confidence, 1)

future_steps = int(future_steps)
future_X = np.arange(len(prices), len(prices) + future_steps).reshape(-1, 1)
future_prices = model.predict(future_X)

# --- Guarantee scalars ---
last_price = float(np.ravel(prices)[-1])
avg_future = float(np.mean(future_prices))
percent_change = ((avg_future - last_price) / last_price) * 100

if percent_change > 0.3:
    signal = "BUY"
elif percent_change < -0.3:
    signal = "SELL"
else:
    signal = "WAIT"

volatility = float(np.std(prices))

if signal == "BUY":
    entry = last_price
    stop_loss = entry - volatility
    take_profit = entry + volatility * 2
elif signal == "SELL":
    entry = last_price
    stop_loss = entry + volatility
    take_profit = entry - volatility * 2
else:
    entry = stop_loss = take_profit = None

# ===== DISPLAY =====
st.subheader("📊 PredictorX Signal")
st.write(f"**Asset:** {symbol}")
st.write(f"**Signal:** {signal}")
st.write(f"**Confidence:** {confidence}%")
st.write(f"**Last Price:** {last_price:.2f}")
st.write(f"**Average Future Price:** {avg_future:.2f}")
st.write(f"**Expected Change:** {percent_change:.2f}%")

if signal != "WAIT":
    st.subheader("🛑 Risk Management")
    st.write(f"**Entry:** {entry:.2f}")
    st.write(f"**Stop-Loss:** {stop_loss:.2f}")
    st.write(f"**Take-Profit:** {take_profit:.2f}")
    st.write("Risk : Reward = 1 : 2")

st.subheader("🔮 Future Predictions")
for i, p in enumerate(future_prices, 1):
    st.write(f"Step {i}: {float(p):.2f}")

st.subheader("📈 Price Chart")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(prices, label="Historical Price")
ax.plot(
    range(len(prices), len(prices) + future_steps),
    future_prices,
    linestyle="--",
    marker="x",
    label="Forecast",
)

if signal != "WAIT":
    ax.axhline(stop_loss, linestyle="--", label="Stop-Loss")
    ax.axhline(take_profit, linestyle="--", label="Take-Profit")

ax.set_title(f"PredictorX Forecast — {symbol}")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)
