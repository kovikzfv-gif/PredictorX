# ==================================================
# PredictorX Web App – Trading Edition (Robust Fix)
# Real Market Data + Multi-Step Forecast + Risk
# ==================================================

import streamlit as st
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="PredictorX", layout="wide")

st.title("🔮 PredictorX Trading AI")
st.write("Real market data • AI forecast • Risk management (paper trading only)")

# ====== USER INPUT ======
symbol = st.text_input("Enter ticker (e.g. AAPL, TSLA, BTC-USD):", value="TSLA").upper()
period = st.selectbox("Select time period:", ["1mo", "3mo", "6mo", "1y", "2y"])
future_steps = st.number_input("Future steps to predict:", min_value=1, max_value=30, value=5)

# ====== FETCH DATA ======
data_load_state = st.text("Fetching market data...")
try:
    data = yf.download(symbol, period=period, progress=False)
    if data is None or data.empty:
        st.error("❌ No data found. Check ticker or period.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error fetching data: {e}")
    st.stop()

# --- Robust Close extraction (fixes your crash) ---
close = data.get("Close")
if close is None:
    st.error("❌ 'Close' column not found in downloaded data.")
    st.stop()

# If Close is a DataFrame (sometimes happens), take the first column
if hasattr(close, "ndim") and close.ndim == 2:
    close = close.iloc[:, 0]

# Convert to clean 1D float array and drop NaNs
prices = np.asarray(close, dtype=float)
prices = prices[np.isfinite(prices)]

if prices.size < 10:
    st.error("❌ Not enough valid price data returned. Try a longer period (e.g. 6mo or 1y).")
    st.stop()

data_load_state.text("✅ Market data loaded!")

# ====== AI FORECAST ======
X = np.arange(len(prices)).reshape(-1, 1)
y = prices

model = LinearRegression()
model.fit(X, y)

confidence = float(r2_score(y, model.predict(X)) * 100)
confidence = max(0.0, min(confidence, 95.0))  # cap at 95%
confidence = round(confidence, 1)

future_X = np.arange(len(prices), len(prices) + int(future_steps)).reshape(-1, 1)
future_prices = model.predict(future_X)

last_price = float(prices[-1])
avg_future = float(np.mean(future_prices))
percent_change = ((avg_future - last_price) / last_price) * 100

# Signal logic
if percent_change > 0.3:
    signal = "BUY"
elif percent_change < -0.3:
    signal = "SELL"
else:
    signal = "WAIT"

# Volatility-based risk
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

# ====== DISPLAY RESULTS ======
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

# ====== PLOT ======
st.subheader("📈 Price Chart")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(prices, label="Historical Price")
ax.plot(
    range(len(prices), len(prices) + int(future_steps)),
    future_prices,
    linestyle="--",
    marker="x",
    label="Forecast",
)

if signal != "WAIT":
    ax.axhline(stop_loss, color="red", linestyle="--", label="Stop-Loss")
    ax.axhline(take_profit, color="green", linestyle="--", label="Take-Profit")

ax.set_title(f"PredictorX Forecast — {symbol}")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)
