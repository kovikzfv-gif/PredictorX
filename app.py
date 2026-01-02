# ==================================================
# PredictorX Web App – Trading Edition
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
    data = yf.download(symbol, period=period)
    if data.empty:
        st.error("❌ No data found. Check ticker or period.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error fetching data: {e}")
    st.stop()
data_load_state.text("✅ Market data loaded!")

prices = data["Close"].values

# ====== AI FORECAST ======
X = np.array([[i] for i in range(len(prices))])
y = prices
model = LinearRegression()
model.fit(X, y)

confidence = float(r2_score(y, model.predict(X)) * 100)
confidence = max(0, min(confidence, 95))  # cap at 95%
confidence = round(confidence, 1)

future_X = np.array([[len(prices) + i] for i in range(1, future_steps + 1)])
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

#
