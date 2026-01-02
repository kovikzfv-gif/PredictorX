# ==================================================
# PredictorX Web App – Pro Gate + Usage Limits
# ==================================================

import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="PredictorX", layout="wide")

# ---------------------------
# PRO GATE + FREE LIMITS
# ---------------------------
FREE_USES_PER_SESSION = 5  # change this if you want

if "is_pro" not in st.session_state:
    st.session_state.is_pro = False
if "free_uses" not in st.session_state:
    st.session_state.free_uses = 0

st.title("🔮 PredictorX Trading AI")
st.caption("AI-powered market analysis • Signals • Risk management (Educational use only)")

st.warning(
    "⚠️ Disclaimer: PredictorX is for educational and paper-trading purposes only. "
    "It does NOT provide financial advice or guarantee profits."
)

with st.expander("Usage Terms"):
    st.markdown("""
- Educational tool only
- Signals are probabilistic
- Market data may be delayed
- Use at your own risk
""")

# Pro / Free status bar
pro_key_saved = st.secrets.get("PRO_KEY", None)

colA, colB, colC = st.columns([1.2, 1.2, 2])
with colA:
    if st.session_state.is_pro:
        st.success("✅ Pro: ON")
    else:
        st.info("Free Mode")

with colB:
    if not st.session_state.is_pro:
        remaining = max(0, FREE_USES_PER_SESSION - st.session_state.free_uses)
        st.write(f"Free uses left: **{remaining} / {FREE_USES_PER_SESSION}**")
    else:
        st.write("Unlimited uses")

with colC:
    with st.popover("Unlock Pro"):
        st.write("Enter your Pro Key to unlock unlimited use.")
        entered = st.text_input("Pro Key", type="password")
        if st.button("Activate Pro"):
            if pro_key_saved and entered == pro_key_saved:
                st.session_state.is_pro = True
                st.success("Pro unlocked ✅")
            else:
                st.error("Invalid Pro Key ❌")

# Hero + Pro banner
st.markdown("""
### 📈 AI Market Snapshot in Seconds
Generate BUY / SELL / WAIT signals, confidence scores, risk levels, and forecasts instantly.
""")

if not st.session_state.is_pro:
    st.info("🚀 PredictorX Pro unlocks unlimited analyses (no daily/session limits).")

# If free limit reached, stop
if (not st.session_state.is_pro) and (st.session_state.free_uses >= FREE_USES_PER_SESSION):
    st.error("❌ Free limit reached for this session. Unlock Pro to continue.")
    st.stop()

# ---------------------------
# USER INPUT
# ---------------------------
symbol = st.text_input("Enter ticker (e.g. AAPL, TSLA, BTC-USD):", value="TSLA").upper()
period = st.selectbox("Select time period:", ["1mo", "3mo", "6mo", "1y", "2y"])
future_steps = st.number_input("Future steps to predict:", min_value=1, max_value=30, value=5)

# Button so we only count “uses” when they actually run it
run = st.button("Run PredictorX")

if not run:
    st.stop()

# Count a free use only when the run button is pressed
if not st.session_state.is_pro:
    st.session_state.free_uses += 1

# ---------------------------
# FETCH DATA
# ---------------------------
st.text("Fetching market data...")
try:
    data = yf.download(symbol, period=period, progress=False)
    if data is None or data.empty:
        st.error("❌ No data found. Check ticker or timeframe.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error fetching data: {e}")
    st.stop()

close = data["Close"]
if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]

prices = np.asarray(close, dtype=float).reshape(-1)
prices = prices[np.isfinite(prices)]

if prices.size < 10:
    st.error("❌ Not enough data. Try a longer timeframe.")
    st.stop()

st.success("Market data loaded!")

# ---------------------------
# AI MODEL
# ---------------------------
X = np.arange(len(prices)).reshape(-1, 1)
y = prices

model = LinearRegression()
model.fit(X, y)

confidence = float(r2_score(y, model.predict(X)) * 100)
confidence = max(0.0, min(confidence, 95.0))
confidence = round(confidence, 1)

future_X = np.arange(len(prices), len(prices) + int(future_steps)).reshape(-1, 1)
future_prices = model.predict(future_X)

last_price = float(prices[-1])
avg_future = float(np.mean(future_prices))
percent_change = ((avg_future - last_price) / last_price) * 100

# Signal
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

# ---------------------------
# OUTPUT
# ---------------------------
st.subheader("📊 PredictorX Signal")
st.write(f"**Asset:** {symbol}")
st.write(f"**Signal:** {signal}")
st.write(f"**Confidence:** {confidence}%")
st.write(f"**Last Price:** {last_price:.2f}")
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
    range(len(prices), len(prices) + int(future_steps)),
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

