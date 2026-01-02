# ==================================================
# PredictorX Web App – Polished + Stripe Checkout (NO FastAPI)
# ==================================================

import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="PredictorX", layout="wide")

# ====== HEADER ======
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

# ====== HERO ======
st.markdown("""
### 📈 AI Market Snapshot in Seconds
Generate BUY / SELL / WAIT signals, confidence scores,
risk levels, and forecasts instantly.
""")

st.info("🚀 PredictorX Pro is in progress (subscriptions via Stripe).")

# ====== STRIPE UPGRADE SECTION ======
st.markdown("## 💳 Upgrade to PredictorX Pro")

stripe_ready = (
    "STRIPE_SECRET_KEY" in st.secrets
    and "STRIPE_PRICE_ID" in st.secrets
    and "APP_URL" in st.secrets
)

if stripe_ready:
    try:
        import stripe
        stripe.api_key = st.secrets["STRIPE_SECRET_KEY"]

        if st.button("Buy PredictorX Pro (Monthly Subscription)"):
            session = stripe.checkout.Session.create(
                payment_method_types=["card"],
                line_items=[{"price": st.secrets["STRIPE_PRICE_ID"], "quantity": 1}],
                mode="subscription",
                success_url=st.secrets["APP_URL"] + "?success=true",
                cancel_url=st.secrets["APP_URL"],
            )
            st.success("Checkout created ✅")
            st.markdown(f"[👉 Click here to complete payment]({session.url})")

        # Optional: show success if redirected back
        if st.query_params.get("success") == ["true"]:
            st.success("✅ Payment successful! (Auto-unlock comes next with webhooks.)")

    except ModuleNotFoundError:
        st.error("Stripe package not installed. Add `stripe` to requirements.txt and redeploy.")
    except Exception as e:
        st.error(f"Stripe error: {e}")

else:
    st.info(
        "Stripe isn’t configured yet. Add these to Streamlit Secrets:\n\n"
        "- STRIPE_SECRET_KEY (sk_test_...)\n"
        "- STRIPE_PRICE_ID (price_...)\n"
        "- APP_URL (your Streamlit app URL)\n"
    )

st.divider()

# ====== USER INPUT ======
symbol = st.text_input("Enter ticker (e.g. AAPL, TSLA, BTC-USD):", value="TSLA").upper()
period = st.selectbox("Select time period:", ["1mo", "3mo", "6mo", "1y", "2y"])
future_steps = st.number_input("Future steps to predict:", min_value=1, max_value=30, value=5)

# ====== FETCH DATA ======
st.text("Fetching market data...")
try:
    data = yf.download(symbol, period=period, progress=False)
    if data is None or data.empty:
        st.error("❌ No data found. Check ticker or timeframe.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error fetching data: {e}")
    st.stop()

# ====== CLEAN CLOSE PRICES ======
close = data["Close"]
if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]

prices = np.asarray(close, dtype=float).reshape(-1)
prices = prices[np.isfinite(prices)]

if prices.size < 10:
    st.error("❌ Not enough data. Try a longer timeframe.")
    st.stop()

st.success("Market data loaded!")

# ====== AI MODEL ======
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

# ====== SIGNAL ======
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

# ====== OUTPUT ======
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

# ====== CHART ======
st.subheader("📈 Price Chart")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(prices, label="Historical Price")
ax.plot(
    range(len(prices), len(prices) + int(future_steps)),
    future_prices,
    linestyle="--",
    marker="x",
    label="Forecast"
)

if signal != "WAIT":
    ax.axhline(stop_loss, linestyle="--", label="Stop-Loss")
    ax.axhline(take_profit, linestyle="--", label="Take-Profit")

ax.set_title(f"PredictorX Forecast — {symbol}")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)
