# ==================================================
# PredictorX – Streamlit Trading App + Stripe Checkout (NO KERAS)
# ==================================================

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="PredictorX", layout="wide")

# ----------------------------
# Header
# ----------------------------
st.title("🔮 PredictorX Trading AI")
st.caption("Real market data • AI forecast • Risk management (Educational use only)")

st.warning(
    "⚠️ Disclaimer: PredictorX is for educational/paper-trading only. "
    "This is NOT financial advice and does not guarantee profits."
)

# ----------------------------
# Session state
# ----------------------------
if "is_pro" not in st.session_state:
    st.session_state.is_pro = False

# ----------------------------
# Stripe Checkout (Subscription)
# ----------------------------
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

        base = st.secrets["APP_URL"].rstrip("/")
        success_url = base + "/?success=true"
        cancel_url = base + "/"

        st.info("🚀 PredictorX Pro (Monthly): unlocks unlimited usage + upcoming Pro features.")

        if st.button("Buy PredictorX Pro (Monthly Subscription)"):
            session = stripe.checkout.Session.create(
                payment_method_types=["card"],
                line_items=[{"price": st.secrets["STRIPE_PRICE_ID"], "quantity": 1}],
                mode="subscription",
                success_url=success_url,
                cancel_url=cancel_url,
            )
            st.success("Checkout created ✅")
            st.markdown(f"[👉 Click here to complete payment]({session.url})")

        if st.query_params.get("success") == ["true"]:
            st.success("✅ Payment successful! Enter your checkout email below to unlock Pro.")

    except ModuleNotFoundError:
        st.error("Stripe package not installed. Add `stripe==10.12.0` to requirements.txt and redeploy.")
    except Exception as e:
        st.error(f"Stripe error: {e}")
else:
    st.info(
        "Stripe not configured yet. Add these secrets:\n\n"
        "- STRIPE_SECRET_KEY\n"
        "- STRIPE_PRICE_ID\n"
        "- APP_URL\n"
    )

# ----------------------------
# Pro unlock by email (Webhook backend) – OPTIONAL
# ----------------------------
st.markdown("### ✅ Pro Access (after purchase)")

email = st.text_input("Email used at checkout:", value="")

if email and "WEBHOOK_BASE_URL" in st.secrets:
    try:
        import requests
        r = requests.get(
            st.secrets["WEBHOOK_BASE_URL"].rstrip("/") + "/pro/check",
            params={"email": email},
            timeout=6
        )
        is_pro = bool(r.json().get("pro", False))
    except Exception:
        is_pro = False

    if is_pro:
        st.success("✅ Pro Active! Unlimited access unlocked.")
        st.session_state.is_pro = True
    else:
        st.info("Not Pro yet for this email. If you just paid, wait 5–10 seconds and try again.")
else:
    if "WEBHOOK_BASE_URL" not in st.secrets:
        st.caption("ℹ️ Auto-unlock not connected (missing WEBHOOK_BASE_URL in Secrets).")
    else:
        st.caption("Enter the same email you used during Stripe Checkout.")

st.divider()

# ==================================================
# Main App
# ==================================================
st.markdown("## 📈 Market Forecast")

symbol = st.text_input("Enter ticker (e.g. AAPL, TSLA, BTC-USD):", value="TSLA").upper()
period = st.selectbox("Select time period:", ["1mo", "3mo", "6mo", "1y", "2y"])
future_steps = st.number_input("Future steps to predict:", min_value=1, max_value=30, value=5)

# Free limit (optional)
if not st.session_state.is_pro:
    future_steps = int(min(future_steps, 10))
    st.caption("Free mode: max 10 forecast steps. Upgrade to Pro for more.")

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

st.success("✅ Market data loaded!")

# ----------------------------
# ML Forecast (simple baseline)
# ----------------------------
X = np.arange(len(prices)).reshape(-1, 1)
y = prices

model = LinearRegression()
model.fit(X, y)

fit_pred = model.predict(X)
confidence = float(r2_score(y, fit_pred) * 100)
confidence = max(0.0, min(confidence, 95.0))
confidence = round(confidence, 1)

future_X = np.arange(len(prices), len(prices) + int(future_steps)).reshape(-1, 1)
future_prices = model.predict(future_X)

last_price = float(prices[-1])
avg_future = float(np.mean(future_prices))
percent_change = ((avg_future - last_price) / last_price) * 100

# ----------------------------
# Signal logic
# ----------------------------
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

# ----------------------------
# Output
# ----------------------------
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
