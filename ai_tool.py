# ==================================================
# PredictorX v1.5 – Trading Edition
# Fully Polished: Real Market Data + Forecast + Risk
# Confidence capped at 95% for realism
# ==================================================

print("=== PredictorX : Trading Edition ===")
print("Real market data • AI forecast • Risk control")
print("Educational / paper trading only")
print("===================================")

import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ====== SAFE MARKET DATA INPUT ======
while True:
    symbol = input("Enter ticker (e.g. AAPL, TSLA, BTC-USD): ").upper()
    period = input("Time period (e.g. 1mo, 3mo, 6mo, 1y): ")

    try:
        data = yf.download(symbol, period=period)
        if data.empty:
            print("❌ No data found. Check ticker or period and try again.\n")
        else:
            break
    except Exception as e:
        print(f"❌ Error fetching data: {e}\nTry again.\n")

prices = data["Close"].values

# ====== FUTURE STEPS INPUT ======
while True:
    try:
        future_steps = int(input("How many future steps to predict? "))
        if future_steps <= 0:
            raise ValueError
        break
    except ValueError:
        print("❌ Enter a positive integer for future steps.")

# ====== LINEAR REGRESSION MODEL ======
X = np.array([[i] for i in range(len(prices))])
y = prices

model = LinearRegression()
model.fit(X, y)

# Confidence (capped at 95%)
confidence = float(r2_score(y, model.predict(X)) * 100)
confidence = max(0, min(confidence, 95))
confidence = round(confidence, 1)

# Multi-step forecast
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

# ====== OUTPUT ======
print("\n📊 PredictorX Signal")
print(f"Asset: {symbol}")
print(f"Signal: {signal}")
print(f"Confidence: {confidence:.1f}%")
print(f"Last Price: {last_price:.2f}")
print(f"Average Future Price: {avg_future:.2f}")
print(f"Expected Change: {percent_change:.2f}%")

if signal != "WAIT":
    print("\n🛑 Risk Management")
    print(f"Entry: {entry:.2f}")
    print(f"Stop-Loss: {stop_loss:.2f}")
    print(f"Take-Profit: {take_profit:.2f}")
    print("Risk : Reward = 1 : 2")

print("\n🔮 Future Predictions:")
for i, price in enumerate(future_prices, 1):
    print(f"Step {i}: {float(price):.2f}")

# ====== PLOT ======
plt.figure(figsize=(10, 6))
plt.plot(prices, label="Historical Price")
plt.plot(
    range(len(prices), len(prices) + future_steps),
    future_prices,
    linestyle="--",
    marker="x",
    label="Forecast",
)

if signal != "WAIT":
    plt.axhline(stop_loss, color="red", linestyle="--", label="Stop-Loss")
    plt.axhline(take_profit, color="green", linestyle="--", label="Take-Profit")

plt.title(f"PredictorX Forecast — {symbol}")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
