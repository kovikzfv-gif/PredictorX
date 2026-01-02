# ==================================================
# PredictorX Web App – Polished Edition
# Landing + Disclaimer + Pro Banner + Trading AI
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

st.caption(
    "AI-powered market analysis • Signals • Risk management "
    "(Educational / paper trading only)"
)

st.warning(
    "⚠️ **Disclaimer:** PredictorX is for educational and paper-trading purposes only. "
    "It does NOT provide financial advice and does NOT guarantee profits. "
    "You are fully responsible for your trading decisions."
)

with st.expander("Usage Terms"):
    st.markdown("""
- Educational tool only — **not financial advice**
- Signals are probabilistic and may be wrong
- Market data may be delayed or incomplete
- Use at your own risk
""")

# ====== HERO SECTION ======
st.markdown(
    """
### 📈 AI Market Snapshot in Seconds
Enter a ticker, select a timeframe, and PredictorX instantly generates:
- **BUY / SELL / WAIT** signal  
- **Confidence score**  
- **Stop-loss & take-profit levels**  
- **Future price forecast**
"""
)

st.info(
    "🚀 **PredictorX Pro** coming soon: advanced indicators, alerts, watchlists, and backtesting."
)

# ====== USER INPUT ======
symbol = st.text_input(
    "Enter ticker (e.g. AAPL, TSLA, BTC-USD):",
    value="TSLA"
).upper()

period =

