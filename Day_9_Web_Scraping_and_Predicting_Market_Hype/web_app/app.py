import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import yfinance as yf
from pathlib import Path

# Page Config
st.set_page_config(page_title="WSB Alpha Tracker", layout="wide")

st.title("WallStreetBets Sentiment & ML Predictor")
st.markdown("""
This dashboard correlates Reddit sentiment (FinBERT) with market price movements 
using a Random Forest Regressor.
""")

# Paths
base_path = Path(__file__).parent.parent
sentiment_path = base_path / "data" / "processed" / "sentiment_trends.csv"
model_path = base_path / "artifacts" / "trend_predictor.pkl"
scaler_path = base_path / "artifacts" / "scaler.pkl"

# Load Data
if not sentiment_path.exists():
    st.error("Data missing. Please run the Scraper and Processor first.")
    st.stop()

df_sent = pd.read_csv(sentiment_path, parse_dates=['date'])

# Layout: 3 Columns for Top Metrics
col1, col2, col3 = st.columns(3)
with col1:
    latest_sent = df_sent['sentiment_mean'].iloc[-1]
    st.metric("Avg Sentiment", f"{latest_sent:.2f}", f"{latest_sent - df_sent['sentiment_mean'].iloc[-2]:.2f}")
with col2:
    vol = df_sent['comment_volume'].iloc[-1]
    st.metric("Comment Volume", int(vol))
with col3:
    st.metric("NLP Engine", "FinBERT-Tone")

# Tabs for visual storytelling
tab1, tab2, tab3 = st.tabs(["Sentiment Analysis", "Market Correlation", "ML Prediction"])

with tab1:
    st.subheader("Reddit Sentiment Over Time")
    fig = px.area(df_sent, x='date', y='sentiment_mean', title="Aggregated WSB Sentiment (Daily)")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    ticker = st.selectbox("Select Ticker for Correlation", ["TSLA", "GME", "AMC", "SPY"])
    prices = yf.download(ticker, start=df_sent['date'].min(), progress=False)['Close']
    
    # Dual axis chart or side-by-side
    st.line_chart(prices)
    st.caption(f"Closing prices for {ticker}")

with tab3:
    if model_path.exists():
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        st.subheader("Predict Next Day Return")
        # Let user simulate sentiment
        user_sent = st.slider("Simulate Sentiment Score", -1.0, 1.0, float(latest_sent))
        
        # Prepare features for the model
        # [sentiment_mean, sentiment_lag1, sentiment_change, comment_volume]
        sim_feat = [[user_sent, df_sent['sentiment_mean'].iloc[-1], user_sent - df_sent['sentiment_mean'].iloc[-1], vol]]
        sim_scaled = scaler.transform(sim_feat)
        pred = model.predict(sim_scaled)[0]
        
        st.write(f"### Predicted Price Change: `{pred:+.2%}`")
        if pred > 0:
            st.success("The model anticipates a Bullish move.")
        else:
            st.error("The model anticipates a Bearish move.")
    else:
        st.info("Train the model to enable predictions.")

st.sidebar.info("Designed for the 100 Days of ML Challenge.")