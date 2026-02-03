import pandas as pd
import joblib
import logging
from pathlib import Path

# Windows-safe logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_artifacts():
    base_path = Path(__file__).parent.parent
    model = joblib.load(base_path / "artifacts" / "trend_predictor.pkl")
    scaler = joblib.load(base_path / "artifacts" / "scaler.pkl")
    return model, scaler

def generate_latest_signal():
    try:
        # 1. Load the processed trends
        df = pd.read_csv("data/processed/sentiment_trends.csv")
        if len(df) < 2:
            print("Insufficient data for technical signal. Need at least 2 days of data.")
            return

        # 2. Extract the 4 features our Random Forest expects:
        # [sentiment_mean, sentiment_lag1, sentiment_change, comment_volume]
        latest_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        feat_sentiment = latest_row['sentiment_mean']
        feat_lag1 = prev_row['sentiment_mean']
        feat_change = feat_sentiment - feat_lag1
        feat_volume = latest_row['comment_volume']

        features = [[feat_sentiment, feat_lag1, feat_change, feat_volume]]

        # 3. Predict
        model, scaler = load_artifacts()
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        # 4. Professional Alert Logic
        print("\n" + "="*40)
        print("MARKET SENTIMENT SIGNAL REPORT")
        print("="*40)
        print(f"Current Sentiment: {feat_sentiment:.3f}")
        print(f"Sentiment Change:  {feat_change:+.3f}")
        print(f"Discussion Volume: {int(feat_volume)} comments")
        print("-" * 40)

        if prediction > 0.02:
            msg = f"STRONG BUY SIGNAL: Expected +{prediction:.2%} move."
        elif prediction > 0:
            msg = f"WEAK BULLISH: Expected +{prediction:.2%} move."
        elif prediction < -0.02:
            msg = f"STRONG SELL ALERT: Expected {prediction:.2%} drop."
        else:
            msg = f"WEAK BEARISH: Expected {prediction:.2%} drop."
        
        print(msg)
        print("="*40 + "\n")

    except Exception as e:
        logging.error(f"Alert generation failed: {e}")

if __name__ == "__main__":
    generate_latest_signal()