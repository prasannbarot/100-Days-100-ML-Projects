import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import logging
from pathlib import Path

# Setup logging without emojis for Windows compatibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StockTrendPredictor:
    def __init__(self, tickers=['GME', 'AMC', 'TSLA']):
        self.tickers = tickers
        self.artifacts_path = Path("artifacts")
        self.artifacts_path.mkdir(exist_ok=True)
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    def fetch_market_data(self, start_date):
        logging.info(f"Fetching market data for {self.tickers} starting {start_date}")
        data = yf.download(self.tickers, start=start_date, progress=False)
        
        # Reshape data
        prices = data['Close'].reset_index().melt(id_vars='Date', var_name='ticker', value_name='price')
        prices.rename(columns={'Date': 'date'}, inplace=True)
        return prices

    def prepare_features(self, sentiment_df, prices_df):
        logging.info("Engineering features: Lags and Returns...")
        
        # Merge sentiment and price
        df = pd.merge(sentiment_df, prices_df, on='date', how='inner')
        
        # TARGET: Next day's percentage return
        df['target_return'] = df.groupby('ticker')['price'].pct_change().shift(-1)
        
        # FEATURE 1: Lagged Sentiment (Yesterday's sentiment affects today)
        df['sentiment_lag1'] = df.groupby('ticker')['sentiment_mean'].shift(1)
        
        # FEATURE 2: Sentiment Momentum (Change in sentiment)
        df['sentiment_change'] = df['sentiment_mean'].diff()
        
        # Drop rows with NaN created by lagging/diffing
        df = df.dropna()
        return df

    def train(self):
        # 1. Load Processed Sentiment
        sentiment_path = Path("data/processed/sentiment_trends.csv")
        if not sentiment_path.exists():
            logging.error("sentiment_trends.csv not found!")
            return
        
        sentiment_df = pd.read_csv(sentiment_path, parse_dates=['date'])
        
        # 2. Fetch Prices
        start_date = sentiment_df['date'].min()
        prices_df = self.fetch_market_data(start_date)
        
        # 3. Feature Engineering
        data = self.prepare_features(sentiment_df, prices_df)
        
        if data.empty:
            logging.error("Not enough data to train. Try scraping a wider date range.")
            return

        # 4. Define X and y
        features = ['sentiment_mean', 'sentiment_lag1', 'sentiment_change', 'comment_volume']
        X = data[features]
        y = data['target_return']
        
        # 5. Scale and Train
        logging.info(f"Training Random Forest on {len(data)} data points...")
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        # 6. Evaluation (Basic)
        preds = self.model.predict(X_scaled)
        mae = mean_absolute_error(y, preds)
        logging.info(f"Model trained. Mean Absolute Error: {mae:.4f}")

        # 7. Save Artifacts
        joblib.dump(self.model, self.artifacts_path / "trend_predictor.pkl")
        joblib.dump(self.scaler, self.artifacts_path / "scaler.pkl")
        logging.info("Model and Scaler saved to artifacts/")

if __name__ == "__main__":
    predictor = StockTrendPredictor()
    predictor.train()