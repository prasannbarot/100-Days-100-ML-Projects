import os
import pandas as pd
import torch
import logging
import re
import warnings
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# 1. Environment Optimization for 8GB RAM & Stability
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60' # Increased timeout for slow connections

# Suppress messy Protobuf/Library warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("processor.log"), logging.StreamHandler()]
)

class FinancialSentimentProcessor:
    def __init__(self, model_name="yiyanghkust/finbert-tone"):
        self.device = self._get_device()
        self.processed_path = Path("data/processed")
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        self.use_transformers = False
        logging.info(f"🚀 Initializing Sentiment Engine...")

        try:
            # Attempt to load High-Performance Transformer
            logging.info(f"Attempting to load {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                low_cpu_mem_usage=True # Optimized for 8GB RAM
            ).to(self.device)
            self.model.eval()
            self.use_transformers = True
            logging.info("✅ Transformer Model (FinBERT) Loaded.")
        except Exception as e:
            # Resilient Failover: Using VADER if network/memory fails
            logging.warning(f"⚠️ Transformer load failed: {e}")
            logging.info("🔄 Falling back to VADER (Rule-based) for stability.")
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.analyzer = SentimentIntensityAnalyzer()

    def _get_device(self):
        if torch.cuda.is_available(): return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")

    def clean_text(self, text):
        if not isinstance(text, str): return ""
        text = re.sub(r'http\S+', '', text) # Remove Links
        text = re.sub(r'[^a-zA-Z0-9\s!?\.]', '', text) # Remove special chars but keep punctuation
        return re.sub(r'\s+', ' ', text).strip()

    def predict_vader(self, texts):
        """Fast fallback sentiment analysis."""
        return [self.analyzer.polarity_scores(t)['compound'] for t in texts]

    def predict_finbert(self, texts, batch_size=16):
        """Advanced transformer-based sentiment analysis."""
        predictions = []
        for i in tqdm(range(0, len(texts), batch_size), desc="NLP Inference"):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=1).cpu().numpy()
            
            for p in probs:
                # yiyanghkust/finbert-tone: 0=Neu, 1=Pos, 2=Neg
                # Formula: Score = Probability(Pos) - Probability(Neg)
                predictions.append(p[1] - p[2])
        return predictions

    def process(self, input_file="data/raw/scraped_posts.csv"):
        if not Path(input_file).exists():
            logging.error("Source file not found. Run scraper first.")
            return

        df = pd.read_csv(input_file)
        logging.info(f"Loaded {len(df)} records. Cleaning text...")
        
        df['cleaned_comment'] = df['comment'].apply(self.clean_text)
        df = df[df['cleaned_comment'].str.len() > 10].copy() # Filter noise

        if self.use_transformers:
            df['sentiment_score'] = self.predict_finbert(df['cleaned_comment'].tolist())
        else:
            df['sentiment_score'] = self.predict_vader(df['cleaned_comment'].tolist())

        # Daily Aggregation
        daily = df.groupby('date').agg({
            'sentiment_score': ['mean', 'std', 'count']
        }).reset_index()
        
        daily.columns = ['date', 'sentiment_mean', 'sentiment_std', 'comment_volume']
        daily['date'] = pd.to_datetime(daily['date'])
        daily.fillna(0, inplace=True)
        
        output_file = self.processed_path / "sentiment_trends.csv"
        daily.to_csv(output_file, index=False)
        logging.info(f"✅ Pipeline Complete. Data saved to {output_file}")

if __name__ == "__main__":
    processor = FinancialSentimentProcessor()
    processor.process()