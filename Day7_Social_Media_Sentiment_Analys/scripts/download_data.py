import pandas as pd
from pathlib import Path
import logging
from kaggle.api.kaggle_api_extended import KaggleApi

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Set up directories
        base_dir = Path(__file__).resolve().parents[1]
        raw_dir = base_dir / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Attempt Kaggle API download
        data_path = raw_dir / "sentiment140_sample.csv"
        try:
            logger.info("Authenticating Kaggle API...")
            api = KaggleApi()
            api.authenticate()
            logger.info("Downloading Sentiment140 dataset...")
            api.dataset_download_files("kazanova/sentiment140", path=raw_dir, unzip=True)
            logger.info(f"Dataset downloaded to {raw_dir}")
        except Exception as e:
            logger.warning(f"Kaggle API download failed: {e}")
            logger.info("Please ensure Kaggle API credentials are set up in ~/.kaggle/kaggle.json")
            
            # Synthetic dataset fallback
            if not data_path.exists():
                logger.info("Generating synthetic dataset...")
                texts = [
                    "I love this product!", "This is terrible.", "Not bad at all.",
                    "Worst experience ever.", "Amazing support team!", "Pretty average service."
                ]
                labels = [1, 0, 1, 0, 1, 0]  # 1: positive, 0: negative (no neutral for Sentiment140)
                pd.DataFrame({"text": texts, "label": labels}).to_csv(data_path, index=False)
                logger.info(f"Synthetic dataset saved to {data_path}")
    
    except Exception as e:
        logger.error(f"Error in dataset preparation: {e}")
        raise

if __name__ == "__main__":
    main()