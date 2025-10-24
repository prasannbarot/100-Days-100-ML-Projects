import pandas as pd
from pathlib import Path
import re
import logging
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('preprocessing.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Download NLTK stopwords
try:
    nltk.download('stopwords', quiet=True)
    STOPWORDS = set(stopwords.words('english'))
except Exception as e:
    logger.error(f"Error downloading NLTK stopwords: {e}")
    STOPWORDS = set()

def clean_text(text):
    """Clean text for NLP tasks."""
    try:
        if not isinstance(text, str) or not text.strip():
            return None
        # Convert to lowercase
        text = text.lower()
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|\#\w+', '', text)
        # Remove emojis and special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Remove stopwords and extra whitespace
        text = ' '.join(word for word in text.split() if word not in STOPWORDS)
        return text if text.strip() else None
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return None

def create_synthetic_dataset(raw_path):
    """Create a synthetic dataset if the main dataset fails to load."""
    try:
        logger.info("Generating synthetic dataset as fallback...")
        texts = [
            "I love this product!", "This is terrible.", "Not bad at all.",
            "Worst experience ever.", "Amazing support team!", "Pretty average service."
        ]
        labels = [1, 0, 1, 0, 1, 0]  # 1: positive, 0: negative
        df = pd.DataFrame({"text": texts, "label": labels})
        df.to_csv(raw_path, index=False)
        logger.info(f"Synthetic dataset saved to {raw_path}")
        return df
    except Exception as e:
        logger.error(f"Error creating synthetic dataset: {e}")
        raise

def main():
    try:
        # Set up directories
        base_dir = Path(__file__).resolve().parents[1]
        raw_path = base_dir / "data" / "raw" / "sentiment140_sample.csv"
        processed_dir = base_dir / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        logger.info(f"Loading dataset from {raw_path}...")
        if not raw_path.exists():
            logger.warning(f"Dataset not found at {raw_path}. Creating synthetic dataset...")
            df = create_synthetic_dataset(raw_path)
        else:
            try:
                df = pd.read_csv(raw_path, encoding='latin-1')
            except UnicodeDecodeError:
                logger.warning("Failed to read dataset with latin-1 encoding. Trying synthetic dataset...")
                df = create_synthetic_dataset(raw_path)
        
        # Validate dataset
        expected_columns = {'text', 'label'}  # Adjust based on actual dataset
        if not expected_columns.issubset(df.columns):
            logger.warning(f"Dataset missing required columns: {expected_columns}. Assuming Sentiment140 structure...")
            df = pd.read_csv(
                raw_path,
                encoding='latin-1',
                names=['target', 'id', 'date', 'flag', 'user', 'text']
            )
            df['label'] = df['target']
            df = df[['text', 'label']]
        
        # Map labels: 0 (negative) -> 0, 4 (positive) -> 1, 2 (neutral) -> 0
        logger.info("Mapping labels...")
        df['label'] = df['label'].map({0: 0, 4: 1, 2: 0, '0': 0, '4': 1, '2': 0})
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)
        
        # Clean text
        logger.info("Cleaning text...")
        df['clean_text'] = df['text'].apply(clean_text)
        df = df.dropna(subset=['clean_text'])
        df = df[df['clean_text'].str.strip() != '']
        
        # Check dataset size
        if len(df) < 10:  # Minimum size for splitting
            logger.warning(f"Dataset too small ({len(df)} rows). Creating synthetic dataset...")
            df = create_synthetic_dataset(raw_path)
        
        # Train-test split (stratified)
        logger.info("Performing train-test split...")
        train_df, test_df = train_test_split(
            df[['clean_text', 'label']],
            test_size=0.2,
            random_state=42,
            stratify=df['label']
        )
        
        # Save processed datasets
        train_path = processed_dir / "train.csv"
        test_path = processed_dir / "test.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        logger.info(f"Preprocessing complete. Saved to {train_path} and {test_path}")
        logger.info(f"Train set: {len(train_df)} rows, Test set: {len(test_df)} rows")
        logger.info(f"Train label distribution:\n{train_df['label'].value_counts()}")
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise

if __name__ == "__main__":
    main()