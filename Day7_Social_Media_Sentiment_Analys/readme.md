# Day 7 — Social Media Sentiment Analysis

## Overview

This project performs sentiment analysis on a large corpus of tweets using both **Logistic Regression** and **Random Forest** classifiers. It includes detailed **exploratory data analysis (EDA)**, **feature engineering**, **model evaluation**, and **interpretability** using SHAP values. The goal is to classify tweets as **Positive** or **Negative**, understand the linguistic and structural patterns behind sentiment, and evaluate model performance with transparency.

---

## Project Structure


Day7_SentimentAnalysis/ 
├── data/ 
│    ├── raw/               # Raw tweet datasets 
│    └── processed/         # Cleaned tweet datasets 
├── scripts/ 
│    ├── download_data.py   # Script to fetch raw data 
│    └── preprocess.py      # Text cleaning and feature extraction 
├── artifacts/              # Saved models, metrics, and visualizations 
├── Day7_SentimentAnalysis.ipynb  # Main notebook 
└── README.md              # Full project documentation

---

## Dataset Summary

- **Total Samples**: ~1.27 million Sentiment140 (20K subset) — labeled tweets for sentiment analysis.
- **Sentiment Classes**:  
  - `0 = Negative` (637,120 samples)  
  - `1 = Positive` (637,004 samples)  
  - `2 = Neutral` (not used in modeling)

- **Balance**: Near-perfect balance between Negative and Positive classes

---

## Exploratory Data Analysis (EDA)

### Sentiment Distribution
- Balanced dataset supports fair binary classification.

### Tweet Length vs Sentiment
- Short tweets skew Positive  
- Long tweets skew Negative  
- Medium tweets are balanced
- Stopword and URL cleaning crucial for model clarity.

### Text Length Analysis
- Negative tweets have higher character count and more variability  
- Word count is similar across classes

### TextBlob Polarity vs True Sentiment
- TextBlob tends to assign slightly positive scores (~0.1) to most tweets  
- Weak correlation with true sentiment labels

### Feature Correlation
| Feature Pair         | Correlation |
|----------------------|-------------|
| char_len & word_count| 0.96        |
| char_len & polarity  | 0.04        |
| word_count & polarity| 0.05        |

- Length features are strongly correlated  
- Polarity is independent of length

---

## Modeling

### Logistic Regression

- **Accuracy**: 63%  
- **Precision/Recall**:
  - Negative: Precision 0.58, Recall 0.96  
  - Positive: Precision 0.88, Recall 0.30  
- **Bias**: Strong bias toward Negative predictions  
- **SHAP Features**:
  - Top: `word_count`, `polarity`, `char_len`, `sad`, `thanks`, `miss`, `love`, `cant`, `hurts`, `sorry`

### Random Forest

- **Accuracy**: 76%  
- **Precision/Recall**:
  - Negative: Precision 0.76, Recall 0.76  
  - Positive: Precision 0.76, Recall 0.76  
- **Balanced performance** across both classes  
- **Improved recall for Positive sentiment**

---

## Sample Predictions

### Logistic Regression
- Correct: 46.67%  
- Positive Recall: 11.1%  
- Bias toward Negative predictions

### Random Forest
- Correct: 80.00%  
- Positive Recall: 66.7%  
- More context-aware and balanced

---

## Key Takeaways

- **Random Forest outperforms Logistic Regression** in both accuracy and recall.
- **Text length and polarity** are strong predictors of sentiment.
- **Keyword presence** (e.g., "sad", "love", "sorry") adds interpretability.
- **SHAP analysis** confirms feature importance and model transparency.
- **TextBlob polarity** is weakly aligned with true sentiment—use with caution.

---

## Future Work

- Integrate transformer-based models (e.g., BERT, RoBERTa)
- Expand multilingual and slang handling
- Add emoji and hashtag sentiment features
- Deploy model via API or dashboard

---

## Author

**Prasann Dineshbhai Barot**  
Data Science & AI Consultant | Business Analyst | Educator  
Specialties: KPI design, analytics integration, behavioral support, and technical mentoring