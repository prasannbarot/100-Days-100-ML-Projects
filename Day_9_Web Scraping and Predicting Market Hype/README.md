#  Day 9: Quantitative Sentiment Analysis & Market Prediction
## *Harnessing Transformer-Based NLP for Retail Volatility Forecasting*

###  The Core Idea
Traditional market models rely on **Efficient Market Hypothesis (EMH)**, assuming all known information is reflected in prices. However, modern "Meme Stock" cycles (TSLA, GME, AMC) prove that **Retail Sentiment**—the collective psychological state of social media communities—acts as a massive driver of short-term volatility.

This project investigates the **Sentiment-to-Price Transmission Mechanism**. By quantifying the "mood" of `r/WallStreetBets`, we attempt to model the psychological momentum that precedes price action. We treat social media not as "noise," but as a leading indicator of liquidity shifts.



---

###  Project Identity
| Attribute | Specification |
| :--- | :--- |
| **Project ID** | `DAY-09` |
| **Domain** | Behavioral Finance / Quantitative Trading |
| **ML Task** | Time-Series Regression / Sequence Classification |
| **Data Velocity** | Real-time Web Scraping (Last 28 Days) |
| **Optimization** | Built for Intel i5-1035G1 (10th Gen) & 8GB RAM |

---

###  Strategic Objectives
1. **High-Fidelity NLP:** Move beyond simple word-counts to **Contextual Embeddings** using `FinBERT`, a BERT model pre-trained specifically on financial corpora (Reuters, TRC2-financial) to understand nuances like "Long," "Short," "Bullish," and "Crash."
2. **Predictive Feature Engineering:** Solve the "simultaneity bias" by implementing **Time-Lagged Features**. Instead of saying "Sentiment is high today and Price is high today," we ask: *"Does high sentiment at T-1 predict a positive return at T+0?"*S
3. **Hardware-Resilient Pipeline:** Implement a **Dual-Engine Failover Strategy**. If the system encounters memory constraints or network timeouts during the FinBERT download, it gracefully degrades to a local **VADER** heuristic engine to ensure 100% uptime.
4. **Actionable Intelligence:** Bridge the gap between "Black Box" machine learning and human decision-making via a **Streamlit Operational Dashboard**.

---

###  The "ML Innovation" for Day 9
Unlike standard sentiment projects that just plot a graph, Day 9 introduces a **Multivariate Random Forest Strategy**. We don't just look at the sentiment score; we analyze:
- **Sentiment Momentum:** Is the community getting happier or sadder *faster*?
- **Discussion Density:** Does a high sentiment score carry more weight if 1,000 people are talking vs. 10?


##  Deep Dive: The NLP Semantic Analysis Engine

The core challenge of Day 9 was transforming chaotic, slang-heavy Reddit comments into structured, statistically significant numerical data. To solve this, the pipeline employs a dual-strategy semantic analysis architecture.



### 1. Primary Engine: FinBERT-Tone (Transformer)
Standard sentiment models (trained on movie reviews or tweets) often fail in financial contexts. For example, the phrase *"The stock hit the floor"* might be seen as neutral by a general model, but **FinBERT** recognizes it as a high-velocity bearish signal.
- **Architecture:** Based on the BERT (Bidirectional Encoder Representations from Transformers) architecture, specifically fine-tuned on the `Financial PhraseBank` dataset.
- **Contextual Awareness:** FinBERT uses "Self-Attention" mechanisms to understand word relationships. It recognizes that "Long" in *"I'm going long on TSLA"* is a bullish position, whereas in *"It's been a long day,"* it is merely a temporal descriptor.

### 2. The Failover Strategy: Heuristic VADER
To ensure the pipeline remains "Production-Ready" on systems with limited VRAM (like an 8GB RAM laptop), I implemented a **Heuristic Failover Mechanism**.
- **The Logic:** If the system detects a resource bottleneck or a timeout while loading the 400MB+ Transformer weights, it automatically initializes the **VADER (Valence Aware Dictionary and sEntiment Reasoner)** engine.
- **Why VADER?** It is optimized for social media. It understands emojis (🚀, 💎, 🙌), capitalization ("HUGE GAINS"), and intensifiers ("very bullish"), providing a reliable, lightweight alternative to deep learning.



###  Data Preprocessing & Vectorization
Raw data undergoes a rigorous 3-step transformation before reaching the ML model:

| Stage | Process | Technical Detail |
| :--- | :--- | :--- |
| **Cleaning** | Noise Reduction | Removal of URLs, bot-generated disclaimers, and non-ASCII artifacts. |
| **Scoring** | Normalized Mapping | Every comment is mapped to a continuous range $[-1.0, +1.0]$, where $-1.0$ is Extreme Panic and $+1.0$ is Extreme Euphoria. |
| **Aggregation** | Resampling | Individual comment scores are resampled into 24-hour windows using a volume-weighted average to produce the `sentiment_mean`. |

---

###  Feature Engineering (The "Alpha" Generation)
To provide the Random Forest with predictive power, we don't just pass raw scores. We engineer **Semantic Derivatives**:

1. **Sentiment Momentum ($\Delta S$):** Calculated as $S_t - S_{t-1}$. A rapid shift from neutral to positive is often more predictive of a "breakout" than a sustained high score.
2. **Lagged Sentiment ($S_{t-1}$):** This addresses the **Information Diffusion** theory—the idea that it takes time for social media hype to translate into actual brokerage orders and price moves.
3. **Engagement Density:** Multiplying sentiment by `log(comment_volume)` to ensure that a sentiment score derived from 2 comments carries less weight than one derived from 2,000.


##  Machine Learning Pipeline & Predictive Methodology

The predictive core of this project transitions from **Descriptive Analytics** (what happened) to **Predictive Modeling** (what will happen). The objective is to map non-linear sentiment features to financial returns.

### 1. The Algorithm: Random Forest Regressor
While deep learning (LSTM/GRU) is popular for time-series, I selected the **Random Forest Regressor** for this project due to:
- **Feature Importance:** It allows us to mathematically rank which features (e.g., Sentiment Momentum vs. Raw Volume) actually drive price changes.
- **Non-Linearity:** Stock markets rarely move in straight lines. Random Forest's ensemble of decision trees captures complex, "step-function" relationships that Linear Regression misses.
- **Robustness:** It is significantly less prone to overfitting on the relatively small datasets (daily aggregations) common in 7-day sentiment windows.



### 2. Multivariate Feature Engineering (The 4-D Vector)
To provide the model with a holistic view of "Community Hype," we construct a 4-dimensional feature vector for every time step $T$:

| Feature | Mathematical Representation | Financial Significance |
| :--- | :--- | :--- |
| **Daily Mean** | $\mu = \frac{1}{n}\sum S_i$ | The aggregate "mood" of the community. |
| **Sentiment Lag** | $S_{t-1}$ | Captured "Information Diffusion"—the delay between a post and a trade. |
| **Momentum** | $\frac{dS}{dt} \approx S_t - S_{t-1}$ | Identifies "Spiking Hype" vs. stagnant sentiment. |
| **Log Volume** | $\log(V)$ | Normalizes massive spikes in discussion (e.g., during an earnings call). |

### 3. Training & Validation Strategy
- **Time-Series Split:** Traditional K-Fold cross-validation is avoided to prevent **Data Leakage**. The model is trained on a strictly chronological basis—ensuring it never "sees the future" during training.
- **Scaling:** A `StandardScaler` is fitted on the training set and persisted in `artifacts/`. This ensures that features with different units (Sentiment $-1$ to $1$ vs. Volume in the thousands) are weighted equally by the model.
- **Target Variable:** The model predicts the **Next-Day Percentage Return** ($R_{t+1}$), calculated as:
$$R_{t+1} = \frac{Price_{t+1} - Price_t}{Price_t}$$



### 4. Model Persistence (The Production Mindset)
In professional ML Engineering, we don't just "run" a model; we "deploy" it.
- **Serialization:** Using `joblib`, the trained model weights and the scaler state are serialized into binary files. 
- **Inference Ready:** This allows the `web_app/` and `alerts.py` scripts to perform real-time predictions without needing to re-fetch the entire training history, significantly reducing latency.


##  System Architecture & Data Engineering Pipeline

The system is architected as a decoupled 4-stage pipeline. Each module is independent, ensuring that a failure in the scraping layer does not corrupt the model training artifacts.


### Project Structure
```text
.
├── artifacts/               # Serialized ML models & scalers
│   ├── scaler.pkl
│   └── trend_predictor.pkl
├── data/
│   ├── raw/                 # Unprocessed data from scraper
│   └── processed/           # Cleaned and structured data
├── scripts/
│   ├── scraper.py           # Real-time product price scraper
│   ├── data_processor.py    # Data cleaning and transformation
│   ├── ml_predictor.py      # ML model inference and trend prediction
│   └── alerts.py            # Alert system for price changes
├── web_app/
│   └── app.py               # Web dashboard (e.g., Streamlit or Flask)
├── Day9_StockSentiment.ipynb  # To run all the files
├── README.md                # Project documentation
├── processor.log            # Logs from data processing
└── scraper.log              # Logs from scraping operations

```


### 1. Data Ingestion Layer (`scraper.py`)
**Task:** High-volume extraction of unstructured social text.
- **Source:** Reddit `r/WallStreetBets` (Daily Discussion & "What Are Your Moves" threads).
- **Mechanism:** RESTful API polling via `requests` with custom User-Agent headers to bypass rate-limiting.
- **Challenges Overcome:** Managed JSON deep-nesting in Reddit's API response and implemented recursive comment fetching to capture community consensus.
- **Output:** `data/raw/scraped_posts.csv` (Timestamped raw text).

### 2. Semantic Analysis Engine (`data_processor.py`)
**Task:** Transforming unstructured prose into quantitative sentiment vectors.
- **Primary Engine:** `FinBERT-Tone` (Transformer-based). Unlike general sentiment models, FinBERT understands that the word *"CRASH"* in a financial context is a negative trend, not just a physical accident.
- **Resilience Strategy:** Implemented a **Heuristic Failover Mechanism**. If the system detects a network timeout or memory overflow (crucial for 8GB RAM environments), it automatically downgrades to **VADER** (Valence Aware Dictionary and sEntiment Reasoner).
- **Feature Aggregation:** Raw comment scores are grouped by 24-hour windows to calculate `sentiment_mean` and `comment_volume`.
- **Output:** `data/processed/sentiment_trends.csv`.



### 3. Machine Learning Core (`ml_predictor.py`)
**Task:** Supervised Learning for Time-Series Forecasting.
- **Algorithm:** `Random Forest Regressor`. Chosen for its ability to handle non-linear relationships and its inherent resistance to overfitting compared to deep neural networks on smaller datasets.
- **Feature Engineering (The "Alpha"):**
    - **Lags:** Shifting sentiment by $T-1$ to ensure the model predicts *future* returns based on *past* hype.
    - **Momentum:** Calculating the derivative of sentiment to find "Spiking Hype."
- **Persistence:** Models and `StandardScaler` objects are serialized via `joblib` into the `artifacts/` directory for production deployment.


##  Operational Layer: The Analyst Dashboard & Real-Time Alerts

A machine learning model is only as valuable as the decisions it enables. For Day 9, I engineered a high-performance **Streamlit Dashboard** and a **CLI-based Alert System** to translate abstract sentiment vectors into actionable market signals.



### 1. The WSB Alpha Dashboard (`web_app/app.py`)
Developed as a centralized command center, the dashboard provides a "Human-in-the-Loop" interface for sentiment-driven trading analysis:
- **Real-Time Sentiment Tracking:** A dynamic visualization layer using `Plotly` that maps the aggregate community mood over a rolling 7-day window.
- **Market Correlation Engine:** Integrated with the `yfinance` API to overlay Reddit sentiment peaks against actual price breakouts for tickers like $TSLA and $GME.
- **Prediction Sandbox:** An interactive "What-If" simulator. Users can adjust sentiment parameters via sliders to observe how the **Random Forest Regressor** shifts its predicted next-day return, providing transparency into the model's logic.

### 2. High-Frequency Signal Generator (`scripts/alerts.py`)
For low-latency environments, the `alerts.py` module serves as a headless inference engine that categorizes model outputs into risk-adjusted trade signals:
-  **Strong Buy:** Predicted return $> 2\%$ with high-momentum sentiment.
-  **Weak Bullish:** Positive sentiment forecast but lower volume confidence.
-  **Bearish:** Negative sentiment drift indicating potential retail exit.
-  **Strong Sell:** High-volume panic or sentiment "crash" ($< -2\%$).


##  Future Research & Iterations (The Roadmap)

To evolve this Day 9 proof-of-concept into a production-grade quantitative tool, the following enhancements are prioritized:
1. **Slang-Aware Tokenization:** Developing custom regex layers to handle Reddit-specific vernacular (e.g., "Diamond Hands," "Tendies," "Mooning") which traditional NLP models often misclassify.
2. **Entity Recognition (NER):** Implementing `spaCy` to automatically detect new trending tickers that aren't currently in the watchlist.
3. **Sentiment Skew Analysis:** Moving beyond the "Mean Score" to analyze the **Sentiment Distribution** (Skewness/Kurtosis) to identify if a community is truly unified or deeply polarized.

