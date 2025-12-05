#  MovieLens Hybrid Recommendation System

##  Project Overview
This project explores **recommender systems** using the MovieLens dataset. It implements three core paradigms, Collaborative Filtering (CF), Matrix Factorization (SVD), and Content-Based Filtering (CB), and integrates them into a **hybrid ensemble**. Each method is grounded in established theory from information retrieval, machine learning, and recommender system design.

This project implements a **hybrid movie recommendation engine** that integrates:
- **Collaborative Filtering (CF)** — user-based similarity.
- **Matrix Factorization (SVD)** — latent factor decomposition.
- **Content-Based Filtering (CB)** — genome and genre similarity.

The pipeline is designed for scalability, interpretability, and reproducibility, with detailed exploratory data analysis (EDA), model evaluation, and ensemble integration.

---

## Dataset Summary

**Source:** [MovieLens 20M (GroupLens)](https://grouplens.org/datasets/movielens/20m/)

**Files Used:**  
- `ratings.csv`: 20,000,263 ratings (userId, movieId, rating, timestamp)  
- `tags.csv`: 465,564 user-applied tags  
- `movies.csv`: 27,278 movie metadata records (title, genres)  
- `links.csv`: 27,278 links to IMDb and TMDb  
- `genome_tags.csv`: 1,128 tag descriptions  
- `genome_scores.csv`: 14,862,528 movie-tag relevance scores  

**Preprocessing Steps:**  
- Sampled top 50,000 active users (~5.2M interactions)  
- Temporal split: 80% training, 20% test  
- Merged datasets for feature engineering  
- Converted timestamps to datetime  
- Handled missing values with imputation  

**Key Statistics:**  
- **Total Ratings:** 13,344,487  
- **Unique Users:** 49,547  
- **Unique Movies:** 11,624  
- **Avg Ratings/User:** 269.3  
- **Avg Ratings/Movie:** 1148.0  
- **Rating Range:** 0.5–5.0  
- **Sparsity:** 97.68%  
- **Period:** 1995–2009 

- **User-Item Matrix:** At the heart of recommender systems lies the sparse matrix of users × items. Each entry represents a rating or implicit interaction. Sparsity is a fundamental challenge, as most users interact with only a small fraction of items.
- **Long-Tail Distribution:** User activity and movie popularity follow a power-law distribution. A few users and movies dominate interactions, while the majority remain sparse. This motivates hybrid approaches to balance popular and niche recommendations.
- **Temporal Dynamics:** Ratings evolve over time, reflecting trends, cultural shifts, and user behavior changes. Time-aware models can capture these dynamics.

  

---

## Exploratory Data Analysis (EDA)

### Rating Behavior
Ratings often exhibit positivity bias—users are more likely to rate items they enjoy. This skews distributions toward higher values, complicating prediction tasks. Normalization (e.g., z-scores) helps mitigate user bias.
- Ratings skewed toward positive values (peaks at 4.0 and 5.0).
- User average ratings cluster around 3.5.

### Temporal Trends
Temporal analysis reveals stability or drift in user behavior. Stable averages suggest consistent rating habits, while growth in rating volume reflects platform adoption. Time-aware recommenders (e.g., decay functions) can exploit these patterns.
- Ratings volume grows steadily from 1996–2008.
- Average rating remains stable (~3.5).

### User Activity
User engagement follows a long-tail distribution. Active users provide rich signals, while sparse users pose cold-start challenges. Hybrid models and content-based features help address this imbalance.
- Long-tail distribution: most users rate few movies, while a minority rate thousands.

### Movie Popularity
Popularity bias occurs when algorithms over-recommend widely rated items. While popularity correlates weakly with quality, Bayesian smoothing stabilizes estimates for low-volume items, preventing unfair penalization.
- Popularity vs quality correlation is weak (Corr ≈ 0.216).
- Long-tail distribution: few movies dominate ratings.

### Genre Preferences
Genre-level analysis supports **content-based filtering** and explainability. Semantic features (genres, tags) provide interpretable signals that complement latent factors.
- Top-rated genres: Musical, Western, Animation, IMAX, Mystery.

---

## Models

### 1. Collaborative Filtering (User-Based)
- CF assumes that users with similar past behaviors will have similar future preferences.
- **User-Based CF:** Finds nearest neighbors in the user space using similarity measures (cosine, Pearson).
- **Weighted Aggregation:** Ratings from similar users are combined, weighted by similarity.
- **Strengths:** Simple, interpretable, effective for dense users.
- **Weaknesses:** Struggles with sparsity and cold-start users.
- Uses similarity matrix over normalized ratings.
- Weighted aggregation from top-K similar users.
- Example recommendations for user 152:
  - *Requiem for a Dream (2000)*  
  - *Amores Perros (2000)*  
  - *Postman, The (1994)*  

### 2. Matrix Factorization (Truncated SVD)
- **Latent Factor Models:** Decompose the user-item matrix into lower-dimensional representations.
- **SVD:** Factorizes the matrix into orthogonal components, capturing hidden structures (e.g., genre affinities, popularity).
- **Explained Variance:** Measures how much of the original rating variance is captured by latent dimensions.
- **Strengths:** Handles sparsity, captures complex patterns, scalable.
- **Weaknesses:** Sensitive to hyperparameters, may underfit if variance explained is low.
- Latent dimension: 50 components.
- Explained variance ratio: 0.3554 (~35.5%).
- User factors: (49,547 × 50), Movie factors: (11,624 × 50).
- Prediction performance:
  - RMSE: 3.048  
  - MAE: 2.833  

### 3. Content-Based Filtering
- CB relies on item metadata (genres, tags, genome scores).
- **Cosine Similarity:** Measures semantic closeness between movies based on feature vectors.
- **User Profile Construction:** Aggregates features of items a user liked to recommend similar items.
- **Strengths:** Cold-start resilience, explainability, diversity.
- **Weaknesses:** Limited by metadata quality, may over-specialize (recommend too-similar items).
- Genome + genre similarity.
- Example recommendations for user 22528:
  - *Babe, The (1992)* (similarity: 0.754)  
  - *Silverado (1985)* (similarity: 0.745)  
  - *Still Crazy (1998)* (similarity: 0.734)  

### 4. Hybrid Ensemble
- Hybrid models combine multiple paradigms to balance strengths and weaknesses.
- **Weighted Ensemble:** Scores from CF, SVD, and CB are linearly combined with tuned weights.
- **Rationale:**  
  - CF captures peer influence.  
  - SVD uncovers latent structures.  
  - CB ensures semantic alignment.  
- **Strengths:** Robustness, diversity, improved coverage.
- **Weaknesses:** Requires careful weight tuning, may dilute strong signals.
- Weighted combination: CF (0.4), SVD (0.35), CB (0.25).
- Example recommendations for user 22528:
  - *Fugitive, The (1993)* (score: 1.102)  
  - *Silence of the Lambs, The (1991)* (score: 1.057)  
  - *Babe, The (1992)* (score: 0.188)  

---

## Evaluation

- **Precision@K:** Fraction of recommended items in top-K that are relevant. Measures accuracy of recommendations.
- **Recall@K:** Fraction of relevant items retrieved in top-K. Measures completeness.
- **Coverage:** Fraction of items/users for which recommendations are generated. Measures diversity and system reach.
- **Consistency Check:** Compares predicted top-N items with actual top-N ratings. Reflects alignment with user preferences.

### Recommendation Metrics (Top 100 Users)
| Model                  | Precision@10 | Recall@10 | Coverage |
|------------------------|--------------|-----------|----------|
| Collaborative Filtering| 0.014        | 0.002     | 0.001    |
| Content-Based          | 0.000        | 0.000     | 0.000    |
| Hybrid Model           | 0.000        | 0.000     | 0.000    |

### Consistency Check (Top 50 Users)
- **Avg Match Rate:** 3.47%  
- **Avg Predicted Rating:** 3.82 / 5  
- **Users Evaluated:** 48  

- Low precision and recall indicate difficulty in predicting explicit ratings.
- Coverage near zero suggests models failed to generalize across users.
- Consistency check shows weak overlap between predicted and actual favorites, highlighting model misalignment.

---

## Conclusion
- **Collaborative Filtering** provides limited but non-zero precision.  
- **SVD** captures latent structure but suffers from high error.  
- **Content-Based** and **Hybrid models** underperform due to sparse tag coverage and weak ensemble integration.  
- **Future Work:**  
  - Tune ensemble weights.  
  - Improve feature engineering (temporal decay, Bayesian smoothing).  
  - Evaluate with ranking metrics (NDCG, MAP).  
  - Expand to MovieLens 25M and implicit feedback.

---
## Project Structure
---
This project is organized into clear modules for **data management**, **scripts**, and **analysis notebooks**, ensuring reproducibility and scalability.

---

#### Root Directory

DAY_8_MOVIE_RECOMMENDATION_SYSTEM/  
 ├── data/   
 │ ├── raw/   
 │ └── processed/   
 ├── scripts/   
 ├── Day8_MovieRecommendation.ipynb   
 ├── README.md  
 └── requirements.txt    

#### `data/raw/` — Original MovieLens Files

data/raw/  
├── genome_scores.csv → Relevance scores for genome tags    
├── genome_tags.csv → Tag vocabulary    
├── links.csv → External IDs (IMDb, TMDb)    
├── movielens-20m-dataset.zip → Original compressed archive    
├── movies.csv → Movie metadata (title, genres)    
├── ratings.csv→ Core user–movie rating data    
└── tags.csv → User-generated tags  

#### `data/processed/` — Preprocessed Artifacts
data/processed/  
├── genome_matrix.npz  
├── genome_tag_index.csv  
├── genome_tags.csv  
├── genres_multi_hot.npz  
├── interactions.npz  
├── movie_id_map.csv  
├── movie_stats.csv  
├── movies_clean.csv  
├── preprocessing_info.txt  
├── test.csv  
├── tfidf_tags.npz  
├── tfidf_vocab.csv  
├── train.csv  
├── user_id_map.csv  
└── user_stats.csv

- **.npz files** → Sparse matrices for genome, genre, TF-IDF features  
- **.csv files** → Cleaned metadata, mappings, and statistics  
- **train.csv / test.csv** → Final rating splits for modeling  
- **preprocessing_info.txt** → Notes on feature engineering steps 
---

## Skills Demonstrated

- Collaborative Filtering (KNN-based)  
- Matrix Factorization (SVD/ALS)  
- Content-Based Filtering  
- Hybrid Ensemble Methods  
- Cold-Start Problem Handling  
- Large-scale dataset processing  
- Feature engineering (user profiles, temporal features)  
- Evaluation metrics (RMSE, MAE, Precision@K, Recall@K, Coverage)  
- Visualization & storytelling

---
## Implementation Highlights

- Modular, memory-safe design (optimized for 8 GB RAM)  
- Clear logging (`[INFO]`, `[WARN]`) for transparency  
- Reviewer-friendly structure with clean variable naming  
- Sparse matrix operations for scalability  

---

## Future Work

- Genre-level consistency diagnostics  
- Cold-start strategies for new users/movies  
- Real-time API deployment (FastAPI/Redis)  
- Advanced models: Neural CF, sequential recommenders (GRU4Rec, SASRec)  
- Monitoring diversity and fairness in recommendations  

---

## References

- Koren, Y., Bell, R., & Volinsky, C. (2009). *Matrix Factorization Techniques for Recommender Systems*  
- Ricci, F., et al. (2011). *Recommender Systems Handbook*  
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)  

---

## Author

**Prasann Barot**  
Lead Data Scientist & AI Consultant  
Day 8 of #100DaysOfML  
Toronto, Canada  