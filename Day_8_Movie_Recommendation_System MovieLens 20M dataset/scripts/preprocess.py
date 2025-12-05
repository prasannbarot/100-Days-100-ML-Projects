"""
Preprocess MovieLens 20M: clean, engineer features, build sparse matrices and content features.
This script keeps a straightforward, top-to-bottom flow (minimal functions),
but produces high-quality artifacts that are reusable for modeling and evaluation.

Outputs (data/processed):
- train.csv, test.csv  → temporally split rating data (cleaned)
- id_maps.csv          → contiguous user/movie index mapping for matrix ops
- user_stats.csv       → per-user statistics (activity, avg, std)
- movie_stats.csv      → per-movie statistics (+ Bayesian average)
- genres_multi_hot.npz → sparse multi-hot matrix [n_movies x n_genres]
- interactions.npz     → sparse user-movie CSR matrix (train only, float)
- tfidf_tags.npz       → sparse TF-IDF matrix [n_movies x vocab]
- genome_matrix.npz    → sparse movie x genome_tag relevance
- genome_tag_index.csv → index mapping for genome tags
- preprocessing_info.txt → summary of decisions and shapes

Notes:
- A small amount of pragmatic filtering is applied to reduce cold-start and memory overhead.
- Adjust thresholds below depending on machine and project goals.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

# Paths and config knobs
base = Path(__file__).resolve().parents[1]
raw_dir = base / "data" / "raw"
proc_dir = base / "data" / "processed"
proc_dir.mkdir(parents=True, exist_ok=True)

# Pragmatic limits to keep memory under control on 8GB RAM
# TODO: tune these based on your machine and target runtime.
MIN_USER_INTERACTIONS = 10         # filter out users with too few ratings
MIN_MOVIE_INTERACTIONS = 10        # filter out very cold movies
TOP_USERS_LIMIT = 60000             # cap active users to focus on dense subgraph
TOP_MOVIES_LIMIT = 25000            # cap movies to those with most interactions
TFIDF_MAX_FEATURES = 25000          # tag vocabulary cap for TF-IDF
RANDOM_SEED = 42

print("[INFO] Loading raw data with memory-friendly dtypes...")

# Explicit dtypes to reduce memory footprint
ratings = pd.read_csv(
    raw_dir / "ratings.csv",
    dtype={"userId": "int32", "movieId": "int32", "rating": "float32"}
)
ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], errors="coerce")
movies = pd.read_csv(raw_dir / "movies.csv", dtype={"movieId": "int32", "title": "string", "genres": "string"})
tags = pd.read_csv(
    raw_dir / "tags.csv",
    dtype={"userId": "int32", "movieId": "int32", "tag": "string"}
)
tags["timestamp"] = pd.to_datetime(tags["timestamp"], errors="coerce")
links = pd.read_csv(raw_dir / "links.csv", dtype={"movieId": "int32", "imdbId": "Int64", "tmdbId": "Int64"})
genome_tags = pd.read_csv(raw_dir / "genome_tags.csv", dtype={"tagId": "int32", "tag": "string"})
genome_scores = pd.read_csv(raw_dir / "genome_scores.csv", dtype={"movieId": "int32", "tagId": "int32", "relevance": "float32"})

print(f"[INFO] Ratings: {ratings.shape}, Movies: {movies.shape}, Tags: {tags.shape}, Genome scores: {genome_scores.shape}")

# Basic cleaning
print("[INFO] Basic cleaning: drop duplicates, coerce genres, strip tags.")
ratings = ratings.drop_duplicates(subset=["userId", "movieId", "timestamp"])
movies["genres"] = movies["genres"].fillna("Unknown")
tags["tag"] = tags["tag"].fillna("").str.lower().str.strip()

# Cold-start and activity filtering
print("[INFO] Filtering cold users/movies and focusing on dense interactions...")

user_activity = ratings.groupby("userId")["movieId"].count().astype("int32")
movie_activity = ratings.groupby("movieId")["userId"].count().astype("int32")

# keep sufficiently active users/movies
active_users = user_activity[user_activity >= MIN_USER_INTERACTIONS].index
active_movies = movie_activity[movie_activity >= MIN_MOVIE_INTERACTIONS].index

ratings = ratings[ratings["userId"].isin(active_users) & ratings["movieId"].isin(active_movies)]

# focus on top users/movies by activity
active_users_sorted = user_activity.loc[active_users].sort_values(ascending=False).head(TOP_USERS_LIMIT).index
active_movies_sorted = movie_activity.loc[active_movies].sort_values(ascending=False).head(TOP_MOVIES_LIMIT).index

ratings = ratings[ratings["userId"].isin(active_users_sorted) & ratings["movieId"].isin(active_movies_sorted)]

print(f"[INFO] After filtering: ratings={ratings.shape}, users={ratings['userId'].nunique()}, movies={ratings['movieId'].nunique()}")

# Merge movie metadata
print("[INFO] Merging movie metadata...")
ratings = ratings.merge(movies, on="movieId", how="left")

# Tag aggregation per movie (user-applied tags)
print("[INFO] Aggregating user tags per movie...")
movie_tags = (
    tags[tags["movieId"].isin(ratings["movieId"].unique())]
    .groupby("movieId")["tag"]
    .apply(lambda x: ", ".join(sorted(set(t for t in x if isinstance(t, str) and t.strip() != ""))))
    .reset_index()
    .rename(columns={"tag": "user_tags"})
)
ratings = ratings.merge(movie_tags, on="movieId", how="left")

# Genome relevance features
print("[INFO] Building genome summary stats per movie...")
genome_subset = genome_scores[genome_scores["movieId"].isin(ratings["movieId"].unique())]
avg_genome = genome_subset.groupby("movieId")["relevance"].agg(["mean", "max", "std"]).reset_index()
avg_genome.columns = ["movieId", "genome_mean", "genome_max", "genome_std"]
ratings = ratings.merge(avg_genome, on="movieId", how="left")

# Fill missing genome stats with global means (sane fallback)
for col in ["genome_mean", "genome_max", "genome_std"]:
    ratings[col] = ratings[col].astype("float32")
    ratings[col] = ratings[col].fillna(ratings[col].mean())

# Bayesian movie popularity prior
print("[INFO] Computing Bayesian movie rating prior (shrinkage).")
movie_stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
movie_stats.columns = ["movieId", "movie_avg_rating", "movie_n_ratings"]
global_mean = ratings["rating"].mean()
m = movie_stats["movie_n_ratings"].median()  # prior strength (median count)
movie_stats["movie_bayesian_avg"] = (
    (movie_stats["movie_n_ratings"] * movie_stats["movie_avg_rating"] + m * global_mean) /
    (movie_stats["movie_n_ratings"] + m)
).astype("float32")
ratings = ratings.merge(movie_stats, on="movieId", how="left")

# User statistics and temporal features
print("[INFO] Engineering user and temporal features...")
user_stats = ratings.groupby("userId")["rating"].agg(["mean", "std", "count"]).reset_index()
user_stats.columns = ["userId", "user_avg_rating", "user_rating_std", "user_n_ratings"]
ratings = ratings.merge(user_stats, on="userId", how="left")

# Timestamp features
ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")
ratings["rating_year"] = ratings["timestamp"].dt.year.astype("int16")
ratings["rating_month"] = ratings["timestamp"].dt.month.astype("int8")
ratings["rating_dow"] = ratings["timestamp"].dt.dayofweek.astype("int8")
ratings["is_weekend"] = ratings["rating_dow"].isin([5, 6]).astype("int8")

# Optional: within-user z-score (good for models that use normalized ratings)
# NOTE: Don't use normalized ratings for RMSE on original scale; keep both.
print("[INFO] Computing within-user normalized rating (z-score).")
ratings["user_rating_std"] = ratings["user_rating_std"].replace(0, np.nan)
ratings["rating_z"] = ((ratings["rating"] - ratings["user_avg_rating"]) / ratings["user_rating_std"]).astype("float32")
ratings["rating_z"] = ratings["rating_z"].fillna(0.0)

# Genre multi-hot (sparse)
print("[INFO] Building sparse multi-hot genre matrix...")
# Determine consistent genre set (exclude '(no genres listed)' if present)
genre_series = movies.loc[movies["movieId"].isin(ratings["movieId"].unique()), "genres"].str.split("|")
unique_genres = pd.Series(np.concatenate(genre_series.dropna().values)).unique()
unique_genres = [g for g in unique_genres if g and g != "(no genres listed)"]
unique_genres = sorted(unique_genres)

# Map movie index for genre matrix
movie_index = pd.Index(sorted(ratings["movieId"].unique()), name="movieId")
movie_id_to_pos = pd.Series(np.arange(len(movie_index), dtype="int32"), index=movie_index)

rows, cols, data = [], [], []
for g_idx, g in enumerate(unique_genres):
    mask = movies["genres"].str.contains(g, na=False) & movies["movieId"].isin(movie_index)
    active_movie_ids = movies.loc[mask, "movieId"].values
    rows.extend(movie_id_to_pos.loc[active_movie_ids].tolist())
    cols.extend([g_idx] * len(active_movie_ids))
    data.extend([1] * len(active_movie_ids))

genres_multi_hot = coo_matrix((data, (rows, cols)), shape=(len(movie_index), len(unique_genres))).tocsr()
save_npz(proc_dir / "genres_multi_hot.npz", genres_multi_hot)

# Contiguous ID mapping and sparse interactions
print("[INFO] Building contiguous ID maps and sparse user-movie interaction matrix (train only).")
# Sort by time for temporal split
ratings = ratings.sort_values("timestamp")
split_idx = int(0.8 * len(ratings))
train = ratings.iloc[:split_idx].copy()
test = ratings.iloc[split_idx:].copy()

# Contiguous indices for users and movies (based on train to avoid leakage)
user_ids = pd.Index(sorted(train["userId"].unique()), name="userId")
movie_ids = pd.Index(sorted(train["movieId"].unique()), name="movieId")
user_id_to_idx = pd.Series(np.arange(len(user_ids), dtype="int32"), index=user_ids)
movie_id_to_idx = pd.Series(np.arange(len(movie_ids), dtype="int32"), index=movie_ids)

# Map train rows
train["user_idx"] = user_id_to_idx.loc[train["userId"]].values
train["movie_idx"] = movie_id_to_idx.loc[train["movieId"]].values

# Build CSR matrix of ratings (could use rating or implicit signal like 1 for watched)
# TODO: consider binarizing as implicit for ranking models.
interactions = coo_matrix(
    (train["rating"].astype("float32"), (train["user_idx"].values, train["movie_idx"].values)),
    shape=(len(user_ids), len(movie_ids))
).tocsr()
save_npz(proc_dir / "interactions.npz", interactions)

# Save ID maps (for downstream models)
id_maps = pd.DataFrame({
    "userId": user_ids.values,
    "user_idx": np.arange(len(user_ids), dtype="int32"),
})
# Save separate maps instead of cross-join
user_map = pd.DataFrame({
    "userId": user_ids.values,
    "user_idx": np.arange(len(user_ids), dtype="int32")
})
movie_map = pd.DataFrame({
    "movieId": movie_ids.values,
    "movie_idx": np.arange(len(movie_ids), dtype="int32")
})

user_map.to_csv(proc_dir / "user_id_map.csv", index=False)
movie_map.to_csv(proc_dir / "movie_id_map.csv", index=False)

# Correction: save separate maps to avoid huge files
pd.DataFrame({"userId": user_ids.values, "user_idx": np.arange(len(user_ids), dtype="int32")}).to_csv(proc_dir / "user_id_map.csv", index=False)
pd.DataFrame({"movieId": movie_ids.values, "movie_idx": np.arange(len(movie_ids), dtype="int32")}).to_csv(proc_dir / "movie_id_map.csv", index=False)

# TF-IDF features from tags (user tags + genres as text)
print("[INFO] Building TF-IDF features from user tags + genres text...")
movie_meta = movies[movies["movieId"].isin(movie_ids)].copy()
movie_meta = movie_meta.merge(movie_tags, on="movieId", how="left")
movie_meta["user_tags"] = movie_meta["user_tags"].fillna("")
movie_meta["genres"] = movie_meta["genres"].fillna("").str.lower().str.replace("|", " ", regex=False)

# Basic cleanup
movie_meta["text"] = (
    movie_meta["user_tags"].str.lower()
    .str.replace(r"[^a-z0-9 ]", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
    + " " +
    movie_meta["genres"].str.lower()
)

# TF-IDF (limit vocab size, remove very common words)
vectorizer = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    min_df=3,
    max_df=0.5,
    ngram_range=(1, 2),
)
tfidf = vectorizer.fit_transform(movie_meta["text"].values)
save_npz(proc_dir / "tfidf_tags.npz", tfidf)

# Save vocabulary (optional, useful for explainability)
vocab = pd.DataFrame({"token": list(vectorizer.vocabulary_.keys()), "index": list(vectorizer.vocabulary_.values())})
vocab.sort_values("index").to_csv(proc_dir / "tfidf_vocab.csv", index=False)

# Genome movie x tag relevance matrix (sparse)
print("[INFO] Building sparse genome matrix (movie x tagId).")
# Restrict to movies in train mapping
genome_m = genome_subset[genome_subset["movieId"].isin(movie_ids)].copy()
tag_ids = pd.Index(sorted(genome_m["tagId"].unique()), name="tagId")
tag_id_to_idx = pd.Series(np.arange(len(tag_ids), dtype="int32"), index=tag_ids)

rows = movie_id_to_idx.loc[genome_m["movieId"]].values
cols = tag_id_to_idx.loc[genome_m["tagId"]].values
vals = genome_m["relevance"].astype("float32").values
genome_matrix = coo_matrix((vals, (rows, cols)), shape=(len(movie_ids), len(tag_ids))).tocsr()
save_npz(proc_dir / "genome_matrix.npz", genome_matrix)
pd.DataFrame({"tagId": tag_ids.values, "tag_idx": np.arange(len(tag_ids), dtype="int32")}).to_csv(proc_dir / "genome_tag_index.csv", index=False)

# Train-test saving and summaries
print("[INFO] Saving train/test splits and stats...")
train.to_csv(proc_dir / "train.csv", index=False)
test.to_csv(proc_dir / "test.csv", index=False)
user_stats.to_csv(proc_dir / "user_stats.csv", index=False)
movie_stats.to_csv(proc_dir / "movie_stats.csv", index=False)
movies.to_csv(proc_dir / "movies_clean.csv", index=False)
genome_tags.to_csv(proc_dir / "genome_tags.csv", index=False)

# Summary file
with open(proc_dir / "preprocessing_info.txt", "w") as f:
    f.write("=== Preprocessing Summary ===\n")
    f.write(f"Global mean rating: {global_mean:.4f}\n")
    f.write(f"Users (after filter): {ratings['userId'].nunique()}\n")
    f.write(f"Movies (after filter): {ratings['movieId'].nunique()}\n")
    f.write(f"Train size: {train.shape}, Test size: {test.shape}\n")
    f.write(f"Interactions CSR: shape={interactions.shape}, nnz={interactions.nnz}\n")
    f.write(f"Genres multi-hot: shape={genres_multi_hot.shape}, nnz={genres_multi_hot.nnz}\n")
    f.write(f"TF-IDF: shape={tfidf.shape}\n")
    f.write(f"Genome matrix: shape={genome_matrix.shape}, nnz={genome_matrix.nnz}\n")
    f.write("\nNotes:\n")
    f.write("- Applied cold-start filters and activity caps to focus on dense segments.\n")
    f.write("- Bayesian movie averages computed for robust popularity priors.\n")
    f.write("- Contiguous ID maps are per-train split to avoid leakage.\n")
    f.write("- Saved sparse artifacts for efficient downstream modeling.\n")

print("[INFO] Preprocessing complete. Artifacts saved to data/processed/")
