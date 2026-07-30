# ==================================================
# Movie Recommendation System
# Recommendation Engine
# ==================================================

import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# Load Cleaned Dataset
# --------------------------------------------------

movies = pd.read_csv("data/cleaned_movies.csv")

print("=" * 60)
print("BUILDING RECOMMENDATION ENGINE")
print("=" * 60)

# --------------------------------------------------
# Prepare Features
# --------------------------------------------------

movies["genres"] = movies["genres"].fillna("")
movies["clean_title"] = movies["clean_title"].fillna("")

movies["features"] = (
    movies["genres"] + " " +
    movies["clean_title"]
)

# --------------------------------------------------
# TF-IDF Vectorization
# --------------------------------------------------

vectorizer = TfidfVectorizer(stop_words="english")

tfidf_matrix = vectorizer.fit_transform(
    movies["features"]
)

# --------------------------------------------------
# Cosine Similarity
# --------------------------------------------------

similarity_matrix = cosine_similarity(
    tfidf_matrix,
    tfidf_matrix
)

# --------------------------------------------------
# Save Model
# --------------------------------------------------

joblib.dump(
    similarity_matrix,
    "models/similarity_matrix.pkl"
)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

movies.to_csv(
    "data/recommendation_movies.csv",
    index=False
)

print("\nRecommendation engine created successfully!")

print(f"Movies Processed : {len(movies):,}")
print(f"Similarity Matrix Shape : {similarity_matrix.shape}")

print("\nGenerated Files:")
print("- similarity_matrix.pkl")
print("- tfidf_vectorizer.pkl")
print("- recommendation_movies.csv")
