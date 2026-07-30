# ==================================================
# Movie Recommendation System
# Data Exploration
# ==================================================

import pandas as pd

# --------------------------------------------------
# Load Cleaned Datasets
# --------------------------------------------------

movies = pd.read_csv("data/cleaned_movies.csv")
ratings = pd.read_csv("data/cleaned_ratings.csv")
master = pd.read_csv("data/cleaned_movielens.csv")

print("=" * 60)
print("DATA EXPLORATION")
print("=" * 60)

# ==================================================
# Dataset Overview
# ==================================================

print("\nDataset Overview")
print("-" * 60)

print(f"Movies : {len(movies):,}")
print(f"Ratings: {len(ratings):,}")
print(f"Users  : {ratings['userId'].nunique():,}")

# ==================================================
# Movie Statistics
# ==================================================

movie_stats = (
    master.groupby(["movieId", "clean_title"])
    .agg(
        Total_Ratings=("rating", "count"),
        Average_Rating=("rating", "mean")
    )
    .reset_index()
)

movie_stats = movie_stats.sort_values(
    "Total_Ratings",
    ascending=False
)

movie_stats.to_csv(
    "data/movie_statistics.csv",
    index=False
)

print("✓ movie_statistics.csv")

# ==================================================
# Top 20 Most Rated Movies
# ==================================================

top20 = movie_stats.head(20)

top20.to_csv(
    "data/most_rated_movies.csv",
    index=False
)

print("✓ most_rated_movies.csv")

# ==================================================
# Highest Rated Movies
# (minimum 50 ratings)
# ==================================================

highest = movie_stats[
    movie_stats["Total_Ratings"] >= 50
]

highest = highest.sort_values(
    "Average_Rating",
    ascending=False
)

highest.to_csv(
    "data/highest_rated_movies.csv",
    index=False
)

print("✓ highest_rated_movies.csv")

# ==================================================
# Rating Distribution
# ==================================================

rating_dist = (
    ratings["rating"]
    .value_counts()
    .sort_index()
    .reset_index()
)

rating_dist.columns = [
    "Rating",
    "Count"
]

rating_dist.to_csv(
    "data/ratings_distribution.csv",
    index=False
)

print("✓ ratings_distribution.csv")

# ==================================================
# Genre Distribution
# ==================================================

genre_dist = (
    movies["genres"]
    .str.split("|")
    .explode()
    .value_counts()
    .reset_index()
)

genre_dist.columns = [
    "Genre",
    "Count"
]

genre_dist.to_csv(
    "data/genre_distribution.csv",
    index=False
)

print("✓ genre_distribution.csv")

# ==================================================
# User Statistics
# ==================================================

user_stats = (
    ratings.groupby("userId")
    .agg(
        Movies_Rated=("movieId", "count"),
        Average_Rating=("rating", "mean")
    )
    .reset_index()
)

user_stats.to_csv(
    "data/user_statistics.csv",
    index=False
)

print("✓ user_statistics.csv")

# ==================================================
# Year Distribution
# ==================================================

year_dist = (
    movies["year"]
    .value_counts()
    .sort_index()
    .reset_index()
)

year_dist.columns = [
    "Year",
    "Movies"
]

year_dist.to_csv(
    "data/year_distribution.csv",
    index=False
)

print("✓ year_distribution.csv")

# ==================================================
# Summary
# ==================================================

print("\n" + "=" * 60)
print("DATA EXPLORATION COMPLETE")
print("=" * 60)

print("\nGenerated Files:")
print("- movie_statistics.csv")
print("- most_rated_movies.csv")
print("- highest_rated_movies.csv")
print("- ratings_distribution.csv")
print("- genre_distribution.csv")
print("- user_statistics.csv")
print("- year_distribution.csv")
