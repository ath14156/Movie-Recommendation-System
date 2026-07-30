# ==================================================
# Movie Recommendation System
# Explore Dataset
# ==================================================

import pandas as pd

# --------------------------------------------------
# Load Datasets
# --------------------------------------------------

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")
links = pd.read_csv("data/links.csv")
tags = pd.read_csv("data/tags.csv")

# --------------------------------------------------
# Movies Dataset
# --------------------------------------------------

print("=" * 60)
print("MOVIES DATASET")
print("=" * 60)

print("\nShape:")
print(movies.shape)

print("\nColumns:")
print(movies.columns.tolist())

print("\nFirst 5 Rows:")
print(movies.head())

print("\nMissing Values:")
print(movies.isnull().sum())

print("\nData Types:")
print(movies.dtypes)

print("\nUnique Genres:")
genres = movies["genres"].str.split("|").explode().unique()
print(sorted(genres))

# --------------------------------------------------
# Ratings Dataset
# --------------------------------------------------

print("\n" + "=" * 60)
print("RATINGS DATASET")
print("=" * 60)

print("\nShape:")
print(ratings.shape)

print("\nColumns:")
print(ratings.columns.tolist())

print("\nFirst 5 Rows:")
print(ratings.head())

print("\nMissing Values:")
print(ratings.isnull().sum())

print("\nData Types:")
print(ratings.dtypes)

print("\nRating Statistics:")
print(ratings["rating"].describe())

# --------------------------------------------------
# Links Dataset
# --------------------------------------------------

print("\n" + "=" * 60)
print("LINKS DATASET")
print("=" * 60)

print("\nShape:")
print(links.shape)

print("\nColumns:")
print(links.columns.tolist())

print("\nFirst 5 Rows:")
print(links.head())

# --------------------------------------------------
# Tags Dataset
# --------------------------------------------------

print("\n" + "=" * 60)
print("TAGS DATASET")
print("=" * 60)

print("\nShape:")
print(tags.shape)

print("\nColumns:")
print(tags.columns.tolist())

print("\nFirst 5 Rows:")
print(tags.head())

print("\nMissing Values:")
print(tags.isnull().sum())

# --------------------------------------------------
# Overall Summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(f"Total Movies: {len(movies):,}")
print(f"Total Ratings: {len(ratings):,}")
print(f"Total Users: {ratings['userId'].nunique():,}")
print(f"Average Rating: {ratings['rating'].mean():.2f}")
