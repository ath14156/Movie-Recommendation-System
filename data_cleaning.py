# ==================================================
# Movie Recommendation System
# Data Cleaning
# ==================================================

import pandas as pd

# --------------------------------------------------
# Load Raw Datasets
# --------------------------------------------------

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")
links = pd.read_csv("data/links.csv")
tags = pd.read_csv("data/tags.csv")

print("=" * 60)
print("STARTING DATA CLEANING")
print("=" * 60)

# ==================================================
# Clean Movies Dataset
# ==================================================

print("\nCleaning movies.csv...")

movies = movies.drop_duplicates()

movies["title"] = movies["title"].str.strip()

# Extract release year
movies["year"] = movies["title"].str.extract(r"\((\d{4})\)")

# Remove year from title
movies["clean_title"] = movies["title"].str.replace(
    r"\s*\(\d{4}\)",
    "",
    regex=True
)

movies["genres"] = movies["genres"].str.strip()

movies.to_csv("data/cleaned_movies.csv", index=False)

print("✓ cleaned_movies.csv saved")

# ==================================================
# Clean Ratings Dataset
# ==================================================

print("\nCleaning ratings.csv...")

ratings = ratings.drop_duplicates()

ratings["rating_date"] = pd.to_datetime(
    ratings["timestamp"],
    unit="s"
)

ratings = ratings.sort_values(
    by=["userId", "movieId"]
)

ratings.to_csv("data/cleaned_ratings.csv", index=False)

print("✓ cleaned_ratings.csv saved")

# ==================================================
# Clean Links Dataset
# ==================================================

print("\nCleaning links.csv...")

links = links.drop_duplicates()

links.to_csv("data/cleaned_links.csv", index=False)

print("✓ cleaned_links.csv saved")

# ==================================================
# Clean Tags Dataset
# ==================================================

print("\nCleaning tags.csv...")

tags = tags.drop_duplicates()

tags["tag"] = (
    tags["tag"]
    .fillna("")
    .str.strip()
    .str.lower()
)

tags["tag_date"] = pd.to_datetime(
    tags["timestamp"],
    unit="s"
)

tags.to_csv("data/cleaned_tags.csv", index=False)

print("✓ cleaned_tags.csv saved")

# ==================================================
# Create Master Dataset
# ==================================================

print("\nCreating master dataset...")

master = ratings.merge(
    movies,
    on="movieId",
    how="left"
)

master.to_csv(
    "data/cleaned_movielens.csv",
    index=False
)

print("✓ cleaned_movielens.csv saved")

# ==================================================
# Summary
# ==================================================

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETE")
print("=" * 60)

print(f"Movies: {len(movies):,}")
print(f"Ratings: {len(ratings):,}")
print(f"Links: {len(links):,}")
print(f"Tags: {len(tags):,}")
print(f"Master Dataset: {len(master):,}")

print("\nGenerated Files:")
print("- cleaned_movies.csv")
print("- cleaned_ratings.csv")
print("- cleaned_links.csv")
print("- cleaned_tags.csv")
print("- cleaned_movielens.csv")
