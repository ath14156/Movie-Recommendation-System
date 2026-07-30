import pandas as pd
import joblib

movies = pd.read_csv("data/recommendation_movies.csv")
similarity = joblib.load("models/similarity_matrix.pkl")


def recommend(movie_name, top_n=10):

    matches = movies[
        movies["clean_title"]
        .str.lower()
        .str.contains(movie_name.lower(), na=False)
    ]

    if matches.empty:
        print("Movie not found.")
        return

    movie_index = matches.index[0]

    scores = list(enumerate(similarity[movie_index]))

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\nRecommendations for: {movies.iloc[movie_index]['clean_title']}\n")

    count = 0

    for index, score in scores[1:]:

        print(
            f"{count+1}. "
            f"{movies.iloc[index]['clean_title']} "
            f"({score:.3f})"
        )

        count += 1

        if count == top_n:
            break


recommend("Toy Story")
