# ==================================================
# Movie Recommendation System
# Streamlit Dashboard
# ==================================================

import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# --------------------------------------------------
# Load Data
# --------------------------------------------------

movies = pd.read_csv("data/recommendation_movies.csv")

movie_stats = pd.read_csv("data/movie_statistics.csv")
genre_stats = pd.read_csv("data/genre_distribution.csv")
rating_dist = pd.read_csv("data/ratings_distribution.csv")
year_dist = pd.read_csv("data/year_distribution.csv")

similarity = joblib.load("models/similarity_matrix.pkl")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("🎬 Navigation")

page = st.sidebar.radio(
    "Choose a Page",
    [
        "🏠 Home",
        "📊 Dataset Analytics",
        "🎬 Recommendation Engine",
        "📈 Recommendation Insights",
        "ℹ️ About"
    ]
)

# ==================================================
# Home
# ==================================================

if page == "🏠 Home":

    st.title("🎬 Movie Recommendation System")

    st.write("""
    Welcome!

    This project demonstrates a **Content-Based Movie Recommendation System**
    built using **TF-IDF Vectorization** and **Cosine Similarity**.

    Select a movie and discover similar movies instantly.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Movies",
        f"{len(movies):,}"
    )

    col2.metric(
        "Genres",
        movies["genres"].str.split("|").explode().nunique()
    )

    col3.metric(
        "Recommendation Model",
        "Content-Based"
    )

    st.divider()

    st.subheader("Top Rated Movies")

    st.dataframe(
        movie_stats.head(10),
        use_container_width=True
    )

# ==================================================
# Dataset Analytics
# ==================================================

elif page == "📊 Dataset Analytics":

    st.title("📊 Dataset Analytics")

    st.subheader("Genre Distribution")

    st.bar_chart(
        genre_stats.set_index("Genre")
    )

    st.subheader("Ratings Distribution")

    st.bar_chart(
        rating_dist.set_index("Rating")
    )

    st.subheader("Movies by Release Year")

    st.line_chart(
        year_dist.set_index("Year")
    )

    st.divider()

    st.subheader("Movie Statistics")

    st.dataframe(
        movie_stats.head(20),
        use_container_width=True
    )
# ==================================================
# Recommendation Engine
# ==================================================

elif page == "🎬 Recommendation Engine":

    st.title("🎬 Movie Recommendation Engine")

    st.write(
        """
        Select a movie below to receive the Top 10 most similar movie recommendations
        using Content-Based Filtering.
        """
    )

    # -----------------------------------------------
    # Movie Selection
    # -----------------------------------------------

    movie_list = sorted(movies["clean_title"].unique())

    selected_movie = st.selectbox(
        "Select a Movie",
        movie_list
    )

    # -----------------------------------------------
    # Recommendation Function
    # -----------------------------------------------

    def recommend(movie_name):

        movie_index = movies[
            movies["clean_title"] == movie_name
        ].index[0]

        similarity_scores = list(
            enumerate(similarity[movie_index])
        )

        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True
        )

        recommendations = []

        for index, score in similarity_scores[1:11]:

            recommendations.append(
                {
                    "Movie": movies.iloc[index]["clean_title"],
                    "Genres": movies.iloc[index]["genres"],
                    "Similarity Score": round(score, 3)
                }
            )

        return pd.DataFrame(recommendations)

    # -----------------------------------------------
    # Recommend Button
    # -----------------------------------------------

    if st.button("🎬 Recommend Movies"):

        recommendations = recommend(selected_movie)

        st.success(
            f"Top recommendations for '{selected_movie}'"
        )

        st.dataframe(
            recommendations,
            use_container_width=True
        )
    

# ==================================================
# Recommendation Insights
# ==================================================

elif page == "📈 Recommendation Insights":

    st.title("📈 Recommendation Insights")

    # -----------------------------------------------
    # Metrics
    # -----------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Movies",
        f"{len(movies):,}"
    )

    col2.metric(
        "Genres",
        genre_stats.shape[0]
    )

    col3.metric(
        "Average Rating",
        f"{movie_stats['Average_Rating'].mean():.2f}"
    )

    st.divider()

    # -----------------------------------------------
    # Top Rated Movies
    # -----------------------------------------------

    st.subheader("⭐ Top Rated Movies")

    st.dataframe(
        movie_stats.sort_values(
            "Average_Rating",
            ascending=False
        ).head(10),
        use_container_width=True
    )

    st.divider()

    # -----------------------------------------------
    # Most Rated Movies
    # -----------------------------------------------

    st.subheader("🎬 Most Rated Movies")

    st.dataframe(
        movie_stats.sort_values(
            "Total_Ratings",
            ascending=False
        ).head(10),
        use_container_width=True
    )

    st.divider()

    # -----------------------------------------------
    # Genre Distribution
    # -----------------------------------------------

    st.subheader("🎭 Genre Distribution")

    st.bar_chart(
        genre_stats.set_index("Genre")
    )

    st.divider()

    # -----------------------------------------------
    # Rating Distribution
    # -----------------------------------------------

    st.subheader("⭐ Rating Distribution")

    st.bar_chart(
        rating_dist.set_index("Rating")
    )
# ==================================================
# About
# ==================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About")

    st.markdown("""
## 🎬 Movie Recommendation System

This project demonstrates a **Content-Based Movie Recommendation System**
built using **Natural Language Processing (TF-IDF)** and
**Cosine Similarity**.

### Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

### Dataset

- MovieLens Latest Small Dataset
- 9,742 Movies
- 100,836 Ratings
- 610 Users

### Features

- Movie Recommendation Engine
- Dataset Analytics
- Recommendation Insights
- Interactive Streamlit Dashboard

### Machine Learning Techniques

- TF-IDF Vectorization
- Cosine Similarity
- Content-Based Filtering

---

Developed as part of an AI / Machine Learning portfolio.
""")
