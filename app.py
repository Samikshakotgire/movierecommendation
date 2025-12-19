import os
import ast
import numpy as np
import pandas as pd
import streamlit as st
import requests

from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# ---------- CONFIG ----------
load_dotenv()  # load .env file
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
DEFAULT_POSTER = "https://via.placeholder.com/300x450?text=No+Image"


# ---------- DATA LOADING ----------
@st.cache_data
def load_data():
    files = os.listdir(".")
    if "tmdb_5000_movies.csv" in files and "tmdb_5000_credits.csv" in files:
        movies = pd.read_csv("tmdb_5000_movies.csv")
        credits = pd.read_csv("tmdb_5000_credits.csv")
        return movies, credits, True

    # Fallback demo data
    demo_movies = pd.DataFrame({
        "title": ["Avatar", "Titanic", "Star Wars", "Avengers", "Inception"],
        "genres": [["Action Adventure SciFi"],
                   ["Romance Drama"],
                   ["SciFi Action"],
                   ["Action Adventure"],
                   ["Thriller SciFi"]],
        "overview": [["futuristic world"],
                     ["ocean disaster"],
                     ["space opera"],
                     ["superheroes"],
                     ["dream heist"]]
    })
    return demo_movies, pd.DataFrame(), False


# ---------- PREPROCESSING ----------
def safe_convert(text, func):
    try:
        return func(ast.literal_eval(text))
    except Exception:
        return []


def convert(text):
    return safe_convert(text, lambda x: [i["name"] for i in x])


def convert3(text):
    L, c = [], 0
    for i in safe_convert(text, lambda x: x):
        if c < 3:
            L.append(i["name"])
            c += 1
        else:
            break
    return L


def fetch_director(text):
    for i in safe_convert(text, lambda x: x):
        if i.get("job") == "Director":
            return [i["name"]]
    return []


def collapse(L):
    return [i.replace(" ", "") for i in L if isinstance(i, str)]


@st.cache_data
def process_data(movies, credits, have_full_data: bool):
    if movies.empty:
        return pd.DataFrame()

    if have_full_data and not credits.empty and "title" in credits.columns:
        movies = movies.merge(credits, on="title", how="inner")

    cols = ["movie_id", "title", "overview", "genres",
            "keywords", "cast", "crew"]
    movies = movies[[c for c in cols if c in movies.columns]]

    movies = movies.dropna(subset=["overview"]).head(2000)

    if "genres" in movies.columns:
        movies["genres"] = movies["genres"].apply(convert)
        movies["genres"] = movies["genres"].apply(collapse)

    if "keywords" in movies.columns:
        movies["keywords"] = movies["keywords"].apply(convert)

    if "cast" in movies.columns:
        movies["cast"] = movies["cast"].apply(convert3)

    if "crew" in movies.columns:
        movies["crew"] = movies["crew"].apply(fetch_director)

    movies["overview"] = movies["overview"].fillna("").apply(
        lambda x: x.split()[:20] if isinstance(x, str) else x
    )

    tag_cols = ["overview"]
    for c in ["genres", "keywords", "cast", "crew"]:
        if c in movies.columns:
            tag_cols.append(c)

    movies["tags"] = movies[tag_cols].apply(lambda row: sum(row, []), axis=1)
    movies["tags"] = movies["tags"].apply(lambda x: " ".join(x).lower())

    return movies


# ---------- MODEL BUILDING ----------
@st.cache_data
def build_model(movies):
    if movies.empty:
        return None, movies

    tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
    vector = tfidf.fit_transform(movies["tags"]).toarray()

    svd = TruncatedSVD(n_components=30, random_state=42)
    vector_reduced = svd.fit_transform(vector)

    similarity = cosine_similarity(vector_reduced)
    return similarity, movies


def recommend(movie, similarity, movies):
    if similarity is None or movies.empty:
        return []

    try:
        index = np.where(
            movies["title"].str.contains(movie, case=False, na=False)
        )[0][0]
        distances = sorted(
            list(enumerate(similarity[index])),
            reverse=True,
            key=lambda x: x[1]
        )
        return [i[0] for i in distances[1:6]]
    except Exception:
        return []


# ---------- POSTER FETCH ----------
def fetch_poster(movie_id: int) -> str:
    if not TMDB_API_KEY:
        return ""

    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get("poster_path")
        if not poster_path:
            return ""
        return "https://image.tmdb.org/t/p/w500" + poster_path
    except Exception:
        return ""


# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="MovieLens", layout="wide")
st.title("🎬 Movie Recommender")
st.markdown("**TF‑IDF + SVD | Uses local TMDB CSV files if present**")

movies_raw, credits_raw, have_full_data = load_data()
movies_df = process_data(movies_raw, credits_raw, have_full_data)
similarity_df, movies_df = build_model(movies_df)

st.sidebar.header("📈 Stats")
st.sidebar.metric("Movies", len(movies_df))
st.sidebar.metric("Vector Dim", 30)

col1, col2 = st.columns([1, 3])

with col1:
    if not movies_df.empty:
        options = movies_df["title"].tolist()
    else:
        options = ["Avatar"]
    selected = st.selectbox("🎥 Movie:", options)

with col2:
    if st.button("🔍 Recommend!", type="primary"):
        rec_indices = recommend(selected, similarity_df, movies_df)
        if rec_indices:
            st.success(f"**Top picks for '{selected}':**")

            cols = st.columns(5)
            for col, idx in zip(cols, rec_indices):
                row = movies_df.iloc[idx]
                title = row["title"]
                movie_id = row.get("movie_id", None)

                with col:
                    st.text(title)
                    if movie_id is not None:
                        poster_url = fetch_poster(int(movie_id))
                        if poster_url:
                            st.image(poster_url, use_container_width=True)
                        else:
                            st.image(DEFAULT_POSTER, use_container_width=True)
        else:
            st.warning("No recommendations found. Try another title.")

st.markdown("---")
