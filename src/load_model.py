import os
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import load_npz


# Find project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACT_DIR = os.path.join(DATA_DIR, "artifacts")


# Load processed movie data
movies = pd.read_pickle(
    os.path.join(ARTIFACT_DIR, "movies_processed.pkl")
)


# Load TF-IDF vectorizer
tfidf = joblib.load(
    os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.pkl")
)


# Load TF-IDF matrix
tfidf_matrix = load_npz(
    os.path.join(ARTIFACT_DIR, "tfidf_matrix.npz")
)


# Load similarity matrices
tfidf_similarity = np.load(
    os.path.join(ARTIFACT_DIR, "tfidf_similarity.npy")
)

semantic_similarity = np.load(
    os.path.join(ARTIFACT_DIR, "semantic_similarity.npy")
)

movie_embeddings = np.load(
    os.path.join(ARTIFACT_DIR, "movie_embeddings.npy")
)

tfidf_normalized = np.load(
    os.path.join(ARTIFACT_DIR, "tfidf_normalized.npy")
)

semantic_normalized = np.load(
    os.path.join(ARTIFACT_DIR, "semantic_normalized.npy")
)

hybrid_similarity = np.load(
    os.path.join(ARTIFACT_DIR, "hybrid_similarity.npy")
)


print("Model artifacts loaded successfully.")
print("Movies:", movies.shape)
print("TF-IDF:", tfidf_matrix.shape)
print("Hybrid:", hybrid_similarity.shape)
print("Embeddings:", movie_embeddings.shape)