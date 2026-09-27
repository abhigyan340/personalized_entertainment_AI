"""
load_model.py
─────────────
Loads the pre-computed ML artifacts from data/artifacts/ into module-level
globals so the recommendation engine can import them directly.

Artifacts actually used at inference time
─────────────────────────────────────────
  movies_processed.pkl   — DataFrame with per-movie metadata
  movie_embeddings.npy   — (4803 × 384) sentence-transformer embeddings
  hybrid_similarity.npy  — (4803 × 4803) 0.5·tfidf + 0.5·semantic cosine matrix

Artifacts NOT loaded here (exist on disk, only needed during pre-processing)
─────────────────────────────────────────────────────────────────────────────
  tfidf_similarity.npy, semantic_similarity.npy, tfidf_normalized.npy,
  semantic_normalized.npy  — intermediate matrices used when building
                             hybrid_similarity in the notebook; ~500 MB total.
  tfidf_vectorizer.pkl, tfidf_matrix.npz — used during tag encoding in the
                             notebook; not referenced by the serving path.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, "data", "artifacts")


def _artifact(filename: str) -> str:
    return os.path.join(ARTIFACT_DIR, filename)


def _load_artifacts():
    """Load and return the three artifacts needed at inference time."""

    required = {
        "movies_processed.pkl":  "Movie metadata DataFrame",
        "movie_embeddings.npy":  "Sentence-transformer embedding matrix",
        "hybrid_similarity.npy": "Pre-computed hybrid similarity matrix",
    }

    missing = [f for f in required if not os.path.exists(_artifact(f))]
    if missing:
        logger.error(
            "Missing required artifact(s): %s\n"
            "Run the preprocessing notebook to generate them:\n"
            "  jupyter notebook notebooks/01_movies_data_exploration.ipynb",
            ", ".join(missing),
        )
        sys.exit(1)

    try:
        _movies = pd.read_pickle(_artifact("movies_processed.pkl"))
        _embeddings = np.load(_artifact("movie_embeddings.npy"))
        _hybrid = np.load(_artifact("hybrid_similarity.npy"))
    except Exception as exc:
        logger.error("Failed to load artifacts: %s", exc)
        sys.exit(1)

    logger.info(
        "Artifacts loaded — movies: %s  embeddings: %s  hybrid: %s",
        _movies.shape,
        _embeddings.shape,
        _hybrid.shape,
    )
    return _movies, _embeddings, _hybrid


# Module-level globals — imported by recommender.py
movies, movie_embeddings, hybrid_similarity = _load_artifacts()
