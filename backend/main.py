"""
backend/main.py
───────────────
FastAPI ML service — runs on port 8000 (internal, not exposed to the browser).

Routes
──────
  GET  /           health check
  GET  /stats      dataset + model metadata (useful for debugging / demos)
  GET  /search     movie title search (TMDB primary, local catalog fallback)
  POST /recommend  main recommendation endpoint
"""

import logging
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.load_model import movie_embeddings, movies, hybrid_similarity
from src.recommender import recommend_detailed, search_catalog, semantic_search
from src.tmdb_client import search_movie, tmdb_available

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Personalized Entertainment AI",
    description=(
        "Hybrid movie recommendation API using Sentence Transformers, "
        "TF-IDF, and metadata re-ranking.  "
        "Supports cold-start for movies outside the training dataset via "
        "live TMDB embedding."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class RecommendationRequest(BaseModel):
    movie: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Movie title to base recommendations on.",
        examples=["The Dark Knight"],
    )
    n: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of recommendations to return (1–20).",
    )
    tmdb_id: Optional[int] = Field(
        default=None,
        gt=0,
        description="TMDB movie id for unambiguous lookup (optional).",
    )


class SemanticSearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description=(
            "Natural-language description of the kind of movie to find. "
            "E.g. 'a heist film set in Paris with a twist ending'."
        ),
        examples=["mind-bending science fiction involving dreams and reality"],
    )
    n: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of results to return (1–50).",
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
def health():
    return {"status": "ok", "service": "Personalized Entertainment AI"}


@app.get("/stats", tags=["meta"])
def stats():
    """
    Return dataset and model metadata.
    Useful for debugging, demos, and confirming artifacts loaded correctly.
    """
    return {
        "dataset_size":       int(len(movies)),
        "embedding_dim":      int(movie_embeddings.shape[1]),
        "hybrid_matrix_shape": list(hybrid_similarity.shape),
        "model":              "all-MiniLM-L6-v2",
        "hybrid_weights":     {"tfidf": 0.5, "semantic": 0.5},
        "ranking_weights":    {"hybrid": 0.9, "metadata": 0.1},
        "metadata_weights":   {
            "genres":   0.40,
            "keywords": 0.20,
            "cast":     0.20,
            "director": 0.20,
        },
        "cold_start": "live TMDB embedding (out-of-dataset movies)",
        "tmdb_available": tmdb_available(),
    }


@app.get("/search", tags=["search"])
def search_movies(
    query: str = Query(..., min_length=1, max_length=200),
):
    """
    Search for movies by title.
    Uses TMDB (primary) with a local catalog fallback.
    """
    if not query.strip():
        return {"results": [], "source": "none"}

    tmdb_results = search_movie(query).get("results") or []
    if tmdb_results:
        return {"results": tmdb_results[:8], "source": "tmdb"}

    return {
        "results": search_catalog(query),
        "source":  "local",
    }


@app.post("/recommend", tags=["recommend"])
def recommend_movies(request: RecommendationRequest):
    """
    Return up to *n* recommendations for the given movie title.
    Logs recommendation latency so performance can be monitored.
    """
    t0      = time.perf_counter()
    results = recommend_detailed(
        request.movie,
        n=request.n,
        tmdb_id=request.tmdb_id,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if not results:
        logger.info(
            "recommend '%s' → no results  (%.1f ms)", request.movie, elapsed_ms
        )
        return {
            "movie":           request.movie,
            "recommendations": [],
            "message":         "Movie not found in dataset or TMDB.",
        }

    logger.info(
        "recommend '%s' → %d results  (%.1f ms)",
        request.movie, len(results), elapsed_ms,
    )
    return {
        "movie":           request.movie,
        "recommendations": results,
        "latency_ms":      round(elapsed_ms, 1),
    }


@app.post("/semantic-search", tags=["search"])
def semantic_search_movies(request: SemanticSearchRequest):
    """
    Search the local movie catalog using a natural-language description.

    The query is encoded with the same SentenceTransformer model (all-MiniLM-L6-v2)
    used to build the movie embedding artifact.  Similarity is computed as cosine
    similarity against all 4,803 pre-computed movie embeddings in a single NumPy
    dot product — no external API calls required.

    Note: movie embeddings were built from each film's overview, genres,
    keywords, top-5 cast, and director. Describe plot, theme, genres,
    or key cast/director terms for best results.

    Limitation: only searches the local 4,803-movie catalog.
    """
    t0 = time.perf_counter()
    results = semantic_search(request.query, n=request.n)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "semantic-search '%s' → %d results  (%.1f ms)",
        request.query[:60], len(results), elapsed_ms,
    )
    return {
        "query":      request.query,
        "results":    results,
        "n":          len(results),
        "latency_ms": round(elapsed_ms, 1),
        "note":       (
            "Results are ranked by semantic similarity of the query to each "
            "movie's plot summary. Scores are cosine similarities in [0, 1] "
            "and do not indicate recommendation confidence."
        ),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
