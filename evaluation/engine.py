"""
evaluation/engine.py
────────────────────
Diagnostic scoring engine for evaluation and ablation experiments.

This module provides a score breakdown for every candidate movie so that
individual signal contributions can be inspected, compared, and logged.

It uses the same scoring functions as the production recommender
(src/recommender.py) — there is NO duplicated logic.  The difference is
that this engine:

  1. Exposes individual tfidf_score, semantic_score, genre_score, etc.
  2. Accepts configurable hybrid weights so ablation experiments can
     swap in any (tfidf_w, semantic_w, meta_w) combination without
     touching production code.
  3. Returns only DataFrame indices — no TMDB calls, no poster fetching.
     This keeps evaluation fast and fully offline.

Usage
─────
    from evaluation.engine import diagnostic_scores, ablation_recommend

    # Score breakdown for one pair:
    info = diagnostic_scores(seed_idx=0, candidate_idx=5)

    # Full recommendation list with configurable weights:
    recs = ablation_recommend(seed_idx=0, n=10, tfidf_w=0.5, semantic_w=0.5, meta_w=0.10)
"""

import os
import functools

import numpy as np

from src.load_model import movies
from src.recommender import (
    genre_similarity,
    keyword_similarity,
    cast_similarity,
    director_similarity,
    metadata_similarity,
)


# ── Lazy-load individual similarity matrices ──────────────────────────────────
# These are NOT loaded by the production recommender.
# lru_cache ensures each matrix is read from disk only once per process.

_ARTIFACT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "artifacts",
)


@functools.lru_cache(maxsize=1)
def _tfidf_sim() -> np.ndarray:
    return np.load(os.path.join(_ARTIFACT_DIR, "tfidf_similarity.npy"))


@functools.lru_cache(maxsize=1)
def _semantic_sim() -> np.ndarray:
    return np.load(os.path.join(_ARTIFACT_DIR, "semantic_similarity.npy"))


# ── Score breakdown for a single pair ────────────────────────────────────────

def diagnostic_scores(seed_idx: int, candidate_idx: int) -> dict:
    """
    Return a full score breakdown for a (seed, candidate) pair.

    The returned dict contains every individual signal plus the derived
    composite scores, making it easy to inspect why a candidate was ranked
    where it was.

    Fields
    ──────
    tfidf_score    : cosine similarity from TF-IDF representation
    semantic_score : cosine similarity from sentence-transformer embedding
    genre_score    : genre overlap (precision-style)
    keyword_score  : keyword overlap (precision-style)
    cast_score     : cast overlap (precision-style)
    director_score : 1.0 if same director else 0.0
    metadata_score : weighted combination (0.40g + 0.20k + 0.20c + 0.20d)
    hybrid_50_50   : 0.50*tfidf + 0.50*semantic  (precomputed hybrid config)
    """
    tfidf_s    = float(_tfidf_sim()[seed_idx, candidate_idx])
    semantic_s = float(_semantic_sim()[seed_idx, candidate_idx])
    genre_s    = genre_similarity(seed_idx, candidate_idx)
    keyword_s  = keyword_similarity(seed_idx, candidate_idx)
    cast_s     = cast_similarity(seed_idx, candidate_idx)
    director_s = director_similarity(seed_idx, candidate_idx)
    meta_s     = metadata_similarity(seed_idx, candidate_idx)

    return {
        "seed_title":      str(movies.iloc[seed_idx]["title"]),
        "candidate_title": str(movies.iloc[candidate_idx]["title"]),
        "tfidf_score":     round(tfidf_s,    6),
        "semantic_score":  round(semantic_s, 6),
        "genre_score":     round(genre_s,    6),
        "keyword_score":   round(keyword_s,  6),
        "cast_score":      round(cast_s,     6),
        "director_score":  round(director_s, 6),
        "metadata_score":  round(meta_s,     6),
        "hybrid_50_50":    round(0.5 * tfidf_s + 0.5 * semantic_s, 6),
    }


# ── Configurable ablation recommender ────────────────────────────────────────

def ablation_recommend(
    seed_idx:       int,
    n:              int   = 10,
    tfidf_w:        float = 0.50,
    semantic_w:     float = 0.50,
    meta_w:         float = 0.10,
    candidate_count: int  = 50,
    exclude_idx:    set[int] | None = None,
) -> list[int]:
    """
    Return the top-n recommendations using configurable signal weights.

    Parameters
    ──────────
    seed_idx        : DataFrame index of the query movie
    n               : number of recommendations to return
    tfidf_w         : weight for TF-IDF similarity  (must sum to 1.0 with semantic_w)
    semantic_w      : weight for semantic similarity
    meta_w          : metadata re-ranking weight.
                      final = (1 - meta_w) * hybrid + meta_w * metadata_sim
                      Set to 0.0 to skip metadata re-ranking entirely.
    candidate_count : number of candidates retrieved before metadata re-ranking
    exclude_idx     : additional indices to exclude (seed is always excluded)

    Returns
    ───────
    List of DataFrame indices, ordered by descending final score.
    Length ≤ n.
    """
    if abs(tfidf_w + semantic_w - 1.0) > 1e-6:
        raise ValueError(
            f"tfidf_w ({tfidf_w}) + semantic_w ({semantic_w}) must equal 1.0"
        )

    excluded = (exclude_idx or set()) | {seed_idx}

    # Step 1 — Compute hybrid retrieval score for all candidates
    hybrid_scores = (
        tfidf_w    * _tfidf_sim()[seed_idx]
        + semantic_w * _semantic_sim()[seed_idx]
    ).copy()
    hybrid_scores[list(excluded)] = -1.0

    # Step 2 — Retrieve top candidates
    top_candidates = np.argsort(hybrid_scores)[::-1]
    candidates = [int(i) for i in top_candidates if i not in excluded][:candidate_count]

    # Step 3 — Optional metadata re-ranking
    if meta_w > 0.0:
        reranked = []
        for cand_idx in candidates:
            h = float(hybrid_scores[cand_idx])
            m = metadata_similarity(seed_idx, cand_idx)
            final = (1.0 - meta_w) * h + meta_w * m
            reranked.append((cand_idx, final))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in reranked[:n]]
    else:
        return candidates[:n]
