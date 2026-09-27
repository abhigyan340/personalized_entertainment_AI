"""
evaluation/relevance.py
───────────────────────
Proxy relevance definition for offline evaluation.

⚠️  DATA LEAKAGE NOTICE
────────────────────────
This evaluation uses metadata from the same dataset that the recommender
uses as features (genre_list, keyword_list, cast_list, director).

Consequence: the evaluation measures *metadata consistency* — whether the
recommender returns movies with similar metadata — rather than *human
preference* or *real-world recommendation quality*.

The evaluation is NOT a substitute for user studies or online A/B testing.
It provides an internal, reproducible consistency check.

The label "relevant" below means:
  "metadata-similar according to our proxy definition"
  NOT "something a real user would enjoy watching".

Relevance definition
────────────────────
proxy_score(seed, candidate) =
    weights["genre"]    × genre_overlap(seed, candidate)
  + weights["keyword"]  × keyword_overlap(seed, candidate)
  + weights["cast"]     × cast_overlap(seed, candidate)
  + weights["director"] × director_match(seed, candidate)

where each overlap is precision-style: |A ∩ B| / |A|  (0.0 if A is empty).

A candidate is binary-relevant if proxy_score >= RELEVANCE_THRESHOLD.

Design choices
──────────────
- Keyword weight (0.55) is the dominant signal because keywords in the
  TMDB 5000 dataset are thematically specific (avg ~15 per movie), making
  keyword overlap a far more discriminating relevance signal than genre
  overlap. Using genre as the primary signal marks 30-70% of the catalog
  as relevant for any seed (e.g., every Action film for The Dark Knight),
  making metrics near-trivially satisfiable.
- Director match is weighted very low (0.05) because most directors have
  only 1-3 films in the dataset; matching on director alone would make the
  relevant set extremely small and unstable.
- The threshold (0.20) yields approximately 50-300 relevant movies per seed,
  giving metrics that are informative and discriminating.

Vectorization
─────────────
`relevant_set_vec()` is a NumPy-vectorized equivalent of `relevant_set()`.
It pre-computes binary indicator matrices (movies × vocabulary) once at
import time and computes all proxy scores for a seed as a single
matrix-vector dot product — reducing per-query time from ~9 seconds to
~2 milliseconds (~4000× speedup on 4,803 movies).

The vectorized and scalar implementations are tested for equivalence in
tests/test_evaluation.py::TestVectorizationEquivalence.
"""

import random
from functools import lru_cache
from typing import NamedTuple

import numpy as np

from evaluation.config import (
    EvalConfig,
    MIN_KEYWORDS,
    MIN_VOTE_COUNT,
    NUM_QUERIES,
    PROXY_WEIGHTS,
    RANDOM_SEED,
    RELEVANCE_THRESHOLD,
)
from src.load_model import movies
from src.recommender import _as_list


# ═══════════════════════════════════════════════════════════════════
# SCALAR REFERENCE IMPLEMENTATION (preserved for correctness checks)
# ═══════════════════════════════════════════════════════════════════

def _genre_overlap(seed_idx: int, cand_idx: int) -> float:
    src = set(_as_list(movies.iloc[seed_idx]["genre_list"]))
    tgt = set(_as_list(movies.iloc[cand_idx]["genre_list"]))
    return len(src & tgt) / len(src) if src else 0.0


def _keyword_overlap(seed_idx: int, cand_idx: int) -> float:
    src = set(_as_list(movies.iloc[seed_idx]["keyword_list"]))
    tgt = set(_as_list(movies.iloc[cand_idx]["keyword_list"]))
    return len(src & tgt) / len(src) if src else 0.0


def _cast_overlap(seed_idx: int, cand_idx: int) -> float:
    src = set(_as_list(movies.iloc[seed_idx]["cast_list"]))
    tgt = set(_as_list(movies.iloc[cand_idx]["cast_list"]))
    return len(src & tgt) / len(src) if src else 0.0


def _director_match(seed_idx: int, cand_idx: int) -> float:
    d1 = movies.iloc[seed_idx]["director"]
    d2 = movies.iloc[cand_idx]["director"]
    return 1.0 if (d1 and d2 and d1 == d2) else 0.0


def proxy_relevance_score(
    seed_idx: int,
    candidate_idx: int,
    weights: dict[str, float] | None = None,
) -> float:
    """
    Scalar proxy relevance score between seed and candidate.

    Parameters
    ----------
    seed_idx      : DataFrame index of the seed/query movie
    candidate_idx : DataFrame index of the candidate movie
    weights       : optional weight override (default: PROXY_WEIGHTS from config)

    Returns a float in [0, 1].
    """
    w = weights if weights is not None else PROXY_WEIGHTS
    return (
        w.get("genre",    0.40) * _genre_overlap(seed_idx, candidate_idx)
        + w.get("keyword",  0.30) * _keyword_overlap(seed_idx, candidate_idx)
        + w.get("cast",     0.20) * _cast_overlap(seed_idx, candidate_idx)
        + w.get("director", 0.10) * _director_match(seed_idx, candidate_idx)
    )


def is_relevant(
    seed_idx: int,
    candidate_idx: int,
    threshold: float = RELEVANCE_THRESHOLD,
    weights: dict[str, float] | None = None,
) -> bool:
    """Return True if the candidate is proxy-relevant for the seed (scalar)."""
    return proxy_relevance_score(seed_idx, candidate_idx, weights) >= threshold


def relevant_set(
    seed_idx: int,
    exclude_self: bool = True,
    threshold: float = RELEVANCE_THRESHOLD,
    weights: dict[str, float] | None = None,
) -> set[int]:
    """
    Scalar (reference) implementation — preserved for equivalence checks.

    Returns the set of all DataFrame indices that are proxy-relevant for seed.
    This is O(N × |vocabulary|) in Python and takes ~9s per seed at N=4803.
    Use relevant_set_vec() for production evaluation.
    """
    result = set()
    for idx in range(len(movies)):
        if exclude_self and idx == seed_idx:
            continue
        if is_relevant(seed_idx, idx, threshold, weights):
            result.add(idx)
    return result


# ═══════════════════════════════════════════════════════════════════
# VECTORIZED IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════

class _ProxyMatrices:
    """
    Pre-computed binary indicator matrices for vectorized proxy scoring.

    Built once at first use (lazy, via build_proxy_matrices()).

    Attributes
    ──────────
    genre_mat    : (N, G)   float32  — binary genre membership
    keyword_mat  : (N, K)   float32  — binary keyword membership
    cast_mat     : (N, C)   float32  — binary cast membership
    genre_sizes  : (N,)     float32  — |genre_list| per movie (denominator)
    keyword_sizes: (N,)     float32  — |keyword_list| per movie
    cast_sizes   : (N,)     float32  — |cast_list| per movie
    director_arr : (N,)     object   — director string per movie
    """

    def __init__(self):
        n = len(movies)

        # ── Build vocabularies ────────────────────────────────────────────────
        genre_vocab   = {}
        keyword_vocab = {}
        cast_vocab    = {}

        for i in range(n):
            row = movies.iloc[i]
            for g in _as_list(row["genre_list"]):
                if g not in genre_vocab:
                    genre_vocab[g] = len(genre_vocab)
            for k in _as_list(row["keyword_list"]):
                if k not in keyword_vocab:
                    keyword_vocab[k] = len(keyword_vocab)
            for c in _as_list(row["cast_list"]):
                if c not in cast_vocab:
                    cast_vocab[c] = len(cast_vocab)

        # ── Build binary indicator matrices ───────────────────────────────────
        G = len(genre_vocab)
        K = len(keyword_vocab)
        C = len(cast_vocab)

        genre_mat   = np.zeros((n, G), dtype=np.float32)
        keyword_mat = np.zeros((n, K), dtype=np.float32)
        cast_mat    = np.zeros((n, C), dtype=np.float32)
        director_arr = np.empty(n, dtype=object)

        for i in range(n):
            row = movies.iloc[i]
            for g in _as_list(row["genre_list"]):
                genre_mat[i, genre_vocab[g]] = 1.0
            for k in _as_list(row["keyword_list"]):
                keyword_mat[i, keyword_vocab[k]] = 1.0
            for c in _as_list(row["cast_list"]):
                cast_mat[i, cast_vocab[c]] = 1.0
            director_arr[i] = row["director"] or ""

        self.genre_mat    = genre_mat
        self.keyword_mat  = keyword_mat
        self.cast_mat     = cast_mat
        self.genre_sizes  = genre_mat.sum(axis=1)    # (N,)
        self.keyword_sizes = keyword_mat.sum(axis=1) # (N,)
        self.cast_sizes   = cast_mat.sum(axis=1)     # (N,)
        self.director_arr = director_arr


# Module-level singleton — built lazily on first call to relevant_set_vec()
_matrices: _ProxyMatrices | None = None


def build_proxy_matrices() -> _ProxyMatrices:
    """
    Build (or return cached) pre-computed binary indicator matrices.

    Building takes ~10-15 seconds once.  All subsequent calls are instant.
    """
    global _matrices
    if _matrices is None:
        import logging
        logging.getLogger(__name__).info(
            "Building proxy matrices (one-time, ~10-15s)..."
        )
        _matrices = _ProxyMatrices()
        logging.getLogger(__name__).info("Proxy matrices ready.")
    return _matrices


def proxy_scores_vec(
    seed_idx: int,
    weights: dict[str, float] | None = None,
    mats: _ProxyMatrices | None = None,
) -> np.ndarray:
    """
    Compute proxy relevance scores for all N movies against a single seed.

    Returns a (N,) float32 array of proxy scores.
    This is the vectorized equivalent of calling proxy_relevance_score(seed, i)
    for every i in range(N).

    Parameters
    ----------
    seed_idx : DataFrame index of the seed movie
    weights  : optional weight override (default: PROXY_WEIGHTS)
    mats     : pre-built _ProxyMatrices (uses/builds module singleton if None)
    """
    w = weights if weights is not None else PROXY_WEIGHTS
    m = mats if mats is not None else build_proxy_matrices()

    n = len(movies)

    # ── Genre overlap: |seed_genres ∩ cand_genres| / |seed_genres| ───────────
    seed_genres = m.genre_mat[seed_idx]          # (G,) binary row
    n_seed_g    = float(m.genre_sizes[seed_idx])
    if n_seed_g > 0:
        # dot product = |intersection| for binary vectors
        genre_scores = (m.genre_mat @ seed_genres) / n_seed_g  # (N,)
    else:
        genre_scores = np.zeros(n, dtype=np.float32)

    # ── Keyword overlap ───────────────────────────────────────────────────────
    seed_kw  = m.keyword_mat[seed_idx]
    n_seed_k = float(m.keyword_sizes[seed_idx])
    if n_seed_k > 0:
        keyword_scores = (m.keyword_mat @ seed_kw) / n_seed_k
    else:
        keyword_scores = np.zeros(n, dtype=np.float32)

    # ── Cast overlap ──────────────────────────────────────────────────────────
    seed_cast = m.cast_mat[seed_idx]
    n_seed_c  = float(m.cast_sizes[seed_idx])
    if n_seed_c > 0:
        cast_scores = (m.cast_mat @ seed_cast) / n_seed_c
    else:
        cast_scores = np.zeros(n, dtype=np.float32)

    # ── Director match ────────────────────────────────────────────────────────
    seed_dir = m.director_arr[seed_idx]
    if seed_dir:
        director_scores = (m.director_arr == seed_dir).astype(np.float32)
    else:
        director_scores = np.zeros(n, dtype=np.float32)

    # ── Weighted sum ──────────────────────────────────────────────────────────
    scores = (
        w.get("genre",    0.0) * genre_scores
        + w.get("keyword",  0.0) * keyword_scores
        + w.get("cast",     0.0) * cast_scores
        + w.get("director", 0.0) * director_scores
    )
    return scores  # (N,) float32


def relevant_set_vec(
    seed_idx: int,
    exclude_self: bool = True,
    threshold: float = RELEVANCE_THRESHOLD,
    weights: dict[str, float] | None = None,
    mats: _ProxyMatrices | None = None,
) -> set[int]:
    """
    Vectorized equivalent of relevant_set().

    Returns the same set of relevant DataFrame indices but in milliseconds
    instead of ~9 seconds, by using pre-computed binary indicator matrices.

    The scalar `relevant_set()` is preserved for correctness verification.
    """
    scores = proxy_scores_vec(seed_idx, weights=weights, mats=mats)
    if exclude_self:
        scores[seed_idx] = -1.0   # ensure seed is never in its own relevant set
    return set(int(i) for i in np.where(scores >= threshold)[0])


# ═══════════════════════════════════════════════════════════════════
# EVALUATION QUERY SET CONSTRUCTION  (uses vectorized path)
# ═══════════════════════════════════════════════════════════════════

class EvalQuery(NamedTuple):
    movie_idx:   int       # DataFrame index
    title:       str       # movie title
    relevant:    set[int]  # proxy-relevant DataFrame indices (excluding self)


def build_query_set(
    n: int               = NUM_QUERIES,
    min_keywords: int    = MIN_KEYWORDS,
    min_votes: int       = MIN_VOTE_COUNT,
    seed: int            = RANDOM_SEED,
    threshold: float     = RELEVANCE_THRESHOLD,
    weights: dict | None = None,
) -> list[EvalQuery]:
    """
    Sample *n* evaluation queries from movies that have sufficient metadata.

    Uses the vectorized relevance implementation (relevant_set_vec).
    Proxy matrices are built once and reused for every query.

    Selection criteria
    ──────────────────
    - At least *min_keywords* keywords (ensures non-trivial proxy relevance)
    - At least *min_votes* vote count (filters very obscure titles)
    - The resulting relevant set must be non-empty (ensures the query is
      evaluable — a movie with zero proxy-relevant matches cannot produce
      meaningful precision/NDCG scores)

    Sampling is deterministic given the same seed.
    """
    rng = random.Random(seed)

    eligible = []
    for idx in range(len(movies)):
        row = movies.iloc[idx]
        kw_count   = len(_as_list(row["keyword_list"]))
        vote_count = row.get("vote_count", 0) or 0
        if kw_count >= min_keywords and vote_count >= min_votes:
            eligible.append(idx)

    rng.shuffle(eligible)

    # Pre-build matrices once for all queries
    mats = build_proxy_matrices()

    queries: list[EvalQuery] = []
    for idx in eligible:
        if len(queries) >= n:
            break
        rel = relevant_set_vec(
            idx, threshold=threshold, weights=weights, mats=mats
        )
        if rel:
            queries.append(EvalQuery(
                movie_idx=idx,
                title=str(movies.iloc[idx]["title"]),
                relevant=rel,
            ))

    if len(queries) < n:
        import logging
        logging.getLogger(__name__).warning(
            "Only %d/%d eligible queries found. "
            "Consider lowering MIN_KEYWORDS, MIN_VOTE_COUNT, or RELEVANCE_THRESHOLD.",
            len(queries), n,
        )

    return queries
