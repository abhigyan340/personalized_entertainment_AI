"""
evaluation/baselines.py
───────────────────────
Simple baseline recommenders used to contextualise the hybrid model's scores.

All recommenders follow the same interface:

    recommend(seed_idx, n, exclude_idx) -> list[int]

where:
  seed_idx    : DataFrame index of the seed/query movie
  n           : maximum number of recommendations to return
  exclude_idx : set of indices to exclude (at minimum the seed itself)

Returns a list of DataFrame indices (length ≤ n), ordered by descending score.

Baselines
─────────
  RandomRecommender    — random sample from the catalog
  GenreRecommender     — rank by genre overlap with the seed
  TFIDFRecommender     — rank by pre-computed TF-IDF cosine similarity
  SemanticRecommender  — rank by pre-computed semantic cosine similarity
"""

import random
import numpy as np

from evaluation.config import RANDOM_SEED
from src.load_model import movies
from src.recommender import _as_list


# ── Load the individual similarity matrices needed by baselines ───────────────
# These are NOT loaded by the production recommender (load_model.py)
# because they are only needed here.  We load them lazily so that importing
# this module does not trigger disk I/O unless a matrix-based baseline runs.

import os
import functools

_ARTIFACT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "artifacts",
)


@functools.lru_cache(maxsize=1)
def _tfidf_sim() -> np.ndarray:
    path = os.path.join(_ARTIFACT_DIR, "tfidf_similarity.npy")
    return np.load(path)


@functools.lru_cache(maxsize=1)
def _semantic_sim() -> np.ndarray:
    path = os.path.join(_ARTIFACT_DIR, "semantic_similarity.npy")
    return np.load(path)


# ── Base interface ────────────────────────────────────────────────────────────

class BaseRecommender:
    """Abstract base — subclasses implement recommend()."""

    name: str = "Base"

    def recommend(
        self,
        seed_idx: int,
        n: int = 10,
        exclude_idx: set[int] | None = None,
    ) -> list[int]:
        raise NotImplementedError


# ── Random baseline ───────────────────────────────────────────────────────────

class RandomRecommender(BaseRecommender):
    """
    Recommend *n* movies chosen uniformly at random from the catalog.

    Purpose: establishes a lower-bound reference point.
    A recommender that cannot consistently outperform random sampling
    on all metrics provides no value.
    """

    name = "Random"

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        self._rng = random.Random(seed)

    def recommend(
        self,
        seed_idx: int,
        n: int = 10,
        exclude_idx: set[int] | None = None,
    ) -> list[int]:
        excluded = (exclude_idx or set()) | {seed_idx}
        pool     = [i for i in range(len(movies)) if i not in excluded]
        return self._rng.sample(pool, min(n, len(pool)))


# ── Genre-only baseline ───────────────────────────────────────────────────────

class GenreRecommender(BaseRecommender):
    """
    Rank all candidates by genre overlap with the seed.

    Overlap = |seed_genres ∩ candidate_genres| / |seed_genres|

    This captures genre similarity without any semantic understanding.
    It represents the simplest possible content-based baseline.
    When multiple candidates share the same overlap score, they are
    broken arbitrarily by DataFrame index (deterministic).
    """

    name = "Genre-only"

    def recommend(
        self,
        seed_idx: int,
        n: int = 10,
        exclude_idx: set[int] | None = None,
    ) -> list[int]:
        excluded   = (exclude_idx or set()) | {seed_idx}
        seed_genres = set(_as_list(movies.iloc[seed_idx]["genre_list"]))

        if not seed_genres:
            # No genre information — fall back to arbitrary order
            return [i for i in range(len(movies)) if i not in excluded][:n]

        scored = []
        for idx in range(len(movies)):
            if idx in excluded:
                continue
            cand_genres = set(_as_list(movies.iloc[idx]["genre_list"]))
            overlap     = len(seed_genres & cand_genres) / len(seed_genres)
            scored.append((idx, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scored[:n]]


# ── TF-IDF only baseline ──────────────────────────────────────────────────────

class TFIDFRecommender(BaseRecommender):
    """
    Rank candidates by pre-computed TF-IDF cosine similarity.

    Uses the same tfidf_similarity.npy matrix produced during preprocessing.
    This isolates the contribution of the TF-IDF lexical signal.
    """

    name = "TF-IDF only"

    def recommend(
        self,
        seed_idx: int,
        n: int = 10,
        exclude_idx: set[int] | None = None,
    ) -> list[int]:
        excluded = (exclude_idx or set()) | {seed_idx}
        scores   = _tfidf_sim()[seed_idx].copy()
        scores[list(excluded)] = -1.0  # mask excluded indices
        top_n = np.argsort(scores)[::-1][:n + len(excluded)]
        return [int(i) for i in top_n if i not in excluded][:n]


# ── Semantic only baseline ────────────────────────────────────────────────────

class SemanticRecommender(BaseRecommender):
    """
    Rank candidates by pre-computed semantic cosine similarity.

    Uses the same semantic_similarity.npy matrix produced during preprocessing.
    This isolates the contribution of the SentenceTransformer semantic signal.
    """

    name = "Semantic only"

    def recommend(
        self,
        seed_idx: int,
        n: int = 10,
        exclude_idx: set[int] | None = None,
    ) -> list[int]:
        excluded = (exclude_idx or set()) | {seed_idx}
        scores   = _semantic_sim()[seed_idx].copy()
        scores[list(excluded)] = -1.0
        top_n = np.argsort(scores)[::-1][:n + len(excluded)]
        return [int(i) for i in top_n if i not in excluded][:n]


# ── Hybrid (no metadata) baseline ────────────────────────────────────────────

class HybridRecommender(BaseRecommender):
    """
    Rank candidates using the pre-computed hybrid similarity matrix only,
    with no metadata re-ranking step.

    hybrid_similarity = 0.5 × tfidf + 0.5 × semantic  (built in the notebook)

    This is the retrieval-only component of the production system.
    Comparing it against Hybrid+Metadata isolates the metadata contribution.
    """

    name = "Hybrid (no metadata)"

    def __init__(self, tfidf_w: float = 0.5, semantic_w: float = 0.5) -> None:
        assert abs(tfidf_w + semantic_w - 1.0) < 1e-6, \
            "tfidf_w + semantic_w must equal 1.0"
        self.tfidf_w    = tfidf_w
        self.semantic_w = semantic_w
        self.name       = f"Hybrid {tfidf_w:.2f}/{semantic_w:.2f} (no meta)"

    def recommend(
        self,
        seed_idx: int,
        n: int = 10,
        exclude_idx: set[int] | None = None,
    ) -> list[int]:
        excluded = (exclude_idx or set()) | {seed_idx}
        scores   = (
            self.tfidf_w    * _tfidf_sim()[seed_idx]
            + self.semantic_w * _semantic_sim()[seed_idx]
        ).copy()
        scores[list(excluded)] = -1.0
        top_n = np.argsort(scores)[::-1][:n + len(excluded)]
        return [int(i) for i in top_n if i not in excluded][:n]


# ── Hybrid + metadata re-ranking ──────────────────────────────────────────────

class HybridMetaRecommender(BaseRecommender):
    """
    Full production-equivalent recommender using the pre-computed matrices
    plus metadata re-ranking.

    final_score = (1 - meta_w) * hybrid + meta_w * metadata_similarity

    This mirrors the production recommender's scoring for in-dataset movies,
    but uses the pre-computed tfidf/semantic matrices directly (no TMDB calls,
    no poster fetching) so it runs efficiently in evaluation.
    """

    name = "Hybrid + Metadata"

    def __init__(
        self,
        tfidf_w:    float = 0.50,
        semantic_w: float = 0.50,
        meta_w:     float = 0.10,
        candidate_count: int = 50,
    ) -> None:
        assert abs(tfidf_w + semantic_w - 1.0) < 1e-6
        self.tfidf_w         = tfidf_w
        self.semantic_w      = semantic_w
        self.meta_w          = meta_w
        self.candidate_count = candidate_count
        self.name            = (
            f"Hybrid {tfidf_w:.2f}/{semantic_w:.2f} + Meta {meta_w:.0%}"
        )

    def recommend(
        self,
        seed_idx: int,
        n: int = 10,
        exclude_idx: set[int] | None = None,
    ) -> list[int]:
        from src.recommender import metadata_similarity

        excluded = (exclude_idx or set()) | {seed_idx}
        hybrid_scores = (
            self.tfidf_w    * _tfidf_sim()[seed_idx]
            + self.semantic_w * _semantic_sim()[seed_idx]
        ).copy()
        hybrid_scores[list(excluded)] = -1.0

        # Retrieve top-candidate_count candidates
        candidates = np.argsort(hybrid_scores)[::-1][:self.candidate_count + len(excluded)]
        candidates = [int(i) for i in candidates if i not in excluded][:self.candidate_count]

        # Re-rank with metadata
        reranked = []
        for cand_idx in candidates:
            h = float(hybrid_scores[cand_idx])
            m = metadata_similarity(seed_idx, cand_idx)
            final = (1.0 - self.meta_w) * h + self.meta_w * m
            reranked.append((cand_idx, final))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in reranked[:n]]
