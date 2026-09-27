"""
evaluation/metrics.py
─────────────────────
Ranking metrics for offline evaluation of the recommendation engine.

All functions operate on plain Python lists so they have no dependency
on the recommendation or data-loading modules.  They can therefore be
unit-tested without loading any ML artifacts.

Metrics implemented
───────────────────
  precision_at_k   — fraction of top-K that are relevant
  ndcg_at_k        — Normalized Discounted Cumulative Gain
  catalog_coverage — fraction of catalog ever recommended
  diversity        — mean intra-list diversity across all queries
"""

import math
from typing import Callable


# ── Precision@K ──────────────────────────────────────────────────────────────

def precision_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int,
) -> float:
    """
    Precision@K = |{recommended[:K]} ∩ relevant| / K

    Parameters
    ----------
    recommended : ordered list of candidate movie indices
    relevant    : set of movie indices considered relevant for this query
    k           : cutoff

    Returns 0.0 if the recommended list is empty or K == 0.
    """
    if not recommended or k == 0:
        return 0.0
    top_k = recommended[:k]
    hits  = sum(1 for idx in top_k if idx in relevant)
    return hits / k


# ── NDCG@K ───────────────────────────────────────────────────────────────────

def dcg_at_k(
    recommended: list[int],
    relevance_fn: Callable[[int], float],
    k: int,
) -> float:
    """
    Discounted Cumulative Gain at K.

    DCG@K = Σ_{i=1}^{K}  rel_i / log2(i + 1)

    Parameters
    ----------
    recommended  : ordered list of candidate movie indices
    relevance_fn : callable that maps a movie index → relevance score ≥ 0
    k            : cutoff
    """
    dcg = 0.0
    for rank, idx in enumerate(recommended[:k], start=1):
        rel  = relevance_fn(idx)
        dcg += rel / math.log2(rank + 1)
    return dcg


def ideal_dcg_at_k(
    relevant_scores: list[float],
    k: int,
) -> float:
    """
    Ideal DCG@K computed from a list of relevance scores
    (already sorted descending by the caller).
    """
    ideal = 0.0
    for rank, rel in enumerate(sorted(relevant_scores, reverse=True)[:k], start=1):
        ideal += rel / math.log2(rank + 1)
    return ideal


def ndcg_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int,
    graded: bool = False,
    grade_fn: Callable[[int], float] | None = None,
) -> float:
    """
    Normalized DCG@K.

    Binary relevance (graded=False, default)
    ─────────────────────────────────────────
    rel_i = 1  if recommended[i] ∈ relevant  else  0
    IDCG  = DCG of the ideal list of |relevant| ones followed by zeros.

    Graded relevance (graded=True)
    ──────────────────────────────
    rel_i = grade_fn(recommended[i])
    Caller must also supply grade_fn.
    All ideal grades are taken from the current query's relevant set.

    Returns 0.0 if IDCG == 0 (no relevant items exist for this query).
    """
    if not recommended or k == 0:
        return 0.0

    if graded and grade_fn is not None:
        rel_fn = grade_fn
        # Ideal grades: apply grade_fn to every relevant item
        ideal_grades = [grade_fn(r) for r in relevant]
    else:
        rel_fn       = lambda idx: 1.0 if idx in relevant else 0.0
        ideal_grades = [1.0] * len(relevant)

    actual_dcg = dcg_at_k(recommended, rel_fn, k)
    ideal      = ideal_dcg_at_k(ideal_grades, k)

    return actual_dcg / ideal if ideal > 0 else 0.0


# ── Catalog Coverage ──────────────────────────────────────────────────────────

def catalog_coverage(
    all_recommendations: list[list[int]],
    catalog_size: int,
) -> float:
    """
    Fraction of the catalog that appears in at least one recommendation list.

    Coverage@∞ = |⋃ recommended_i| / |catalog|

    High coverage indicates the recommender surfaces diverse catalog items.
    Low coverage (coverage bias) indicates the recommender repeatedly
    recommends the same popular subset.

    Note: this metric does NOT measure quality — high coverage with poor
    relevance is not desirable.

    Parameters
    ----------
    all_recommendations : list of recommendation lists (one per query),
                          each a list of movie indices
    catalog_size        : total number of movies in the catalog

    Returns 0.0 if catalog_size == 0.
    """
    if catalog_size == 0:
        return 0.0
    unique_recommended = set()
    for recs in all_recommendations:
        unique_recommended.update(recs)
    return len(unique_recommended) / catalog_size


# ── Intra-List Diversity ──────────────────────────────────────────────────────

def intra_list_diversity(
    recommended: list[int],
    similarity_fn: Callable[[int, int], float],
) -> float:
    """
    Mean pairwise dissimilarity within a single recommendation list.

    ILD = (2 / (K*(K-1))) * Σ_{i<j} (1 - similarity(i, j))

    A value near 1.0 means all recommended items are very different.
    A value near 0.0 means all recommended items are very similar.

    Parameters
    ----------
    recommended   : list of movie indices (the recommendation list)
    similarity_fn : callable (idx_a, idx_b) → float in [0, 1]
    """
    k = len(recommended)
    if k < 2:
        return 0.0

    total    = 0.0
    n_pairs  = 0
    for i in range(k):
        for j in range(i + 1, k):
            sim     = similarity_fn(recommended[i], recommended[j])
            total  += (1.0 - sim)
            n_pairs += 1

    return total / n_pairs if n_pairs > 0 else 0.0


def mean_diversity(
    all_recommendations: list[list[int]],
    similarity_fn: Callable[[int, int], float],
) -> float:
    """
    Mean intra-list diversity across all queries.

    Parameters
    ----------
    all_recommendations : list of recommendation lists (one per query)
    similarity_fn       : callable (idx_a, idx_b) → float in [0, 1]
    """
    if not all_recommendations:
        return 0.0
    scores = [
        intra_list_diversity(recs, similarity_fn)
        for recs in all_recommendations
        if len(recs) >= 2
    ]
    return sum(scores) / len(scores) if scores else 0.0


# ── Aggregate helper ──────────────────────────────────────────────────────────

def mean_metric(values: list[float]) -> float:
    """Return arithmetic mean of a list of metric values, or 0.0 if empty."""
    return sum(values) / len(values) if values else 0.0
