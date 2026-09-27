"""
evaluation/config.py
────────────────────
Central configuration for all offline evaluation experiments.

Changing values here propagates to every experiment and result file,
making the evaluation reproducible and easy to audit.
"""

from dataclasses import dataclass, field


# ── Reproducibility ───────────────────────────────────────────────────────────

RANDOM_SEED: int = 42

# ── Evaluation dataset ────────────────────────────────────────────────────────

# Number of seed movies to use as evaluation queries.
# Each seed movie generates one recommendation list that is then measured.
# 100 seeds gives statistically stable averages without excessive runtime.
NUM_QUERIES: int = 100

# Minimum number of keywords a seed movie must have.
# Movies with no keywords produce uninformative proxy-relevance assessments.
MIN_KEYWORDS: int = 3

# Minimum vote count — filters out low-visibility movies
# that may have sparse metadata.
MIN_VOTE_COUNT: int = 50

# ── Metric configuration ──────────────────────────────────────────────────────

# K values for Precision@K and NDCG@K
K_VALUES: list[int] = [5, 10]

# ── Proxy relevance thresholds ────────────────────────────────────────────────
# A candidate is considered "relevant" if its proxy relevance score
# meets or exceeds RELEVANCE_THRESHOLD.
#
# The proxy score is computed as a weighted combination of
# genre overlap, keyword overlap, cast overlap, and director match.
# This mirrors the metadata_similarity function used by the recommender,
# so it is subject to data leakage — see evaluation/README.md.

# Weights for the proxy relevance score (must sum to 1.0)
#
# Design rationale
# ─────────────────
# Genres are too coarse: a threshold based only on genre overlap would mark
# 30–70% of the catalog as "relevant" for any seed movie (e.g., all Action
# movies for The Dark Knight), making the metric near-trivially satisfiable.
#
# Keywords are more specific (21 keywords for Avatar vs 4 genres), so a
# high keyword weight produces a much more discriminating relevance signal.
#
# Director is downweighted (0.05) because most directors have very few films
# in the dataset, making director match a near-binary signal that distorts
# aggregate metrics.
PROXY_WEIGHTS: dict[str, float] = {
    "genre":    0.20,
    "keyword":  0.55,   # dominant weight — keywords are thematically specific
    "cast":     0.20,
    "director": 0.05,
}

# Binary relevance threshold: proxy_score >= THRESHOLD → relevant
# At 0.20 with keyword-dominant weights, a movie must share a meaningful
# fraction of keywords (or combined genre+cast signals) to be relevant.
# Empirically yields ~50–300 relevant movies per seed (vs 1000–3500 at 0.10).
RELEVANCE_THRESHOLD: float = 0.20

# ── Recommendations per query ─────────────────────────────────────────────────

N_RECOMMENDATIONS: int = 10   # max list length returned per query

# ── Ablation experiment configurations ───────────────────────────────────────

# Each entry: (label, tfidf_weight, semantic_weight, metadata_weight)
# tfidf_weight + semantic_weight must equal 1.0
# metadata_weight is the share of the final score given to metadata re-ranking.
# final_score = (1 - metadata_weight) * hybrid + metadata_weight * metadata_sim
# where hybrid = tfidf_weight * tfidf_sim + semantic_weight * semantic_sim

ABLATION_CONFIGS: list[dict] = [
    # Pure retrieval signals
    {"label": "TF-IDF only",          "tfidf_w": 1.00, "semantic_w": 0.00, "meta_w": 0.00},
    {"label": "Semantic only",         "tfidf_w": 0.00, "semantic_w": 1.00, "meta_w": 0.00},

    # Hybrid weight sweep (no metadata)
    {"label": "Hybrid 25/75",          "tfidf_w": 0.25, "semantic_w": 0.75, "meta_w": 0.00},
    {"label": "Hybrid 50/50",          "tfidf_w": 0.50, "semantic_w": 0.50, "meta_w": 0.00},
    {"label": "Hybrid 75/25",          "tfidf_w": 0.75, "semantic_w": 0.25, "meta_w": 0.00},

    # Effect of metadata re-ranking on the current hybrid (50/50)
    {"label": "Hybrid 50/50 + Meta",   "tfidf_w": 0.50, "semantic_w": 0.50, "meta_w": 0.10},

    # Production configuration
    {"label": "Production (50/50+10%)", "tfidf_w": 0.50, "semantic_w": 0.50, "meta_w": 0.10},
]


@dataclass
class EvalConfig:
    """Snapshot of active evaluation configuration for reproducibility."""
    random_seed:         int              = RANDOM_SEED
    num_queries:         int              = NUM_QUERIES
    min_keywords:        int              = MIN_KEYWORDS
    min_vote_count:      int              = MIN_VOTE_COUNT
    k_values:            list[int]        = field(default_factory=lambda: list(K_VALUES))
    proxy_weights:       dict[str, float] = field(default_factory=lambda: dict(PROXY_WEIGHTS))
    relevance_threshold: float            = RELEVANCE_THRESHOLD
    n_recommendations:   int              = N_RECOMMENDATIONS
    embedding_model:     str              = "all-MiniLM-L6-v2"
    dataset:             str              = "TMDB 5000 Movie Dataset"
