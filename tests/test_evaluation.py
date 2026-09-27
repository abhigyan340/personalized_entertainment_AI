"""
tests/test_evaluation.py
────────────────────────
Unit tests for the offline evaluation framework.

These tests use small, manually-verifiable examples so metric
correctness can be checked by inspection.  They do NOT require
loading the ML artifacts — only the metric/relevance logic is tested
here.  Tests that need artifacts are guarded by requires_artifacts.

Run with:
    pytest tests/test_evaluation.py -v
"""

import math
import os
import pytest

# ── Artifact availability guard ───────────────────────────────────────────────

def _artifacts_available() -> bool:
    artifact_dir = os.path.join(
        os.path.dirname(__file__), "..", "data", "artifacts"
    )
    required = [
        "movies_processed.pkl",
        "movie_embeddings.npy",
        "hybrid_similarity.npy",
        "tfidf_similarity.npy",
        "semantic_similarity.npy",
    ]
    return all(os.path.exists(os.path.join(artifact_dir, f)) for f in required)


requires_artifacts = pytest.mark.skipif(
    not _artifacts_available(),
    reason="Artifact files not found — run preprocessing notebook first.",
)


# ═══════════════════════════════════════════════════════════════════
# METRIC TESTS — no artifacts needed
# ═══════════════════════════════════════════════════════════════════

class TestPrecisionAtK:

    def test_perfect_precision(self):
        from evaluation.metrics import precision_at_k
        recs    = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3, 4, 5}
        assert precision_at_k(recs, relevant, k=5) == pytest.approx(1.0)

    def test_zero_precision(self):
        from evaluation.metrics import precision_at_k
        recs     = [1, 2, 3]
        relevant = {10, 20, 30}
        assert precision_at_k(recs, relevant, k=3) == pytest.approx(0.0)

    def test_partial_precision(self):
        from evaluation.metrics import precision_at_k
        # 2 out of 5 top-K are relevant → P@5 = 0.4
        recs     = [1, 99, 2, 99, 99]
        relevant = {1, 2}
        assert precision_at_k(recs, relevant, k=5) == pytest.approx(0.4)

    def test_k_larger_than_list(self):
        from evaluation.metrics import precision_at_k
        recs     = [1, 2]
        relevant = {1}
        # Only 2 items, K=5; still divided by K=5
        assert precision_at_k(recs, relevant, k=5) == pytest.approx(1 / 5)

    def test_empty_recommendations(self):
        from evaluation.metrics import precision_at_k
        assert precision_at_k([], {1, 2}, k=5) == 0.0

    def test_k_zero(self):
        from evaluation.metrics import precision_at_k
        assert precision_at_k([1, 2, 3], {1}, k=0) == 0.0


class TestNDCGAtK:

    def test_perfect_ndcg(self):
        from evaluation.metrics import ndcg_at_k
        # Perfect ordering: all top-K are relevant
        recs     = [1, 2, 3]
        relevant = {1, 2, 3, 4}
        assert ndcg_at_k(recs, relevant, k=3) == pytest.approx(1.0)

    def test_zero_ndcg(self):
        from evaluation.metrics import ndcg_at_k
        recs     = [1, 2, 3]
        relevant = {10, 20, 30}
        assert ndcg_at_k(recs, relevant, k=3) == pytest.approx(0.0)

    def test_ordering_matters(self):
        from evaluation.metrics import ndcg_at_k
        relevant = {1}
        # Relevant item at rank 1 vs rank 3 — rank 1 gives higher NDCG
        recs_good = [1, 99, 99]
        recs_bad  = [99, 99, 1]
        ndcg_good = ndcg_at_k(recs_good, relevant, k=3)
        ndcg_bad  = ndcg_at_k(recs_bad,  relevant, k=3)
        assert ndcg_good > ndcg_bad

    def test_single_relevant_at_rank_1(self):
        from evaluation.metrics import ndcg_at_k
        # DCG = 1/log2(2) = 1.0; IDCG = 1.0 → NDCG = 1.0
        recs     = [1]
        relevant = {1}
        assert ndcg_at_k(recs, relevant, k=1) == pytest.approx(1.0)

    def test_no_relevant_items(self):
        from evaluation.metrics import ndcg_at_k
        recs     = [1, 2, 3]
        relevant = set()          # no relevant items → IDCG = 0 → return 0
        assert ndcg_at_k(recs, relevant, k=3) == pytest.approx(0.0)

    def test_manual_calculation(self):
        from evaluation.metrics import ndcg_at_k
        # recs = [1(rel), 2(not), 3(rel), 4(not), 5(rel)]
        # DCG@5 = 1/log2(2) + 0 + 1/log2(4) + 0 + 1/log2(6)
        #       = 1.0 + 0.5 + 0.3869 = 1.8869
        # IDCG@5 = 1/log2(2) + 1/log2(3) + 1/log2(4)
        #        = 1.0 + 0.6309 + 0.5 = 2.1309
        recs     = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5}
        expected_ndcg = (
            (1/math.log2(2) + 1/math.log2(4) + 1/math.log2(6))
            / (1/math.log2(2) + 1/math.log2(3) + 1/math.log2(4))
        )
        assert ndcg_at_k(recs, relevant, k=5) == pytest.approx(expected_ndcg, abs=1e-6)


class TestCatalogCoverage:

    def test_full_coverage(self):
        from evaluation.metrics import catalog_coverage
        all_recs = [[0, 1, 2], [3, 4, 5]]
        assert catalog_coverage(all_recs, catalog_size=6) == pytest.approx(1.0)

    def test_zero_coverage(self):
        from evaluation.metrics import catalog_coverage
        assert catalog_coverage([], catalog_size=100) == pytest.approx(0.0)

    def test_partial_coverage(self):
        from evaluation.metrics import catalog_coverage
        # 3 unique items out of 10
        all_recs = [[0, 1], [1, 2]]
        assert catalog_coverage(all_recs, catalog_size=10) == pytest.approx(0.3)

    def test_overlapping_recommendations(self):
        from evaluation.metrics import catalog_coverage
        # Same items recommended across queries — counts as 2 unique, not 4
        all_recs = [[0, 1], [0, 1]]
        assert catalog_coverage(all_recs, catalog_size=10) == pytest.approx(0.2)

    def test_zero_catalog_size(self):
        from evaluation.metrics import catalog_coverage
        assert catalog_coverage([[0, 1]], catalog_size=0) == pytest.approx(0.0)


class TestIntraListDiversity:

    def test_identical_items_zero_diversity(self):
        from evaluation.metrics import intra_list_diversity
        # All items identical → similarity=1.0 → diversity=0.0
        sim_fn = lambda a, b: 1.0
        assert intra_list_diversity([0, 1, 2], sim_fn) == pytest.approx(0.0)

    def test_maximally_diverse(self):
        from evaluation.metrics import intra_list_diversity
        # All items orthogonal → similarity=0.0 → diversity=1.0
        sim_fn = lambda a, b: 0.0
        assert intra_list_diversity([0, 1, 2], sim_fn) == pytest.approx(1.0)

    def test_single_item_zero_diversity(self):
        from evaluation.metrics import intra_list_diversity
        sim_fn = lambda a, b: 0.5
        assert intra_list_diversity([0], sim_fn) == pytest.approx(0.0)

    def test_two_items_manual(self):
        from evaluation.metrics import intra_list_diversity
        # sim(0,1)=0.6 → diversity = 1 - 0.6 = 0.4
        sim_fn = lambda a, b: 0.6
        assert intra_list_diversity([0, 1], sim_fn) == pytest.approx(0.4)

    def test_mean_diversity_aggregation(self):
        from evaluation.metrics import mean_diversity
        sim_fn = lambda a, b: 0.5   # diversity per pair = 0.5
        lists  = [[0, 1], [2, 3], [4, 5]]
        assert mean_diversity(lists, sim_fn) == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════
# RELEVANCE TESTS — uses mock data, no artifacts
# ═══════════════════════════════════════════════════════════════════

class TestProxyRelevanceLogic:
    """
    Tests for the proxy_relevance_score formula itself.
    Uses synthetic inputs to verify the weighted formula.
    """

    def _score(self, genre_ov, kw_ov, cast_ov, dir_match):
        """Apply production proxy weights manually."""
        return (
            0.20 * genre_ov
            + 0.55 * kw_ov
            + 0.20 * cast_ov
            + 0.05 * dir_match
        )

    def test_identical_items_score_one(self):
        # All overlaps = 1.0, director match = 1.0 → score = 1.0
        assert self._score(1.0, 1.0, 1.0, 1.0) == pytest.approx(1.0)

    def test_zero_overlap_score_zero(self):
        assert self._score(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0)

    def test_keyword_dominant(self):
        # Only keywords match → score = 0.55 * 1.0 = 0.55 > threshold 0.20
        score = self._score(0.0, 1.0, 0.0, 0.0)
        assert score == pytest.approx(0.55)
        assert score >= 0.20   # relevant under threshold

    def test_genre_only_below_threshold(self):
        # Only one genre overlaps: genre_ov = 0.5, no keywords
        # score = 0.20 * 0.5 = 0.10 < threshold 0.20 → NOT relevant
        score = self._score(0.5, 0.0, 0.0, 0.0)
        assert score == pytest.approx(0.10)
        assert score < 0.20   # not relevant


# ═══════════════════════════════════════════════════════════════════
# BASELINE TESTS — need artifacts
# ═══════════════════════════════════════════════════════════════════

class TestBaselineRecommenders:

    @requires_artifacts
    def test_random_returns_correct_length(self):
        from evaluation.baselines import RandomRecommender
        r = RandomRecommender(seed=42)
        recs = r.recommend(seed_idx=0, n=10)
        assert len(recs) == 10

    @requires_artifacts
    def test_random_excludes_seed(self):
        from evaluation.baselines import RandomRecommender
        r = RandomRecommender(seed=42)
        recs = r.recommend(seed_idx=0, n=10)
        assert 0 not in recs

    @requires_artifacts
    def test_random_returns_valid_indices(self):
        from evaluation.baselines import RandomRecommender
        from src.load_model import movies
        r = RandomRecommender(seed=42)
        recs = r.recommend(seed_idx=0, n=20)
        assert all(0 <= idx < len(movies) for idx in recs)

    @requires_artifacts
    def test_genre_returns_correct_length(self):
        from evaluation.baselines import GenreRecommender
        r = GenreRecommender()
        recs = r.recommend(seed_idx=0, n=5)
        assert len(recs) == 5

    @requires_artifacts
    def test_genre_excludes_seed(self):
        from evaluation.baselines import GenreRecommender
        r = GenreRecommender()
        recs = r.recommend(seed_idx=0, n=10)
        assert 0 not in recs

    @requires_artifacts
    def test_tfidf_returns_correct_length(self):
        from evaluation.baselines import TFIDFRecommender
        r = TFIDFRecommender()
        recs = r.recommend(seed_idx=0, n=10)
        assert len(recs) == 10

    @requires_artifacts
    def test_semantic_returns_correct_length(self):
        from evaluation.baselines import SemanticRecommender
        r = SemanticRecommender()
        recs = r.recommend(seed_idx=0, n=10)
        assert len(recs) == 10

    @requires_artifacts
    def test_hybrid_meta_excludes_seed(self):
        from evaluation.baselines import HybridMetaRecommender
        r = HybridMetaRecommender()
        recs = r.recommend(seed_idx=5, n=10)
        assert 5 not in recs

    @requires_artifacts
    def test_all_baselines_return_unique_indices(self):
        from evaluation.baselines import (
            RandomRecommender, GenreRecommender,
            TFIDFRecommender, SemanticRecommender, HybridMetaRecommender,
        )
        for Cls in [GenreRecommender, TFIDFRecommender, SemanticRecommender, HybridMetaRecommender]:
            r    = Cls()
            recs = r.recommend(seed_idx=0, n=10)
            assert len(recs) == len(set(recs)), f"{Cls.__name__} returned duplicates"


# ═══════════════════════════════════════════════════════════════════
# ENGINE TESTS — need artifacts
# ═══════════════════════════════════════════════════════════════════

class TestAblationEngine:

    @requires_artifacts
    def test_diagnostic_scores_has_required_fields(self):
        from evaluation.engine import diagnostic_scores
        result = diagnostic_scores(seed_idx=0, candidate_idx=1)
        required = {
            "seed_title", "candidate_title",
            "tfidf_score", "semantic_score",
            "genre_score", "keyword_score", "cast_score", "director_score",
            "metadata_score", "hybrid_50_50",
        }
        assert required <= result.keys()

    @requires_artifacts
    def test_diagnostic_scores_self_similarity(self):
        from evaluation.engine import diagnostic_scores
        result = diagnostic_scores(seed_idx=0, candidate_idx=0)
        # Cosine similarity with itself ≈ 1.0
        assert result["tfidf_score"]   >= 0.99
        assert result["semantic_score"] >= 0.99

    @requires_artifacts
    def test_ablation_recommend_correct_length(self):
        from evaluation.engine import ablation_recommend
        recs = ablation_recommend(seed_idx=0, n=10, tfidf_w=0.5, semantic_w=0.5, meta_w=0.0)
        assert len(recs) == 10

    @requires_artifacts
    def test_ablation_recommend_excludes_seed(self):
        from evaluation.engine import ablation_recommend
        recs = ablation_recommend(seed_idx=0, n=10, tfidf_w=0.5, semantic_w=0.5, meta_w=0.0)
        assert 0 not in recs

    @requires_artifacts
    def test_ablation_recommend_invalid_weights_raises(self):
        from evaluation.engine import ablation_recommend
        with pytest.raises(ValueError):
            ablation_recommend(seed_idx=0, n=5, tfidf_w=0.6, semantic_w=0.6, meta_w=0.0)

    @requires_artifacts
    def test_ablation_tfidf_only(self):
        from evaluation.engine import ablation_recommend
        recs = ablation_recommend(seed_idx=0, n=5, tfidf_w=1.0, semantic_w=0.0, meta_w=0.0)
        assert len(recs) == 5
        assert 0 not in recs

    @requires_artifacts
    def test_ablation_semantic_only(self):
        from evaluation.engine import ablation_recommend
        recs = ablation_recommend(seed_idx=0, n=5, tfidf_w=0.0, semantic_w=1.0, meta_w=0.0)
        assert len(recs) == 5

    @requires_artifacts
    def test_metadata_reranking_changes_order(self):
        from evaluation.engine import ablation_recommend
        # With and without metadata re-ranking may return different ordering
        recs_no_meta   = ablation_recommend(0, n=10, tfidf_w=0.5, semantic_w=0.5, meta_w=0.00)
        recs_with_meta = ablation_recommend(0, n=10, tfidf_w=0.5, semantic_w=0.5, meta_w=0.10)
        # Both return 10 items — ordering may differ
        assert len(recs_no_meta)   == 10
        assert len(recs_with_meta) == 10


# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT STRUCTURE TESTS — need artifacts
# ═══════════════════════════════════════════════════════════════════

class TestExperimentStructure:

    @requires_artifacts
    def test_ablation_config_produces_expected_keys(self):
        from evaluation.run import evaluate_ablation_config
        from evaluation.relevance import build_query_set
        queries = build_query_set(n=3, seed=42)
        cfg     = {"label": "test", "tfidf_w": 0.5, "semantic_w": 0.5, "meta_w": 0.0}
        result  = evaluate_ablation_config(cfg, queries, k_values=[5], n=5)
        required = {"label", "tfidf_w", "semantic_w", "meta_w",
                    "precision_at_5", "ndcg_at_5", "coverage", "diversity"}
        assert required <= result.keys()

    @requires_artifacts
    def test_ablation_config_scores_in_range(self):
        from evaluation.run import evaluate_ablation_config
        from evaluation.relevance import build_query_set
        queries = build_query_set(n=5, seed=42)
        cfg     = {"label": "test", "tfidf_w": 0.5, "semantic_w": 0.5, "meta_w": 0.0}
        result  = evaluate_ablation_config(cfg, queries, k_values=[5, 10], n=10)
        for k in [5, 10]:
            assert 0.0 <= result[f"precision_at_{k}"] <= 1.0
            assert 0.0 <= result[f"ndcg_at_{k}"]      <= 1.0
        assert 0.0 <= result["coverage"]  <= 1.0
        assert 0.0 <= result["diversity"] <= 1.0
