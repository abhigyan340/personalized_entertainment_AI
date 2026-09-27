"""
tests/test_recommender.py
─────────────────────────
Unit and integration tests for the recommendation engine.

Run with:
    pytest tests/ -v

These tests load the real pre-computed artifacts from data/artifacts/.
They will be skipped automatically if the artifacts are missing (i.e., the
preprocessing notebook hasn't been run yet).
"""

import pytest

# ── Fixtures / skip guard ─────────────────────────────────────────────────────

def _artifacts_available() -> bool:
    """Return True only if the required artifact files exist on disk."""
    import os
    artifact_dir = os.path.join(
        os.path.dirname(__file__), "..", "data", "artifacts"
    )
    required = [
        "movies_processed.pkl",
        "movie_embeddings.npy",
        "hybrid_similarity.npy",
    ]
    return all(os.path.exists(os.path.join(artifact_dir, f)) for f in required)


requires_artifacts = pytest.mark.skipif(
    not _artifacts_available(),
    reason="Artifact files not found — run the preprocessing notebook first.",
)


@pytest.fixture(scope="module")
def recommender():
    """Import the recommender module once for the whole test module."""
    from src import recommender as rec
    return rec


# ── Helper / utility tests ────────────────────────────────────────────────────

class TestUtilityHelpers:
    """Tests for the small helper functions that don't need artifacts."""

    def test_as_list_with_none(self):
        from src.recommender import _as_list
        assert _as_list(None) == []

    def test_as_list_with_list(self):
        from src.recommender import _as_list
        assert _as_list(["a", "b"]) == ["a", "b"]

    def test_as_list_with_numpy_array(self):
        import numpy as np
        from src.recommender import _as_list
        arr = np.array([1, 2, 3])
        assert _as_list(arr) == [1, 2, 3]

    def test_safe_int_with_none(self):
        from src.recommender import _safe_int
        assert _safe_int(None) is None

    def test_safe_int_with_nan(self):
        import math
        from src.recommender import _safe_int
        assert _safe_int(float("nan")) is None

    def test_safe_int_normal(self):
        from src.recommender import _safe_int
        assert _safe_int(19995) == 19995
        assert _safe_int(19995.0) == 19995

    def test_safe_int_invalid_string(self):
        from src.recommender import _safe_int
        assert _safe_int("abc") is None


# ── Recommendation engine tests ───────────────────────────────────────────────

class TestRecommendDetailed:

    @requires_artifacts
    def test_returns_list(self, recommender):
        results = recommender.recommend_detailed("The Dark Knight", n=5)
        assert isinstance(results, list)

    @requires_artifacts
    def test_correct_count(self, recommender):
        results = recommender.recommend_detailed("The Dark Knight", n=5)
        assert len(results) == 5

    @requires_artifacts
    def test_does_not_recommend_itself(self, recommender):
        results = recommender.recommend_detailed("The Dark Knight", n=5)
        titles = [r["title"] for r in results]
        assert "The Dark Knight" not in titles

    @requires_artifacts
    def test_scores_in_valid_range(self, recommender):
        results = recommender.recommend_detailed("Avatar", n=5)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0, (
                f"score {r['score']} out of [0, 1] for '{r['title']}'"
            )

    @requires_artifacts
    def test_result_has_required_fields(self, recommender):
        required_fields = {
            "title", "score", "hybrid_score", "metadata_score",
            "genres", "cast", "director", "overview",
            "release_year", "rating", "runtime", "reasons",
        }
        results = recommender.recommend_detailed("Inception", n=3)
        assert results, "Expected at least one result for 'Inception'"
        for r in results:
            missing = required_fields - r.keys()
            assert not missing, f"Result missing fields: {missing}"

    @requires_artifacts
    def test_scores_are_descending(self, recommender):
        results = recommender.recommend_detailed("Interstellar", n=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), (
            "Results should be sorted by score descending"
        )

    @requires_artifacts
    def test_unknown_movie_returns_list(self, recommender):
        """An unrecognised title should return an empty list, not raise."""
        results = recommender.recommend_detailed("xyzTotallyFakeMovie99999", n=5)
        assert isinstance(results, list)

    @requires_artifacts
    def test_n_equals_one(self, recommender):
        results = recommender.recommend_detailed("The Matrix", n=1)
        assert len(results) == 1

    @requires_artifacts
    def test_tmdb_id_lookup(self, recommender):
        """Providing a TMDB id should still return valid recommendations."""
        # Avatar's TMDB id is 19995
        results = recommender.recommend_detailed("Avatar", n=5, tmdb_id=19995)
        assert isinstance(results, list)
        assert len(results) > 0


# ── Metadata similarity tests ─────────────────────────────────────────────────

class TestMetadataSimilarity:

    @requires_artifacts
    def test_same_movie_is_one(self, recommender):
        """A movie compared with itself should have metadata similarity of 1.0
        (assuming it shares all genres/keywords/cast with itself)."""
        # Use index 0 — any movie will do.
        score = recommender.metadata_similarity(0, 0)
        assert score == pytest.approx(1.0, abs=0.01)

    @requires_artifacts
    def test_returns_float_in_range(self, recommender):
        score = recommender.metadata_similarity(0, 1)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ── Explanation tests ─────────────────────────────────────────────────────────

class TestExplainRecommendation:

    @requires_artifacts
    def test_returns_non_empty_list(self, recommender):
        reasons = recommender.explain_recommendation(0, 1)
        assert isinstance(reasons, list)
        assert len(reasons) >= 1

    @requires_artifacts
    def test_reasons_are_strings(self, recommender):
        reasons = recommender.explain_recommendation(0, 1)
        assert all(isinstance(r, str) for r in reasons)

    @requires_artifacts
    def test_shared_genres_listed_by_name(self, recommender):
        """Shared genre reasons should list genre names, not just a count."""
        results = recommender.recommend_detailed("The Dark Knight", n=5)
        for rec in results:
            for reason in rec["reasons"]:
                if reason.startswith("Shares") and "genre" in reason:
                    # e.g. "Shares 2 genre(s): Action, Crime"
                    assert ":" in reason, (
                        f"Genre reason should list names after ':', got: {reason}"
                    )

    @requires_artifacts
    def test_shared_keywords_listed_by_name(self, recommender):
        """Shared keyword reasons should list keyword names, not just a count."""
        results = recommender.recommend_detailed("Inception", n=5)
        for rec in results:
            for reason in rec["reasons"]:
                if reason.startswith("Shares") and "keyword" in reason:
                    assert ":" in reason, (
                        f"Keyword reason should list names after ':', got: {reason}"
                    )


# ── Movie lookup tests ────────────────────────────────────────────────────────

class TestMovieLookup:

    @requires_artifacts
    def test_find_by_exact_title(self, recommender):
        idx = recommender.find_local_unique_match("Avatar")
        assert idx is not None
        assert isinstance(idx, int)

    @requires_artifacts
    def test_find_by_tmdb_id(self, recommender):
        # Avatar's TMDB id
        idx = recommender.find_movie_by_tmdb_id(19995)
        assert idx is not None

    @requires_artifacts
    def test_unknown_title_returns_none(self, recommender):
        idx = recommender.find_local_unique_match("xyzTotallyFakeMovie99999")
        assert idx is None

    @requires_artifacts
    def test_find_movie_index_prefers_tmdb_id(self, recommender):
        """When a valid tmdb_id is given, it should be used over the title."""
        idx_by_id    = recommender.find_movie_by_tmdb_id(19995)
        idx_combined = recommender.find_movie_index("Avatar", tmdb_id=19995)
        assert idx_by_id == idx_combined


# ── Search catalog tests ──────────────────────────────────────────────────────

class TestSearchCatalog:

    @requires_artifacts
    def test_returns_list(self, recommender):
        results = recommender.search_catalog("batman")
        assert isinstance(results, list)

    @requires_artifacts
    def test_empty_query_returns_empty(self, recommender):
        assert recommender.search_catalog("") == []
        assert recommender.search_catalog("   ") == []

    @requires_artifacts
    def test_results_have_required_fields(self, recommender):
        results = recommender.search_catalog("avatar", limit=3)
        for r in results:
            assert "id"    in r
            assert "title" in r

    @requires_artifacts
    def test_limit_respected(self, recommender):
        results = recommender.search_catalog("the", limit=4)
        assert len(results) <= 4


# ── Semantic search tests ─────────────────────────────────────────────────────

class TestSemanticSearch:
    """
    Tests for semantic_search() in src/recommender.py.

    semantic_search() encodes a natural-language query with the same
    SentenceTransformer used to build movie_embeddings, then returns the
    top-N catalog movies ranked by cosine similarity.

    The movie embeddings were built from each film's overview (plot summary),
    so queries that describe plot and theme produce the most relevant results.
    """

    @requires_artifacts
    def test_valid_query_returns_list(self, recommender):
        results = recommender.semantic_search("space adventure with aliens", n=5)
        assert isinstance(results, list)

    @requires_artifacts
    def test_correct_count(self, recommender):
        results = recommender.semantic_search("dark crime thriller", n=7)
        assert len(results) == 7

    @requires_artifacts
    def test_n_equals_one(self, recommender):
        results = recommender.semantic_search("animated fairy tale", n=1)
        assert len(results) == 1

    @requires_artifacts
    def test_result_has_required_fields(self, recommender):
        required = {"title", "tmdb_id", "release_year", "overview", "genres",
                    "rating", "semantic_score"}
        results = recommender.semantic_search("romantic comedy in Paris", n=3)
        assert results, "Expected at least one result"
        for r in results:
            missing = required - r.keys()
            assert not missing, f"Result missing fields: {missing}"

    @requires_artifacts
    def test_scores_are_numeric_in_range(self, recommender):
        results = recommender.semantic_search("psychological horror", n=5)
        for r in results:
            score = r["semantic_score"]
            assert isinstance(score, float), f"score is not float: {type(score)}"
            assert -1.0 <= score <= 1.0, f"cosine score {score} out of [-1, 1]"

    @requires_artifacts
    def test_scores_descending(self, recommender):
        results = recommender.semantic_search("superhero action film", n=10)
        scores = [r["semantic_score"] for r in results]
        assert scores == sorted(scores, reverse=True), \
            "Results should be ordered by descending semantic_score"

    @requires_artifacts
    def test_no_duplicate_movies(self, recommender):
        results = recommender.semantic_search("war drama historical", n=10)
        titles = [r["title"] for r in results]
        assert len(titles) == len(set(titles)), "Duplicate titles found in results"

    @requires_artifacts
    def test_deterministic_same_query(self, recommender):
        """Same query should always return the same ranking."""
        q = "mind-bending science fiction involving dreams"
        r1 = recommender.semantic_search(q, n=5)
        r2 = recommender.semantic_search(q, n=5)
        assert [r["title"] for r in r1] == [r["title"] for r in r2]
        assert [r["semantic_score"] for r in r1] == [r["semantic_score"] for r in r2]

    @requires_artifacts
    def test_empty_query_returns_empty(self, recommender):
        """Empty / whitespace-only query returns empty list, not an error."""
        assert recommender.semantic_search("", n=5) == []
        assert recommender.semantic_search("   ", n=5) == []

    @requires_artifacts
    def test_movie_own_tags_ranks_first(self, recommender):
        """
        When the query is a movie's own tags text (overview + metadata),
        that movie should rank #1 — the strongest possible retrieval signal.
        This tests that the dot-product similarity actually correlates with
        embedding distance in the expected direction.
        """
        from src.load_model import movies
        # Use Avatar (index 0) — take its tags text as the query
        seed_row  = movies.iloc[0]
        seed_tags = str(seed_row["tags"])[:500]   # first 500 chars of tags
        seed_title = seed_row["title"]

        results = recommender.semantic_search(seed_tags, n=5)
        assert results, "Expected results for a known movie's own tags text"
        assert results[0]["title"] == seed_title, (
            f"Expected '{seed_title}' at rank 1 for its own tags text, "
            f"got '{results[0]['title']}'"
        )


# ── Vectorization equivalence tests ─────────────────────────────────────────

class TestVectorizationEquivalence:
    """
    Verify that the vectorized relevant_set_vec() produces identical results
    to the scalar reference relevant_set() for representative seeds.

    These tests guard against any silent regression in the proxy relevance
    definition caused by future changes to relevance.py.
    """

    @requires_artifacts
    def test_equivalence_on_five_seeds(self):
        """Vectorized and scalar relevant sets must be identical."""
        from evaluation.relevance import (
            relevant_set,
            relevant_set_vec,
            build_proxy_matrices,
        )
        import random
        from src.load_model import movies

        mats  = build_proxy_matrices()
        rng   = random.Random(99)
        seeds = rng.sample(range(len(movies)), 5)

        for seed_idx in seeds:
            scalar_rel = relevant_set(seed_idx)
            vec_rel    = relevant_set_vec(seed_idx, mats=mats)
            assert scalar_rel == vec_rel, (
                f"Mismatch at seed {seed_idx} "
                f"(scalar_only={scalar_rel - vec_rel}, "
                f"vec_only={vec_rel - scalar_rel})"
            )

    @requires_artifacts
    def test_individual_scores_match(self):
        """Scalar proxy_relevance_score and vectorized scores agree to 1e-5."""
        from evaluation.relevance import (
            proxy_relevance_score,
            proxy_scores_vec,
            build_proxy_matrices,
        )
        import random
        from src.load_model import movies

        mats      = build_proxy_matrices()
        rng       = random.Random(7)
        seed_idx  = rng.randint(0, len(movies) - 1)
        cand_idxs = rng.sample(range(len(movies)), 20)

        vec_scores = proxy_scores_vec(seed_idx, mats=mats)
        for c in cand_idxs:
            scalar = proxy_relevance_score(seed_idx, c)
            vec    = float(vec_scores[c])
            assert abs(scalar - vec) < 1e-5, (
                f"Score mismatch seed={seed_idx} cand={c}: "
                f"scalar={scalar:.7f} vec={vec:.7f}"
            )

    @requires_artifacts
    def test_self_is_excluded_by_default(self):
        """relevant_set_vec must not include the seed itself."""
        from evaluation.relevance import relevant_set_vec, build_proxy_matrices
        mats = build_proxy_matrices()
        seed = 0
        rel  = relevant_set_vec(seed, mats=mats)
        assert seed not in rel

    @requires_artifacts
    def test_include_self_option(self):
        """With exclude_self=False, seed may appear in the relevant set."""
        from evaluation.relevance import relevant_set_vec, build_proxy_matrices
        mats = build_proxy_matrices()
        seed = 0
        rel  = relevant_set_vec(seed, exclude_self=False, mats=mats)
        # seed should be relevant to itself (score=1.0 > threshold)
        assert seed in rel
