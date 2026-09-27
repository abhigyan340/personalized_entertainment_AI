"""
evaluation/run.py
─────────────────
Main entry point for the offline evaluation pipeline.

Usage
─────
    python -m evaluation.run              # full evaluation
    python -m evaluation.run --fast       # 30-query smoke test
    python -m evaluation.run --no-save    # print only, no JSON output

What this script does
─────────────────────
  1. Loads pre-computed artifacts (no TMDB calls, no web server needed).
  2. Builds a reproducible evaluation query set (100 seed movies).
  3. Runs every recommender / ablation configuration on every query.
  4. Computes Precision@5, Precision@10, NDCG@5, NDCG@10, Coverage, Diversity.
  5. Prints a human-readable comparison table.
  6. Saves full results to evaluation/results/.
  7. Runs failure analysis on a small subset of interesting cases.

Runtime
───────
  Full run (~100 queries × 9 configurations): ~3–8 minutes
  Fast run (~30 queries × 9 configurations): ~1–2 minutes
  Most time is spent on metadata re-ranking (O(candidate_count) per query).
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np

# Ensure project root is on the path when run as a module
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.baselines import (
    GenreRecommender,
    HybridMetaRecommender,
    HybridRecommender,
    RandomRecommender,
    SemanticRecommender,
    TFIDFRecommender,
)
from evaluation.config import (
    ABLATION_CONFIGS,
    EvalConfig,
    K_VALUES,
    N_RECOMMENDATIONS,
    NUM_QUERIES,
    PROXY_WEIGHTS,
    RANDOM_SEED,
    RELEVANCE_THRESHOLD,
)
from evaluation.engine import ablation_recommend, diagnostic_scores
from evaluation.metrics import (
    catalog_coverage,
    intra_list_diversity,
    mean_metric,
    ndcg_at_k,
    precision_at_k,
)
from evaluation.relevance import build_query_set
from src.load_model import movie_embeddings, movies

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Results directory ─────────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Diversity similarity function ─────────────────────────────────────────────

def _embedding_similarity(idx_a: int, idx_b: int) -> float:
    """
    Cosine similarity between two movie embeddings.
    Used to compute intra-list diversity (1 - similarity).
    """
    a = movie_embeddings[idx_a]
    b = movie_embeddings[idx_b]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ── Core evaluation loop ──────────────────────────────────────────────────────

def evaluate_recommender(
    recommender,
    queries,
    k_values: list[int],
    n: int,
) -> dict:
    """
    Run *recommender* on every query and compute all metrics.

    Returns a dict with the following keys:
      precision_at_{k}   for each k
      ndcg_at_{k}        for each k
      coverage           catalog fraction ever recommended
      diversity          mean intra-list diversity (embedding-based)
      mean_relevant_set_size
      n_queries
      recommendations    list of lists (for coverage computation)
    """
    all_recs:       list[list[int]] = []
    p_scores:       dict[int, list[float]] = {k: [] for k in k_values}
    ndcg_scores:    dict[int, list[float]] = {k: [] for k in k_values}
    div_scores:     list[float] = []
    rel_set_sizes:  list[int]   = []

    for query in queries:
        recs = recommender.recommend(
            seed_idx=query.movie_idx,
            n=n,
            exclude_idx=None,
        )
        all_recs.append(recs)
        rel_set_sizes.append(len(query.relevant))

        for k in k_values:
            p_scores[k].append(
                precision_at_k(recs, query.relevant, k)
            )
            ndcg_scores[k].append(
                ndcg_at_k(recs, query.relevant, k)
            )

        div_scores.append(
            intra_list_diversity(recs, _embedding_similarity)
        )

    result = {
        "n_queries":               len(queries),
        "mean_relevant_set_size":  round(mean_metric(rel_set_sizes), 1),
        "coverage":                round(catalog_coverage(all_recs, len(movies)), 6),
        "diversity":               round(mean_metric(div_scores),      6),
    }
    for k in k_values:
        result[f"precision_at_{k}"] = round(mean_metric(p_scores[k]),    6)
        result[f"ndcg_at_{k}"]      = round(mean_metric(ndcg_scores[k]), 6)

    return result


# ── Ablation evaluation loop ──────────────────────────────────────────────────

def evaluate_ablation_config(
    cfg: dict,
    queries,
    k_values: list[int],
    n: int,
) -> dict:
    """
    Run a single ablation configuration (from ABLATION_CONFIGS) on all queries.
    """
    all_recs:    list[list[int]] = []
    p_scores:    dict[int, list[float]] = {k: [] for k in k_values}
    ndcg_scores: dict[int, list[float]] = {k: [] for k in k_values}
    div_scores:  list[float] = []

    tfidf_w    = cfg["tfidf_w"]
    semantic_w = cfg["semantic_w"]
    meta_w     = cfg["meta_w"]

    for query in queries:
        recs = ablation_recommend(
            seed_idx=query.movie_idx,
            n=n,
            tfidf_w=tfidf_w,
            semantic_w=semantic_w,
            meta_w=meta_w,
        )
        all_recs.append(recs)

        for k in k_values:
            p_scores[k].append(precision_at_k(recs, query.relevant, k))
            ndcg_scores[k].append(ndcg_at_k(recs, query.relevant, k))

        div_scores.append(intra_list_diversity(recs, _embedding_similarity))

    result = {
        "label":      cfg["label"],
        "tfidf_w":    tfidf_w,
        "semantic_w": semantic_w,
        "meta_w":     meta_w,
        "coverage":   round(catalog_coverage(all_recs, len(movies)), 6),
        "diversity":  round(mean_metric(div_scores),                  6),
    }
    for k in k_values:
        result[f"precision_at_{k}"] = round(mean_metric(p_scores[k]),    6)
        result[f"ndcg_at_{k}"]      = round(mean_metric(ndcg_scores[k]), 6)
    return result


# ── Failure analysis ──────────────────────────────────────────────────────────

def run_failure_analysis(queries, n_cases: int = 10) -> list[dict]:
    """
    Identify and document interesting failure / edge cases.

    Selects queries where the production hybrid+metadata recommender
    produces the lowest Precision@5 scores, then for each of those queries
    records the full score breakdown for every recommended movie.
    """
    from evaluation.baselines import HybridMetaRecommender as HMR
    hmr = HMR(tfidf_w=0.50, semantic_w=0.50, meta_w=0.10)

    scored_queries = []
    for query in queries:
        recs = hmr.recommend(seed_idx=query.movie_idx, n=5)
        p5   = precision_at_k(recs, query.relevant, 5)
        scored_queries.append((p5, query, recs))

    # Sort ascending — worst cases first
    scored_queries.sort(key=lambda x: x[0])
    worst = scored_queries[:n_cases]

    cases = []
    for p5, query, recs in worst:
        breakdown = []
        for cand_idx in recs:
            scores = diagnostic_scores(query.movie_idx, cand_idx)
            scores["is_relevant"] = (cand_idx in query.relevant)
            breakdown.append(scores)

        cases.append({
            "seed_title":       query.title,
            "seed_idx":         query.movie_idx,
            "precision_at_5":   round(p5, 4),
            "relevant_set_size": len(query.relevant),
            "recommendations":  breakdown,
        })
    return cases


# ── Human-readable report ─────────────────────────────────────────────────────

def print_report(
    baseline_results: list[dict],
    ablation_results: list[dict],
    config: EvalConfig,
) -> None:
    width = 90
    print("\n" + "=" * width)
    print("  PERSONALIZED ENTERTAINMENT AI — OFFLINE EVALUATION REPORT")
    print("=" * width)
    print(f"  Dataset         : {config.dataset}")
    print(f"  Catalog size    : {len(movies):,} movies")
    print(f"  Evaluation queries: {config.num_queries}")
    print(f"  Embedding model : {config.embedding_model}")
    print(f"  Random seed     : {config.random_seed}")
    print(f"  K values        : {config.k_values}")
    print(f"  Relevance threshold: {config.relevance_threshold}")
    print(f"  Proxy weights   : {config.proxy_weights}")
    print(f"  Date            : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()
    print("  ⚠  DATA LEAKAGE NOTE")
    print("  The proxy relevance signal uses genre/keyword/cast/director metadata —")
    print("  the same features used by the content-based recommender.")
    print("  This evaluation measures *metadata consistency*, not human preference.")
    print("=" * width)

    # Baseline comparison table
    k5, k10 = K_VALUES[0], K_VALUES[1]
    header = f"  {'Method':<32}  {'P@5':>6}  {'P@10':>6}  {'NDCG@5':>7}  {'NDCG@10':>8}  {'Cover':>7}  {'Divers':>7}"
    sep    = "  " + "-" * (len(header) - 2)
    print(f"\n  BASELINE COMPARISON  (n_recommendations={N_RECOMMENDATIONS})")
    print(sep)
    print(header)
    print(sep)
    for r in baseline_results:
        print(
            f"  {r['name']:<32}  "
            f"{r[f'precision_at_{k5}']:>6.4f}  "
            f"{r[f'precision_at_{k10}']:>6.4f}  "
            f"{r[f'ndcg_at_{k5}']:>7.4f}  "
            f"{r[f'ndcg_at_{k10}']:>8.4f}  "
            f"{r['coverage']:>7.4f}  "
            f"{r['diversity']:>7.4f}"
        )
    print(sep)

    # Ablation table
    print(f"\n  ABLATION STUDY")
    print(sep)
    print(header.replace("Method", "Configuration"))
    print(sep)
    for r in ablation_results:
        print(
            f"  {r['label']:<32}  "
            f"{r[f'precision_at_{k5}']:>6.4f}  "
            f"{r[f'precision_at_{k10}']:>6.4f}  "
            f"{r[f'ndcg_at_{k5}']:>7.4f}  "
            f"{r[f'ndcg_at_{k10}']:>8.4f}  "
            f"{r['coverage']:>7.4f}  "
            f"{r['diversity']:>7.4f}"
        )
    print(sep)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(fast: bool = False, save: bool = True) -> None:
    config = EvalConfig()
    n_queries = 30 if fast else config.num_queries
    logger.info("Building evaluation query set (%d queries)...", n_queries)

    t0     = time.perf_counter()
    queries = build_query_set(
        n=n_queries,
        seed=config.random_seed,
        threshold=config.relevance_threshold,
        weights=config.proxy_weights,
    )
    logger.info(
        "Query set ready: %d queries  (%.1fs)", len(queries),
        time.perf_counter() - t0,
    )

    # ── Define recommenders ───────────────────────────────────────────────────
    baselines = [
        RandomRecommender(seed=RANDOM_SEED),
        GenreRecommender(),
        TFIDFRecommender(),
        SemanticRecommender(),
        HybridRecommender(tfidf_w=0.5, semantic_w=0.5),
        HybridMetaRecommender(tfidf_w=0.5, semantic_w=0.5, meta_w=0.10),
    ]

    # ── Baseline evaluation ───────────────────────────────────────────────────
    baseline_results = []
    for rec in baselines:
        logger.info("Evaluating: %s", rec.name)
        t1 = time.perf_counter()
        result = evaluate_recommender(rec, queries, config.k_values, N_RECOMMENDATIONS)
        result["name"] = rec.name
        elapsed = time.perf_counter() - t1
        logger.info(
            "  P@5=%.4f  NDCG@5=%.4f  Coverage=%.4f  Diversity=%.4f  (%.1fs)",
            result["precision_at_5"], result["ndcg_at_5"],
            result["coverage"], result["diversity"], elapsed,
        )
        baseline_results.append(result)

    # ── Ablation evaluation ───────────────────────────────────────────────────
    ablation_results = []
    logger.info("Running ablation study (%d configs)...", len(ABLATION_CONFIGS))
    for cfg in ABLATION_CONFIGS:
        logger.info("  Config: %s", cfg["label"])
        t2 = time.perf_counter()
        result = evaluate_ablation_config(cfg, queries, config.k_values, N_RECOMMENDATIONS)
        elapsed = time.perf_counter() - t2
        logger.info(
            "    P@5=%.4f  NDCG@5=%.4f  (%.1fs)",
            result["precision_at_5"], result["ndcg_at_5"], elapsed,
        )
        ablation_results.append(result)

    # ── Failure analysis ──────────────────────────────────────────────────────
    logger.info("Running failure analysis...")
    failure_cases = run_failure_analysis(queries, n_cases=10)

    # ── Print report ──────────────────────────────────────────────────────────
    print_report(baseline_results, ablation_results, config)

    # ── Save results ──────────────────────────────────────────────────────────
    if save:
        eval_output = {
            "meta": {
                "dataset":            config.dataset,
                "catalog_size":       len(movies),
                "n_queries":          len(queries),
                "embedding_model":    config.embedding_model,
                "random_seed":        config.random_seed,
                "k_values":           config.k_values,
                "relevance_threshold": config.relevance_threshold,
                "proxy_weights":      config.proxy_weights,
                "n_recommendations":  N_RECOMMENDATIONS,
                "run_date":           datetime.now().isoformat(),
                "data_leakage_note":  (
                    "Proxy relevance uses genre/keyword/cast/director metadata — "
                    "the same features used by the recommender. This evaluation "
                    "measures metadata consistency, not human preference."
                ),
            },
            "baseline_results": baseline_results,
            "failure_cases":    failure_cases,
        }
        ablation_output = {
            "meta": {
                "dataset":         config.dataset,
                "catalog_size":    len(movies),
                "n_queries":       len(queries),
                "random_seed":     config.random_seed,
                "k_values":        config.k_values,
                "run_date":        datetime.now().isoformat(),
            },
            "ablation_results": ablation_results,
        }

        eval_path    = os.path.join(RESULTS_DIR, "evaluation_results.json")
        ablation_path = os.path.join(RESULTS_DIR, "ablation_results.json")

        with open(eval_path,    "w", encoding="utf-8") as f:
            json.dump(eval_output,    f, indent=2)
        with open(ablation_path, "w", encoding="utf-8") as f:
            json.dump(ablation_output, f, indent=2)

        logger.info("Results saved to %s", RESULTS_DIR)

    total = time.perf_counter() - t0
    logger.info("Evaluation complete in %.1f seconds.", total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run offline evaluation for Personalized Entertainment AI"
    )
    parser.add_argument(
        "--fast",    action="store_true",
        help="Use 30 queries instead of 100 (smoke test)"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Print report only, do not write JSON result files"
    )
    args = parser.parse_args()
    main(fast=args.fast, save=not args.no_save)
