# Evaluation — Personalized Entertainment AI

Offline evaluation framework for the hybrid movie recommendation engine.

---

## Quick start

```bash
# Full evaluation (100 queries, ~18 min)
python -m evaluation.run

# Fast smoke test (30 queries, ~2 min)
python -m evaluation.run --fast

# Print only — do not write JSON files
python -m evaluation.run --no-save
```

Results are written to `evaluation/results/`.

---

## What this evaluates

The recommendation engine is a **content-based information retrieval system**.
It is not a collaborative filter and does not use user interaction data.
Every user querying the same movie receives identical results.

The evaluation measures how consistently the recommender surfaces
**metadata-similar** movies — not how much a real user would enjoy the results.

---

## Proxy relevance definition

We have no human relevance judgements for this dataset.
Instead, we define a proxy relevance score from metadata already in the corpus:

```
proxy_score(seed, candidate) =
    0.20 × genre_overlap(seed, candidate)
  + 0.55 × keyword_overlap(seed, candidate)
  + 0.20 × cast_overlap(seed, candidate)
  + 0.05 × director_match(seed, candidate)
```

where each overlap is `|A ∩ B| / |A|` (precision-style, not Jaccard).

A candidate is **relevant** if `proxy_score ≥ 0.20`.

**Why keyword-dominant?**
Genres are too coarse: a threshold based primarily on genre would mark
30–70% of the catalog as relevant for any seed (every Action film for
The Dark Knight). Keywords are thematically specific — Avatar has 21
keywords — making keyword overlap a far more discriminating signal.

### ⚠ Data leakage notice

The proxy relevance signal uses genre, keyword, cast, and director metadata —
**the same features used by the content-based recommender**.

Consequence: this evaluation measures *metadata consistency*, not
*human preference* or *real-world recommendation quality*.
High scores do not mean users will enjoy the recommendations.
The evaluation cannot substitute for user studies or A/B testing.

---

## Metrics

| Metric | Description |
|--------|-------------|
| **Precision@K** | Fraction of top-K recommendations that are proxy-relevant |
| **NDCG@K** | Normalized Discounted Cumulative Gain — rewards relevant items ranked higher |
| **Coverage** | Fraction of the 4,803-movie catalog ever recommended across all queries |
| **Diversity** | Mean intra-list diversity — average pairwise embedding dissimilarity within each list |

K values: **5** and **10**.

---

## Baselines

| Recommender | Description |
|-------------|-------------|
| **Random** | Uniform random sample — lower bound |
| **Genre-only** | Ranked by genre overlap — simple content filter |
| **TF-IDF only** | Pre-computed TF-IDF cosine similarity |
| **Semantic only** | Pre-computed sentence-transformer cosine similarity |
| **Hybrid (no meta)** | `0.5 × TF-IDF + 0.5 × semantic`, no re-ranking |
| **Hybrid + Metadata** | Hybrid retrieval + metadata re-ranking (production config) |

---

## Ablation configurations

| Configuration | TF-IDF weight | Semantic weight | Metadata weight |
|---------------|--------------|-----------------|-----------------|
| TF-IDF only | 1.00 | 0.00 | 0.00 |
| Semantic only | 0.00 | 1.00 | 0.00 |
| Hybrid 25/75 | 0.25 | 0.75 | 0.00 |
| Hybrid 50/50 | 0.50 | 0.50 | 0.00 |
| Hybrid 75/25 | 0.75 | 0.25 | 0.00 |
| Hybrid 50/50 + Meta | 0.50 | 0.50 | 0.10 |
| Production | 0.50 | 0.50 | 0.10 |

---

## Results (100 queries, seed=42)

### Baseline comparison

| Method | P@5 | P@10 | NDCG@5 | NDCG@10 | Coverage | Diversity |
|--------|-----|------|--------|---------|----------|-----------|
| Random | 0.086 | 0.091 | 0.087 | 0.090 | 0.188 | 0.391 |
| Genre-only | 0.988 | 0.966 | 0.992 | 0.980 | 0.072 | 0.340 |
| TF-IDF only | 0.252 | 0.213 | 0.273 | 0.239 | 0.059 | 0.339 |
| Semantic only | 0.420 | 0.374 | 0.445 | 0.405 | 0.173 | 0.266 |
| Hybrid 50/50 (no meta) | 0.448 | 0.396 | 0.481 | 0.434 | 0.142 | 0.246 |
| **Hybrid + Metadata** | **0.654** | **0.589** | **0.676** | **0.624** | 0.143 | 0.244 |

### Ablation study

| Configuration | P@5 | NDCG@5 | Coverage | Diversity |
|---------------|-----|--------|----------|-----------|
| TF-IDF only | 0.252 | 0.273 | 0.059 | 0.339 |
| Semantic only | 0.420 | 0.445 | 0.173 | 0.266 |
| Hybrid 25/75 | 0.478 | 0.499 | 0.165 | 0.249 |
| Hybrid 50/50 | 0.448 | 0.481 | 0.142 | 0.246 |
| Hybrid 75/25 | 0.400 | 0.431 | 0.106 | 0.258 |
| Hybrid 50/50 + Meta | 0.654 | 0.676 | 0.143 | 0.244 |

---

## File structure

```
evaluation/
├── __init__.py
├── config.py        # all evaluation parameters (seed, thresholds, weights)
├── metrics.py       # precision@k, ndcg@k, coverage, diversity
├── relevance.py     # proxy relevance definition + query set builder
├── baselines.py     # all baseline recommenders
├── engine.py        # diagnostic score breakdown + ablation engine
├── run.py           # main entry point
├── README.md        # this file
└── results/
    ├── evaluation_results.json   # baseline results + failure cases
    └── ablation_results.json     # ablation experiment results
```

---

## Reproducibility

All experiments are fully deterministic given the same `RANDOM_SEED` (42).
To reproduce exactly:

```bash
python -m evaluation.run
```

Configuration lives in `evaluation/config.py`.
Changing any parameter there propagates to all experiments and output files.
