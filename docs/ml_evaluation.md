# ML Evaluation Report — Personalized Entertainment AI

**System type**: Content-based movie recommendation (information retrieval)
**Dataset**: TMDB 5000 Movie Dataset (4,803 films)
**Evaluation date**: 2026-09-25
**Evaluation queries**: 100 seed movies (random seed 42)

---

## 1. Problem definition

The system is a **content-based recommendation engine**. Given a seed movie, it retrieves and ranks the most similar movies from a fixed catalog of 4,803 films.

It is **not** a collaborative filter. There is no user interaction data, no training objective, and no learned user preference model. Every user querying the same movie receives identical results.

### Recommendation pipeline (in-dataset path)

```
Seed movie title
      ↓
Three-tier lookup (TMDB id → fuzzy title → TMDB API)
      ↓
Retrieve: np.argsort(hybrid_similarity[seed_idx])[::-1][1:21]
      |
      ├─ hybrid_similarity = 0.50 × tfidf_sim + 0.50 × semantic_sim
      |   ├─ tfidf_sim:    TF-IDF cosine similarity (precomputed, 4803×4803)
      |   └─ semantic_sim: all-MiniLM-L6-v2 cosine similarity (precomputed, 4803×4803)
      ↓
Re-rank top-20 candidates:
  final = 0.90 × hybrid + 0.10 × metadata
      |
      └─ metadata = 0.40×genre + 0.20×keyword + 0.20×cast + 0.20×director
      ↓
Top-N recommendations with per-signal explanation
```

### Input representation

Each movie is encoded as a "tag string":
```
genres + keywords + top-5 cast + director + overview
```

This string is fed to both:
- `TfidfVectorizer(max_features=5000)` → 5,000-dim sparse vector
- `SentenceTransformer("all-MiniLM-L6-v2")` → 384-dim dense vector

Both similarity matrices are precomputed offline across all 4,803 × 4,803 pairs.

---

## 2. Evaluation methodology

### Offline proxy evaluation

No human relevance labels exist for this dataset. The evaluation uses a **proxy relevance signal** constructed from the same metadata in the corpus.

This is a standard approach in academic recommender systems research when human judgements are unavailable (e.g., Herlocker et al. 2004, Cremonesi et al. 2010). The key requirement is to be explicit about what it measures and what it does not.

### Query set construction

100 seed movies were sampled deterministically (random seed 42) from the catalog with:
- At least 3 keywords (ensures non-trivial proxy relevance)
- At least 50 TMDB votes (filters low-visibility films with sparse metadata)
- At least 1 proxy-relevant neighbor (ensures the query is evaluable)

---

## 3. Relevance definition

```
proxy_score(seed, candidate) =
    0.20 × genre_overlap(seed, candidate)
  + 0.55 × keyword_overlap(seed, candidate)
  + 0.20 × cast_overlap(seed, candidate)
  + 0.05 × director_match(seed, candidate)
```

where `overlap(A, B) = |A ∩ B| / |A|` (precision-style, not Jaccard).

A candidate is **relevant** if `proxy_score ≥ 0.20`.

### Design rationale

Keyword overlap dominates (0.55) because:
- Genres are too coarse: Action, Adventure, Sci-Fi apply to hundreds of films. A genre-dominant proxy marks 30–70% of the catalog as relevant for any seed, making P@K near-trivially satisfiable.
- Keywords are thematically specific: Avatar has 21 keywords including "alien planet", "space colony", "anti war". Sharing several of these is a meaningful similarity signal.

Director weight is low (0.05) because most directors have only 1–3 films in the dataset; director-match alone would produce a relevant set too small to be meaningful.

At threshold 0.20, the average relevant set size is **~120 movies per seed** (range: 8–400), giving metrics that are informative and discriminating.

### ⚠ Data leakage notice

The proxy relevance signal uses genre, keyword, cast, and director — the **same features** the recommender uses. This creates unavoidable measurement circularity:

> The evaluation measures *metadata consistency* — whether the recommender returns metadata-similar movies — not *human preference* or *real-world recommendation quality*.

This is the fundamental limitation of offline content-based evaluation without human labels. It must be stated clearly when presenting results. A high P@5 score in this framework does not prove users will enjoy the recommendations.

---

## 4. Metrics

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| **Precision@K** | `|top-K ∩ relevant| / K` | Fraction of recommendations that are proxy-relevant |
| **NDCG@K** | Normalized DCG with binary relevance | Ranking quality — rewards relevant items appearing earlier |
| **Coverage** | `|unique recommended| / 4803` | Fraction of catalog ever surfaced across all queries |
| **Diversity** | Mean `(1 - cosine_sim)` within each list | How different the recommendations in a single list are |

K values: 5 and 10.

Binary relevance (not graded) is used for NDCG because the proxy score is not calibrated on a meaningful human scale. Graded relevance would imply a precision of measurement that the proxy does not support.

---

## 5. Baselines

| Recommender | Description | Purpose |
|-------------|-------------|---------|
| **Random** | Uniform random sample | Lower bound — trivially uninformed |
| **Genre-only** | Ranked by genre overlap | Coarse content filter |
| **TF-IDF only** | Pre-computed TF-IDF cosine similarity | Isolates lexical signal |
| **Semantic only** | Pre-computed semantic cosine similarity | Isolates semantic signal |
| **Hybrid (no meta)** | 0.5×TF-IDF + 0.5×semantic | Retrieval without re-ranking |
| **Hybrid + Metadata** | Hybrid + metadata re-ranking | Full production configuration |

---

## 6. Ablation experiments

Seven configurations were tested to isolate each signal's contribution:

| Configuration | TF-IDF | Semantic | Metadata | Purpose |
|---------------|--------|----------|----------|---------|
| TF-IDF only | 1.00 | 0.00 | 0% | Lexical signal alone |
| Semantic only | 0.00 | 1.00 | 0% | Semantic signal alone |
| Hybrid 25/75 | 0.25 | 0.75 | 0% | Semantic-dominant hybrid |
| Hybrid 50/50 | 0.50 | 0.50 | 0% | Equal hybrid, no metadata |
| Hybrid 75/25 | 0.75 | 0.25 | 0% | TF-IDF-dominant hybrid |
| Hybrid 50/50 + Meta | 0.50 | 0.50 | 10% | Production configuration |
| Production | 0.50 | 0.50 | 10% | Confirms prod = above |

---

## 7. Results

All results are from 100 evaluation queries, random seed 42.

### 7.1 Baseline comparison

| Method | P@5 | P@10 | NDCG@5 | NDCG@10 | Coverage | Diversity |
|--------|-----|------|--------|---------|----------|-----------|
| Random | 0.086 | 0.091 | 0.087 | 0.090 | **0.188** | **0.391** |
| Genre-only | 0.988 | 0.966 | 0.992 | 0.980 | 0.072 | 0.340 |
| TF-IDF only | 0.252 | 0.213 | 0.273 | 0.239 | 0.059 | 0.339 |
| Semantic only | 0.420 | 0.374 | 0.445 | 0.405 | 0.173 | 0.266 |
| Hybrid 50/50 (no meta) | 0.448 | 0.396 | 0.481 | 0.434 | 0.142 | 0.246 |
| **Hybrid + Metadata** | **0.654** | **0.589** | **0.676** | **0.624** | 0.143 | 0.244 |

### 7.2 Ablation study

| Configuration | P@5 | P@10 | NDCG@5 | NDCG@10 | Coverage | Diversity |
|---------------|-----|------|--------|---------|----------|-----------|
| TF-IDF only | 0.252 | 0.213 | 0.273 | 0.239 | 0.059 | 0.339 |
| Semantic only | 0.420 | 0.374 | 0.445 | 0.405 | 0.173 | 0.266 |
| Hybrid 25/75 | **0.478** | 0.402 | **0.499** | 0.439 | 0.165 | 0.249 |
| Hybrid 50/50 | 0.448 | 0.396 | 0.481 | 0.434 | 0.142 | 0.246 |
| Hybrid 75/25 | 0.400 | 0.355 | 0.431 | 0.391 | 0.106 | 0.258 |
| Hybrid 50/50 + Meta | 0.654 | 0.589 | 0.676 | 0.624 | 0.143 | 0.244 |

---

## 8. Failure analysis

The 5 queries with the lowest P@5 under the production configuration (all P@5 = 0.0):

| Seed movie | Relevant set size | Failure mode |
|------------|------------------|--------------|
| The Perfect Host | 54 | keyword_score=0.000 for all top-5 recs; genre/embedding similarity alone |
| Bless the Child | 15 | Very small relevant set; hybrid finds genre-similar but keyword-different films |
| West Side Story | 8 | Extremely small relevant set (8 movies); keyword leakage only to musicals |
| Michael Clayton | 63 | Legal-drama keywords sparse; legal thrillers don't share keywords despite genre match |
| Severance | 27 | Horror keywords very specific; recommender finds adjacent-genre movies |

### Observed failure patterns

**Pattern 1 — Keyword gap**
All 5 failure cases show `keyword_score = 0.000` for most recommended movies. The hybrid retrieval ranks movies by semantic/TF-IDF similarity, but these signals sometimes surface genre-adjacent films that share no specific keywords with the seed. Since the proxy relevance uses keyword overlap as the dominant signal (0.55 weight), a keyword miss is difficult to recover from.

**Pattern 2 — Small relevant set**
West Side Story (8 relevant movies) and Bless the Child (15) have very few proxy-relevant neighbors in the dataset. Even a perfect top-K list would have low absolute P@5 for these seeds because the relevant set is too small relative to K.

**Pattern 3 — Domain specificity**
Michael Clayton and Severance represent niche subgenres (legal procedural, workplace horror) with highly specific keyword vocabularies. The hybrid model surfaces movies from adjacent broader categories (thriller, drama, horror) that are semantically close but keyword-distant.

---

## 9. Limitations

### L1 — Data leakage (critical)
The proxy relevance signal is derived from the same metadata the recommender uses as features. Any metric improvement when adding metadata re-ranking partially reflects the recommender aligning with its own evaluation proxy. This is unavoidable in offline evaluation of content-based systems without human labels.

### L2 — Genre-only anomaly
Genre-only achieves P@5 = 0.988 — near-perfect. This is expected: genre overlap contributes to the proxy score (0.20 weight), and the genre recommender returns movies that always share genres. This result demonstrates the evaluation's leakage, not that genre-only is a good recommender. In a human preference study, recommending only genre-similar movies would likely produce monotonous, low-quality lists.

### L3 — No user preference signal
The system has no user interaction data. "Relevant" means "metadata-similar", not "something this particular user would enjoy". Personalization is not evaluated here.

### L4 — Static dataset
The evaluation covers 4,803 films from TMDB 5000. Films released after this dataset was collected are only supported via the out-of-dataset path (live TMDB embedding), which is not evaluated here due to the need for TMDB API calls at evaluation time.

### L5 — Binary relevance approximation
NDCG uses binary relevance (relevant/not relevant). The proxy score is a continuous value in [0, 1] and could support graded relevance, but the score threshold of 0.20 is not calibrated on human judgements, so treating it as a graded scale would imply measurement precision the proxy does not support.

### L6 — No novelty / serendipity metrics
Precision and NDCG measure relevance consistency. They do not capture novelty (recommending non-obvious items) or serendipity (surprising but appropriate recommendations). The system likely has low novelty due to the similarity-maximizing objective.

---

## 10. Conclusions

All conclusions are drawn directly from experimental results.

**1. Metadata re-ranking provides the largest single improvement.**
Adding 10% metadata re-ranking to the Hybrid 50/50 configuration increases P@5 from 0.448 to 0.654 — a 46% relative improvement. This is the most impactful component after retrieval.

**2. Semantic similarity consistently outperforms TF-IDF alone.**
Semantic only (P@5=0.420) beats TF-IDF only (P@5=0.252) by a substantial margin across all metrics. TF-IDF is limited to exact term overlap; the embedding model captures paraphrase and theme.

**3. Combining TF-IDF and semantic signals improves over either alone — but the optimal hybrid weight leans semantic.**
Hybrid 25/75 (TF-IDF 25%, Semantic 75%) achieves the highest P@5 (0.478) among hybrid-only configurations. The production 50/50 split (0.448) is slightly lower. This suggests that slightly higher semantic weight may be preferable, though the difference is small and may not be robust to a different evaluation set.

**4. Coverage-diversity trade-off is real.**
Methods with higher P@K (Hybrid + Metadata) have lower diversity (0.244) and lower coverage (0.143) than semantic-heavy configurations. The production system concentrates recommendations around a narrower, more metadata-consistent set of films.

**5. Random baseline confirms the evaluation is non-trivial.**
Random achieves P@5=0.086, well below all content-based methods. This confirms the evaluation dataset and proxy are non-trivially discriminating — the content-based signals provide genuine signal over chance.

**6. Genre-only near-perfection is an artifact of proxy leakage, not recommendation quality.**
Genre-only P@5=0.988 directly reflects that genre contributes to the proxy relevance score. This result should not be interpreted as "genre-only is the best recommender" — it demonstrates the proxy's leakage rather than real-world quality.

**7. The production 50/50 weight split is not demonstrably optimal.**
The ablation shows Hybrid 25/75 outperforms 50/50 on this evaluation proxy. However, given the evaluation limitations (data leakage, proxy approximation), this finding is suggestive rather than conclusive. A human preference study would be needed to validate the optimal hybrid weight.

---

## Reproducibility

```bash
# Reproduce all results
python -m evaluation.run

# Configuration in: evaluation/config.py
# Results saved to: evaluation/results/
```

Parameters: seed=42, queries=100, threshold=0.20, proxy_weights={genre:0.20, keyword:0.55, cast:0.20, director:0.05}
