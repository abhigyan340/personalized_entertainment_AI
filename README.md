# Personalized Entertainment AI

Hybrid, explainable movie recommendations using TMDB 5000, Sentence Transformers, TF-IDF, metadata re-ranking, and live TMDB data.

## Run locally

You need two servers. Open the site at **http://127.0.0.1:3000** (prefer `127.0.0.1` over `localhost` if WiFi DNS/IPv6 is flaky).

1. Create a `.env` in the project root with:

```
TMDB_API_KEY=your_key
TMDB_READ_ACCESS_TOKEN=your_read_token
```

2. Install Python dependencies (Python 3.10+ recommended):

```
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install fastapi uvicorn sentence-transformers scikit-learn pandas numpy joblib python-dotenv requests
```

3. Install Node dependencies:

```
cd node_backend
npm install
```

4. **Generate the similarity matrices** (first run only — these are not in the repo due to file size):

```
jupyter notebook notebooks/01_movies_data_exploration.ipynb
```

Run all cells. This produces `data/artifacts/*.npy` and `data/artifacts/*.pkl`.

5. Start the ML backend (from the project root, with the venv active):

```
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

6. Start the Node API + frontend:

```
cd node_backend
npm start
```

If TMDB is blocked on a WiFi network, search falls back to the local 4,803-movie catalog and in-dataset recommendations still work. A mobile hotspot usually reaches TMDB if the WiFi does not.

## Architecture

- Frontend: HTML, CSS, JS served by Express
- Node.js API on port 3000: `/api/search`, `/api/recommend`, `/api/semantic-search`
- FastAPI on port 8000: hybrid recommender, semantic search, embeddings, TF-IDF, metadata re-ranking

## API reference (FastAPI — port 8000)

| Endpoint           | Method | Description                                 |
| ------------------ | ------ | ------------------------------------------- |
| `/`                | GET    | Health check                                |
| `/stats`           | GET    | Dataset + model metadata                    |
| `/search`          | GET    | Title search (TMDB primary, local fallback) |
| `/recommend`       | POST   | Hybrid movie recommendations                |
| `/semantic-search` | POST   | Natural-language movie search               |

## Semantic Search

The `/semantic-search` endpoint accepts a natural-language description and returns catalog movies ranked by semantic similarity to the query.

```bash
curl -X POST http://127.0.0.1:8000/semantic-search \
  -H "Content-Type: application/json" \
  -d '{"query": "mind-bending science fiction involving dreams and reality", "n": 5}'
```

**How it works:** the query is encoded with the same `all-MiniLM-L6-v2` SentenceTransformer model used to build the movie embedding artifact. Cosine similarity is computed against all 4,803 pre-computed movie embeddings in a single NumPy dot product — no external API calls, no re-encoding of catalog movies.

**What was embedded:** each movie's clean embedding text combines its overview, genres, keywords, top-5 cast, and director.

**Limitation:** operates only against the local 4,803-movie catalog. Movies outside this dataset are not searched.

## Evaluation

The recommendation engine is evaluated offline using a reproducible proxy relevance framework. No human relevance labels exist, so proxy relevance is derived from genre, keyword, cast, and director overlap.

```bash
# Run full evaluation (100 queries, ~3 sec)
python -m evaluation.run

# Quick smoke test (30 queries)
python -m evaluation.run --fast
```

### Results summary (100 queries, seed=42)

| Method                             | P@5       | NDCG@5    | Coverage | Diversity |
| ---------------------------------- | --------- | --------- | -------- | --------- |
| Random                             | 0.086     | 0.087     | 0.188    | 0.391     |
| TF-IDF only                        | 0.252     | 0.273     | 0.059    | 0.339     |
| Semantic only                      | 0.420     | 0.445     | 0.173    | 0.266     |
| Hybrid 50/50 (no metadata)         | 0.448     | 0.481     | 0.142    | 0.246     |
| **Hybrid + Metadata (production)** | **0.654** | **0.676** | 0.143    | 0.244     |

### Key findings

* Metadata re-ranking increased P@5 from 0.448 to 0.654 (+46% relative)
* Semantic similarity consistently outperforms TF-IDF alone (0.420 vs 0.252 P@5)
* Combining both signals outperforms either alone
* Ablation shows Hybrid 25/75 (semantic-dominant) slightly outperforms 50/50 without metadata re-ranking

### Limitations

The evaluation uses metadata-derived proxy relevance — the same features the recommender uses. It measures *metadata consistency*, not *human preference*. High scores do not guarantee users will enjoy the recommendations. See [`docs/ml_evaluation.md`](docs/ml_evaluation.md) for the full technical report.
