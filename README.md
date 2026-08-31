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
- Node.js API on port 3000: `/api/search`, `/api/recommend`
- FastAPI on port 8000: hybrid recommender, embeddings, TF-IDF, metadata re-ranking
