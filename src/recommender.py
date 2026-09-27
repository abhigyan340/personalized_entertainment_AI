"""
recommender.py
──────────────
Core recommendation engine.

Two recommendation paths
────────────────────────
  Path A — In-dataset movie (~5 ms)
    1. Retrieve: np.argsort(hybrid_similarity[idx]) → top-20 candidates
    2. Re-rank:  final = 0.90 × hybrid + 0.10 × metadata
    3. Explain:  shared genres / keywords / cast / director

  Path B — Out-of-dataset movie (~150–300 ms)
    1. Fetch TMDB metadata (genres, keywords, credits, overview)
    2. Encode tag string with SentenceTransformer("all-MiniLM-L6-v2")
    3. Retrieve: cosine_similarity(query_emb, movie_embeddings) → top-20
    4. Re-rank and explain same as Path A

Metadata re-ranking weights
────────────────────────────
  genres    40 %
  keywords  20 %
  cast      20 %
  director  20 %
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sentence_transformers import SentenceTransformer

from src.load_model import movies, hybrid_similarity, movie_embeddings
from src.tag_builder import build_tag_from_tmdb, _as_list
from src.tmdb_client import (
    search_movie,
    get_movie_details,
    get_poster_url,
    get_poster_by_tmdb_id,
)

logger = logging.getLogger(__name__)

# Loaded once per process — ~80 MB, shared across all requests.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

_poster_cache: dict = {}


# ── Utility helpers ────────────────────────────────────────────────────────────


def _safe_int(value):
    """Convert value to int, returning None on NaN / None / error."""
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _tmdb_id_for_row(row) -> int | None:
    """Return the TMDB id from a DataFrame row, checking both column names."""
    for col in ("movie_id", "id"):
        val = _safe_int(row.get(col) if hasattr(row, "get") else row[col])
        if val:
            return val
    return None


def _release_year(release_date) -> str | None:
    return str(release_date)[:4] if release_date else None


# ── Per-pair metadata similarity ───────────────────────────────────────────────

def genre_similarity(movie_idx: int, candidate_idx: int) -> float:
    """
    Precision-style overlap: |intersection| / |input_genres|.
    Returns the fraction of the input movie's genres shared with the candidate.
    """
    src = set(_as_list(movies.iloc[movie_idx]["genre_list"]))
    tgt = set(_as_list(movies.iloc[candidate_idx]["genre_list"]))
    return len(src & tgt) / len(src) if src else 0.0


def keyword_similarity(movie_idx: int, candidate_idx: int) -> float:
    src = set(_as_list(movies.iloc[movie_idx]["keyword_list"]))
    tgt = set(_as_list(movies.iloc[candidate_idx]["keyword_list"]))
    return len(src & tgt) / len(src) if src else 0.0


def cast_similarity(movie_idx: int, candidate_idx: int) -> float:
    src = set(_as_list(movies.iloc[movie_idx]["cast_list"]))
    tgt = set(_as_list(movies.iloc[candidate_idx]["cast_list"]))
    return len(src & tgt) / len(src) if src else 0.0


def director_similarity(movie_idx: int, candidate_idx: int) -> float:
    d1 = movies.iloc[movie_idx]["director"]
    d2 = movies.iloc[candidate_idx]["director"]
    return 1.0 if (d1 and d2 and d1 == d2) else 0.0


def metadata_similarity(movie_idx: int, candidate_idx: int) -> float:
    """Weighted combination of genre, keyword, cast and director overlap."""
    return (
        0.40 * genre_similarity(movie_idx, candidate_idx)
        + 0.20 * keyword_similarity(movie_idx, candidate_idx)
        + 0.20 * cast_similarity(movie_idx, candidate_idx)
        + 0.20 * director_similarity(movie_idx, candidate_idx)
    )


# ── Movie lookup ───────────────────────────────────────────────────────────────

def find_movie_by_tmdb_id(tmdb_id) -> int | None:
    """Return the DataFrame index for a given TMDB movie id."""
    tmdb_id = _safe_int(tmdb_id)
    if not tmdb_id:
        return None
    for col in ("movie_id", "id"):
        matches = movies[movies[col] == tmdb_id]
        if not matches.empty:
            return int(matches.index[0])
    return None


def find_local_unique_match(query: str) -> int | None:
    """
    Fuzzy title match against the local dataset.
    Returns a DataFrame index only when the match is unambiguous.
    Priority: exact → exact original_title → unique prefix → unique substring.
    """
    q = (query or "").strip().lower()
    if not q:
        return None

    titles    = movies["title"].fillna("").str.lower()
    originals = movies["original_title"].fillna("").str.lower()

    exact = movies[titles == q]
    if not exact.empty:
        return int(exact.index[0])

    exact_orig = movies[originals == q]
    if not exact_orig.empty:
        return int(exact_orig.index[0])

    starts = movies[titles.str.startswith(q)]
    if len(starts) == 1:
        return int(starts.index[0])

    contains = movies[titles.str.contains(re.escape(q), regex=True, na=False)]
    if len(contains) == 1:
        return int(contains.index[0])

    return None


def _find_movie_from_tmdb(movie_title: str) -> int | None:
    """Search TMDB and attempt to match the result to a local dataset entry."""
    results = search_movie(movie_title)
    if not results.get("results"):
        return None
    return find_movie_by_tmdb_id(results["results"][0].get("id"))


def find_movie_index(movie_title: str, tmdb_id=None) -> int | None:
    """
    Three-tier movie lookup:
      1. Exact TMDB id match in the local dataset
      2. Fuzzy title match in the local dataset
      3. TMDB API search → local id match
    Returns the DataFrame index, or None if not found.
    """
    idx = find_movie_by_tmdb_id(tmdb_id)
    if idx is not None:
        return idx

    idx = find_local_unique_match(movie_title)
    if idx is not None:
        return idx

    return _find_movie_from_tmdb(movie_title)


# ── Local catalog search (used as TMDB fallback in /search) ──────────────────

def search_catalog(query: str, limit: int = 8) -> list[dict]:
    """Return movies from the local dataset whose title matches the query."""
    q = (query or "").strip().lower()
    if not q:
        return []

    titles    = movies["title"].fillna("").str.lower()
    originals = movies["original_title"].fillna("").str.lower()
    mask      = (
        titles.str.contains(re.escape(q), regex=True, na=False)
        | originals.str.contains(re.escape(q), regex=True, na=False)
    )
    ranked = movies[mask].copy()
    ranked["_rank"] = (
        ranked["title"].fillna("").str.lower().str.startswith(q).astype(int)
    )
    ranked = ranked.sort_values(by=["_rank", "popularity"], ascending=[False, False])

    results = []
    for _, row in ranked.head(limit).iterrows():
        release_date = row.get("release_date")
        vote = row.get("vote_average")
        results.append({
            "id":           _tmdb_id_for_row(row),
            "title":        row["title"],
            "release_date": str(release_date)[:10] if release_date else "",
            "overview":     row.get("overview") or "",
            "poster_path":  None,
            "vote_average": float(vote) if (vote == vote) else None,
            "source":       "local",
        })
    return results


# ── Explanation generation ─────────────────────────────────────────────────────

def explain_recommendation(movie_idx: int, candidate_idx: int) -> list[str]:
    """
    Generate human-readable reasons why a candidate was recommended.
    Shared keywords are listed explicitly (not just counted).
    """
    reasons = []

    shared_genres = (
        set(_as_list(movies.iloc[movie_idx]["genre_list"]))
        & set(_as_list(movies.iloc[candidate_idx]["genre_list"]))
    )
    if shared_genres:
        reasons.append(
            f"Shares {len(shared_genres)} genre(s): {', '.join(sorted(shared_genres))}"
        )

    shared_keywords = (
        set(_as_list(movies.iloc[movie_idx]["keyword_list"]))
        & set(_as_list(movies.iloc[candidate_idx]["keyword_list"]))
    )
    if shared_keywords:
        # Show up to 5 keywords so the explanation stays concise.
        sample = sorted(shared_keywords)[:5]
        more   = len(shared_keywords) - len(sample)
        label  = ", ".join(sample) + (f" (+{more} more)" if more else "")
        reasons.append(f"Shares {len(shared_keywords)} keyword(s): {label}")

    shared_cast = (
        set(_as_list(movies.iloc[movie_idx]["cast_list"]))
        & set(_as_list(movies.iloc[candidate_idx]["cast_list"]))
    )
    if shared_cast:
        reasons.append(
            f"Shares {len(shared_cast)} cast member(s): {', '.join(sorted(shared_cast))}"
        )

    d1 = movies.iloc[movie_idx]["director"]
    d2 = movies.iloc[candidate_idx]["director"]
    if d1 and d2 and d1 == d2:
        reasons.append(f"Same director: {d1}")

    if not reasons:
        reasons.append("Strong semantic similarity based on movie content")

    return reasons


def explain_external(movie_data: dict, candidate_idx: int) -> list[str]:
    """Explanation for an out-of-dataset (TMDB-sourced) input movie."""
    ext_genres   = set(g["name"] for g in movie_data.get("genres", []))
    ext_keywords = set(
        k["name"] for k in movie_data.get("keywords", {}).get("keywords", [])
    )
    ext_cast = set(
        m["name"] for m in movie_data.get("credits", {}).get("cast", [])[:5]
    )
    ext_director = ""
    for m in movie_data.get("credits", {}).get("crew", []):
        if m.get("job") == "Director":
            ext_director = m["name"]
            break

    reasons = []

    shared_genres = ext_genres & set(_as_list(movies.iloc[candidate_idx]["genre_list"]))
    if shared_genres:
        reasons.append(
            f"Shares {len(shared_genres)} genre(s): {', '.join(sorted(shared_genres))}"
        )

    shared_keywords = ext_keywords & set(_as_list(movies.iloc[candidate_idx]["keyword_list"]))
    if shared_keywords:
        sample = sorted(shared_keywords)[:5]
        more   = len(shared_keywords) - len(sample)
        label  = ", ".join(sample) + (f" (+{more} more)" if more else "")
        reasons.append(f"Shares {len(shared_keywords)} keyword(s): {label}")

    shared_cast = ext_cast & set(_as_list(movies.iloc[candidate_idx]["cast_list"]))
    if shared_cast:
        reasons.append(
            f"Shares {len(shared_cast)} cast member(s): {', '.join(sorted(shared_cast))}"
        )

    cand_director = movies.iloc[candidate_idx]["director"]
    if ext_director and cand_director and ext_director == cand_director:
        reasons.append(f"Same director: {ext_director}")

    if not reasons:
        reasons.append("Recommended via semantic similarity to the TMDB movie")

    return reasons


# ── Candidate card builder ─────────────────────────────────────────────────────

def _candidate_card(
    candidate_idx: int,
    score: float,
    hybrid_score: float,
    metadata_score: float,
    reasons: list[str],
) -> dict:
    row = movies.iloc[candidate_idx]
    return {
        "index":          int(candidate_idx),
        "tmdb_id":        _tmdb_id_for_row(row),
        "title":          row["title"],
        "score":          float(score),
        "hybrid_score":   float(hybrid_score),
        "metadata_score": float(metadata_score),
        "release_year":   _release_year(row["release_date"]),
        "rating":         float(row["vote_average"]) if row["vote_average"] == row["vote_average"] else None,
        "runtime":        _safe_int(row["runtime"]),
        "genres":         _as_list(row["genre_list"]),
        "overview":       row["overview"] or "",
        "cast":           _as_list(row["cast_list"])[:5],
        "director":       row["director"] or "",
        "poster_url":     None,
        "reasons":        reasons,
    }


# ── Poster fetching ────────────────────────────────────────────────────────────

def _cached_poster_by_id(tmdb_id) -> str | None:
    tmdb_id = _safe_int(tmdb_id)
    if not tmdb_id:
        return None
    if tmdb_id not in _poster_cache:
        summary = get_poster_by_tmdb_id(tmdb_id)
        _poster_cache[tmdb_id] = summary
    return _poster_cache[tmdb_id]


def _attach_posters(recommendations: list[dict]) -> list[dict]:
    """Fetch poster URLs in parallel and attach them to each recommendation."""
    ids = [item.get("tmdb_id") for item in recommendations]
    with ThreadPoolExecutor(max_workers=5) as pool:
        posters = list(pool.map(_cached_poster_by_id, ids))
    for item, poster_url in zip(recommendations, posters):
        item["poster_url"] = poster_url
    return recommendations


# ── Path A: In-dataset recommendation ─────────────────────────────────────────

def _hybrid_recommendations(
    movie_idx: int,
    n: int = 5,
    candidate_count: int = 20,
) -> list[dict]:
    """
    Two-stage recommendation process:
      Stage 1 — Candidate Retrieval: Retrieve the top candidate pool (default: 20)
                from the pre-computed hybrid similarity matrix using np.argsort,
                explicitly excluding the seed movie itself.
      Stage 2 — Metadata Re-ranking: Re-rank only the retrieved candidate pool
                (not all 4,803 catalog movies) using weighted metadata similarity
                (0.90 * hybrid + 0.10 * metadata).
      Final: Return the top-n results with explanation cards and posters.
    """
    distances = hybrid_similarity[movie_idx]

    # np.argsort is significantly faster than sorted(list(enumerate(...)))
    # for a 4803-element array.
    sorted_indices = np.argsort(distances)[::-1]
    # Explicitly exclude the seed movie itself from the candidate pool.
    candidate_indices = [
        i for i in sorted_indices if i != movie_idx
    ][:candidate_count]

    cards = []
    for cand_idx in candidate_indices:
        h_score = float(distances[cand_idx])
        m_score = metadata_similarity(movie_idx, cand_idx)
        final   = 0.90 * h_score + 0.10 * m_score
        cards.append(
            _candidate_card(
                cand_idx, final, h_score, m_score,
                explain_recommendation(movie_idx, cand_idx),
            )
        )

    cards.sort(key=lambda c: c["score"], reverse=True)
    return _attach_posters(cards[:n])


# ── Path B: Out-of-dataset recommendation ─────────────────────────────────────

def _external_recommendations(
    movie_data: dict,
    n: int = 5,
    candidate_count: int = 20,
) -> list[dict]:
    """
    Encode the TMDB movie on-the-fly and compare against all local embeddings.
    Used when the input movie is not in the local dataset.

    Two-stage process
    ─────────────────
    Stage 1 — Retrieval: encode the TMDB movie and compute cosine similarity
    against all 4,803 pre-computed unit-normalized movie embeddings via a
    single dot product.  Selects the top `candidate_count` candidates.

    Stage 2 — Metadata re-ranking: within those candidates, applies a
    weighted metadata signal (genres 40 %, keywords 20 %, cast 20 %,
    director 20 %) blended with the semantic score at 90 % / 10 %.
    Returns the top-n after re-ranking.
    """
    # build_tag_from_tmdb is the canonical tag constructor shared with
    # scripts/regenerate_embeddings.py — ensures query and catalog use
    # the same text representation.
    tags  = build_tag_from_tmdb(movie_data)
    q_emb = embedding_model.encode([tags])[0]

    # Normalize query to unit vector.
    # movie_embeddings are already L2-normalized (norm ≈ 1.0), so the
    # dot product equals cosine similarity without computing mat_norm.
    q_norm = float(np.linalg.norm(q_emb))
    if q_norm == 0.0:
        return []
    q_vec  = q_emb / q_norm
    scores = movie_embeddings @ q_vec  # (4803,) — vectorized dot product

    candidate_indices = np.argsort(scores)[::-1][:candidate_count]

    # Build external metadata sets for explanation / re-ranking
    ext_genres   = set(g["name"] for g in movie_data.get("genres", []))
    ext_keywords = set(k["name"] for k in movie_data.get("keywords", {}).get("keywords", []))
    ext_cast     = set(m["name"] for m in movie_data.get("credits", {}).get("cast", [])[:5])
    ext_director = ""
    for m in movie_data.get("credits", {}).get("crew", []):
        if m.get("job") == "Director":
            ext_director = m["name"]
            break

    cards = []
    for cand_idx in candidate_indices:
        row     = movies.iloc[cand_idx]
        c_genres   = set(_as_list(row["genre_list"]))
        c_keywords = set(_as_list(row["keyword_list"]))
        c_cast     = set(_as_list(row["cast_list"]))
        c_director = row["director"]

        g_score = len(ext_genres & c_genres)   / len(ext_genres)   if ext_genres   else 0.0
        k_score = len(ext_keywords & c_keywords) / len(ext_keywords) if ext_keywords else 0.0
        c_score = len(ext_cast & c_cast)        / len(ext_cast)     if ext_cast     else 0.0
        d_score = float(bool(ext_director and c_director and ext_director == c_director))

        m_score  = 0.40 * g_score + 0.20 * k_score + 0.20 * c_score + 0.20 * d_score
        s_score  = float(scores[cand_idx])
        final    = 0.90 * s_score + 0.10 * m_score

        cards.append(
            _candidate_card(
                int(cand_idx), final, s_score, m_score,
                explain_external(movie_data, int(cand_idx)),
            )
        )

    cards.sort(key=lambda c: c["score"], reverse=True)
    return _attach_posters(cards[:n])


# ── Public entry point ─────────────────────────────────────────────────────────

def recommend_detailed(
    movie: str,
    n: int = 5,
    candidate_count: int = 20,
    tmdb_id: int | None = None,
) -> list[dict]:
    """
    Return up to *n* detailed recommendation cards for *movie*.

    Tries Path A (pre-computed hybrid matrix) first; falls back to
    Path B (live TMDB embedding) if the movie is outside the dataset.
    """
    # Path A — in-dataset via TMDB id or title lookup
    movie_idx = find_movie_index(movie, tmdb_id=tmdb_id)
    if movie_idx is not None:
        logger.debug("Path A (in-dataset) for '%s' → index %d", movie, movie_idx)
        return _hybrid_recommendations(movie_idx, n=n, candidate_count=candidate_count)

    # Path B — out-of-dataset: fetch live TMDB metadata + embed on-the-fly
    logger.debug("Path B (out-of-dataset) for '%s'", movie)
    results = search_movie(movie)
    if not results.get("results"):
        logger.warning("No TMDB results for '%s'", movie)
        return []

    tmdb_movie = results["results"][0]
    details    = get_movie_details(tmdb_movie["id"])
    if not details:
        logger.warning("TMDB details unavailable for id %s", tmdb_movie["id"])
        return []

    # One more id-based check: TMDB may return an id that IS in the dataset
    in_dataset = find_movie_by_tmdb_id(details.get("id"))
    if in_dataset is not None:
        logger.debug(
            "Path A after TMDB lookup for '%s' → index %d", movie, in_dataset
        )
        return _hybrid_recommendations(in_dataset, n=n, candidate_count=candidate_count)

    return _external_recommendations(details, n=n, candidate_count=candidate_count)


# ── Semantic natural-language search ──────────────────────────────────────────

def semantic_search(
    query: str,
    n: int = 10,
) -> list[dict]:
    """
    Search the local 4,803-movie catalog using a natural-language description.

    The query is encoded with the same SentenceTransformer model used to build
    the movie_embeddings artifact.  Cosine similarity is computed against all
    pre-computed embeddings using a single NumPy dot product — no Python loop
    over individual movies.

    Important: the movie embeddings were built from clean metadata using the
    canonical tag representation (see src/tag_builder.build_tag_from_row):
        "{overview} {genres} {keywords} {cast_top5} {director}"  (lowercased)
    All fields come from the pre-extracted clean columns in movies_processed.pkl
    (genre_list, keyword_list, cast_list top-5, director, overview).
    Queries describing plot, theme, genre, or key cast/director terms
    produce the most relevant results.

    Limitation: operates only against the local 4,803-movie catalog.
    Movies outside this dataset are not searched.

    Parameters
    ----------
    query : natural-language description (e.g. "a heist film set in Paris")
    n     : number of top results to return (1–50)

    Returns
    -------
    List of result dicts, ordered by descending semantic_score.
    Each dict contains: title, tmdb_id, release_year, overview, genres,
    rating, semantic_score.
    """
    query = query.strip()
    if not query:
        return []

    # Encode query and normalize to unit vector
    q_vec  = embedding_model.encode([query])[0]
    q_norm = float(np.linalg.norm(q_vec))
    if q_norm == 0.0:
        return []
    q_vec = q_vec / q_norm  # (384,) unit vector

    # Cosine similarity: movie_embeddings are already L2-normalized (norm≈1.0),
    # so dot product = cosine similarity.
    scores = movie_embeddings @ q_vec  # (4803,) — fully vectorized, no loop

    top_indices = np.argsort(scores)[::-1][:n]

    results = []
    for idx in top_indices:
        row = movies.iloc[int(idx)]
        results.append({
            "title":          str(row["title"]),
            "tmdb_id":        _tmdb_id_for_row(row),
            "release_year":   _release_year(row["release_date"]),
            "overview":       str(row["overview"] or ""),
            "genres":         _as_list(row["genre_list"]),
            "rating":         float(row["vote_average"]) if row["vote_average"] == row["vote_average"] else None,
            "semantic_score": round(float(scores[idx]), 6),
        })
    return results
