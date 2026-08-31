import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sentence_transformers import SentenceTransformer

from src.load_model import movies, hybrid_similarity, movie_embeddings
from src.tmdb_client import (
    search_movie,
    get_movie_details,
    get_poster_url,
    get_poster_by_tmdb_id,
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

_poster_cache = {}


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    try:
        return list(value)
    except TypeError:
        return []


def _safe_int(value):
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _tmdb_id_for_row(row):
    for column in ("movie_id", "id"):
        value = _safe_int(row.get(column) if hasattr(row, "get") else row[column])
        if value:
            return value
    return None


def genre_similarity(movie_index, candidate_index):
    movie_genres = set(_as_list(movies.iloc[movie_index]["genre_list"]))
    candidate_genres = set(_as_list(movies.iloc[candidate_index]["genre_list"]))

    if not movie_genres:
        return 0

    return len(movie_genres & candidate_genres) / len(movie_genres)


def keyword_similarity(movie_index, candidate_index):
    movie_keywords = set(_as_list(movies.iloc[movie_index]["keyword_list"]))
    candidate_keywords = set(_as_list(movies.iloc[candidate_index]["keyword_list"]))

    if not movie_keywords:
        return 0

    return len(movie_keywords & candidate_keywords) / len(movie_keywords)


def cast_similarity(movie_index, candidate_index):
    movie_cast = set(_as_list(movies.iloc[movie_index]["cast_list"]))
    candidate_cast = set(_as_list(movies.iloc[candidate_index]["cast_list"]))

    if not movie_cast:
        return 0

    return len(movie_cast & candidate_cast) / len(movie_cast)


def director_similarity(movie_index, candidate_index):
    movie_director = movies.iloc[movie_index]["director"]
    candidate_director = movies.iloc[candidate_index]["director"]

    if not movie_director or not candidate_director:
        return 0

    return float(movie_director == candidate_director)


def metadata_similarity(movie_index, candidate_index):
    return (
        0.40 * genre_similarity(movie_index, candidate_index)
        + 0.20 * keyword_similarity(movie_index, candidate_index)
        + 0.20 * cast_similarity(movie_index, candidate_index)
        + 0.20 * director_similarity(movie_index, candidate_index)
    )


def find_movie_by_tmdb_id(tmdb_id):
    tmdb_id = _safe_int(tmdb_id)
    if not tmdb_id:
        return None

    for column in ("movie_id", "id"):
        matches = movies[movies[column] == tmdb_id]
        if not matches.empty:
            return int(matches.index[0])

    return None


def find_movie_from_tmdb(movie_title):
    results = search_movie(movie_title)
    if not results.get("results"):
        return None

    return find_movie_by_tmdb_id(results["results"][0].get("id"))


def find_local_unique_match(query):
    q = (query or "").strip().lower()
    if not q:
        return None

    titles = movies["title"].fillna("").str.lower()
    exact = movies[titles == q]
    if not exact.empty:
        return int(exact.index[0])

    originals = movies["original_title"].fillna("").str.lower()
    exact_original = movies[originals == q]
    if not exact_original.empty:
        return int(exact_original.index[0])

    starts = movies[titles.str.startswith(q)]
    if len(starts) == 1:
        return int(starts.index[0])

    contains = movies[titles.str.contains(re.escape(q), regex=True, na=False)]
    if len(contains) == 1:
        return int(contains.index[0])

    return None


def find_movie_index(movie_title, tmdb_id=None):
    local_id_match = find_movie_by_tmdb_id(tmdb_id)
    if local_id_match is not None:
        return local_id_match

    local_title_match = find_local_unique_match(movie_title)
    if local_title_match is not None:
        return local_title_match

    return find_movie_from_tmdb(movie_title)


def search_catalog(query, limit=8):
    q = (query or "").strip().lower()
    if not q:
        return []

    titles = movies["title"].fillna("").str.lower()
    originals = movies["original_title"].fillna("").str.lower()
    mask = titles.str.contains(re.escape(q), regex=True, na=False) | originals.str.contains(
        re.escape(q), regex=True, na=False
    )
    ranked = movies[mask].copy()
    ranked["_rank"] = ranked["title"].fillna("").str.lower().str.startswith(q).astype(int)
    ranked = ranked.sort_values(by=["_rank", "popularity"], ascending=[False, False])

    results = []
    for _, row in ranked.head(limit).iterrows():
        release_date = row.get("release_date")
        results.append(
            {
                "id": _tmdb_id_for_row(row),
                "title": row["title"],
                "release_date": str(release_date)[:10] if release_date else "",
                "overview": row.get("overview") or "",
                "poster_path": None,
                "vote_average": float(row["vote_average"]) if row.get("vote_average") == row.get("vote_average") else None,
                "source": "local",
            }
        )
    return results


def build_tmdb_tags(movie_data):
    genres = [genre["name"] for genre in movie_data.get("genres", [])]
    keywords = [
        keyword["name"]
        for keyword in movie_data.get("keywords", {}).get("keywords", [])
    ]
    cast = [
        member["name"]
        for member in movie_data.get("credits", {}).get("cast", [])[:5]
    ]

    director = ""
    for member in movie_data.get("credits", {}).get("crew", []):
        if member.get("job") == "Director":
            director = member["name"]
            break

    overview = movie_data.get("overview") or ""

    return " ".join(genres + keywords + cast + [director, overview])


def _external_metadata(movie_data):
    genres = set(genre["name"] for genre in movie_data.get("genres", []))
    keywords = set(
        keyword["name"]
        for keyword in movie_data.get("keywords", {}).get("keywords", [])
    )
    cast = set(
        member["name"]
        for member in movie_data.get("credits", {}).get("cast", [])[:5]
    )

    director = ""
    for member in movie_data.get("credits", {}).get("crew", []):
        if member.get("job") == "Director":
            director = member["name"]
            break

    return genres, keywords, cast, director


def recommend_external_movie(movie_data, n=5, candidate_count=20):
    tags = build_tmdb_tags(movie_data)
    movie_embedding = embedding_model.encode([tags])[0]

    embedding_norm = np.linalg.norm(movie_embedding)
    movie_matrix_norm = np.linalg.norm(movie_embeddings, axis=1)
    semantic_scores = (movie_embeddings @ movie_embedding) / (
        movie_matrix_norm * embedding_norm
    )

    candidate_indices = np.argsort(semantic_scores)[::-1][:candidate_count]
    external_genres, external_keywords, external_cast, external_director = _external_metadata(
        movie_data
    )

    reranked = []
    for candidate_index in candidate_indices:
        movie_data_local = movies.iloc[candidate_index]
        candidate_genres = set(_as_list(movie_data_local["genre_list"]))
        candidate_keywords = set(_as_list(movie_data_local["keyword_list"]))
        candidate_cast = set(_as_list(movie_data_local["cast_list"]))
        candidate_director = movie_data_local["director"]

        genre_score = (
            len(external_genres & candidate_genres) / len(external_genres)
            if external_genres
            else 0
        )
        keyword_score = (
            len(external_keywords & candidate_keywords) / len(external_keywords)
            if external_keywords
            else 0
        )
        cast_score = (
            len(external_cast & candidate_cast) / len(external_cast)
            if external_cast
            else 0
        )
        director_score = float(
            bool(external_director)
            and bool(candidate_director)
            and external_director == candidate_director
        )

        metadata_score = (
            0.40 * genre_score
            + 0.20 * keyword_score
            + 0.20 * cast_score
            + 0.20 * director_score
        )
        semantic_score = float(semantic_scores[candidate_index])
        final_score = 0.90 * semantic_score + 0.10 * metadata_score

        reranked.append(
            (candidate_index, final_score, semantic_score, metadata_score)
        )

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:n]


def recommend(movie, n=5, candidate_count=20):
    movie_index = find_movie_index(movie)
    if movie_index is None:
        print(f"Movie '{movie}' not found.")
        return []

    distances = hybrid_similarity[movie_index]
    candidates = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1],
    )[1 : candidate_count + 1]

    reranked = []
    for candidate_index, hybrid_score in candidates:
        metadata_score = metadata_similarity(movie_index, candidate_index)
        final_score = 0.90 * hybrid_score + 0.10 * metadata_score
        reranked.append((candidate_index, final_score))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return [movies.iloc[i]["title"] for i, _ in reranked[:n]]


def get_movie_poster(title):
    try:
        results = search_movie(title)
        if not results.get("results"):
            return None
        return get_poster_url(results["results"][0].get("poster_path"))
    except Exception as error:
        print(f"TMDB poster lookup failed for '{title}':", error)
        return None


def cached_poster_by_id(tmdb_id):
    tmdb_id = _safe_int(tmdb_id)
    if not tmdb_id:
        return None
    if tmdb_id in _poster_cache:
        return _poster_cache[tmdb_id]

    poster_url = get_poster_by_tmdb_id(tmdb_id)
    _poster_cache[tmdb_id] = poster_url
    return poster_url


def _attach_posters(recommendations):
    ids = [item.get("tmdb_id") for item in recommendations]

    def lookup(tmdb_id):
        return cached_poster_by_id(tmdb_id)

    with ThreadPoolExecutor(max_workers=5) as pool:
        posters = list(pool.map(lookup, ids))

    for item, poster_url in zip(recommendations, posters):
        item["poster_url"] = poster_url

    return recommendations


def _release_year(release_date):
    if release_date:
        return str(release_date)[:4]
    return None


def _candidate_card(
    candidate_index,
    score,
    hybrid_score,
    metadata_score,
    reasons,
):
    movie_data = movies.iloc[candidate_index]
    runtime = movie_data["runtime"]
    rating = movie_data["vote_average"]

    return {
        "index": int(candidate_index),
        "tmdb_id": _tmdb_id_for_row(movie_data),
        "title": movie_data["title"],
        "score": float(score),
        "hybrid_score": float(hybrid_score),
        "metadata_score": float(metadata_score),
        "release_year": _release_year(movie_data["release_date"]),
        "rating": float(rating) if rating == rating else None,
        "runtime": _safe_int(runtime),
        "genres": _as_list(movie_data["genre_list"]),
        "overview": movie_data["overview"] or "",
        "cast": _as_list(movie_data["cast_list"])[:5],
        "director": movie_data["director"] or "",
        "poster_url": None,
        "reasons": reasons,
    }


def explain_recommendation(movie_index, candidate_index):
    reasons = []

    shared_genres = set(_as_list(movies.iloc[movie_index]["genre_list"])) & set(
        _as_list(movies.iloc[candidate_index]["genre_list"])
    )
    if shared_genres:
        reasons.append(
            f"Shares {len(shared_genres)} genre(s): "
            + ", ".join(sorted(shared_genres))
        )

    shared_keywords = set(_as_list(movies.iloc[movie_index]["keyword_list"])) & set(
        _as_list(movies.iloc[candidate_index]["keyword_list"])
    )
    if shared_keywords:
        reasons.append(f"Shares {len(shared_keywords)} keyword(s)")

    shared_cast = set(_as_list(movies.iloc[movie_index]["cast_list"])) & set(
        _as_list(movies.iloc[candidate_index]["cast_list"])
    )
    if shared_cast:
        reasons.append(f"Shares {len(shared_cast)} cast member(s)")

    movie_director = movies.iloc[movie_index]["director"]
    candidate_director = movies.iloc[candidate_index]["director"]
    if movie_director and candidate_director and movie_director == candidate_director:
        reasons.append(f"Same director: {movie_director}")

    if not reasons:
        reasons.append("Strong semantic similarity based on movie content")

    return reasons


def explain_external(movie_data, candidate_index):
    genres, keywords, cast, director = _external_metadata(movie_data)
    reasons = []

    shared_genres = genres & set(_as_list(movies.iloc[candidate_index]["genre_list"]))
    if shared_genres:
        reasons.append(
            f"Shares {len(shared_genres)} genre(s): "
            + ", ".join(sorted(shared_genres))
        )

    shared_keywords = keywords & set(
        _as_list(movies.iloc[candidate_index]["keyword_list"])
    )
    if shared_keywords:
        reasons.append(f"Shares {len(shared_keywords)} keyword(s)")

    shared_cast = cast & set(_as_list(movies.iloc[candidate_index]["cast_list"]))
    if shared_cast:
        reasons.append(f"Shares {len(shared_cast)} cast member(s)")

    candidate_director = movies.iloc[candidate_index]["director"]
    if director and candidate_director and director == candidate_director:
        reasons.append(f"Same director: {director}")

    if not reasons:
        reasons.append("Recommended using semantic similarity to the TMDB movie")

    return reasons


def _hybrid_recommendations(movie_index, n=5, candidate_count=20):
    distances = hybrid_similarity[movie_index]
    candidates = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1],
    )[1 : candidate_count + 1]

    reranked = []
    for candidate_index, hybrid_score in candidates:
        metadata_score = metadata_similarity(movie_index, candidate_index)
        final_score = 0.90 * hybrid_score + 0.10 * metadata_score
        reranked.append(
            _candidate_card(
                candidate_index,
                final_score,
                hybrid_score,
                metadata_score,
                explain_recommendation(movie_index, candidate_index),
            )
        )

    reranked.sort(key=lambda x: x["score"], reverse=True)
    return _attach_posters(reranked[:n])


def _external_recommendations(movie_data, n=5, candidate_count=20):
    ranked = recommend_external_movie(movie_data, n=n, candidate_count=candidate_count)
    cards = [
        _candidate_card(
            candidate_index,
            final_score,
            semantic_score,
            metadata_score,
            explain_external(movie_data, candidate_index),
        )
        for candidate_index, final_score, semantic_score, metadata_score in ranked
    ]
    return _attach_posters(cards)


def recommend_detailed(movie, n=5, candidate_count=20, tmdb_id=None):
    movie_index = find_movie_index(movie, tmdb_id=tmdb_id)

    if movie_index is not None:
        return _hybrid_recommendations(movie_index, n=n, candidate_count=candidate_count)

    results = search_movie(movie)
    if not results.get("results"):
        return []

    tmdb_movie = results["results"][0]
    details = get_movie_details(tmdb_movie["id"])
    if not details:
        return []

    in_dataset = find_movie_by_tmdb_id(details.get("id"))
    if in_dataset is not None:
        return _hybrid_recommendations(in_dataset, n=n, candidate_count=candidate_count)

    return _external_recommendations(details, n=n, candidate_count=candidate_count)
