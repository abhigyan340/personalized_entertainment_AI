"""
src/tag_builder.py
──────────────────
Canonical tag-string construction for semantic movie embeddings.

This is the single source of truth for how a movie's metadata is
converted into the plain-text string that gets passed to the
SentenceTransformer encoder.

Format
──────
    "{overview} {genres} {keywords} {cast_top5} {director}"

All text is lowercased and stripped of leading/trailing whitespace.
Fields come from the pre-extracted clean columns in movies_processed.pkl
(genre_list, keyword_list, cast_list, director, overview), or from live TMDB
metadata for cold-start movies.

Used by
───────
    src/recommender._external_recommendations()  — encodes TMDB movie data
    scripts/regenerate_embeddings.py             — encodes catalog rows
"""

from __future__ import annotations


def _as_list(value) -> list:
    """Coerce any value (list, numpy array, None, …) to a plain Python list."""
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


def build_tag_text(
    overview: str = "",
    genres: list | None = None,
    keywords: list | None = None,
    cast: list | None = None,
    director: str = "",
) -> str:
    """
    Construct the canonical plain-text representation:
        "{overview} {genres} {keywords} {cast_top5} {director}" (lowercased)

    Parameters
    ----------
    overview : str or None
    genres : list-like of str or None
    keywords : list-like of str or None
    cast : list-like of str or None (top 5 cast members)
    director : str or None

    Returns
    -------
    Canonical lowercased, whitespace-normalised tag string.
    """
    ov = str(overview or "")
    g = " ".join(str(x) for x in _as_list(genres) if x)
    k = " ".join(str(x) for x in _as_list(keywords) if x)
    c = " ".join(str(x) for x in _as_list(cast)[:5] if x)
    d = str(director or "")
    return f"{ov} {g} {k} {c} {d}".strip().lower()


def build_tag_from_row(row) -> str:
    """
    Build a canonical tag string from a DataFrame row (movies_processed.pkl).

    Parameters
    ----------
    row : pandas Series or dict-like with keys:
          overview, genre_list, keyword_list, cast_list, director

    Returns
    -------
    Lowercased, whitespace-normalised tag string.
    """
    return build_tag_text(
        overview=row.get("overview") if hasattr(row, "get") else row["overview"],
        genres=row.get("genre_list") if hasattr(row, "get") else row["genre_list"],
        keywords=row.get("keyword_list") if hasattr(row, "get") else row["keyword_list"],
        cast=row.get("cast_list") if hasattr(row, "get") else row["cast_list"],
        director=row.get("director") if hasattr(row, "get") else row["director"],
    )


def build_tag_from_tmdb(movie_data: dict) -> str:
    """
    Build a canonical tag string from a TMDB API movie-details response.

    Used by the out-of-dataset recommendation path to encode a live TMDB
    movie into the same embedding space as the pre-computed catalog.

    Parameters
    ----------
    movie_data : dict returned by TMDB /movie/{id}?append_to_response=credits,keywords

    Returns
    -------
    Lowercased, whitespace-normalised tag string matching build_tag_from_row().
    """
    overview = movie_data.get("overview") or ""
    genres = [g["name"] for g in movie_data.get("genres", []) if isinstance(g, dict) and "name" in g]
    keywords = [k["name"] for k in movie_data.get("keywords", {}).get("keywords", []) if isinstance(k, dict) and "name" in k]
    cast = [m["name"] for m in movie_data.get("credits", {}).get("cast", [])[:5] if isinstance(m, dict) and "name" in m]
    director = ""
    for m in movie_data.get("credits", {}).get("crew", []):
        if isinstance(m, dict) and m.get("job") == "Director":
            director = m.get("name", "")
            break
    return build_tag_text(
        overview=overview,
        genres=genres,
        keywords=keywords,
        cast=cast,
        director=director,
    )
