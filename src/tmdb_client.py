"""
tmdb_client.py
──────────────
Thin wrapper around the TMDB v4 REST API.

All requests use the Read Access Token (Bearer auth).
IPv6 is disabled at the urllib3 level because some networks advertise
broken IPv6 routes; forcing IPv4 improves reliability.
"""

import logging
import os

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Prefer IPv4 — prevents hangs on networks with broken IPv6 routing.
try:
    from urllib3.util import connection as _urllib3_connection
    _urllib3_connection.HAS_IPV6 = False
except Exception:
    pass

load_dotenv()

_TOKEN   = os.getenv("TMDB_READ_ACCESS_TOKEN")
_TIMEOUT = 12  # seconds


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_TOKEN}",
        "accept": "application/json",
    }


def tmdb_available() -> bool:
    """Return True if a TMDB Read Access Token is configured."""
    return bool(_TOKEN)


# ── Search ────────────────────────────────────────────────────────────────────

def search_movie(movie_title: str) -> dict:
    """
    Search TMDB for *movie_title*.
    Returns a dict with a 'results' key (list of movie objects).
    Returns {'results': []} on any error or if TMDB is unavailable.
    """
    if not movie_title or not tmdb_available():
        return {"results": []}

    try:
        response = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            headers=_headers(),
            params={"query": movie_title},
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        data.setdefault("results", [])
        return data
    except Exception as exc:
        logger.warning("TMDB search failed for '%s': %s", movie_title, exc)
        return {"results": []}


# ── Movie details ─────────────────────────────────────────────────────────────

def get_movie_details(movie_id: int) -> dict | None:
    """
    Fetch full movie details including credits and keywords.
    Used by the out-of-dataset recommendation path to build the tag string.
    Returns None on error.
    """
    if not movie_id or not tmdb_available():
        return None

    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            headers=_headers(),
            params={"append_to_response": "credits,keywords"},
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        logger.warning("TMDB details failed for id %s: %s", movie_id, exc)
        return None


# ── Poster helpers ─────────────────────────────────────────────────────────────

def get_poster_url(poster_path: str | None) -> str | None:
    """Convert a TMDB poster_path slug to a full w500 image URL."""
    if not poster_path:
        return None
    return f"https://image.tmdb.org/t/p/w500{poster_path}"


def get_poster_by_tmdb_id(movie_id: int) -> str | None:
    """
    Return the w500 poster URL for a movie given its TMDB id.
    Fetches only the base movie object (no credits / keywords).
    """
    if not movie_id or not tmdb_available():
        return None

    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            headers=_headers(),
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        return get_poster_url(response.json().get("poster_path"))
    except Exception as exc:
        logger.warning("TMDB poster fetch failed for id %s: %s", movie_id, exc)
        return None
