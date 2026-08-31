import os
import requests
from dotenv import load_dotenv

try:
    from urllib3.util import connection as urllib3_connection

    # Some WiFi networks advertise broken IPv6; mobile hotspots often
    # work because they stay on IPv4. Prefer IPv4 for TMDB.
    urllib3_connection.HAS_IPV6 = False
except Exception:
    pass


load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_READ_ACCESS_TOKEN = os.getenv("TMDB_READ_ACCESS_TOKEN")
TMDB_TIMEOUT = 12


def _headers():
    return {
        "Authorization": f"Bearer {TMDB_READ_ACCESS_TOKEN}",
        "accept": "application/json",
    }


def tmdb_available():
    return bool(TMDB_READ_ACCESS_TOKEN)


def search_movie(movie_title):
    if not movie_title or not tmdb_available():
        return {"results": []}

    try:
        response = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            headers=_headers(),
            params={"query": movie_title},
            timeout=TMDB_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        data.setdefault("results", [])
        return data
    except Exception as error:
        print("TMDB search failed:", error)
        return {"results": []}


def get_movie_details(movie_id):
    if not movie_id or not tmdb_available():
        return None

    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            headers=_headers(),
            params={"append_to_response": "credits,keywords"},
            timeout=TMDB_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except Exception as error:
        print(f"TMDB details failed for {movie_id}:", error)
        return None


def get_movie_summary(movie_id):
    if not movie_id or not tmdb_available():
        return None

    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            headers=_headers(),
            timeout=TMDB_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except Exception as error:
        print(f"TMDB summary failed for {movie_id}:", error)
        return None


def get_poster_url(poster_path):
    if not poster_path:
        return None

    return f"https://image.tmdb.org/t/p/w500{poster_path}"


def get_poster_by_tmdb_id(movie_id):
    summary = get_movie_summary(movie_id)
    if not summary:
        return None
    return get_poster_url(summary.get("poster_path"))
