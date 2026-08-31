from src.tmdb_client import (
    search_movie,
    get_movie_details,
    get_poster_url
)
from src.load_model import movies

# Search for the movie
results = search_movie("Avatar")

print("Number of results:", len(results["results"]))


# Get the first matching movie
movie = results["results"][0]

print("\nFirst result:")
print(movie["title"])
print(movie["release_date"])
print("TMDB ID:", movie["id"])


# Get complete movie details
details = get_movie_details(movie["id"])


print("\nMovie Details:")
print("Title:", details["title"])
print("Overview:", details["overview"])
print("Runtime:", details["runtime"])
print("Rating:", details["vote_average"])
print("Vote count:", details["vote_count"])

poster_url = get_poster_url(
    details["poster_path"]
)

print("Poster URL:", poster_url)