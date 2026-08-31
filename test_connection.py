import requests

url = "https://api.themoviedb.org/3/search/movie"

params = {
    "query": "Avatar"
}

response = requests.get(
    url,
    params=params,
    timeout=15
)

print("Status:", response.status_code)
print(response.text[:500])