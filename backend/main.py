from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.recommender import recommend_detailed, search_catalog
from src.tmdb_client import search_movie


app = FastAPI(
    title="Personalized Entertainment AI",
    description="Movie recommendation API powered by a hybrid ML recommender.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendationRequest(BaseModel):
    movie: str
    n: int = 5
    tmdb_id: Optional[int] = None


@app.get("/")
def home():
    return {
        "message": "Personalized Entertainment AI API is running"
    }


@app.get("/search")
def search_movies(query: str):
    if not query.strip():
        return {"results": []}

    tmdb = search_movie(query)
    tmdb_results = tmdb.get("results") or []
    if tmdb_results:
        return {"results": tmdb_results[:8], "source": "tmdb"}

    return {
        "results": search_catalog(query),
        "source": "local",
    }


@app.post("/recommend")
def recommend_movies(request: RecommendationRequest):
    results = recommend_detailed(
        request.movie,
        n=request.n,
        tmdb_id=request.tmdb_id,
    )

    if not results:
        return {
            "movie": request.movie,
            "recommendations": [],
            "message": "Movie not found",
        }

    return {
        "movie": request.movie,
        "recommendations": results,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
