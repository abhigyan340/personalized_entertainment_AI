const express = require("express");
const axios = require("axios");
const cors = require("cors");
const dotenv = require("dotenv");
const path = require("path");
const dns = require("dns");

dns.setDefaultResultOrder("ipv4first");

dotenv.config({
    path: path.join(__dirname, "..", ".env")
});

const app = express();

const PORT = 3000;
const ML_SERVICE_URL = "http://127.0.0.1:8000";

const TMDB_READ_ACCESS_TOKEN = process.env.TMDB_READ_ACCESS_TOKEN;

const frontendDir = path.join(__dirname, "..", "frontend");

app.use(express.json());
app.use(cors());


// ── Health ────────────────────────────────────────────────
app.get("/api/health", (req, res) => {
    res.json({ message: "Personalized Entertainment AI Node.js API is running" });
});


// ── Search ────────────────────────────────────────────────
app.get("/api/search", async (req, res) => {
    try {
        const { query } = req.query;

        if (!query || !String(query).trim()) {
            return res.status(400).json({ error: "Search query is required" });
        }

        if (TMDB_READ_ACCESS_TOKEN) {
            try {
                const tmdbResponse = await axios.get(
                    "https://api.themoviedb.org/3/search/movie",
                    {
                        headers: {
                            Authorization: `Bearer ${TMDB_READ_ACCESS_TOKEN}`,
                            accept: "application/json"
                        },
                        params: { query },
                        timeout: 12000,
                        family: 4
                    }
                );
                const results = tmdbResponse.data.results || [];
                if (results.length > 0) {
                    return res.json({ results, source: "tmdb" });
                }
            } catch (tmdbError) {
                console.error("TMDB search error:", tmdbError.response?.data || tmdbError.message);
            }
        }

        const localResponse = await axios.get(
            `${ML_SERVICE_URL}/search`,
            { params: { query }, timeout: 15000, family: 4 }
        );
        return res.json({
            results: localResponse.data.results || [],
            source: localResponse.data.source || "local"
        });
    } catch (error) {
        console.error("Search error:", error.message);
        res.status(500).json({ error: "Unable to search movies" });
    }
});


// ── Recommend ─────────────────────────────────────────────
// Returns ML recommendations immediately — no TMDB blocking.
// The frontend calls /api/enrich in the background to load
// posters, IMDb URLs, and Wikipedia URLs asynchronously.
app.post("/api/recommend", async (req, res) => {
    try {
        const { movie, n = 5, tmdb_id = null } = req.body;

        if (!movie) {
            return res.status(400).json({ error: "Movie name is required" });
        }

        const response = await axios.post(
            `${ML_SERVICE_URL}/recommend`,
            { movie, n, tmdb_id },
            { timeout: 120000, family: 4 }
        );

        res.json(response.data);
    } catch (error) {
        console.error("ML service error:", error.message);
        res.status(500).json({
            error: "Unable to get recommendations from ML service. Make sure the FastAPI backend is running on port 8000."
        });
    }
});


// ── Enrich ────────────────────────────────────────────────
// Called by the frontend after cards render.
// Accepts: { tmdb_id, title, release_year }
// Returns: { poster_url, imdb_url, wikipedia_url }
app.get("/api/enrich", async (req, res) => {
    const { tmdb_id, title, release_year } = req.query;

    if (!title) {
        return res.status(400).json({ error: "title is required" });
    }

    const result = {
        poster_url: null,
        imdb_url: null,
        wikipedia_url: null
    };

    // Always set Wikipedia and IMDb fallback from title immediately
    const wikiSlug = String(title).replace(/ /g, "_");
    result.wikipedia_url = `https://en.wikipedia.org/wiki/${encodeURIComponent(wikiSlug)}`;
    result.imdb_url = `https://www.imdb.com/find/?q=${encodeURIComponent(title)}&s=tt`;

    if (!TMDB_READ_ACCESS_TOKEN) {
        return res.json(result);
    }

    try {
        // Use TMDB search — proven fastest endpoint on this machine
        const searchRes = await axios.get(
            "https://api.themoviedb.org/3/search/movie",
            {
                headers: {
                    Authorization: `Bearer ${TMDB_READ_ACCESS_TOKEN}`,
                    accept: "application/json"
                },
                params: {
                    query: title,
                    ...(release_year ? { year: release_year } : {})
                },
                timeout: 10000,
                family: 4
            }
        );

        const results = searchRes.data.results || [];
        const tid = tmdb_id ? Number(tmdb_id) : null;
        const match = (tid && results.find(r => r.id === tid)) || results[0];

        if (match) {
            // Poster
            if (match.poster_path) {
                result.poster_url = `https://image.tmdb.org/t/p/w500${match.poster_path}`;
            }
            // Wikipedia from confirmed TMDB title
            const confirmedSlug = (match.title || title).replace(/ /g, "_");
            result.wikipedia_url = `https://en.wikipedia.org/wiki/${encodeURIComponent(confirmedSlug)}`;

            // Fetch external_ids to get the real IMDb tt-id
            try {
                const extRes = await axios.get(
                    `https://api.themoviedb.org/3/movie/${match.id}/external_ids`,
                    {
                        headers: {
                            Authorization: `Bearer ${TMDB_READ_ACCESS_TOKEN}`,
                            accept: "application/json"
                        },
                        timeout: 10000,
                        family: 4
                    }
                );
                if (extRes.data.imdb_id) {
                    result.imdb_url = `https://www.imdb.com/title/${extRes.data.imdb_id}/`;
                }
            } catch (_) {
                // keep the search-based IMDb fallback already set
            }
        }
    } catch (err) {
        console.error(`Enrich failed for "${title}":`, err.code || err.message);
        // fallback URLs are already set above — just return them
    }

    res.json(result);
});


app.use(express.static(frontendDir));

// Global JSON error handler
app.use((err, req, res, next) => {
    console.error("Unhandled error:", err.stack || err.message);
    if (res.headersSent) return next(err);
    res.status(err.status || 500).json({ error: err.message || "Internal server error" });
});

app.listen(PORT, "0.0.0.0", () => {
    console.log(`Node.js server running on http://127.0.0.1:${PORT}`);
});
