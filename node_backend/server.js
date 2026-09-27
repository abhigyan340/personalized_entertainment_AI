/**
 * server.js
 * ─────────
 * Node.js / Express API gateway + static file server.
 *
 * Responsibilities
 * ─────────────────
 *  - Serve the frontend (HTML/CSS/JS) as static files
 *  - Proxy /api/recommend and /api/search to the FastAPI ML service
 *  - Handle TMDB enrichment (poster URL, IMDb link, Wikipedia link)
 *    using a single TMDB call per movie with in-memory caching
 *
 * Environment variables (see ../.env.example)
 * ────────────────────────────────────────────
 *  PORT             Node server port          (default: 3000)
 *  ML_SERVICE_URL   FastAPI base URL          (default: http://127.0.0.1:8000)
 *  TMDB_READ_ACCESS_TOKEN   TMDB v4 token     (required for TMDB features)
 */

"use strict";

const express = require("express");
const axios   = require("axios");
const cors    = require("cors");
const dotenv  = require("dotenv");
const path    = require("path");
const dns     = require("dns");

// Prefer IPv4 — prevents hangs on networks with broken IPv6 routing.
dns.setDefaultResultOrder("ipv4first");

dotenv.config({ path: path.join(__dirname, "..", ".env") });

// ── Configuration ─────────────────────────────────────────────────────────────
const PORT             = parseInt(process.env.PORT            || "3000", 10);
const ML_SERVICE_URL   = process.env.ML_SERVICE_URL           || "http://127.0.0.1:8000";
const TMDB_TOKEN       = process.env.TMDB_READ_ACCESS_TOKEN   || "";
const FRONTEND_DIR     = path.join(__dirname, "..", "frontend");

const TMDB_HEADERS = {
    Authorization: `Bearer ${TMDB_TOKEN}`,
    accept: "application/json",
};

// ── App setup ─────────────────────────────────────────────────────────────────
const app = express();
app.use(express.json());
app.use(cors());


// ── Health ────────────────────────────────────────────────────────────────────
app.get("/api/health", (_req, res) => {
    res.json({
        status:       "ok",
        service:      "Personalized Entertainment AI (Node)",
        ml_service:   ML_SERVICE_URL,
        tmdb_enabled: Boolean(TMDB_TOKEN),
    });
});


// ── Search ────────────────────────────────────────────────────────────────────
app.get("/api/search", async (req, res) => {
    const { query } = req.query;

    if (!query || !String(query).trim()) {
        return res.status(400).json({ error: "query parameter is required" });
    }

    // Primary: TMDB live search (includes poster thumbnails for the dropdown)
    if (TMDB_TOKEN) {
        try {
            const tmdbRes = await axios.get(
                "https://api.themoviedb.org/3/search/movie",
                {
                    headers: TMDB_HEADERS,
                    params:  { query },
                    timeout: 12000,
                    family:  4,
                }
            );
            const results = tmdbRes.data.results || [];
            if (results.length > 0) {
                return res.json({ results, source: "tmdb" });
            }
        } catch (err) {
            console.error("TMDB search error:", err.response?.data || err.message);
        }
    }

    // Fallback: local catalog via FastAPI
    try {
        const localRes = await axios.get(`${ML_SERVICE_URL}/search`, {
            params:  { query },
            timeout: 15000,
            family:  4,
        });
        return res.json({
            results: localRes.data.results || [],
            source:  localRes.data.source  || "local",
        });
    } catch (err) {
        console.error("Local search error:", err.message);
        return res.status(500).json({ error: "Unable to search movies" });
    }
});


// ── Recommend ─────────────────────────────────────────────────────────────────
// Returns ML recommendations immediately — no TMDB blocking.
// The frontend calls /api/enrich per card to load posters + links async.
app.post("/api/recommend", async (req, res) => {
    const { movie, n = 5, tmdb_id = null } = req.body;

    if (!movie || !String(movie).trim()) {
        return res.status(400).json({ error: "movie is required" });
    }
    if (n !== undefined && (n < 1 || n > 20)) {
        return res.status(400).json({ error: "n must be between 1 and 20" });
    }

    try {
        const mlRes = await axios.post(
            `${ML_SERVICE_URL}/recommend`,
            { movie, n, tmdb_id },
            { timeout: 120000, family: 4 }
        );
        return res.json(mlRes.data);
    } catch (err) {
        console.error("ML service error:", err.message);
        return res.status(502).json({
            error: `Unable to reach the ML service at ${ML_SERVICE_URL}. `
                 + "Make sure the FastAPI backend is running.",
        });
    }
});


// ── Semantic search ───────────────────────────────────────────────────────────
// Proxies to FastAPI POST /semantic-search.
// Accepts { query, n } and forwards the response directly.
app.post("/api/semantic-search", async (req, res) => {
    const { query, n = 10 } = req.body;

    if (!query || !String(query).trim()) {
        return res.status(400).json({ error: "query is required" });
    }
    if (n !== undefined && (n < 1 || n > 50)) {
        return res.status(400).json({ error: "n must be between 1 and 50" });
    }

    try {
        const mlRes = await axios.post(
            `${ML_SERVICE_URL}/semantic-search`,
            { query, n },
            { timeout: 30000, family: 4 }
        );
        return res.json(mlRes.data);
    } catch (err) {
        console.error("Semantic search error:", err.message);
        return res.status(502).json({
            error: `Unable to reach the ML service at ${ML_SERVICE_URL}. `
                 + "Make sure the FastAPI backend is running.",
        });
    }
});


// ── Enrich ────────────────────────────────────────────────────────────────────
// Called by the frontend after each recommendation card renders.
// Returns { poster_url, imdb_url, wikipedia_url } for a single movie.
// Uses a single TMDB /movie/{id} call (poster_path + imdb_id in one response).
// Results are cached in-memory by tmdb_id (or title as fallback key).

/** @type {Map<string, {poster_url:string|null, imdb_url:string, wikipedia_url:string}>} */
const enrichCache = new Map();

app.get("/api/enrich", async (req, res) => {
    const { tmdb_id, title, release_year } = req.query;

    if (!title || !String(title).trim()) {
        return res.status(400).json({ error: "title is required" });
    }

    // Fallback result — always safe to return even if TMDB fails
    const wikiSlug = String(title).replace(/ /g, "_");
    const result = {
        poster_url:    null,
        imdb_url:      `https://www.imdb.com/find/?q=${encodeURIComponent(title)}&s=tt`,
        wikipedia_url: `https://en.wikipedia.org/wiki/${encodeURIComponent(wikiSlug)}`,
    };

    if (!TMDB_TOKEN) {
        return res.json(result);
    }

    const cacheKey = tmdb_id ? String(tmdb_id) : String(title).toLowerCase();
    if (enrichCache.has(cacheKey)) {
        return res.json(enrichCache.get(cacheKey));
    }

    try {
        if (tmdb_id) {
            // Single call: /movie/{id} returns poster_path + imdb_id together
            const detailRes = await axios.get(
                `https://api.themoviedb.org/3/movie/${tmdb_id}`,
                { headers: TMDB_HEADERS, timeout: 10000, family: 4 }
            );
            const d = detailRes.data;
            if (d.poster_path) {
                result.poster_url = `https://image.tmdb.org/t/p/w500${d.poster_path}`;
            }
            if (d.imdb_id) {
                result.imdb_url = `https://www.imdb.com/title/${d.imdb_id}/`;
            }
            const slug = (d.title || title).replace(/ /g, "_");
            result.wikipedia_url = `https://en.wikipedia.org/wiki/${encodeURIComponent(slug)}`;
        } else {
            // No tmdb_id — search by title (1 call; no imdb_id available from search)
            const searchRes = await axios.get(
                "https://api.themoviedb.org/3/search/movie",
                {
                    headers: TMDB_HEADERS,
                    params:  { query: title, ...(release_year ? { year: release_year } : {}) },
                    timeout: 10000,
                    family:  4,
                }
            );
            const match = (searchRes.data.results || [])[0];
            if (match) {
                if (match.poster_path) {
                    result.poster_url = `https://image.tmdb.org/t/p/w500${match.poster_path}`;
                }
                const slug = (match.title || title).replace(/ /g, "_");
                result.wikipedia_url = `https://en.wikipedia.org/wiki/${encodeURIComponent(slug)}`;
            }
        }
    } catch (err) {
        console.error(`Enrich failed for "${title}":`, err.code || err.message);
        // Return fallback URLs already set above
    }

    enrichCache.set(cacheKey, result);
    return res.json(result);
});


// ── Static frontend ───────────────────────────────────────────────────────────
app.use(express.static(FRONTEND_DIR));


// ── Global error handler ─────────────────────────────────────────────────────
// Express 5 forwards unhandled async errors here.
// Without this, Express returns an HTML error page which the frontend
// cannot parse as JSON, causing "Unexpected end of JSON input".
app.use((err, _req, res, next) => {
    console.error("Unhandled error:", err.stack || err.message);
    if (res.headersSent) return next(err);
    res.status(err.status || 500).json({
        error: err.message || "Internal server error",
    });
});


// ── Start ─────────────────────────────────────────────────────────────────────
app.listen(PORT, "0.0.0.0", () => {
    console.log(`Node server  →  http://127.0.0.1:${PORT}`);
    console.log(`ML service   →  ${ML_SERVICE_URL}`);
    console.log(`TMDB         →  ${TMDB_TOKEN ? "enabled" : "disabled (no token)"}`);
});
