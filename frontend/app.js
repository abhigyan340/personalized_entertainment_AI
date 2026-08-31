const API_BASE =
    window.location.protocol === "file:"
        ? "http://127.0.0.1:3000"
        : "";

const movieInput = document.getElementById("movieInput");
const recommendButton = document.getElementById("recommendButton");
const searchResults = document.getElementById("searchResults");

let selectedTMDBMovie = null;
let highlightedIndex = -1;
let currentSuggestions = [];


function posterUrl(posterPath) {
    if (!posterPath) {
        return null;
    }

    if (String(posterPath).startsWith("http")) {
        return posterPath;
    }

    return "https://image.tmdb.org/t/p/w92" + posterPath;
}


async function searchMovies(query) {
    const response = await fetch(
        API_BASE + "/api/search?query=" + encodeURIComponent(query)
    );

    if (!response.ok) {
        const text = await response.text();
        let message = "Search failed";
        try { message = JSON.parse(text).error || message; } catch (_) { if (text) message = text; }
        throw new Error(message);
    }

    const data = await response.json();
    return data.results || [];
}


function hideSuggestions() {
    if (!searchResults) {
        return;
    }

    searchResults.innerHTML = "";
    searchResults.style.display = "none";
    highlightedIndex = -1;
    currentSuggestions = [];
}


function selectSuggestion(movie) {
    selectedTMDBMovie = movie;
    movieInput.value = movie.title;
    hideSuggestions();
}

// Selects a suggestion AND immediately navigates to the results page.
function pickSuggestion(movie) {
    selectSuggestion(movie);
    goToResults();
}


function displaySearchResults(movies) {
    currentSuggestions = (movies || []).slice(0, 6);
    highlightedIndex = -1;

    if (!searchResults) {
        return;
    }

    searchResults.innerHTML = "";

    if (currentSuggestions.length === 0) {
        searchResults.style.display = "none";
        return;
    }

    currentSuggestions.forEach((movie, index) => {
        const item = document.createElement("div");
        item.className = "search-result-item";
        item.setAttribute("data-index", String(index));

        const year = movie.release_date
            ? movie.release_date.substring(0, 4)
            : "";

        const image = posterUrl(movie.poster_path);

        if (image) {
            const img = document.createElement("img");
            img.alt = movie.title;
            img.className = "search-result-thumb";
            // Show placeholder immediately, swap in real image when loaded
            img.style.background = "rgba(255,255,255,0.06)";
            img.onload = () => img.classList.add("loaded");
            img.onerror = () => { img.style.display = "none"; };
            img.src = image; // browser fetches in parallel — no extra fetch call
            item.appendChild(img);
        } else {
            const placeholder = document.createElement("div");
            placeholder.className = "search-result-placeholder";
            placeholder.textContent = "✦";
            item.appendChild(placeholder);
        }

        const info = document.createElement("div");
        info.className = "search-result-info";

        const title = document.createElement("strong");
        title.textContent = movie.title;

        const meta = document.createElement("span");
        meta.textContent = year || "Unknown year";

        info.appendChild(title);
        info.appendChild(meta);
        item.appendChild(info);

        // mousedown fires before blur — navigates directly to results
        item.addEventListener("mousedown", (event) => {
            event.preventDefault();
            pickSuggestion(movie);
        });

        searchResults.appendChild(item);
    });

    searchResults.style.display = "block";
}


function highlightSuggestion(nextIndex) {
    const items = searchResults
        ? searchResults.querySelectorAll(".search-result-item")
        : [];

    if (!items.length) {
        return;
    }

    highlightedIndex =
        (nextIndex + items.length) % items.length;

    items.forEach((item, index) => {
        item.classList.toggle("active", index === highlightedIndex);
    });
}


function goToResults() {
    const movie = movieInput.value.trim();

    if (!movie) {
        movieInput.focus();
        return;
    }

    localStorage.setItem("selectedMovie", movie);

    if (selectedTMDBMovie && selectedTMDBMovie.id) {
        localStorage.setItem(
            "selectedMovieTMDBId",
            String(selectedTMDBMovie.id)
        );
    } else {
        localStorage.removeItem("selectedMovieTMDBId");
    }

    window.location.href = "results.html";
}


if (movieInput && recommendButton) {
    let searchTimeout;

    movieInput.addEventListener("input", () => {
        const query = movieInput.value.trim();
        selectedTMDBMovie = null;
        clearTimeout(searchTimeout);

        if (!query) {
            hideSuggestions();
            return;
        }

        searchTimeout = setTimeout(async () => {
            try {
                const results = await searchMovies(query);
                displaySearchResults(results);
            } catch (error) {
                console.error("TMDB search error:", error);
                hideSuggestions();
            }
        }, 150);
    });

    movieInput.addEventListener("keydown", (event) => {
        const items = searchResults
            ? searchResults.querySelectorAll(".search-result-item")
            : [];

        if (event.key === "ArrowDown" && items.length) {
            event.preventDefault();
            highlightSuggestion(highlightedIndex + 1);
            return;
        }

        if (event.key === "ArrowUp" && items.length) {
            event.preventDefault();
            highlightSuggestion(highlightedIndex - 1);
            return;
        }

        if (event.key === "Escape") {
            hideSuggestions();
            return;
        }

        if (event.key === "Enter") {
            event.preventDefault();

            if (highlightedIndex >= 0 && currentSuggestions[highlightedIndex]) {
                pickSuggestion(currentSuggestions[highlightedIndex]);
                return;
            }

            if (currentSuggestions.length === 1) {
                selectSuggestion(currentSuggestions[0]);
            }

            goToResults();
        }
    });

    movieInput.addEventListener("blur", () => {
        setTimeout(hideSuggestions, 150);
    });

    recommendButton.addEventListener("click", goToResults);

    document.querySelectorAll(".example-movie").forEach((button) => {
        button.addEventListener("click", async () => {
            const movie = button.textContent.trim();
            movieInput.value = movie;
            selectedTMDBMovie = null;

            try {
                const results = await searchMovies(movie);
                if (results.length > 0) {
                    selectSuggestion(results[0]);
                    movieInput.value = results[0].title;
                }
            } catch (error) {
                console.error("Example search error:", error);
            }

            movieInput.focus();
        });
    });
}


const resultsGrid = document.getElementById("resultsGrid");

if (resultsGrid) {
    loadRecommendations();
}


async function loadRecommendations() {
    const movie = localStorage.getItem("selectedMovie");
    const tmdbId = localStorage.getItem("selectedMovieTMDBId");

    if (!movie) {
        window.location.href = "index.html";
        return;
    }

    const searchedMovie = document.getElementById("searchedMovie");

    if (searchedMovie) {
        searchedMovie.textContent = `"${movie}"`;
    }

    resultsGrid.innerHTML = `
        <div class="empty-state">
            <div class="empty-icon">✦</div>
            <h3>Finding your next movies...</h3>
            <p>Our AI recommendation engine is analyzing "${movie}".</p>
        </div>
    `;

    try {
        const payload = {
            movie: movie,
            n: 5
        };

        if (tmdbId) {
            payload.tmdb_id = Number(tmdbId);
        }

        const response = await fetch(API_BASE + "/api/recommend", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const text = await response.text();
            let message = "Unable to get recommendations.";
            try { message = JSON.parse(text).error || message; } catch (_) { if (text) message = text; }
            throw new Error(message);
        }

        const data = await response.json();

        displayRecommendations(data.recommendations);
    } catch (error) {
        console.error(error);

        resultsGrid.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">!</div>
                <h3>Something went wrong</h3>
                <p>${error.message}</p>
            </div>
        `;
    }
}


function displayRecommendations(recommendations) {
    if (!recommendations || recommendations.length === 0) {
        resultsGrid.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">?</div>
                <h3>No recommendations found</h3>
                <p>Try searching for another movie, or pick a title from the suggestions.</p>
            </div>
        `;
        return;
    }

    resultsGrid.innerHTML = "";

    recommendations.forEach((movie, index) => {
        const card = document.createElement("article");
        card.className = "movie-card";

        const matchPercentage = Math.round((movie.score || 0) * 100);
        const hybridPercentage = Math.round((movie.hybrid_score || 0) * 100);
        const metadataPercentage = Math.round((movie.metadata_score || 0) * 100);
        const genres = movie.genres || [];
        const cast = (movie.cast || []).slice(0, 5);

        const genresHTML = genres
            .map((genre) => `<span class="genre-tag">${genre}</span>`)
            .join("");

        const castHTML = cast
            .map((actor) => `<span class="cast-name">${actor}</span>`)
            .join(", ");

        const poster = movie.poster_url
            ? `<img src="${movie.poster_url}" alt="${movie.title}" class="movie-poster-image">`
            : `<div class="poster-placeholder"><span>${index + 1}</span></div>`;

        card.innerHTML = `
            <div class="movie-rank">#${index + 1}</div>
            <div class="movie-poster" id="poster-${index}">${poster}</div>
            <div class="movie-info">
                <h3>${movie.title}</h3>
                <div class="movie-meta">
                    <span>${movie.release_year || "—"}</span>
                    <span>⭐ ${movie.rating ?? "—"}</span>
                    <span>${movie.runtime ? movie.runtime + " min" : "—"}</span>
                </div>
                <div class="genre-list">${genresHTML}</div>
                <p class="movie-overview">${movie.overview || ""}</p>
                <div class="match-header">
                    <span>AI MATCH</span>
                    <strong>${matchPercentage}%</strong>
                </div>
                <div class="match-bar">
                    <div class="match-fill" style="width: ${matchPercentage}%"></div>
                </div>
                <div class="reason">${formatReasons(movie.reasons)}</div>
                <div class="model-scores">
                    <div class="model-score">
                        <span>Hybrid similarity</span>
                        <strong>${hybridPercentage}%</strong>
                    </div>
                    <div class="model-score">
                        <span>Metadata similarity</span>
                        <strong>${metadataPercentage}%</strong>
                    </div>
                </div>
                <div class="movie-cast">
                    <span class="detail-label">CAST</span>
                    <p>${castHTML || "—"}</p>
                </div>
                <div class="movie-director">
                    <span class="detail-label">DIRECTOR</span>
                    <p>${movie.director || "Unknown"}</p>
                </div>
                <div class="movie-links">
                    <a
                        id="imdb-link-${index}"
                        href="https://www.imdb.com/find/?q=${encodeURIComponent(movie.title)}&s=tt"
                        target="_blank"
                        rel="noopener noreferrer"
                        class="movie-link-btn imdb-btn"
                        title="View on IMDb"
                    >
                        <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14" aria-hidden="true"><path d="M14.31 9.588v.005c-.077-.048-.227-.07-.42-.07v4.815c.27 0 .44-.057.5-.17.062-.115.095-.4.095-.865V10.6c0-.42-.022-.694-.062-.817a.344.344 0 0 0-.114-.195zM12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zM7.18 16.3H5.5V7.7h1.68V16.3zm4.252 0H9.832v-.522c-.347.405-.73.608-1.145.608-.332 0-.57-.106-.712-.315-.143-.21-.213-.54-.213-.992V9.578h1.597v5.13c0 .207.008.335.025.382.036.098.113.147.226.147.148 0 .3-.075.45-.225V9.578h1.37V16.3zm4.212-.783c0 .44-.05.76-.15.96-.14.285-.4.43-.79.43-.31 0-.617-.13-.92-.39V16.3H13.1V7.7h1.684v2.55c.293-.246.593-.37.9-.37.38 0 .644.148.79.443.103.208.154.54.154.99V15.517zm3.876-3.074h-1.614v.95c0 .454.015.727.042.82.028.09.1.135.21.135.142 0 .232-.057.27-.174.037-.116.056-.4.056-.853v-.36h1.037v.394c0 .47-.014.8-.043.99-.028.19-.11.37-.247.54-.136.17-.32.3-.552.39-.23.09-.5.135-.8.135-.29 0-.556-.04-.793-.12a1.27 1.27 0 0 1-.546-.367 1.4 1.4 0 0 1-.258-.548c-.044-.2-.065-.508-.065-.922v-2.14c0-.455.024-.784.072-.987.047-.204.155-.386.323-.544.168-.157.37-.278.61-.36.237-.083.504-.124.8-.124.302 0 .572.045.81.134.237.09.426.213.564.37.14.156.227.33.264.518.038.19.057.49.057.9v.764z"/></svg>
                        IMDb
                    </a>
                    <a
                        id="wiki-link-${index}"
                        href="https://en.wikipedia.org/wiki/${encodeURIComponent(movie.title.replace(/ /g, '_'))}"
                        target="_blank"
                        rel="noopener noreferrer"
                        class="movie-link-btn wiki-btn"
                        title="View on Wikipedia"
                    >
                        <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14" aria-hidden="true"><path d="M12.09 13.119c-.936 1.932-2.217 4.548-2.853 5.728-.616 1.074-1.127.993-1.688.58-.28-.196-1.333-1.713-1.333-1.713C5.083 17.48 5 14.418 5 13.559c0-4.844 2.37-5.667 4.578-5.667-.022.18-.073 1.3.198 1.867.14.294.418.44.783.44.757 0 1.178-.516 1.38-1.055.105-.29.08-.572.008-.803-.07-.22-.19-.434-.346-.63a.67.67 0 0 1-.15-.5c0-.428.504-.713.965-.713.766 0 2.253.756 2.253 3.713 0 1.48-.252 2.984-.578 4.108zm1.14.636C13.89 11.44 14.33 9.35 14.33 7.94c0-2.917-1.583-4.68-3.714-4.68-.65 0-1.46.217-2.02.652a1.93 1.93 0 0 0-.777 1.578c0 .663.302 1.23.729 1.708.03.033.057.066.08.1.018.03.032.063.044.097.033.09.02.187-.017.27a2.127 2.127 0 0 1-.407.564c-.266.265-.69.62-1.218.62-.69 0-1.17-.487-1.33-1.167C5.58 6.39 5.5 5.63 5.5 4.87 5.5 2.185 7.695 0 12 0s6.5 2.185 6.5 4.87c0 3.73-2.61 7.15-5.27 9.885zm.98 6.245L12 24l-2.21-4h4.42z"/></svg>
                        Wikipedia
                    </a>
                </div>
            </div>
        `;

        resultsGrid.appendChild(card);

        // Asynchronously enrich this card with poster + direct links
        enrichCard(index, movie);
    });
}


// Fetches poster + direct IMDb/Wikipedia URLs for a single card
// after it has already been rendered, then updates the DOM in-place.
async function enrichCard(index, movie) {
    try {
        const params = new URLSearchParams({ title: movie.title });
        if (movie.tmdb_id) params.set("tmdb_id", movie.tmdb_id);
        if (movie.release_year) params.set("release_year", movie.release_year);

        const response = await fetch(API_BASE + "/api/enrich?" + params.toString());
        if (!response.ok) return;

        const data = await response.json();

        // Update poster
        if (data.poster_url) {
            const posterEl = document.getElementById(`poster-${index}`);
            if (posterEl) {
                posterEl.innerHTML = `<img src="${data.poster_url}" alt="${movie.title}" class="movie-poster-image">`;
            }
        }

        // Update IMDb link to direct page
        if (data.imdb_url) {
            const imdbEl = document.getElementById(`imdb-link-${index}`);
            if (imdbEl) imdbEl.href = data.imdb_url;
        }

        // Update Wikipedia link to direct article
        if (data.wikipedia_url) {
            const wikiEl = document.getElementById(`wiki-link-${index}`);
            if (wikiEl) wikiEl.href = data.wikipedia_url;
        }
    } catch (_) {
        // silently skip — fallback links already in place
    }
}


function formatReasons(reasons) {
    if (!reasons || reasons.length === 0) {
        return `
            <p class="reason-empty">
                Recommended based on overall movie similarity.
            </p>
        `;
    }

    return reasons
        .map(
            (reason) => `
                <div class="reason-item">
                    <span class="reason-check">✓</span>
                    <span>${reason}</span>
                </div>
            `
        )
        .join("");
}
