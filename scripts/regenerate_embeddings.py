"""
scripts/regenerate_embeddings.py
─────────────────────────────────
Regenerate data/artifacts/movie_embeddings.npy using the clean metadata
representation from src.tag_builder.build_tag_from_row().

Clean representation (canonical — see src/tag_builder.py):
    "{overview} {genres} {keywords} {cast_top5} {director}"  (lowercased)

All fields come from the already-extracted clean columns in movies_processed.pkl:
    genre_list, keyword_list, cast_list (top-5), director, overview

Why this is different from the original tags column:
    The original tags column contained the full cast JSON array (80+ members with
    cast_id, credit_id, character, gender, order fields) which consumed the
    256-token limit of all-MiniLM-L6-v2, leaving no budget for keywords/director.
    This version contains only semantically meaningful terms.

Run:
    python scripts/regenerate_embeddings.py

Outputs:
    data/artifacts/movie_embeddings.npy  (4803 × 384, float32, L2-normalized)

Does NOT rerun:
    TF-IDF generation, hybrid similarity, or any other artifact.
"""

import os
import sys
import time
import numpy as np

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd
from sentence_transformers import SentenceTransformer

from src.tag_builder import build_tag_from_row


# ── Configuration ─────────────────────────────────────────────────────────────

ARTIFACT_DIR = os.path.join(ROOT, "data", "artifacts")
PKL_PATH     = os.path.join(ARTIFACT_DIR, "movies_processed.pkl")
OUTPUT_PATH  = os.path.join(ARTIFACT_DIR, "movie_embeddings.npy")
MODEL_NAME   = "all-MiniLM-L6-v2"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Load movies
    print(f"Loading {PKL_PATH} ...")
    movies = pd.read_pickle(PKL_PATH)
    print(f"  {len(movies)} movies loaded  (expected: 4803)")
    assert len(movies) == 4803, f"Unexpected movie count: {len(movies)}"

    # 2. Build clean tags using the canonical helper
    print("Building clean tag strings ...")
    tags = [build_tag_from_row(movies.iloc[i]) for i in range(len(movies))]
    print(f"  Sample [0]: {tags[0][:120]}")
    print(f"  Sample [100]: {tags[100][:120]}")

    # Verify no JSON brackets remain
    json_count = sum(1 for t in tags if "{" in t or "}" in t)
    print(f"  Tags with raw JSON brackets: {json_count}  (should be 0)")
    assert json_count == 0, "JSON brackets found in clean tags — check build_clean_tag()"

    # 3. Load model
    print(f"\nLoading SentenceTransformer('{MODEL_NAME}') ...")
    model = SentenceTransformer(MODEL_NAME)
    print("  Model ready.")

    # 4. Encode
    print(f"\nEncoding {len(tags)} movies ...")
    t0 = time.perf_counter()
    embeddings = model.encode(
        tags,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,   # L2-normalize → cosine sim = dot product
        convert_to_numpy=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Encoding time: {elapsed:.1f}s")

    # 5. Validate
    print(f"\nValidating output ...")
    assert embeddings.shape == (4803, 384), f"Wrong shape: {embeddings.shape}"
    assert embeddings.dtype == np.float32,  f"Wrong dtype: {embeddings.dtype}"

    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), \
        f"Embeddings not unit-normalized. min={norms.min():.5f} max={norms.max():.5f}"

    print(f"  Shape:  {embeddings.shape}  ✓")
    print(f"  Dtype:  {embeddings.dtype}  ✓")
    print(f"  Norms:  min={norms.min():.6f}  max={norms.max():.6f}  mean={norms.mean():.6f}  ✓")

    # 6. Row alignment check
    # Verify row 0 is Avatar, row alignment is intact
    assert movies.iloc[0]["title"] == "Avatar", \
        f"Row 0 is not Avatar: {movies.iloc[0]['title']}"
    print(f"  Row alignment:  row 0 = '{movies.iloc[0]['title']}'  ✓")

    # 7. Save
    print(f"\nSaving to {OUTPUT_PATH} ...")
    np.save(OUTPUT_PATH, embeddings)
    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"  Saved  ({size_mb:.2f} MB)")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
