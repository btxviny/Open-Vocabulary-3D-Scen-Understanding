"""
ChromaDB vector store from a processed .npz point cloud.

Usage (from repo root):
    uv run python -m src.scene_search.vectorstore \\
        --npz_file output/dense_pointcloud.npz \\
        --collection_name my_scene_points
"""

import argparse
from pathlib import Path

import chromadb
import numpy as np
from tqdm import tqdm

from .utils import sanitize_collection_name

_MAX_BATCH = 5461  # ChromaDB hard limit per add() call


def _setup_collection(npz_path: str, collection_name: str) -> chromadb.Collection:
    collection_name = sanitize_collection_name(collection_name)
    store_dir = str(Path(npz_path).parent / "vector_store")
    Path(store_dir).mkdir(exist_ok=True)

    client = chromadb.PersistentClient(path=store_dir)
    if any(c.name == collection_name for c in client.list_collections()):
        client.delete_collection(collection_name)

    return client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def create_vectorstore(npz_path: str, collection_name: str) -> None:
    """Index every point as a separate ChromaDB entry keyed by its array index."""
    data = np.load(npz_path)
    embeddings = data["embeddings"]

    valid = np.linalg.norm(embeddings, axis=1) > 1e-3
    n_removed = (~valid).sum()
    if n_removed:
        print(f"Skipping {n_removed} zero-norm embeddings")

    valid_ids  = np.where(valid)[0]
    embeddings = embeddings[valid]
    print(f"Indexing {len(valid_ids):,} points…")

    collection = _setup_collection(npz_path, collection_name)
    for i in tqdm(range(0, len(valid_ids), _MAX_BATCH), desc="Uploading"):
        batch_ids  = valid_ids[i:i + _MAX_BATCH]
        batch_embs = embeddings[i:i + _MAX_BATCH]
        collection.add(
            ids=[str(j) for j in batch_ids],
            documents=[f"point_{j}" for j in batch_ids],
            embeddings=batch_embs.tolist(),
        )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ChromaDB vector store from point cloud")
    parser.add_argument("--npz_file",        required=True)
    parser.add_argument("--collection_name", required=True)
    args = parser.parse_args()
    create_vectorstore(args.npz_file, args.collection_name)
