"""
Build a ChromaDB vector store from a processed point cloud .npz file.

Usage (from repo root):
    # Index individual points
    uv run python -m scene_search.vectorstore \
        --npz_file output/dense_pointcloud.npz \
        --logic points \
        --collection_name my_scene_points

    # Index cluster centroids
    uv run python -m scene_search.vectorstore \
        --npz_file output/dense_pointcloud.npz \
        --logic clusters \
        --collection_name my_scene_clusters
"""

import argparse
import os
from pathlib import Path

import chromadb
import numpy as np
from tqdm import tqdm

_MAX_BATCH = 5461  # ChromaDB hard limit per add() call


def _setup_collection(npz_path: str, collection_name: str) -> tuple[chromadb.Client, chromadb.Collection]:
    vector_store_dir = str(Path(npz_path).parent / "vector_store")
    Path(vector_store_dir).mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=vector_store_dir)
    # Drop existing collection so re-runs are idempotent
    if any(c.name == collection_name for c in client.list_collections()):
        client.delete_collection(collection_name)
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


def _filter_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Return boolean mask of embeddings with non-zero norm."""
    norms = np.linalg.norm(embeddings, axis=1)
    mask = norms > 1e-3
    n_removed = (~mask).sum()
    if n_removed:
        print(f"Removing {n_removed} zero-norm embeddings ({len(embeddings)} total)")
    return mask


def create_point_vectorstore(npz_path: str, collection_name: str | None = None) -> None:
    """Index every point as a separate entry, keyed by its index in the array."""
    if collection_name is None:
        collection_name = os.path.splitext(os.path.basename(npz_path))[0] + "_points"

    data = np.load(npz_path)
    embeddings = data["embeddings"]
    mask = _filter_embeddings(embeddings)

    valid_ids  = np.where(mask)[0]
    embeddings = embeddings[mask]
    print(f"Indexing {len(valid_ids)} points…")

    _, collection = _setup_collection(npz_path, collection_name)

    for i in tqdm(range(0, len(valid_ids), _MAX_BATCH), desc="Uploading (points)"):
        batch_ids  = valid_ids[i:i + _MAX_BATCH]
        batch_embs = embeddings[i:i + _MAX_BATCH]
        collection.add(
            ids=[str(j) for j in batch_ids],
            documents=[f"point_{j}" for j in batch_ids],
            embeddings=batch_embs.tolist(),
        )
    print("Done.")


def create_cluster_vectorstore(npz_path: str, collection_name: str | None = None) -> None:
    """Index one averaged embedding per cluster, keyed by cluster ID."""
    if collection_name is None:
        collection_name = os.path.splitext(os.path.basename(npz_path))[0] + "_clusters"

    data = np.load(npz_path)
    embeddings   = data["embeddings"]
    cluster_ids  = data["cluster_labels"]
    mask = _filter_embeddings(embeddings)

    embeddings  = embeddings[mask]
    cluster_ids = cluster_ids[mask]

    unique_ids = np.unique(cluster_ids)
    D = embeddings.shape[1]
    cluster_embs = np.zeros((len(unique_ids), D), dtype=np.float32)

    for i, cid in enumerate(unique_ids):
        avg = embeddings[cluster_ids == cid].mean(axis=0)
        cluster_embs[i] = avg / (np.linalg.norm(avg) + 1e-8)

    print(f"Indexing {len(unique_ids)} clusters…")
    _, collection = _setup_collection(npz_path, collection_name)
    collection.add(
        ids=[str(cid) for cid in unique_ids],
        documents=[f"cluster_{cid}" for cid in unique_ids],
        embeddings=cluster_embs.tolist(),
    )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_file",        required=True)
    parser.add_argument("--logic",           choices=["points", "clusters"], required=True)
    parser.add_argument("--collection_name", default=None)
    args = parser.parse_args()

    if args.logic == "points":
        create_point_vectorstore(args.npz_file, args.collection_name)
    else:
        create_cluster_vectorstore(args.npz_file, args.collection_name)
