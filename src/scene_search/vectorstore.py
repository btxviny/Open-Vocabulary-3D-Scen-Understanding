"""
Example usage:
python create_vectorstore_w_clusters.py --collection_name clip_pointcloud --pointcloud_path clip_pointcloud_expanded_patches.npz
"""
import os
import argparse
import numpy as np
import chromadb
from pathlib import Path

from tqdm import tqdm

MAX_BATCH_SIZE = 5461  # ChromaDB batch limit

def create_point_vectorstore(pointcloud_path: str, collection_name: str = None, max_batch_size: int = MAX_BATCH_SIZE):
    # Get parent directory (scene directory) and create vector_store subdirectory
    scene_dir = Path(pointcloud_path).parent
    client_path = str(scene_dir / 'vector_store')  # Convert to string for chromadb
    Path(client_path).mkdir(exist_ok=True)

    if collection_name is None:
        collection_name = os.path.splitext(os.path.basename(pointcloud_path))[0] + "_points"

    # Load data
    data = np.load(pointcloud_path)
    points = data['points']
    embeddings = data['embeddings']
    del data

    # Compute norms and mask out near-zero vectors
    norms = np.linalg.norm(embeddings, axis=1)
    nonzero_mask = norms > 1e-3

    num_filtered = (~nonzero_mask).sum()
    print(f"Removing {num_filtered} zero-norm embeddings out of {len(embeddings)}")

    # Track original indices before filtering
    original_indices = np.arange(len(embeddings))
    filtered_indices = original_indices[nonzero_mask]
    embeddings = embeddings[nonzero_mask]
    points = points[nonzero_mask]

    print(f"Remaining {len(points)} valid entries after filtering...")

    # Setup ChromaDB
    chroma_client = chromadb.PersistentClient(path=client_path)
    for collection in chroma_client.list_collections():
        if collection.name == collection_name:
            chroma_client.delete_collection(collection.name)

    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Upload in batches
    for i in tqdm(range(0, len(points), max_batch_size), desc="Uploading", unit="batch"):
        batch_embeddings = embeddings[i:i + max_batch_size]
        batch_ids = filtered_indices[i:i + max_batch_size]

        documents = [f'point_{j}' for j in batch_ids]
        ids = [str(j) for j in batch_ids]

        collection.add(
            documents=documents,
            embeddings=batch_embeddings.tolist(),
            ids=ids
        )

    print("Vector store creation complete.")



def create_cluster_vectorstore(pointcloud_path: str, collection_name: str = None, max_batch_size: int = MAX_BATCH_SIZE):
    # Get parent directory (scene directory) and create vector_store subdirectory
    scene_dir = Path(pointcloud_path).parent
    client_path = str(scene_dir / 'vector_store')  # Convert to string for chromadb
    Path(client_path).mkdir(exist_ok=True)

    if collection_name is None:
        collection_name = os.path.splitext(os.path.basename(pointcloud_path))[0] + "_clusters"
    data = np.load(pointcloud_path)
    embeddings = data['embeddings']
    cluster_ids = data['cluster_labels']

    # Compute norms and mask out near-zero vectors
    norms = np.linalg.norm(embeddings, axis=1)
    nonzero_mask = norms > 1e-3

    num_filtered = (~nonzero_mask).sum()
    print(f"Removing {num_filtered} zero-norm embeddings out of {len(embeddings)}")

    # Filter embeddings and update cluster IDs
    embeddings = embeddings[nonzero_mask]
    cluster_ids = cluster_ids[nonzero_mask]

    print(f"Remaining {len(embeddings)} valid entries after filtering...")

    # Setup ChromaDB
    chroma_client = chromadb.PersistentClient(path=client_path)
    for collection in chroma_client.list_collections():
        if collection.name == collection_name:
            chroma_client.delete_collection(collection.name)

    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    unique_cluster_ids = np.unique(cluster_ids)
    final_embeddings = np.zeros(shape=(len(unique_cluster_ids), embeddings.shape[1]), dtype=np.float32)

    for idx, cluster_id in enumerate(unique_cluster_ids):
        # Get embeddings for this cluster
        if cluster_id == -1:
            continue
        cluster_indices = np.where(cluster_ids == cluster_id)[0]
        cluster_embeddings = embeddings[cluster_indices]
        
        # Simple average of embeddings
        avg_embedding = np.mean(cluster_embeddings, axis=0)
        
        # Normalize to unit length
        norm = np.linalg.norm(avg_embedding)
        avg_embedding /= norm + 1e-8
        
        final_embeddings[idx] = avg_embedding

    # Add to collection
    documents = [f'cluster_{i}' for i in unique_cluster_ids]
    final_embeddings = final_embeddings.tolist()
    ids = [str(i) for i in unique_cluster_ids]

    collection.add(
        documents=documents,
        embeddings=final_embeddings,
        ids=ids,
    )
    print("Vector store creation complete.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a vector store from a pointcloud.')
    parser.add_argument('--npz_file', type=str, required=True, help='Path to the pointcloud file.')
    parser.add_argument('--logic', type=str, help='Available options: "points" or "clusters')
    parser.add_argument('--collection_name', type=str, help='Optional custom collection name')
    args = parser.parse_args()
    if args.logic == "points":
        create_point_vectorstore(args.npz_file, args.collection_name)
    elif args.logic == "clusters":
        create_cluster_vectorstore(args.npz_file, args.collection_name)
    else:
        raise ValueError(f"Invalid logic: {args.logic}")
