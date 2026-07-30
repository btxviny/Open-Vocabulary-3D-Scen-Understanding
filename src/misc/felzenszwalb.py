import argparse
import numpy as np
import open3d as o3d
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import SpectralClustering
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------- Union-Find & Graph Segmentation ----------

class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)
        self.size = np.ones(n, dtype=int)
        self.int_diff = np.zeros(n)

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v, weight):
        u_root, v_root = self.find(u), self.find(v)
        if u_root == v_root:
            return
        new_root = u_root if self.size[u_root] > self.size[v_root] else v_root
        other_root = v_root if new_root == u_root else u_root

        self.parent[other_root] = new_root
        self.size[new_root] += self.size[other_root]
        self.int_diff[new_root] = max(self.int_diff[u_root], self.int_diff[v_root], weight)

    def component_threshold(self, comp_id, k):
        return self.int_diff[comp_id] + k / self.size[comp_id]


def positional_encoding_3d(coords, num_freqs=10):
    freq_bands = np.logspace(0.0, np.log10(10000.0), num=num_freqs)
    pe = []

    for i in range(3):  # x, y, z
        for freq in freq_bands:
            pe.append(np.sin(coords[:, i] / freq))
            pe.append(np.cos(coords[:, i] / freq))

    return np.stack(pe, axis=1).flatten()

def build_graph(points, embeddings, k=10, alpha=1.0, beta=1.0):
    N = points.shape[0]
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(points)
    pairs = nn.kneighbors(points, return_distance=False)
    edges = []

    for i in tqdm(range(N), desc="Building graph"):
        for j in pairs[i][1:]:
            points1_encoded = positional_encoding_3d(points[i].reshape(1, -1))
            points1_encoded /= (np.linalg.norm(points1_encoded) + 1e-8)
            points2_encoded = positional_encoding_3d(points[j].reshape(1, -1))
            points2_encoded /= (np.linalg.norm(points2_encoded) + 1e-8)
            distance1 = 1 - np.dot(points1_encoded , points2_encoded)

            distance2 = 1 - np.dot(embeddings[i], embeddings[j])  # embeddings assumed normalized
            combined_weight = alpha * distance1 + beta * distance2
            edges.append((i, j, combined_weight))

    return edges


def efficient_graph_segmentation(points, embeddings, k=10, seg_k=0.5, min_size=20, alpha=1.0, beta=1.0):
    N = points.shape[0]
    edges = build_graph(points, embeddings, k=k, alpha=alpha, beta=beta)
    edges.sort(key=lambda e: e[2])
    uf = UnionFind(N)

    for u, v, w in tqdm(edges, desc="Merging components (1st pass)"):
        Cu, Cv = uf.find(u), uf.find(v)
        if Cu != Cv:
            if w <= min(uf.component_threshold(Cu, seg_k), uf.component_threshold(Cv, seg_k)):
                uf.union(Cu, Cv, w)

    for u, v, w in tqdm(edges, desc="Merging small components (2nd pass)"):
        Cu, Cv = uf.find(u), uf.find(v)
        if Cu != Cv and (uf.size[Cu] < min_size or uf.size[Cv] < min_size):
            uf.union(Cu, Cv, w)

    labels = np.array([uf.find(i) for i in range(N)])
    _, seg_ids = np.unique(labels, return_inverse=True)
    return seg_ids

# ---------- Downsampling & Propagation ----------

def uniform_downsample_point_cloud(points, every_k_points=100):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downsampled_pcd = pcd.uniform_down_sample(every_k_points=every_k_points)
    indices = np.arange(points.shape[0])[::every_k_points]
    return np.asarray(downsampled_pcd.points), indices


# def ncut(embeddings, labels, affinity='cosine'):
#     """
#     Applies NCut (Spectral Clustering) on the cluster-level mean embeddings to merge similar clusters.
    
#     Args:
#         embeddings: (N, D) array of normalized embeddings.
#         labels: (N,) array of integer cluster labels from initial segmentation.
#         n_clusters: desired number of clusters (optional).
#         affinity: similarity measure ('cosine' is typical for embeddings).

#     Returns:
#         (N,) array of refined cluster labels.
#     """
#     unique_labels = np.unique(labels)
#     n_components = len(unique_labels)

#     mean_embeddings = np.zeros(shape=(n_components, embeddings.shape[1]),dtype=np.float32)
#     for idx, label in enumerate(unique_labels):
#         indices = np.where(labels == label)[0]
#         mean_embeddings[idx] = np.mean(embeddings[indices], axis=0)

#     # Apply Spectral Clustering
#     spectral = SpectralClustering(
#         affinity=affinity,
#         assign_labels='kmeans',
#         random_state=42
#     )
#     new_segment_labels = spectral.fit_predict(mean_embeddings)

#     # Map old cluster labels to new spectral ones
#     label_map = {old: new_segment_labels[i] for i, old in enumerate(unique_labels)}
#     refined_labels = np.array([label_map[l] for l in labels])
    return refined_labels


def propagate_labels(full_points, down_points, down_labels, k=101, batch_size=5000):
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(down_points)
    N = full_points.shape[0]
    full_labels = np.zeros(N, dtype=int)

    for start in tqdm(range(0, N, batch_size), desc="Propagating labels"):
        end = min(start + batch_size, N)
        batch_points = full_points[start:end]
        idxs = nn.kneighbors(batch_points, return_distance=False)

        if k == 1:
            full_labels[start:end] = down_labels[idxs[:, 0]]
        else:
            neighbor_labels = down_labels[idxs]
            mode_labels = mode(neighbor_labels, axis=1, keepdims=False).mode
            full_labels[start:end] = mode_labels

    return full_labels

# ---------- Main Pipeline ----------

def segment_point_cloud(points, embeddings, every_k_points=100, k_graph=100, seg_k=0.1,
                        min_size=500, k_label=5, batch_size=5000, alpha=0.5, beta=0.5):
    down_points, indices = uniform_downsample_point_cloud(points, every_k_points=every_k_points)
    down_embeddings = embeddings[indices]

    print(f"Original points: {len(points)}, Downsampled: {len(down_points)}")

    down_labels = efficient_graph_segmentation(down_points, down_embeddings, k=k_graph, seg_k=seg_k,
                                                min_size=min_size, alpha=alpha, beta=beta)
    print(f'Segmented {len(np.unique(down_labels))} instances.')
    
    #ncut_labels = ncut(embeddings, down_labels, affinity='cosine')
    #print(f"Refined to {len(np.unique(ncut_labels))} clusters using NCut.")

    full_labels = propagate_labels(points, down_points, down_labels, k=k_label, batch_size=batch_size)

    return full_labels

# ---------- CLI Entry Point ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment a point cloud using a geometric+embedding graph segmentation.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input point cloud .npz file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output .npz file.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Weight for cosine embedded coordinates distance.")
    parser.add_argument("--beta", type=float, default=0.9, help="Weight for cosine clip embedding distance.")
    args = parser.parse_args()

    data = np.load(args.input)
    points = data['points']
    colors = data['colors']
    embeddings = data['embeddings']  # already normalized
    min_cluster_size = 0.0001 * points.shape[0]

    cluster_labels = segment_point_cloud(
        points,
        embeddings=embeddings,
        min_size=min_cluster_size,
        batch_size=5000,
        alpha=args.alpha,
        beta=args.beta
    )

    np.savez(
        args.output,
        points=points,
        colors=colors,
        embeddings=embeddings,
        cluster_labels=cluster_labels
    )