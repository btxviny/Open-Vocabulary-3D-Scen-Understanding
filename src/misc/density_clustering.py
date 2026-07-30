import numpy as np
import argparse
import open3d as o3d
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA

import yaml
with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)['clustering']
num_freqs = config['num_freqs']
downsampling_k = config['downsampling_k']
pca_dim = config['pca_dim']
min_cluster_ratio = config['min_cluster_ratio']


def positional_encoding_3d(coords, num_freqs=10):
    freq_bands = np.logspace(0.0, np.log10(10000.0), num=num_freqs)
    pe = []

    for i in range(3):  # x, y, z
        for freq in freq_bands:
            pe.append(np.sin(coords[:, i] / freq))
            pe.append(np.cos(coords[:, i] / freq))

    return np.stack(pe, axis=1)


def cluster_point_cloud(dense_npz_file):
    # Load dense point cloud
    k = downsampling_k
    data = np.load(dense_npz_file)
    dense_points = data['points']
    dense_colors = data['colors']
    dense_embeddings = data['embeddings']
    del data

    #downsample
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(dense_points)
    downsampled_pcd = pcd.uniform_down_sample(every_k_points=k)
    sample_points = np.asarray(downsampled_pcd.points)
    downsample_indices = np.arange(0, dense_points.shape[0], k)
    sample_embeddings = dense_embeddings[downsample_indices]

    print(f"""
    Clustering {len(dense_points)} points,
    with k={k} uniform sampling, 
    positional_encoding with {num_freqs} frequency bands, 
    PCA with {pca_dim} components,
    and HDBSCAN with min_cluster_size={int(min_cluster_ratio * len(dense_points))}"""
    )

    print("Clustering downsampled point cloud...")
    sample_encoded = positional_encoding_3d(sample_points, num_freqs=num_freqs)
    sample_fused = np.concatenate((sample_encoded, sample_embeddings), axis=-1)
    pca = PCA(n_components=pca_dim)
    reduced_sample = pca.fit_transform(sample_fused)
    print(reduced_sample.shape)
    min_cluster_size = int(min_cluster_ratio * len(dense_points))
    labels_sample = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(reduced_sample)
    unique_labels = set(labels_sample)
    print(f"Found {len(unique_labels) - (1 if -1 in unique_labels else 0)} clusters")

    print("Propagating labels to full dense cloud using k-NN...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(reduced_sample, labels_sample)

    dense_encoded = positional_encoding_3d(dense_points, num_freqs=num_freqs)
    dense_fused = np.concatenate((dense_encoded, dense_embeddings), axis=-1)
    reduced_dense = pca.transform(dense_fused)
    labels_dense = knn.predict(reduced_dense)

    np.savez(dense_npz_file,
             points=dense_points,
             embeddings=dense_embeddings,
             colors=dense_colors,
             cluster_labels=labels_dense)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster downsampled dense point cloud and propagate labels.")
    parser.add_argument('--npz_file', type=str, required=True, help='Dense point cloud .npz')
    args = parser.parse_args()
    cluster_point_cloud(args.npz_file)