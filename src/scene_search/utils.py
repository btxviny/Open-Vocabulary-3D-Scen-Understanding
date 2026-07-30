import numpy as np
import hashlib
import open3d as o3d
import matplotlib.pyplot as plt
import torchvision.transforms as T

from PIL import Image
from open3d.utility import Vector3dVector



def cluster_color(cluster_id):
    if cluster_id == -1:
        return [0.5, 0.5, 0.5]  # gray for noise
    hash_digest = hashlib.md5(str(cluster_id).encode()).hexdigest()
    r = int(hash_digest[0:2], 16) / 255.0
    g = int(hash_digest[2:4], 16) / 255.0
    b = int(hash_digest[4:6], 16) / 255.0
    return [r, g, b]


def load_pointcloud(path):
    data = np.load(path)
    points, colors = data['points'], data['colors']
    pcd = o3d.geometry.PointCloud()
    pcd.points = Vector3dVector(points)
    pcd.colors = Vector3dVector(colors)
    return pcd

def preprocess_image(image):
    preprocess = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711])
    ])
    return preprocess(image)

def cosine_to_color(score, min_score=0.0, max_score=1.0, colormap='plasma', num_bins=10):
    norm_score = (score - min_score) / (max_score - min_score + 1e-8)
    norm_score = np.clip(norm_score, 0, 1)
    
    # Quantize
    norm_score = np.round(norm_score * (num_bins - 1)) / (num_bins - 1)

    cmap = plt.get_cmap(colormap)
    return list(cmap(norm_score)[:3])
