"""
SAM2 per-frame 2D segmentation.

For each frame directory, runs SAM2AutomaticMaskGenerator and saves:
    mask_<id>.png      — binary segmentation mask (uint8, 0 / 255)
    object_<id>.pcd    — world-space 3D points for that mask (colours in 0–1)

Masks with fewer than MIN_POINTS valid depth pixels are skipped.

Usage (from repo root):
    uv run python -m src.scene_search.segment \\
        --base_dir ScanNet/RGBD/scene0001_00 \\
        [--checkpoint checkpoints/sam2.1_hiera_base_plus.pt]
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from tqdm import tqdm

from .utils import load_config, load_intrinsics, load_pose

_cfg = load_config()
_frame_stride = _cfg.get("frame_stride", 10)
_point_stride = _cfg.get("point_stride", 10)
_min_depth    = _cfg.get("min_depth", 0.5)
_max_depth    = _cfg.get("max_depth", 6.0)

_sam2_cfg        = _cfg.get("sam2", {})
_POINTS_PER_SIDE = _sam2_cfg.get("points_per_side", 32)
_PRED_IOU_THRESH = _sam2_cfg.get("pred_iou_thresh", 0.80)
_STAB_THRESH     = _sam2_cfg.get("stability_score_thresh", 0.80)
_MIN_MASK_AREA   = _sam2_cfg.get("min_mask_region_area", 200)

_fallback        = _cfg.get("default_intrinsics", {})
_MIN_POINTS      = 50
_DEFAULT_CKPT    = str(Path(__file__).parents[2] / "checkpoints" / "sam2.1_hiera_base_plus.pt")


def _project_mask(
    mask: np.ndarray,
    depth: np.ndarray,
    color: np.ndarray,
    pose: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    stride: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Unproject mask pixels to world space. Returns (pts N×3, colors N×3) or None."""
    ys, xs = np.where(mask)
    xs, ys = xs[::stride], ys[::stride]
    zs = depth[ys, xs]

    valid = (zs > _min_depth) & (zs <= _max_depth)
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    if len(zs) < _MIN_POINTS:
        return None

    pts_cam = np.stack([(xs - cx) / fx * zs,
                        (ys - cy) / fy * zs,
                        zs,
                        np.ones_like(zs)], axis=1)
    pts_world = (pose @ pts_cam.T).T[:, :3]

    fin = np.isfinite(pts_world).all(axis=1)
    if fin.sum() < _MIN_POINTS:
        return None

    cols = color[ys[fin], xs[fin]].astype(np.float32) / 255.0
    return pts_world[fin].astype(np.float32), cols


def run_segmentation(
    base_dir: str,
    checkpoint: str = _DEFAULT_CKPT,
    frame_stride: int = _frame_stride,
    point_stride: int = _point_stride,
) -> None:
    fx, fy, cx, cy = load_intrinsics(base_dir, **_fallback)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading SAM2…")
    model = build_sam2("configs/sam2.1/sam2.1_hiera_b+.yaml", checkpoint, device=device)
    mask_gen = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=_POINTS_PER_SIDE,
        pred_iou_thresh=_PRED_IOU_THRESH,
        stability_score_thresh=_STAB_THRESH,
        output_mode="binary_mask",
        crop_n_layers=0,
        min_mask_region_area=_MIN_MASK_AREA,
    )

    frame_dirs = sorted(
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    )[::frame_stride]

    for frame_dir in tqdm(frame_dirs, desc="SAM2 segmenting"):
        rgb_path = os.path.join(frame_dir, "rgb_image.png")
        dep_path = os.path.join(frame_dir, "depth_image.png")
        if not os.path.exists(rgb_path) or not os.path.exists(dep_path):
            continue

        pose = load_pose(frame_dir)
        if pose is None:
            continue

        image_np = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        depth = cv2.imread(dep_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        depth = depth.astype(np.float32) / 1000.0

        H, W = image_np.shape[:2]
        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

        for mask_id, md in enumerate(mask_gen.generate(image_np)):
            result = _project_mask(md["segmentation"], depth, image_np,
                                   pose, fx, fy, cx, cy, point_stride)
            if result is None:
                continue
            pts, cols = result

            cv2.imwrite(
                os.path.join(frame_dir, f"mask_{mask_id}.png"),
                md["segmentation"].astype(np.uint8) * 255,
            )
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(cols)
            o3d.io.write_point_cloud(os.path.join(frame_dir, f"object_{mask_id}.pcd"), pcd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2 per-frame segmentation → mask_*.png + object_*.pcd")
    parser.add_argument("--base_dir",     required=True)
    parser.add_argument("--checkpoint",   default=_DEFAULT_CKPT)
    parser.add_argument("--frame_stride", type=int, default=_frame_stride)
    parser.add_argument("--point_stride", type=int, default=_point_stride)
    args = parser.parse_args()
    run_segmentation(args.base_dir, args.checkpoint, args.frame_stride, args.point_stride)
