"""
SAM3D: per-frame SAM2 masks projected into 3D, then merged into object clusters.

Usage (from repo root):
    uv run python -m scene_search.sam3d \
        --base_dir /path/to/scene_frames \
        --save_path output/segmented.npz \
        --camera_type scannet
"""

import json
import os
import sys
import copy
import argparse
from pathlib import Path
from os.path import join

import cv2
import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm

from . import _config

# ── Config ──────────────────────────────────────────────────────────────────
_cfg = _config.load()

voxel_size        = _cfg.get("sam3d", {}).get("voxel_size", 0.01)
point_stride      = _cfg.get("point_stride", 10)
frame_stride      = _cfg.get("frame_stride", 10)
th                = _cfg.get("sam3d", {}).get("th", 1000)
min_depth         = _cfg.get("min_depth", 0.5)
max_depth         = _cfg.get("max_depth", 6.0)
group_overlap_ratio    = _cfg.get("sam3d", {}).get("group_overlap_ratio", 0.2)
voxel_search_multiplier = _cfg.get("sam3d", {}).get("voxel_search_multiplier", 1.5)

_sam_cfg = {
    "points_per_side":        _cfg.get("sam3d", {}).get("points_per_side", 32),
    "pred_iou_thresh":        _cfg.get("sam3d", {}).get("pred_iou_thresh", 0.8),
    "stability_score_thresh": _cfg.get("sam3d", {}).get("stability_score_thresh", 0.8),
    "min_mask_region_area":   _cfg.get("sam3d", {}).get("min_mask_region_area", 200),
}

# ── SAM2 import (installed as a package) ────────────────────────────────────
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

# ── SegmentAnything3D util (requires pointops to be built) ──────────────────
def _find_sa3d():
    """Find SegmentAnything3D directory: env var → inside repo root."""
    env = os.environ.get("SEGMENT_ANYTHING_3D_PATH")
    if env:
        return Path(env)
    # parents[2] = src/ → open-3d-scene-understanding/ (repo root)
    repo_root = Path(__file__).parents[2]
    return repo_root / "SegmentAnything3D"

_sa3d = _find_sa3d()
if not _sa3d.exists():
    raise RuntimeError(
        f"SegmentAnything3D not found at {_sa3d}. "
        "Run setup.sh or set SEGMENT_ANYTHING_3D_PATH."
    )
if str(_sa3d) not in sys.path:
    sys.path.insert(0, str(_sa3d))

from util import Voxelize, pairwise_indices, num_to_natural, remove_small_group, get_matching_indices  # noqa: E402


# ── Camera helpers ───────────────────────────────────────────────────────────

_default_intr = _cfg.get("default_intrinsics", {})
_FALLBACK_FX  = _default_intr.get("fx",  577.87)
_FALLBACK_FY  = _default_intr.get("fy",  577.87)
_FALLBACK_PPX = _default_intr.get("cx",  319.5)
_FALLBACK_PPY = _default_intr.get("cy",  239.5)


def _load_scene_intrinsics(base_dir: str) -> dict | None:
    """Load per-scene intrinsics from intrinsics.json (written by prepare_scene)."""
    p = Path(base_dir) / "intrinsics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _k_inv(base_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (K_inv 3×3, identity 4×4).  Intrinsics from intrinsics.json or fallback."""
    intr = _load_scene_intrinsics(base_dir)
    if intr is not None:
        fx, fy   = intr["fx"],  intr["fy"]
        ppx, ppy = intr["cx"],  intr["cy"]
        print(f"Intrinsics: fx={fx:.2f} fy={fy:.2f} cx={ppx:.1f} cy={ppy:.1f}")
    else:
        fx, fy, ppx, ppy = _FALLBACK_FX, _FALLBACK_FY, _FALLBACK_PPX, _FALLBACK_PPY
        print("Warning: no intrinsics.json found — using fallback intrinsics")
    K_inv = np.array([
        [1/fx,    0, -ppx/fx],
        [   0, 1/fy, -ppy/fy],
        [   0,    0,       1],
    ])
    return K_inv, np.eye(4, dtype=np.float64)


def _load_pose(frame_path: str):
    """Load camera-to-world pose; tries pose.npy / pose.txt / extrinsic_matrix.npy."""
    candidates = ["pose.npy", "pose.txt", "extrinsic_matrix.npy"]
    for fname in candidates:
        p = join(frame_path, fname)
        if not os.path.exists(p):
            continue
        try:
            ext = np.loadtxt(p).reshape(4, 4) if fname.endswith(".txt") else np.load(p)
            if ext.shape != (4, 4):
                continue
            if np.any(np.isnan(ext)) or np.any(np.isinf(ext)):
                continue
            if abs(np.linalg.det(ext[:3, :3]) - 1.0) > 0.1:
                continue
            return ext
        except Exception:
            continue
    return None


# ── Per-frame processing ─────────────────────────────────────────────────────

def get_pcd(frame_path, mask_generator, point_stride, K_inv, cam_to_imu):
    extrinsic = _load_pose(frame_path)
    if extrinsic is None:
        return dict(coord=np.empty((0, 3)), color=np.empty((0, 3)), group=np.array([]))

    depth_img = cv2.imread(join(frame_path, "depth_image.png"), -1)
    if depth_img is None:
        return dict(coord=np.empty((0, 3)), color=np.empty((0, 3)), group=np.array([]))
    depth_img = depth_img.astype(np.float32) / 1000.0

    color_image = cv2.imread(join(frame_path, "rgb_image.png"))
    if color_image is None:
        return dict(coord=np.empty((0, 3)), color=np.empty((0, 3)), group=np.array([]))
    color_image = cv2.resize(color_image, (640, 480))

    depth_mask = (depth_img > min_depth) & (depth_img <= max_depth)
    if not np.any(depth_mask):
        return dict(coord=np.empty((0, 3)), color=np.empty((0, 3)), group=np.array([]))

    # SAM2 segmentation or load cached masks
    if mask_generator is not None:
        masks = mask_generator.generate(color_image)
        group_ids = np.full((480, 640), -1, dtype=int)
        for i in reversed(range(len(masks))):
            group_ids[masks[i]["segmentation"]] = i
        _save_masks(masks, depth_img, color_image, extrinsic, frame_path, point_stride, K_inv, cam_to_imu)
    else:
        group_ids = np.full((480, 640), -1, dtype=int)
        for mf in sorted(os.listdir(frame_path)):
            if mf.startswith("mask_") and mf.endswith(".png"):
                idx = int(mf.split("_")[1].split(".")[0])
                m = cv2.imread(join(frame_path, mf), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    group_ids[m > 127] = idx

    # Unproject
    H, W = depth_img.shape
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    flat_mask = depth_mask.flatten()
    xs_f = xs.flatten()[flat_mask]
    ys_f = ys.flatten()[flat_mask]
    depth_f = depth_img.flatten()[flat_mask]

    pixels = np.stack([xs_f, ys_f, np.ones_like(xs_f)], axis=0)  # (3, N)
    pts_cam = (K_inv @ pixels) * depth_f                           # (3, N)
    pts_cam_h = np.vstack([pts_cam, np.ones((1, pts_cam.shape[1]))])  # (4, N)
    pts_world = (extrinsic @ cam_to_imu @ pts_cam_h).T[:, :3]

    valid = np.isfinite(pts_world).all(axis=1)
    pts_world = pts_world[valid]

    colors = color_image.reshape(-1, 3)[flat_mask][valid].astype(np.float32)[:, ::-1] / 255.0
    group_out = group_ids.flatten()[flat_mask][valid]

    return dict(coord=pts_world[::point_stride],
                color=colors[::point_stride],
                group=group_out[::point_stride])


def _save_masks(masks, depth_img, color_image, extrinsic, save_dir, point_stride, K_inv, cam_to_imu):
    """Persist per-mask PNG and object .pcd files."""
    os.makedirs(save_dir, exist_ok=True)
    for mask_id, md in enumerate(masks):
        mask = md["segmentation"]
        ys, xs = np.where(mask)
        xs, ys = xs[::point_stride], ys[::point_stride]
        depths = depth_img[ys, xs]
        valid = (depths > min_depth) & (depths <= max_depth)
        xs, ys, depths = xs[valid], ys[valid], depths[valid]
        if len(depths) < 50:
            continue

        pixels = np.stack([xs, ys, np.ones_like(xs)], axis=0)
        pts_cam = (K_inv @ pixels) * depths
        pts_cam_h = np.vstack([pts_cam, np.ones((1, pts_cam.shape[1]))])
        pts_world = (extrinsic @ cam_to_imu @ pts_cam_h).T[:, :3]

        valid_c = np.isfinite(pts_world).all(axis=1)
        pts_world = pts_world[valid_c]
        cols = color_image[ys[valid_c], xs[valid_c]].astype(np.float32)[:, ::-1] / 255.0

        if len(pts_world) < 50:
            continue

        cv2.imwrite(join(save_dir, f"mask_{mask_id}.png"), mask.astype(np.uint8) * 255)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_world)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        o3d.io.write_point_cloud(join(save_dir, f"object_{mask_id}.pcd"), pcd)


# ── Multi-frame merging ───────────────────────────────────────────────────────

def _make_o3d_pcd(d, th):
    d["group"] = remove_small_group(d["group"], th)
    if np.isnan(d["coord"]).any():
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(d["coord"])
    return pcd


def cal_group(d0, d1, match_inds, ratio=None):
    ratio = ratio or group_overlap_ratio
    g0 = d0["group"]
    g1 = d1["group"].copy()
    if isinstance(g0, torch.Tensor):
        g0 = g0.cpu().numpy()
    if isinstance(g1, torch.Tensor):
        g1 = g1.cpu().numpy()

    if len(np.unique(g0)) > 0:
        g1[g1 != -1] += np.max(g0) + 1

    cnt0 = dict(zip(*np.unique(g0, return_counts=True)))
    cnt1 = dict(zip(*np.unique(g1, return_counts=True)))

    overlap = {}
    for i, j in match_inds:
        if i >= len(g1) or j >= len(g0):
            continue
        gi, gj = g1[i], g0[j]
        if gi == -1:
            g1[i] = gj
            continue
        if gj == -1:
            continue
        overlap.setdefault(gi, {}).setdefault(gj, 0)
        overlap[gi][gj] += 1

    for gi, ovlp in overlap.items():
        if not ovlp:
            continue
        gj = max(ovlp, key=ovlp.get)
        count = ovlp[gj]
        if gj in cnt0 and gi in cnt1:
            total = min(cnt0[gj], cnt1[gi])
            if count / float(total) >= ratio:
                g1[g1 == gi] = gj
    return g1


def cal_2_scenes(pcd_list, index, voxel_size, voxelize, th=50):
    if len(index) == 1:
        return pcd_list[index[0]]

    d0, d1 = pcd_list[index[0]], pcd_list[index[1]]
    p0 = _make_o3d_pcd(copy.deepcopy(d0), th)
    p1 = _make_o3d_pcd(copy.deepcopy(d1), th)

    if p0 is None:
        return d1 if p1 is not None else None
    if p1 is None:
        return d0

    d0["group"] = d0["group"][:len(p0.points)]
    d1["group"] = d1["group"][:len(p1.points)]

    mi = get_matching_indices(p1, p0, voxel_search_multiplier * voxel_size, 1)
    if not mi:
        return d0
    g1_new = cal_group(d0, d1, mi)

    mi2 = get_matching_indices(p0, p1, voxel_search_multiplier * voxel_size, 1)
    if not mi2:
        return d0
    d1["group"] = g1_new
    g0_new = cal_group(d1, d0, mi2)

    merged = dict(
        coord=np.concatenate([d0["coord"], d1["coord"]]),
        color=np.concatenate([d0["color"], d1["color"]]),
        group=num_to_natural(np.concatenate([g0_new, g1_new])),
    )
    return voxelize(merged)


def seg_pcd(frame_paths, save_path, mask_generator, voxel_size, voxelizer, th, point_stride, base_dir):
    K_inv, cam_to_imu = _k_inv(base_dir)

    pcd_list = []
    for fp in tqdm(frame_paths, desc="SAM3D: processing frames"):
        d = get_pcd(fp, mask_generator, point_stride, K_inv, cam_to_imu)
        if len(d["coord"]) == 0:
            continue
        d = voxelizer(d)
        pcd_list.append(d)

    while len(pcd_list) > 1:
        new_list = []
        for idx in pairwise_indices(len(pcd_list)):
            merged = cal_2_scenes(pcd_list, idx, voxel_size=voxel_size, voxelize=voxelizer, th=th)
            if merged is not None:
                new_list.append(merged)
        pcd_list = new_list

    seg = pcd_list[0]
    seg["group"] = num_to_natural(remove_small_group(seg["group"], th))

    valid = np.isfinite(seg["coord"]).all(axis=1)
    seg["coord"] = seg["coord"][valid]
    seg["color"] = seg["color"][valid]
    seg["group"] = seg["group"][valid]

    np.savez(save_path,
             points=seg["coord"],
             colors=seg["color"],
             cluster_labels=seg["group"])
    print(f"Saved {save_path} ({len(seg['coord'])} points, {len(np.unique(seg['group']))} clusters)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAM3D: segment a scene into 3D object clusters")
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--checkpoint", default=str(Path(__file__).parents[2] / "checkpoints" / "sam2.1_hiera_base_plus.pt"))
    parser.add_argument("--voxel_size", type=float, default=voxel_size)
    parser.add_argument("--th", type=int, default=th)
    parser.add_argument("--frame_stride", type=int, default=frame_stride)
    parser.add_argument("--point_stride", type=int, default=point_stride)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_sam2("configs/sam2.1/sam2.1_hiera_b+.yaml", ckpt_path=args.checkpoint, device=device)
    mask_gen = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=_sam_cfg["points_per_side"],
        pred_iou_thresh=_sam_cfg["pred_iou_thresh"],
        stability_score_thresh=_sam_cfg["stability_score_thresh"],
        output_mode="binary_mask",
        crop_n_layers=0,
        min_mask_region_area=_sam_cfg["min_mask_region_area"],
    )

    frame_paths = sorted(
        p for p in (Path(args.base_dir) / d for d in sorted(os.listdir(args.base_dir)))
        if p.is_dir()
    )[::args.frame_stride]

    voxelize = Voxelize(voxel_size=args.voxel_size, mode="train", keys=("coord", "color", "group"))
    seg_pcd(frame_paths, args.save_path, mask_gen, args.voxel_size,
            voxelize, args.th, args.point_stride, base_dir=args.base_dir)


if __name__ == "__main__":
    main()
