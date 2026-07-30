"""
Streamlit app: 3D scene search via CLIP embeddings, visualised with Rerun.

Run from repo root:
    uv run streamlit run app.py
"""

import json
from pathlib import Path

import chromadb
import clip
import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import streamlit as st
import torch
import yaml
from PIL import Image

from src.scene_search.utils import preprocess_image

# ── Config ───────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent

with open(_ROOT / "src" / "config.yaml") as f:
    _cfg = yaml.safe_load(f)
WEIGHTS = _cfg.get("weights", "ViT-B/32")
SCENES_CONFIG = _ROOT / "scenes_config.yaml"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.classes.__path__ = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_scenes() -> dict:
    if not SCENES_CONFIG.exists():
        return {}
    with open(SCENES_CONFIG) as f:
        return (yaml.safe_load(f) or {}).get("scenes", {})


def query_collection(model, collection, text=None, image: Image.Image = None, top_k=1000):
    if text is not None:
        tokens = clip.tokenize([text]).to(DEVICE)
        with torch.no_grad():
            emb = model.encode_text(tokens).cpu().numpy().flatten().tolist()
    elif image is not None:
        t = preprocess_image(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = model.encode_image(t).cpu().numpy().flatten().tolist()
    else:
        raise ValueError("Provide text or image.")

    res = collection.query(query_embeddings=[emb], n_results=top_k, include=["distances"])
    raw = np.array([1 - d for d in res["distances"][0]])

    # Normalise to [~0.5, 1]
    mu, sigma = raw.mean(), raw.std() + 1e-8
    norm = np.clip(0.75 + 0.1 * (raw - mu) / sigma, 0, 1)
    return [{"id": eval(i), "score": float(s)} for i, s in zip(res["ids"][0], norm)]


def visualise(matches, points, base_colors, cluster_labels=None, alpha=0.8, n_bins=200):
    rr.log("points/overlay", rr.Clear(recursive=False))
    cols = base_colors.copy()
    if cols.max() > 1.0:
        cols = cols / 255.0

    logic = st.session_state.logic
    scores = np.array([m["score"] for m in matches])
    ids    = np.array([m["id"] for m in matches])

    min_s, max_s = scores.min(), scores.max()
    ns = np.clip((scores - min_s) / (max_s - min_s + 1e-8), 0, 1)
    ns = np.round(ns * (n_bins - 1)) / (n_bins - 1)
    cmap_rgb = plt.get_cmap("plasma")(ns)[:, :3]

    overlay = np.concatenate([cols, np.ones((len(cols), 1))], axis=1)  # RGBA

    if logic == "points":
        overlay[ids, :3] = alpha * cmap_rgb + (1 - alpha) * cols[ids]
    elif logic == "clusters" and cluster_labels is not None:
        for label, color in zip(ids, cmap_rgb):
            idx = np.where(cluster_labels == label)[0]
            overlay[idx, :3] = alpha * color + (1 - alpha) * cols[idx]

    rr.log("points/overlay", rr.Points3D(
        positions=points.astype(np.float32),
        colors=overlay.astype(np.float32),
    ))


# ── App ────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="3D Scene Search", layout="centered")
    st.title("3D Scene Understanding")

    st.sidebar.markdown(
        """
        **CLIP-based 3D semantic search.**

        - Select a processed scene
        - Choose point-level or cluster-level search
        - Query by text or image
        - Results rendered live in the Rerun viewer
        """
    )
    st.sidebar.image(str(_ROOT / "plasma_colormap.png"),
                     caption="Similarity colormap: purple = low, yellow = high")

    scenes = load_scenes()
    if not scenes:
        st.warning("No scenes found. Run the pipeline first, then reload this page.")
        return

    col1, col2 = st.columns(2)
    with col1:
        scene = st.selectbox("Scene", list(scenes.keys()))
    with col2:
        logic = st.selectbox("Mode", ["points", "clusters"], key="_logic")

    cfg = scenes[scene][logic]

    if st.button("Load scene"):
        with st.spinner("Loading model, vector store and point cloud…"):
            try:
                model, _ = clip.load(WEIGHTS, device=DEVICE)
                client     = chromadb.PersistentClient(cfg["chromadb_path"])
                collection = client.get_collection(cfg["collection"])
                data       = np.load(cfg["npz_file"])

                st.session_state.update(
                    model=model, collection=collection, logic=logic,
                    points=data["points"], colors=data["colors"],
                    cluster_labels=data.get("cluster_labels", np.array([])),
                    ready=True,
                )

                rr.init("scene_search", spawn=True)
                rr.log("points", rr.Clear(recursive=True))
                pts  = data["points"].astype(np.float32)
                cols = data["colors"].astype(np.float32)
                if cols.max() > 1.0:
                    cols /= 255.0
                rr.log("points/base", rr.Points3D(positions=pts, colors=cols))
                st.success(f"Loaded {len(pts):,} points.")
            except Exception as e:
                st.session_state.ready = False
                st.error(f"Failed to load scene: {e}")

    if not st.session_state.get("ready"):
        st.info("Load a scene to enable search.")
        return

    # ── Query UI ──────────────────────────────────────────────────────────────
    st.markdown("---")
    qtype = st.radio("Query type", ["Text", "Image"])
    text_q = img_q = None
    if qtype == "Text":
        text_q = st.text_input("Text prompt")
    else:
        up = st.file_uploader("Image", type=["png", "jpg", "jpeg"])
        if up:
            img_q = Image.open(up).convert("RGB")

    total = (st.session_state.points.shape[0] if logic == "points"
             else len(np.unique(st.session_state.cluster_labels)))
    default_k = min(max(1000, total // 5), total) if logic == "points" else min(5, total)
    top_k = st.slider("Results", 1 if logic == "clusters" else 1000, total, default_k,
                      step=1 if logic == "clusters" else 1000)

    if st.button("Search"):
        if qtype == "Text" and not (text_q or "").strip():
            st.warning("Enter a text prompt.")
            return
        if qtype == "Image" and img_q is None:
            st.warning("Upload an image.")
            return
        with st.spinner("Running search…"):
            hits = query_collection(
                st.session_state.model,
                st.session_state.collection,
                text=text_q, image=img_q, top_k=top_k,
            )
        if not hits:
            st.error("No results.")
        else:
            visualise(hits, st.session_state.points, st.session_state.colors,
                      st.session_state.cluster_labels, alpha=0.8)
            st.success(f"{len(hits)} results visualised.")


if __name__ == "__main__":
    main()
