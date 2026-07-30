import os
import streamlit as st
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import torch
import clip
import chromadb
from PIL import Image
import rerun as rr
import matplotlib.pyplot as plt
import json
import yaml
from pathlib import Path


from src.scene_search.utils import preprocess_image
with open('./src/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
weights = config.get('weights',"ViT-B/16")


def load_scenes_config():
    """Load scenes configuration from YAML file."""
    config_path = Path("scenes_config.yaml")
    if not config_path.exists():
        st.error("❌ Scene configuration file not found!")
        return {}
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get("scenes", {})

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.classes.__path__ = []

# Load scenes configuration
scenes = load_scenes_config()

def cosine_to_color(score, min_score=0.0, max_score=1.0, colormap='inferno', num_bins=1000):
    norm_score = ((score - min_score) / (max_score - min_score + 1e-8)) ** 2
    # Quantize
    norm_score = np.round(norm_score * (num_bins - 1)) / (num_bins - 1)
    cmap = plt.get_cmap(colormap)
    return list(cmap(norm_score)[:3])


def query_vectorstore(model, collection, query=None, image: Image.Image = None, top_k=10):
    if query is not None:
        tokens = clip.tokenize([query]).to(device)
        with torch.no_grad():
            embedding = model.encode_text(tokens).cpu().numpy().flatten().tolist()
    elif image is not None:
        img_tensor = preprocess_image(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(img_tensor).cpu().numpy().flatten().tolist()
    else:
        raise ValueError("Either text query or image must be provided.")

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["distances"]
    )

    # Convert distances to similarity scores
    raw_scores = np.array([1 - d for d in results['distances'][0]])

    # --- Diagnostic Statistics ---
    st.write("📊 **Similarity Score Stats:**")
    st.write(f"- Mean: {raw_scores.mean():.4f}")
    st.write(f"- Std: {raw_scores.std():.4f}")
    st.write(f"- Min: {raw_scores.min():.4f}")
    st.write(f"- Max: {raw_scores.max():.4f}")

    # Normalize scores to mean 0.5, std 0.25
    mean = raw_scores.mean()
    std = raw_scores.std() + 1e-8  # prevent divide-by-zero
    standardized = (raw_scores - mean) / std
    normalized_scores = 0.75 + 0.1 * standardized
    normalized_scores = np.clip(normalized_scores, 0, 1)
    # Return formatted matches
    matches = []
    for idx, score in zip(results['ids'][0], normalized_scores):
        matches.append({"id": eval(idx), "score": float(score)})
    return matches


# --- Visualization ---
def visualize(matches, points, base_colors, cluster_labels=None, alpha=0.4, num_bins = 200):
    # Only clear the overlay points
    rr.log("points/overlay", rr.Clear(recursive=False))

    if base_colors.max() > 1.0:
        base_colors = base_colors / 255.0

    if st.session_state.logic == "points":
        overlay_colors = np.zeros((base_colors.shape[0], 4))  # RGBA
        overlay_colors[:, :3] = base_colors
        overlay_colors[:, 3] = 1.0

        indices = np.array([m["id"] for m in matches])
        scores = np.array([m["score"] for m in matches])
        min_score, max_score = scores.min(), scores.max()

        # Vectorized conversion of scores to colors:
        # Map all scores to colors via cosine_to_color in a vectorized manner:
        norm_scores = (scores - min_score) / (max_score - min_score + 1e-8)
        norm_scores = np.clip(norm_scores, 0, 1)

        # Quantize to bins
        norm_scores = np.round(norm_scores * (num_bins - 1)) / (num_bins - 1)

        cmap = plt.get_cmap('plasma')
        colors_rgb = cmap(norm_scores)[:, :3]

        # Blend colors
        overlay_colors[indices, :3] = alpha * colors_rgb + (1 - alpha) * base_colors[indices]
        overlay_colors[indices, 3] = 1.0

        # Ensure proper data types for RerunSDK
        points_array = np.asarray(points, dtype=np.float32)
        overlay_colors_array = np.asarray(overlay_colors, dtype=np.float32)
        rr.log("points/overlay", rr.Points3D(positions=points_array, colors=overlay_colors_array))

    elif st.session_state.logic == "clusters":
        overlay_rgba = np.zeros((points.shape[0], 4))
        overlay_rgba[:, :3] = base_colors
        overlay_rgba[:, 3] = 1.0

        labels = np.array([m["id"] for m in matches])
        scores = np.array([m["score"] for m in matches])
        min_score, max_score = scores.min(), scores.max()

        norm_scores = (scores - min_score) / (max_score - min_score + 1e-8)
        norm_scores = np.clip(norm_scores, 0, 1)
        norm_scores = np.round(norm_scores * (num_bins - 1)) / (num_bins - 1)
        cmap = plt.get_cmap('plasma')
        colors_rgb = cmap(norm_scores)[:, :3]
        for label, color_rgb in zip(labels, colors_rgb):
            indices = np.where(cluster_labels == label)[0]
            overlay_rgba[indices, :3] = alpha * color_rgb + (1 - alpha) * base_colors[indices]
            overlay_rgba[indices, 3] = 1.0

        # Ensure proper data types for RerunSDK
        points_array = np.asarray(points, dtype=np.float32)
        overlay_rgba_array = np.asarray(overlay_rgba, dtype=np.float32)
        rr.log("points/overlay", rr.Points3D(positions=points_array, colors=overlay_rgba_array))


def load_tags(tags_file):
    """Load tags from a JSON file."""
    try:
        with open(tags_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not load tags file: {e}")
        return []

# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="CLIP PointCloud Search", layout="centered")
    st.sidebar.markdown(
        """
        This is a **3D Scene Understanding** application built on top of:

        - **Segment Anything (SAM)** for object detection  
        - **CLIP embeddings** for semantic search  
        - RealSense sensor for **RGB-D data**  
        - **LIDAR** for localization  

        You can:
        - Load different scenes (bedroom, office, warehouse)
        - Choose between point-level and cluster-level (experimental) logic
        - Run semantic search via text or image query

        The visual results are shown via **Rerun viewer**.
        """
    )
    st.sidebar.image('./plasma_colormap.png', caption="Color Map for Similarity, Purple Low - Yellow High")
    st.title("""🤖 SPARC: 3D Scene Understanding""")

    # Scene & logic selection
    column1, column2 = st.columns(2)
    with column1:
        selected_scene = st.selectbox("Select Scene", list(scenes.keys()), key="selected_scene")
    with column2:
        selected_logic = st.selectbox("Select Logic", ["points", "clusters"], key="_logic")

    # Derived paths
    npz_file = scenes[selected_scene][selected_logic]["npz_file"]
    chromadb_path = scenes[selected_scene][selected_logic]["chromadb_path"]
    collection_name = scenes[selected_scene][selected_logic]["collection"]

    if st.button("Get Scene"):
        with st.spinner("Loading model, vector store, and point cloud..."):
            try:
                # Load resources directly without caching
                model, _ = clip.load(weights, device=device)
                chroma_client = chromadb.PersistentClient(chromadb_path)
                collection = chroma_client.get_collection(name=collection_name)
                data = np.load(npz_file)
                points = data["points"]
                colors = data["colors"]
                cluster_labels = data.get("cluster_labels", [])

                st.session_state.model = model
                st.session_state.collection = collection
                st.session_state.points = points
                st.session_state.colors = colors
                st.session_state.cluster_labels = cluster_labels
                st.session_state.collection_ready = True
                st.session_state.logic = selected_logic
                
                # Initialize Rerun viewer and clear all previous visualizations
                rr.init("clip_pointcloud_query", spawn=True)
                rr.log("points", rr.Clear(recursive=True))  # Clear everything recursively
                
                # Show base point cloud
                # Ensure proper data types for RerunSDK
                points_array = np.asarray(points, dtype=np.float32)
                colors_array = np.asarray(colors, dtype=np.float32)
                
                # Normalize colors if needed
                if colors_array.max() > 1.0:
                    colors_array = colors_array / 255.0
                
                # Ensure colors are in the correct range [0, 1]
                colors_array = np.clip(colors_array, 0.0, 1.0)
                
                rr.log("points/base", rr.Points3D(positions=points_array, colors=colors_array))
                
                st.success(f"✅ Scene loaded successfully with {len(points)} points.")
                st.write("Rerun viewer launched in a separate window/tab.")
            except Exception as e:
                st.session_state.collection_ready = False
                st.error(f"❌ Failed to load scene data.\n{e}")

    if st.session_state.get("collection_ready", False):
        st.markdown("### Enter a text prompt or upload an image to search clusters.")
        
        query_type = st.radio("Select query type:", ("Text", "Image"))

        text_query = None
        image_query = None
        
        if query_type == "Text":
            text_query = st.text_input("Text Prompt")
        else:
            image_query = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

        match st.session_state.logic:
            case "points":
                total_points = st.session_state.points.shape[0]
                default = max(1000, int(0.2 * total_points / 100) * 100)
                top_k = st.slider("Number of Results", 
                                min_value=1000, 
                                max_value=total_points, 
                                value=default,
                                step=1000)
            case "clusters":
                total_clusters = np.unique(st.session_state.cluster_labels).shape[0]
                top_k = st.slider("Number of Results", 
                                min_value=1, 
                                max_value=total_clusters, 
                                value= min(5,total_clusters), 
                                step=1)

        if st.button("Run Query"):
            if query_type == "Text" and not text_query.strip():
                st.warning("Please enter a text prompt.")
            elif query_type == "Image" and image_query is None:
                st.warning("Please upload an image.")
            else:
                st.info("Running query...")
                image = Image.open(image_query).convert("RGB") if image_query else None
                results = query_vectorstore(
                    st.session_state.model,
                    st.session_state.collection,
                    query=text_query,
                    image=image,
                    top_k=top_k
                )
                if not results:
                    st.error("❌ No matches found.")
                else:
                    st.success("✅ Visualizing results...")
                    visualize(results, st.session_state.points, st.session_state.colors, st.session_state.cluster_labels, alpha=0.8)
                    st.write("Rerun viewer launched in a separate window/tab.")
    else:
        st.info("ℹ️ Please load the scene collection to enable querying.")

if __name__ == "__main__":
    main()