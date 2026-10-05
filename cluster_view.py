"""Exploratory clusters for the local archive. Run: python cluster_view.py

K-means groups image embeddings by visual similarity, not verified land-use classes.
This separate UI runs on localhost:7862 without the VLM server.
"""
import json
from pathlib import Path
import faiss
import gradio as gr
import numpy as np
from sklearn.cluster import KMeans
from archive_search import ROOT, OUT


def clusters(k=4):
    index = faiss.read_index(str(OUT / "tiles.faiss"))
    rows = json.loads((OUT / "items.json").read_text(encoding="utf-8"))
    if len(rows) != index.ntotal:
        raise RuntimeError("Index/items mismatch. Rebuild the index first.")
    if not 2 <= k <= len(rows):
        raise ValueError("Choose 2 to the number of indexed images clusters.")
    # IndexFlatIP stores original vectors and supports reconstruct_n.
    vectors = index.reconstruct_n(0, index.ntotal)
    groups = KMeans(n_clusters=int(k), random_state=42, n_init=10).fit_predict(vectors)
    return [(row, int(cluster)) for row, cluster in zip(rows, groups)]


def show(k, selected):
    assigned = clusters(int(k))
    selected = int(selected.split()[-1])
    matches = [(str(ROOT / row["path"]), f'{row.get("label", "unknown")} | {row["path"]}')
               for row, group in assigned if group == selected and (ROOT / row["path"]).is_file()]
    counts = [sum(group == i for _, group in assigned) for i in range(int(k))]
    return f"Exploratory groups of {len(assigned)} indexed tiles. Cluster sizes: {counts}. No ground-truth validation.", matches


if __name__ == "__main__":
    with gr.Blocks(title="SatQuery archive clusters") as demo:
        gr.Markdown("# Local archive clusters\nExploratory K-means on the cached image embeddings. Clusters are NOT verified change types or land-use labels.")
        k = gr.Slider(2, 6, value=4, step=1, label="Number of clusters")
        group = gr.Dropdown([f"Cluster {i}" for i in range(6)], value="Cluster 0", label="Show cluster")
        button = gr.Button("Show cluster")
        summary = gr.Markdown()
        gallery = gr.Gallery(label="Images in selected cluster", columns=5, height=480)
        button.click(show, [k, group], [summary, gallery])
    demo.launch(server_name="127.0.0.1", server_port=7862, inbrowser=True)
