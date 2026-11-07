from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .labels import load_labels, to_indices


def run_pca(emb_path: Path, out_path: Path, labels_path: Path):
    feats = np.load(emb_path)

    # Project all embeddings together
    pca = PCA(n_components=2)
    feats_2d = pca.fit_transform(feats)

    # Assign labels to every sampled frame
    labels = np.full(len(feats), "unknown", dtype=object)

    df = load_labels(labels_path)
    for row in df.itertuples():
        first, last = to_indices(row.Start, row.End, len(feats))  # type: ignore
        labels[first:last] = row.Name  # type: ignore

    # Create dataframe
    results = pd.DataFrame(
        {
            "second": np.arange(len(feats)),
            "PC1": feats_2d[:, 0],
            "PC2": feats_2d[:, 1],
            "label": labels,
        }
    )

    # Exclude unknown labels
    plot_data = results[results["label"] != "unknown"]

    fig, ax = plt.subplots(figsize=(12, 8))

    for label, group in plot_data.groupby("label"):
        ax.scatter(group["PC1"], group["PC2"], label=label, s=15, alpha=0.7)

    ax.set_title("DINOv2 PCA - Annotated Frames")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()
    plt.tight_layout()

    out_path.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path / f"{emb_path.name}_pca.png", dpi=150)
    plt.close(fig)
