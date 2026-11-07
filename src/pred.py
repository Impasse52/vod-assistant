from pathlib import Path

import joblib
import numpy as np


def pred_frames(emb_path: Path, model_path: Path):
    bundle = joblib.load(model_path)
    model, classes = bundle["model"], bundle["classes"]

    X = np.load(emb_path)
    pred = model.predict(X)  # X: (N, 768) embeddings
    labels = [classes[i] for i in pred]

    return labels
