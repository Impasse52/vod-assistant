from pathlib import Path

import joblib
import numpy as np


def pred_frames(emb_path: Path, model_path: Path):
    bundle = joblib.load(model_path)
    model, classes = bundle["model"], bundle["classes"]

    X = np.load(emb_path)  # (N, 768)
    proba = model.predict_proba(X)  # (N, n_classes)

    pred = proba.argmax(axis=1)
    conf = proba[np.arange(len(pred)), pred]
    labels = [classes[i] for i in pred]

    return labels, conf
