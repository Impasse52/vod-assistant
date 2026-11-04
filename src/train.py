import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

UNLABELED = -1


def train_logreg(
    emb_path,
    csv_path,
    fps=29.97,
):
    """Train a logistic regression on labeled seconds only (80/20 split).

    The split is chronological (first 80% train, last 20% test) to limit
    leakage between neighboring, nearly identical frames.
    """
    emb = np.load(emb_path)  # (N, D), one row per second
    labels = pd.read_csv(csv_path)

    # Get unique classes
    classes = list(labels.Name.unique())

    # Build per-second labels; -1 means unlabeled
    y = np.full(len(emb), -1, dtype=int)
    for row in labels.itertuples():
        first = int(np.ceil(row.Start / fps))
        last = int(np.ceil(row.End / fps))
        y[first:last] = classes.index(row.Name)

    # Skip unlabeled embeddings
    mask = y != -1
    X, y = emb[mask], y[mask]

    print(X.shape)
    print(y.shape)
    print(len(emb))

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(scaler.transform(X_train), y_train)

    pred = clf.predict(scaler.transform(X_test))
    present = sorted(set(y_test) | set(pred))
    print(
        classification_report(
            y_test,
            pred,
            labels=present,
            target_names=[classes[i] for i in present],
            zero_division=0,
        )
    )
    return clf, scaler
