import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

UNLABELED = -1


def load_vod(emb_path, csv_path, fps, classes):
    """Return (X, y) keeping only the seconds covered by a label."""
    emb = np.load(emb_path)  # (N, D), one row per second
    labels = pd.read_csv(csv_path)

    y = np.full(len(emb), UNLABELED, dtype=int)
    for row in labels.itertuples():
        # frame -> second; second i is labeled if it falls inside [start, end)
        first = int(np.ceil(row.Start / fps))
        last = int(np.ceil(row.End / fps))
        y[first:last] = classes.index(row.Name)

    mask = y != UNLABELED  # skip unlabeled embeddings
    return emb[mask], y[mask]


def train_logreg(
    pairs,
    classes,
    fps=29.97,
):
    """Train a logistic regression on labeled seconds only (80/20 split).

    The split is chronological to limit leakage between
    neighboring, nearly identical frames.
    """
    vods = []
    for pair in pairs:
        emb_path, csv_path = pair.split(":")
        vods.append(load_vod(emb_path, csv_path, fps, classes))

    # Split by VOD, never by frame: the last VOD is held out for evaluation.
    if len(vods) < 2:
        raise SystemExit("Need at least 2 VODs (train + held-out eval).")

    train, test = vods[:-1], vods[-1]
    X_train = np.concatenate([x for x, _ in train])
    y_train = np.concatenate([y for _, y in train])
    X_test, y_test = test

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
