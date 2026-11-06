import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from .labels import load_labels, to_indices

UNLABELED = -1


def load_vod(emb_path, csv_path, classes, name_map=None):
    emb = np.load(emb_path)  # (N, D), one row per second
    labels = load_labels(csv_path, n_seconds=len(emb))

    y = np.full(len(emb), UNLABELED, dtype=int)
    for row in labels.itertuples():
        name = name_map[row.Name] if name_map else row.Name  # type: ignore
        first, last = to_indices(row.Start, row.End, len(emb))  # type: ignore
        y[first:last] = classes.index(name)

    mask = y != UNLABELED
    return emb[mask], y[mask]


def segment_report(clf, scaler, emb_path, csv_path, classes, name_map=None):
    """Per-segment accuracy of the predictions on one VOD."""
    emb = np.load(emb_path)
    pred = clf.predict(scaler.transform(emb))
    df = pd.read_csv(csv_path)

    rows = []
    for r in df.itertuples():
        a, b = to_indices(r.Start, r.End, len(emb))  # type: ignore

        if b <= a:
            continue

        true = classes.index(name_map[r.Name] if name_map else r.Name)  # type: ignore
        rows.append((r.Name, a, b - a, (pred[a:b] == true).mean()))  # type: ignore

    df = pd.DataFrame(rows, columns=["name", "start_s", "dur_s", "acc"])

    print(df.sort_values("dur_s", ascending=False).round(2).to_string(index=False))


def train_logreg(pairs, classes, fps=29.97, name_map=None):
    """Train a logistic regression on labeled seconds only.

    Split by VOD, never by frame: the last VOD in `pairs` is held out for
    evaluation, all the others are used for training. `classes` are the
    target classes after applying `name_map`.
    """
    vods = []
    for pair in pairs:
        emb_path, csv_path = pair.split(":")
        vods.append(load_vod(emb_path, csv_path, classes, name_map))

    if len(vods) < 2:
        raise SystemExit("Need at least 2 VODs (train + held-out eval).")

    train, test = vods[:-1], vods[-1]
    X_train = np.concatenate([x for x, _ in train])
    y_train = np.concatenate([y for _, y in train])
    X_test, y_test = test

    print("train class counts:", np.bincount(y_train, minlength=len(classes)))
    print("test class counts: ", np.bincount(y_test, minlength=len(classes)))

    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.0001)
    clf.fit(scaler.transform(X_train), y_train)

    print("train acc:", clf.score(scaler.transform(X_train), y_train))

    y_pred = clf.predict(scaler.transform(X_test))
    present = sorted(set(y_test) | set(y_pred))
    print(
        classification_report(
            y_test,
            y_pred,
            labels=present,
            target_names=[classes[i] for i in present],
            zero_division=0,
        )
    )

    # segment_report(clf, scaler, emb_path, csv_path, fps, classes, name_map)

    return clf, scaler
