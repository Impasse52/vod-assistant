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


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def _make_clf():
    return LogisticRegression(max_iter=1000, class_weight="balanced", C=0.0001)


def train_logreg(pairs, classes, name_map=None, n_splits=None):
    """Evaluate a logistic regression with GroupKFold (group = VOD), then
    refit on all VODs and return the final model.

    Split by VOD, never by frame: every second of a VOD lands either in the
    train or in the test side of a fold, never both.
    """
    vods = []
    for pair in pairs:
        emb_path, csv_path = pair.split(":")
        vods.append(load_vod(emb_path, csv_path, classes, name_map))

    if len(vods) < 2:
        raise SystemExit("Need at least 2 VODs for cross-validation.")

    X = np.concatenate([x for x, _ in vods])
    y = np.concatenate([y for _, y in vods])
    groups = np.concatenate([np.full(len(yv), i) for i, (_, yv) in enumerate(vods)])

    # n_splits cannot exceed the number of VODs; with n_splits == len(vods)
    # this is leave-one-VOD-out.
    n_splits = min(n_splits or 5, len(vods))
    print("Total class counts:", np.bincount(y, minlength=len(classes)))
    print(f"GroupKFold: {n_splits} folds over {len(vods)} VODs")

    y_oof = np.empty_like(y)
    accs, f1s = [], []

    for fold, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        # Fit the scaler on the training fold only.
        scaler = StandardScaler().fit(X[tr])
        clf = _make_clf().fit(scaler.transform(X[tr]), y[tr])
        pred = clf.predict(scaler.transform(X[te]))
        y_oof[te] = pred

        present = sorted(set(y[te]) | set(pred))
        acc = accuracy_score(y[te], pred)
        f1 = f1_score(y[te], pred, labels=present, average="macro", zero_division=0)
        accs.append(acc)
        f1s.append(f1)
        print(
            f"Fold {fold}: test VODs={sorted(set(groups[te]))} "
            f"acc={acc:.3f} macro-F1={f1:.3f}"
        )

    print(f"\nMean acc:      {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"Mean macro-F1: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}\n")

    # Pooled report over all out-of-fold predictions (more stable than
    # per-fold reports when single VODs have few examples of some classes).
    present = sorted(set(y) | set(y_oof))
    print(
        classification_report(
            y,
            y_oof,
            labels=present,
            target_names=[classes[i] for i in present],
            zero_division=0,
        )
    )

    # Final model: refit on everything.
    scaler = StandardScaler().fit(X)
    clf = _make_clf().fit(scaler.transform(X), y)

    # segment_report(clf, scaler, emb_path, csv_path, fps, classes, name_map)

    return clf, scaler
