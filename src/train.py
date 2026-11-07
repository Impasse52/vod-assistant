from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .labels import load_labels, to_indices

UNLABELED = -1


def load_vod(emb_path, csv_path, classes, name_map=None):
    X = np.load(emb_path)  # (N, D), one row per second
    labels = load_labels(csv_path)

    # Build matrix of label indexes
    y = np.full(len(X), UNLABELED, dtype=int)
    for row in labels.itertuples():
        name = name_map[row.Name] if name_map else row.Name  # type: ignore
        first, last = to_indices(row.Start, row.End, len(X))  # type: ignore
        y[first:last] = classes.index(name)

    mask = y != UNLABELED
    return X[mask], y[mask]


def _make_clf():
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def train_logreg(
    pairs, classes, name_map=None, n_splits=None, out_path=Path("./models")
):
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

    # Needed for pooled eval
    y_oof = np.empty_like(y)

    accs, f1s = [], []

    for fold, samples in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        train, test = samples

        # Fit the scaler on the training fold only
        scaler = StandardScaler().fit(X[train])
        clf = _make_clf().fit(scaler.transform(X[train]), y[train])
        pred = clf.predict(scaler.transform(X[test]))
        y_oof[test] = pred

        # Compute metrics
        curr_labels = sorted(set(y[test]) | set(pred))
        acc = accuracy_score(y[test], pred)
        f1 = f1_score(
            y[test], pred, labels=curr_labels, average="macro", zero_division=0
        )
        accs.append(acc)
        f1s.append(f1)
        print(
            f"Fold {fold}: test VODs={sorted(set(groups[test]))} "
            f"acc={acc:.3f} macro-F1={f1:.3f}"
        )
        print(
            classification_report(
                y[test],
                pred,
                labels=curr_labels,
                target_names=[classes[i] for i in curr_labels],
                zero_division=0,
            )
        )

    print(f"\nMean acc:      {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"Mean macro-F1: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}\n")

    # Pooled report over all out-of-fold predictions
    curr_labels = sorted(set(y) | set(y_oof))
    print(
        classification_report(
            y,
            y_oof,
            labels=curr_labels,
            target_names=[classes[i] for i in curr_labels],
            zero_division=0,
        )
    )

    # Final model: refit on everything.
    scaler = StandardScaler().fit(X)
    clf = _make_clf().fit(scaler.transform(X), y)

    # Save the model together with the class names, so inference
    # does not depend on the order of a list defined elsewhere.
    out_path.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "classes": classes}, "models/logreg.joblib")

    return clf, scaler
