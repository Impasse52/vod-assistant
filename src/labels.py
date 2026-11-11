"""Label loading and conversion to per-second embedding indices.

Labels are expressed in seconds (floats). Embeddings are sampled at 1 fps,
so row ``i`` of an embedding array is the frame at time ``i`` seconds.
This module is the single place where seconds become row indices, so that
training, evaluation and PCA always agree.
"""

from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd


class Segment(NamedTuple):
    start: float
    end: float
    name: str


def load_labels(csv_path: str | Path) -> list[Segment]:
    """Read a label CSV with columns Start, End, Name (Start/End in seconds)."""
    df = pd.read_csv(csv_path)
    return [
        Segment(float(s), float(e), str(n))
        for s, e, n in zip(df["Start"], df["End"], df["Name"])
    ]


def to_indices(start: float, end: float, n: int) -> tuple[int, int]:
    """Map the interval [start, end) in seconds to embedding rows [first, last).

    Row i belongs to the label iff start <= i < end, hence the two ceil calls.
    The result is clamped to [0, n].
    """
    first = max(0, int(np.ceil(start)))
    last = min(n, int(np.ceil(end)))

    return first, last


def to_intervals(
    labels: np.ndarray, conf: list, ignore: Any = None
) -> list[tuple[int, int, Any, float]]:
    """Group consecutive equal values of a per-second array into intervals.

    Row i covers the second [i, i + 1), so a run over rows first..last - 1
    becomes the interval [first, last) in seconds, the inverse of to_indices.
    Labels can be integers, strings or booleans and are returned unchanged.
    Returns (start, end, label) tuples sorted by start. Runs whose label equals
    `ignore` are skipped. For a boolean mask use ignore=False.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be 1-D")
    if len(labels) == 0:
        return []

    # Rows where the value differs from the previous one start a new run.
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(labels)]))

    return [
        (int(s), int(e), labels[s].item(), c)
        for s, e, c in zip(starts, ends, conf)
        if ignore is None or labels[s] != ignore
    ]


def get_dataset_info(data_path: Path):
    rows = {}
    for csv_path in sorted(Path(data_path).glob("*.csv")):
        df = pd.read_csv(csv_path)

        # Number of samples covered by each segment (End treated as exclusive)
        df["n_samples"] = (df["End"] - df["Start"]).round().astype(int)
        rows[csv_path.stem] = df.groupby("Name")["n_samples"].sum()

    # Missing labels in a CSV become 0
    table = pd.DataFrame.from_dict(rows, orient="index").fillna(0).astype(int)
    table.index.name = "dataset"

    table["total"] = table.sum(axis=1)
    table.loc["total"] = table.sum(axis=0)

    return table.sort_index(axis=1)
