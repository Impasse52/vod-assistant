"""Label loading and conversion to per-second embedding indices.

Labels are expressed in seconds (floats). Embeddings are sampled at 1 fps,
so row ``i`` of an embedding array is the frame at time ``i`` seconds.
This module is the single place where seconds become row indices, so that
training, evaluation and PCA always agree.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_labels(csv_path: str | Path, n_seconds: int | None = None) -> pd.DataFrame:
    """Read a label CSV with columns Start, End, Name (Start/End in seconds).

    If ``n_seconds`` is given, raise when a label lies past the end of the
    VOD. This is the typical symptom of a CSV that is still in frames, which
    would otherwise be silently ignored by array slicing.
    """
    df = pd.read_csv(csv_path)
    df["Start"] = df["Start"].astype(float)
    df["End"] = df["End"].astype(float)

    if (df["End"] < df["Start"]).any():
        raise ValueError(f"{csv_path}: found a label with End < Start")

    return df


def to_indices(start: float, end: float, n: int) -> tuple[int, int]:
    """Map the interval [start, end) in seconds to embedding rows [first, last).

    Row i belongs to the label iff start <= i < end, hence the two ceil calls.
    The result is clamped to [0, n].
    """
    first = max(0, int(np.ceil(start)))
    last = min(n, int(np.ceil(end)))

    return first, last
