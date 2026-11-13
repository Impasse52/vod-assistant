from pathlib import Path

import numpy as np
import pandas as pd

from .labels import load_labels

CLASS_MAP = {
    "ssbu": "game",
    "melee": "game",
    "game": "game",
    "playercam": "camera",
    "comms": "camera",
    "camera": "camera",
    "menu": "none",
    "layout": "none",
    "none": "none",
}


def game_runs(segments, merge_gap: float = 3.0) -> list[list[float]]:
    """Merge consecutive game segments separated by <= merge_gap seconds."""
    runs: list[list[float]] = []
    for s in sorted(segments, key=lambda s: s.start):
        if CLASS_MAP[s.name] != "game":
            continue
        if runs and s.start - runs[-1][1] <= merge_gap:
            runs[-1][1] = max(runs[-1][1], s.end)
        else:
            runs.append([s.start, s.end])
    return runs


def gap_composition(segments, a: float, b: float) -> dict[str, float]:
    """Seconds spent in each class inside the window [a, b)."""
    out = {"camera": 0.0, "none": 0.0}
    for s in segments:
        cls = CLASS_MAP[s.name]
        if cls == "game":
            continue
        overlap = min(b, s.end) - max(a, s.start)
        if overlap > 0:
            out[cls] += overlap
    return out


def draft_table(csv_path: Path, guess_gap: float = 120.0) -> pd.DataFrame:
    """One row per game run, with features of the gap before it.

    `new_set` is only a draft guess and is meant to be manually checked.
    """
    segments = load_labels(csv_path)
    runs = game_runs(segments)
    rows = []
    for i, (start, end) in enumerate(runs):
        gap = start - runs[i - 1][1] if i else float("nan")
        comp = gap_composition(segments, runs[i - 1][1], start) if i else {}
        total = max(gap, 1e-9) if i else 1.0
        rows.append(
            {
                # "run": i,
                "start": round(start, 1),
                "end": round(end, 1),
                "gap_before": round(gap, 1) if i else None,
                "camera_frac": round(comp.get("camera", 0) / total, 2) if i else None,
                "none_frac": round(comp.get("none", 0) / total, 2) if i else None,
                "new_set": int(i == 0 or gap > guess_gap),
            }
        )

    return pd.DataFrame(rows)


def runs_to_sets(csv_path: Path, lead: float = 7.0) -> pd.DataFrame:
    runs = pd.read_csv(csv_path)
    opens = runs["new_set"].fillna(0).astype(bool)

    sets = runs.groupby(opens.cumsum()).agg(
        start=("start", "min"), end=("end", "max"), n_runs=("start", "size")
    )

    sets["start"] = (sets["start"] - lead).clip(lower=0)
    sets["end"] = sets["end"] + lead

    return sets.reset_index(drop=True)
