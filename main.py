import argparse
import os
from glob import glob
from pathlib import Path

import pandas as pd

from src.labels import to_intervals

LABELS = ["ssbu", "melee", "playercam", "comms", "layout", "menu", "none"]

# Map CSV class names to target classes. Every CSV name must be listed.
MERGES = {
    "game-only": {
        "ssbu": "ssbu",
        "melee": "melee",
        "playercam": "other",
        "comms": "other",
        "layout": "other",
        "menu": "other",
        "none": "other",
    },
    "simple": {
        "ssbu": "ssbu",
        "melee": "melee",
        "playercam": "camera",
        "comms": "camera",
        "layout": "none",
        "menu": "none",
        "none": "none",
    },
}


def pairs_from_dirs(spec: str) -> list[str]:
    """'EMBEDS_DIR:LABELS_DIR' -> ['embeds/x.npy:labels/x.csv', ...], matched by stem."""
    emb_dir, labels_dir = map(Path, spec.split(":"))
    embs = {p.stem: p for p in emb_dir.glob("*.npy")}
    labels = {p.stem: p for p in labels_dir.glob("*.csv")}

    if embs.keys() != labels.keys():
        raise SystemExit(
            "Mismatch between embeddings and labels, aborting training.\n"
            f"  Embeddings without labels: {sorted(embs.keys() - labels.keys())}\n"
            f"  Labels without embeddings: {sorted(labels.keys() - embs.keys())}"
        )
    return [f"{embs[s]}:{labels[s]}" for s in sorted(embs)]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vod-assistant")
    sub = p.add_subparsers(dest="command", required=True)

    # Embed command
    e = sub.add_parser("embed", help="embed 1 fps frames of VODs into .npy files")
    e.add_argument("--video-path", type=Path)
    e.add_argument("--out-dir", type=Path, default=Path("embeddings"))
    e.add_argument(
        "--model",
        default="facebook/dinov2-small",
        help="HF model id (default: facebook/dinov2-small)",
    )
    e.add_argument("--force", action="store_true", help="recompute existing embeddings")
    e.add_argument("--batch-size", type=int, default=64)

    # Train command
    t = sub.add_parser("train", help="train a per-second scene classifier")

    src = t.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--pairs",
        nargs="+",
        metavar="EMBEDS.npy:LABELS.csv",
        help="embeds.npy:labels.csv, one per VOD",
    )
    src.add_argument(
        "--dirs",
        metavar="LABELS_DIR:EMBEDS_DIR",
        help="folder of label CSVs and folder of embeddings, matched by filename stem",
    )

    t.add_argument("--classes", nargs="+", default=LABELS)
    t.add_argument(
        "--merge",
        choices=sorted(MERGES),
        help="collapse CSV classes into target classes (overrides --classes)",
    )

    # PCA command
    a = sub.add_parser("pca", help="PCA plot of the embeddings")
    a.add_argument("--emb-path", type=Path, required=True)
    a.add_argument(
        "--labels-path", type=Path, required=True, help="Color points by label"
    )
    a.add_argument("--out-dir", type=Path, default=Path("pca"))

    # Predict frames command
    t = sub.add_parser("pred-frames", help="Predict frames on embeds using a model")
    t.add_argument("--emb-path", help="Path of embeds file", type=Path, required=True)
    t.add_argument("--model-path", type=Path, required=True)
    t.add_argument("--out-dir", required=False, default=Path("output"))
    t.add_argument("--classes", nargs="+", default=LABELS)

    d = sub.add_parser("dataset-info", help="Show dataset info")
    d.add_argument(
        "--data-path", help="Path of data files", type=Path, default=Path("data")
    )

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "embed":
        import numpy as np

        from src.embed import embed

        args.out_dir.mkdir(parents=True, exist_ok=True)
        dest = args.out_dir / f"{args.video_path.stem}.npy"
        if dest.exists() and not args.force:
            print(f"{dest} exists, skipping (use --force)")
            return

        paths = [args.video_path]
        if os.path.isdir(args.video_path):
            paths = glob(f"{args.video_path}/*.mp4")

        for path in paths:
            print(f"Computing embeds for VOD: {path}")
            emb = embed(path, batch_size=args.batch_size, model_name=args.model)
            tmp = dest.with_suffix(".tmp.npy")
            np.save(tmp, emb)
            tmp.replace(dest)

            print(f"Saved {dest} {emb.shape}")

    if args.command == "pca":
        from src.pca import run_pca

        run_pca(args.emb_path, args.out_dir, args.labels_path)

    if args.command == "dataset-info":
        from src.labels import get_dataset_info

        get_dataset_info(args.data_path)

    if args.command == "train":
        from src.train import train_logreg

        name_map = MERGES[args.merge] if args.merge else None
        # Unique target classes, in order of first appearance.
        classes = list(dict.fromkeys(name_map.values())) if name_map else args.classes
        print("Classes:", classes)

        pairs = pairs_from_dirs(args.dirs) if args.dirs else args.pairs

        train_logreg(pairs, classes, name_map)

    if args.command == "pred-frames":
        from src.pred import pred_frames

        labels, conf = pred_frames(args.emb_path, args.model_path)
        intervals = to_intervals(labels, conf)
        df = pd.DataFrame(intervals, columns=["Start", "End", "Name", "Confidence"])

        args.out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(
            Path(args.out_dir / args.emb_path.stem).with_suffix(".csv"), index=False
        )


if __name__ == "__main__":
    main()
