import argparse
from pathlib import Path

# Map CSV class names to target classes. Every CSV name must be listed.
MERGES = {
    "game-vs-other": {
        "game": "game",
        "playercam": "other",
        "camera": "other",
        "comms": "other",
        "layout": "other",
    },
}


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
    t.add_argument(
        "--pairs", nargs="+", help="embeds.npy:labels.csv, one per VOD", required=True
    )
    t.add_argument(
        "--classes", nargs="+", default=["game", "playercam", "comms", "layout"]
    )
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

        emb = embed(args.video_path, batch_size=args.batch_size, model_name=args.model)
        tmp = dest.with_suffix(".tmp.npy")
        np.save(tmp, emb)
        tmp.replace(dest)

        print(f"Saved {dest} {emb.shape}")

    if args.command == "pca":
        from src.pca import run_pca

        run_pca(args.emb_path, args.out_dir, args.labels_path)

    if args.command == "train":
        from src.train import train_logreg

        name_map = MERGES[args.merge] if args.merge else None
        # Unique target classes, in order of first appearance.
        classes = list(dict.fromkeys(name_map.values())) if name_map else args.classes
        print("classes:", classes)
        train_logreg(args.pairs, classes, name_map)


if __name__ == "__main__":
    main()
