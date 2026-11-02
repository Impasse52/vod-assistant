import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from transformers import AutoModel

W, H = 392, 224  # multipli di 14, mantiene il rapporto 16:9
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
from pathlib import Path

from tqdm import tqdm


def duration(path: str) -> float | None:
    """Return the video duration in seconds, or None if ffprobe cannot read it."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            path,
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return None


def frames(path: str, fps: int = 1):
    """Yield RGB frames of shape (H, W, 3), one every 1/fps seconds.

    Frames are decoded and resized by ffmpeg and read from its stdout, so
    nothing is written to disk. The video frame rate does not matter: the
    ``fps`` filter always keeps ``fps`` frames per second of footage.
    """
    dur = duration(path)
    total = int(dur * fps) if dur else None

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        path,
        "-vf",
        f"fps={fps},scale={W}:{H}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = W * H * 3  # bytes per frame: 3 bytes (R, G, B) per pixel

    try:
        with tqdm(total=total, unit="frame", desc=Path(path).name) as pbar:
            while True:
                buf = proc.stdout.read(size)
                if len(buf) < size:  # end of the video
                    break
                pbar.update(1)
                yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    finally:
        # Do not leave ffmpeg running if the consumer stops early (e.g. Ctrl+C).
        if proc.poll() is None:
            proc.kill()
        proc.wait()


# Do not track gradients during training
@torch.inference_mode()
def embed(path: str, model_name: str = "facebook/dinov2-small", batch_size: int = 64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    if device == "cuda":
        # Half precision
        model = model.half()

    mean, std = MEAN.to(device), STD.to(device)
    features, buf = [], []

    def flush():
        x = torch.from_numpy(np.stack(buf)).to(device).permute(0, 3, 1, 2).float() / 255
        x = (x - mean) / std
        if device == "cuda":
            x = x.half()  # Half precision
        h = model(pixel_values=x).last_hidden_state  # (B, 1 + patch, D)

        # CLS + media delle patch: la combinazione usata nel paper per il linear probing
        features.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1).float().cpu())
        buf.clear()

    # GPU is more efficient handling 64 images at once
    for frame in frames(path):
        buf.append(frame)
        if len(buf) == batch_size:
            flush()
    if buf:
        flush()
    return torch.cat(features).numpy()  # (n_secondi, 2 * D)


def load_timestamps(path: Path, fps_original: float, fps_target: float = 1.0):
    df = pd.read_csv(path)

    df["Start"] = (df["Start"] / fps_original * fps_target).round().astype(int)
    df["End"] = (df["End"] / fps_original * fps_target).round().astype(int)

    return df


if __name__ == "__main__":
    video, dest = sys.argv[1], sys.argv[2]

    fps_original = 30000 / 1001
    ts_path = Path("./data/genesis_x3_top64_ssbu.mp4.csv")
    df = load_timestamps(ts_path, fps_original)
    print(df)

    ## Load embeddings
    feats = np.load(dest + ".npy")
