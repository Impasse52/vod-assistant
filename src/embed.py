from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel

from src.video import frames

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


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
