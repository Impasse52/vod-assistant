import subprocess
from pathlib import Path

import numpy as np
from tqdm import tqdm

W, H = 392, 224  # multiples of 14 to maintain 16:9 ratio


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
        check=False,
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
