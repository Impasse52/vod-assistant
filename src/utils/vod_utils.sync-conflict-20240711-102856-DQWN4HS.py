from copy import deepcopy
from datetime import timedelta
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import tensorflow as tf
import ffmpeg


def preprocess_image(img_dir):
    img = tf.keras.preprocessing.image.load_img(
        img_dir,
        target_size=(224, 224),  # inputs directory  # resizes images
    )

    # (height, width, channels)
    img_tensor = tf.keras.preprocessing.image.img_to_array(img)

    # (1, height, width, channels), add a dimension because the model
    # expects this shape: (batch_size, height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)

    # normalization
    img_tensor /= 255.0

    return img_tensor


def predict_screen(model, img_dir, labels) -> tuple:
    img_tensor = preprocess_image(img_dir)

    predictions = model.predict(img_tensor)

    # gets label with highest confidence
    pred_idx = np.argmax(predictions, axis=1)[0]
    label = labels[pred_idx]

    # uses original file's name as output name
    # keeping the filename as it is useful to track timestamps
    name = img_dir.split("/")[-1]

    return (name, label, predictions[0][pred_idx])


def video_to_frames(input_file: Path, output_dir: Path) -> None:
    try:
        (
            ffmpeg.input(input_file)
            .filter("fps", fps=1)
            .output(
                f"{output_dir}/%d.jpg",
                s="426x240",
                start_number=0,
            )
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print("stdout:", e.stdout.decode("utf8"))
        print("stderr:", e.stderr.decode("utf8"))


def setup_dirs(input_file: str) -> tuple:
    try:
        os.mkdir("./data")
    except FileExistsError:
        print("Root data directory already exists, skipping.")

    dataframes_dir = Path(f"./data/dataframes/{input_file}")
    timestamps_dir = Path(f"./data/timestamps/{input_file}")
    snapshots_dir = Path(f"./data/snapshots/{input_file}")

    try:
        os.mkdir(dataframes_dir)
        os.mkdir(timestamps_dir)
        os.mkdir(snapshots_dir)
    except FileExistsError:
        print(f"Data directories already exist for {input_file}")

    return (dataframes_dir, timestamps_dir, snapshots_dir)


def group_frames(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df
    tmp = tmp.groupby("group").first()
    tmp["end_ts"] = df.groupby("group").last()["timestamp"]

    return tmp


def apply_thresholding(df: pd.DataFrame, col: str, threshold: int) -> pd.DataFrame:
    df = df.drop(df[df[col] < threshold].index)

    # re-apply grouping after thresholding
    if "group" in df.columns:
        df["group"] = (df["label"] != df["label"].shift()).cumsum()

    return df


def plot_1d_scatter(
    df: pd.DataFrame,
    col: str,
    label: str | None = None,
    c: str | None = None,
    edgecolors: str | None = None,
) -> None:
    plt.scatter(
        x=df[col],
        y=[0 for _ in range(df.shape[0])],
        marker="s",
        s=35,
        label=label,
        c=df[c] if c else None,
        edgecolors=edgecolors,
        linewidth=1,
    )  # type: ignore


def extract_vod_timestamps(
    df: pd.DataFrame,
    output_dir: str,
    title: str = "title",
    buffer: int = 7,
    override: bool = False,
) -> None:
    # create tmp variable to ensure that time buffers are applied only once
    tmp = deepcopy(df)
    timestamps_file = ""

    tmp["timestamp"] = df["timestamp"].apply(
        lambda x: max(pd.Timestamp(0), x - timedelta(seconds=7))
    )
    tmp["end_ts"] = df["end_ts"].apply(
        lambda x: min(x + timedelta(seconds=buffer), max(df["end_ts"]))
    )

    if os.path.exists(output_dir) and not override:
        with open(output_dir) as output:
            print("Timestamps file already exists:\n")
            print(output.read())
            return

    for i, idx in enumerate(tmp["cluster"].unique()):
        start = tmp[tmp["cluster"] == idx].iloc[0]["timestamp"]
        end = tmp[tmp["cluster"] == idx].iloc[-1]["end_ts"]

        timestamp_line = f"{title} | round - player1 (char1) vs player2 (char2) -- {str(start)[11:]} - {str(end)[11:]} -- {i + 1} \n"
        timestamps_file += timestamp_line

        print(timestamp_line, end="")

    if override:
        with open(output_dir, "w") as output:
            output.writelines(timestamps_file)


def frames_to_games(predictions: list) -> pd.DataFrame:
    df = pd.DataFrame(predictions, columns=["frame", "label", "confidence"])
    frame_idxs = df["frame"].apply(lambda x: x.replace(".jpg", ""))
    df["timestamp"] = pd.to_datetime(frame_idxs, unit="s")

    df["group"] = (df["label"] != df["label"].shift()).cumsum()
    df["duration"] = df.groupby("group")["group"].transform("count")
    # predictions_df = predictions_df.drop(columns=["group_id"])

    return df


def games_to_sets(df: pd.DataFrame, n_clusters: int):
    # exclude junk labels
    df = df[df["label"] == "gameplay"]

    # setup clustering algorithm
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="single")

    # convert datetime to int
    tmp = deepcopy(df)
    tmp["timestamp"] = tmp["timestamp"].astype("int")
    tmp["end_ts"] = tmp["end_ts"].astype("int")

    # specify X and Y axis and compute clusters
    labels = clustering.fit_predict(tmp[["timestamp", "end_ts"]])

    # add clusters to dataframe for easy handling
    df.loc[:, "cluster"] = labels

    return df
