import json
import os
import re
import sys

from utils.trim_utils import replace_accented_with_quote


def crop(start, end, input, output) -> None:
    os.system(rf'ffmpeg -i "{input}" -ss {start} -to {end} -c copy "{output}"')


def tokenize_titles(filename: str) -> list:
    with open(filename) as vod_file:
        vod_file = "".join(vod_file.readlines())

    title_re = r"(.*) \| (.*) - (.*) \((.*?)\) vs (.*) \((.*?)\) -- (\d+:\d\d:\d\d) - (\d+:\d\d:\d\d) -- (\d+)"
    title = re.findall(title_re, vod_file)
    print(title)

    return title


def trim_vods(titles: list, input_filename: str, videos_dir: str) -> None:
    for t in titles:
        output_filename = f"{(t[2] + t[4]).lower()}".replace(" ", "")
        print(output_filename)

        n = os.listdir().count(f"{output_filename}.mp4")

        if n > 0:
            output_filename = f"{output_filename}{n + 1}"

        start_time = t[6]
        end_time = t[7]

        crop(
            start_time, end_time, input_filename, f"{videos_dir}/{output_filename}.mp4"
        )


if __name__ == "__main__":
    from thumbfair.thumbfair import gen_thumbnail

    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
        timestamps_dir = sys.argv[2]
    else:
        input_dir = "/home/impasse/Code/Projects/VOD-Assistant/u.mkv"
        timestamps_dir = "/home/impasse/Code/Projects/VOD-Assistant/data/timestamps/2024-09-22_2257511253/2024-09-22_2257511253.md"

    with open(
        "/home/impasse/Code/Projects/Thumbfair/thumbfair/resources/skins.json"
    ) as skins:
        skins = json.load(skins)

    titles = tokenize_titles(timestamps_dir)
    t_name = titles[0][0]
    try:
        os.mkdir(f"output/thumbnails/{t_name}/")
    except FileExistsError:
        print(f"Thumbnail directory already exists for {t_name}, skipping.")

    for t in titles:
        _, r_name, p1_nick, p1_char, p2_nick, p2_char, _, _, _ = t

        # only get the first character for each player
        p1_char = p1_char.split(", ")[0]
        p2_char = p2_char.split(", ")[0]

        output = gen_thumbnail(
            p1_nick,
            p2_nick,
            r_name,
            p1_char,
            p2_char,
            skins[p1_char].get(p1_nick, "1"),
            skins[p2_char].get(p2_nick, "1"),
        )
        output.save(f"output/thumbnails/{t_name}/{p1_nick}_{p2_nick}_{r_name}.png")

    vods_dir = f"./output/vods/{t_name}"
    # os.mkdir(vods_dir)
    # trim_vods(titles, input_dir, vods_dir)
