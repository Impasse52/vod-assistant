# vod-assistant

This project contains a set of scripts that aim at simplifying VODs-cutting workflows by automatically providing timestamps for beginning and end of each tournament set.

The classifier is trained on DINOv2 embeddings, created from tournament VODs, and an annotated dataset (see `data/`) which labels every frame of the VOD. The classifier then learns to distinguish game frames from other scenes such as streaming layout or menus. 

The predicted game frames can then be used to infer beginning/end of sets (currently naively implemented through a simple rule), which will be used to produce a dataframe of sets. 


## Usage
**Setup**: sync the environment by cloning this repo then running `uv sync`.

To **predict frames**:
1. Run `pred-frames` as follows: `uv run main.py pred-frames --model-path <model.joblib> --emb-path <embeddings.npy>`, which will produce a `.csv` file of predictions for each frame.
2. Merge consecutive frames into runs: `uv run main.py runs --data-path <frames.csv>`
3. Convert runs to sets: `uv run main.py runs-to-sets --data-path <runs.csv> --lead <n>`, where `n` is the number of starting/leading seconds to add to each set.

It is suggested to import the output into software such as [LosslessCut](https://github.com/mifi/lossless-cut) in order to double check the results before cutting and uploading.


**Training**: assuming embeddings at `embeddings/` and CSVs in `data/`, training can be performed by simply calling `uv run main.py train --dirs embeddings:data`


## Dataset

All VODs have been sampled at 480p1fps and embedded using `dinov2-small` (see `src/embed.py`).

|VOD| Link | 
|---|---|
| Genesis X3 - Top 64 SSBU          | https://www.youtube.com/watch?v=JFomLqNnixU  | 
| Pound 2022 - SSBU/Melee Finals    | https://www.twitch.tv/videos/1465710386      | 
| Smashcon 2023 - SSBU/Melee Top 8  | https://www.twitch.tv/videos/1897725006       |
| Arena dei Pupi |  https://www.twitch.tv/videos/2910329470  |
| Ducato: Binding Blade | https://www.twitch.tv/videos/2884528491 |
| Double Down 2022  | https://www.twitch.tv/videos/1526959037 |
| Glitch: Konami Code  | https://www.twitch.tv/videos/1159903417  |
| Dehscension XLV |  https://www.twitch.tv/videos/2873000878 |

### Labels

```
                                             comms  layout  melee  menu  none  playercam   ssbu   total
dataset
arenadeipupi                                  2982       0      0  2599   130          0   7457   13168
bindingblade                                   376       0      0   158     0       1054   2065    3653
dehscensionxlv                                   0    1064      0  2816     0          0   3470    7350
doubledown                                    1181    2444      0     0   391       1648   4380   10044
genesis_x3_top64_ssbu                         3824    5238      0     0   759       9721  23353   42895
glitchkonamicode                               928    1129      0     0   325       1058   3847    7287
pound2022_ssbu_melee_finals                   5667    4801   9178     0  1439       4838  16976   42899
smashcon2023_ssbutop32_melee_top8_ssbu_top8   5257    5983   7470     0   426       9643  17485   46264
total                                        20215   20659  16648  5573  3470      27962  79033  173560
```


## Results
Results when labels are limited to `game`, `camera` and `none`, obtained by merging (see the `--merge default` option for `train`).

```
Mean acc:      0.958 ± 0.033
Mean macro-F1: 0.944 ± 0.039

              precision    recall  f1-score   support

        game       0.98      0.99      0.98     95683
      camera       0.98      0.93      0.96     48173
        none       0.91      0.94      0.93     29700

    accuracy                           0.97    173556
   macro avg       0.95      0.96      0.95    173556
weighted avg       0.97      0.97      0.97    173556
```


Results when training on every label:

```
Mean acc:      0.846 ± 0.137
Mean macro-F1: 0.731 ± 0.189

              precision    recall  f1-score   support

        ssbu       0.98      0.99      0.99     79037
       melee       0.99      0.92      0.95     16646
   playercam       0.85      0.96      0.90     27959
       comms       0.95      0.76      0.84     20214
      layout       0.95      0.43      0.60     20660
        menu       0.32      0.46      0.37      5570
        none       0.20      0.76      0.32      3470

    accuracy                           0.87    173556
   macro avg       0.75      0.75      0.71    173556
weighted avg       0.92      0.87      0.87    173556
```