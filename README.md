# vod-assistant

DINOv2 -> Classifier pipeline that aims to automatically provide timestamps for efficient workflows for VODs.

## Dataset

All VODs have been sampled at 480p1fps and embedded using `dinov2-small`.

|VOD| Link | Manual Annotation |
|---|---|---|
| Genesis X3 - Top 64 SSBU          | https://www.youtube.com/watch?v=JFomLqNnixU  | 5/12 hours |
| Pound 2022 - SSBU/Melee Finals    | https://www.twitch.tv/videos/1465710386      | 3/12 hours |
| Smashcon 2023 - SSBU/Melee Top 8  | https://www.twitch.tv/videos/1897725006       | 3/13 hours |

## Results
WIP

```
Mean acc:      0.964 ± 0.036
Mean macro-F1: 0.930 ± 0.078

              precision    recall  f1-score   support

        game       0.98      1.00      0.99     33703
      camera       0.85      0.98      0.91     17830
        none       0.98      0.63      0.76      9467

    accuracy                           0.94     61000
   macro avg       0.94      0.87      0.89     61000
weighted avg       0.94      0.94      0.93     61000
```