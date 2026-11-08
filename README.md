# vod-assistant

DINOv2 -> Classifier pipeline that aims to automatically provide timestamps for efficient workflows for VODs.

## Dataset

All VODs have been sampled at 480p1fps and embedded using `dinov2-small`.

|VOD| Link | 
|---|---|
| Genesis X3 - Top 64 SSBU          | https://www.youtube.com/watch?v=JFomLqNnixU  | 
| Pound 2022 - SSBU/Melee Finals    | https://www.twitch.tv/videos/1465710386      | 
| Smashcon 2023 - SSBU/Melee Top 8  | https://www.twitch.tv/videos/1897725006       |

## Results
WIP

```
Mean acc:      0.986 ± 0.005
Mean macro-F1: 0.900 ± 0.113

              precision    recall  f1-score   support

        ssbu       0.99      1.00      0.99     57817
       melee       0.98      0.99      0.98     16646
      camera       0.99      0.97      0.98     38946
        none       0.95      0.98      0.97     18646

    accuracy                           0.99    132055
   macro avg       0.98      0.98      0.98    132055
weighted avg       0.99      0.99      0.99    132055
```