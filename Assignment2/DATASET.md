# Dataset Card

The project uses a self-collected five-class static hand-gesture dataset with three participants.

Classes: `fist`, `open_palm`, `peace`, `point_up`, `thumbs_up`.

## Current split

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| fist | 76 | 19 | 5 | 100 |
| open_palm | 60 | 15 | 26 | 101 |
| peace | 54 | 14 | 35 | 103 |
| point_up | 76 | 19 | 9 | 104 |
| thumbs_up | 61 | 15 | 26 | 102 |
| **Total** | **327** | **82** | **101** | **510** |

`person_003` is held out for final cross-person testing.

Raw participant images are not stored in this public repository for privacy reasons. The notebook expects the private dataset ZIP at `MyDrive/CV_project/hand_gestures.zip`.
