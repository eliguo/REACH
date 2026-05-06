# mediapipe

Holistic landmark detection on infant–caregiver interaction videos using Google's [MediaPipe](https://developers.google.com/mediapipe). Extracts face mesh (468 landmarks), body pose, and hand keypoints from a single frame.

The notebook here was used to produce the MediaPipe Holistic example shown in the report (still-face experiment frame). It illustrates both what MediaPipe Holistic can do and where it falls short on infant subjects (oblique viewpoints, multi-person scenes), which motivated our choice of FiDIP for body pose and PyAFAR for facial AUs in the main pipeline.

## Setup

```bash
pip install mediapipe opencv-python
```

## Required downloads (not in repo)

**MediaPipe pretrained models** — download from the official MediaPipe model zoo and place under `models/`:

- [`face_landmarker_v2_with_blendshapes.task`](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
- [`hand_landmarker.task`](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- [`pose_landmarker.task`](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) (full variant)

**Input videos** — public YouTube videos used as demo inputs. Download with `yt-dlp` (or any tool you prefer) and place under `data/`. The exact clips used in the report:

- *Still face experiment* (Lise-Lotte Austad)
- *Motor Video — Get Up & Go*
- *Motor Video — Reach To Eat*

## Expected layout (after downloads)

```
mediapipe/
├── MediaPipe_Holistic_Video_Landmarker.ipynb
├── models/
│   ├── face_landmarker_v2_with_blendshapes.task
│   ├── hand_landmarker.task
│   └── pose_landmarker.task
├── data/                # input videos
└── outputs/             # generated annotated videos + landmark JSON
```

Edit the model and video paths at the top of the notebook before running.
