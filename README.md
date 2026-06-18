# SoccerAnalytics — Football Tracking with YOLOv8 & OpenCV

An end-to-end football match analysis system that detects and tracks players,
referees, and the ball, estimates camera movement, measures player speed and
distance, and assigns team possession — all from a single input video.

---

## Features

| Feature | Description |
|---|---|
| **Object Detection** | Detects players, referees, and the ball with a fine-tuned YOLOv8 model |
| **Multi-Object Tracking** | Maintains consistent player and referee IDs across frames via ByteTrack |
| **Team Assignment** | Separates the two teams using KMeans clustering on shirt colours |
| **Ball Possession** | Assigns the ball to the nearest player each frame and tracks team control |
| **Camera Movement** | Compensates for camera pan/tilt using Lucas-Kanade optical flow |
| **Perspective Transform** | Maps pixel coordinates to real-world metres using a homography |
| **Speed & Distance** | Computes each player's speed (km/h) and total distance covered (m) |
| **Annotated Output** | Produces a fully annotated output video with overlays for all metrics |

---

## Project Structure

```
SoccerAnalytics/
├── main.py                          # Entry point — orchestrates the full pipeline
├── yolo_inference.py                # Standalone YOLO inference helper
├── requirements.txt                 # Python dependencies
│
├── utils/                           # Shared utility functions
│   ├── video_utils.py               #   read_video / save_video
│   └── bbox_utils.py                #   bbox helpers, distance functions
│
├── trackers/
│   └── tracker.py                   # YOLO detection + ByteTrack + annotation
├── team_assigner/
│   └── team_assigner.py             # KMeans-based team colour clustering
├── player_ball_assigner/
│   └── player_ball_assigner.py      # Nearest-player ball possession
├── camera_movement_estimator/
│   └── camera_movement_estimator.py # Optical-flow camera motion estimation
├── view_transformer/
│   └── view_transformer.py          # Perspective transform to real-world coords
├── speed_and_distance_estimator/
│   └── speed_and_distance_estimator.py
│
├── models/                          # Put your trained model here (best.pt)
├── stubs/                           # Auto-generated pickle cache files
├── input_videos/                    # Source videos
├── output_videos/                   # Annotated output videos
└── training/                        # YOLOv8 training notebook & dataset
```

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- A CUDA-capable GPU is recommended but not required

### 1 — Clone the repository

```bash
git clone https://github.com/hackolite/SoccerAnalytics.git
cd SoccerAnalytics
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Add your model and input video

1. Place your trained YOLOv8 model at `models/best.pt`.  
   *(See `training/football_training_yolo_v5.ipynb` to train your own, or use
   a pre-trained checkpoint from [Roboflow Universe](https://universe.roboflow.com/).)*

2. Place your input video at `input_videos/trimmed_live.mp4`  
   *(or update the path in `main.py`)*.

### 4 — Run the pipeline

```bash
python main.py
```

The annotated video will be saved to `output_videos/trimmed_live.mp4`.

---

## Usage Details

### Stub caching

On the first run, tracking and camera-movement results are computed and cached
as pickle files in `stubs/`.  Subsequent runs load from the cache to save time.
To force recomputation, delete the `.pkl` files or set `read_from_stub=False`
in `main.py`.

### Standalone inference

```bash
python yolo_inference.py
```

Runs raw YOLOv8 inference on `input_videos/08fd33_4.mp4` and prints detected
bounding boxes.

### Training a custom model

Open `training/football_training_yolo_v5.ipynb` in Jupyter and follow the
instructions to fine-tune YOLOv8 on the bundled Roboflow dataset.

---

## Pipeline Overview

```
Input video
    │
    ▼
Object Detection (YOLOv8)
    │
    ▼
Multi-Object Tracking (ByteTrack)
    │
    ├─► Team Assignment (KMeans on shirt colour)
    │
    ├─► Ball Possession (nearest-player distance)
    │
    ├─► Camera Movement Estimation (Lucas-Kanade optical flow)
    │       └─► Adjusted positions (camera-compensated)
    │
    ├─► Perspective Transformation (homography → metres)
    │
    └─► Speed & Distance Estimation
    │
    ▼
Annotated Output Video
```

---

## Known Limitations

- The perspective transform vertices (`view_transformer.py`) are hard-coded for
  a specific camera angle and resolution. Adjust `pixel_vertices` for your own
  footage.
- Player ID `91` is hard-coded to team 1 in `team_assigner.py` — remove or
  generalise this if it does not apply to your video.
- The `draw_team_ball_control` overlay assumes a 1920 × 1080 output resolution.

---

## Contributing

Contributions are welcome! Please open an issue to discuss your idea before
submitting a pull request.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE)
file for details.

---

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [Roboflow](https://roboflow.com/) for dataset tooling
- [supervision](https://github.com/roboflow/supervision) for the ByteTrack wrapper
- [OpenCV](https://opencv.org/) for computer vision primitives
