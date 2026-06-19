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
| **Player Interaction Graph** | Draws proximity lines between nearest teammates for each team, showing tactical structure in real time |
| **Minimap** | Top-down 2-D pitch overlay showing every player, referee and ball in real-world coordinates via homography |
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
├── minimap/
│   └── minimap.py                   # Top-down minimap overlay (homography)
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

### 2 — Create and activate a virtual environment

**Option A — venv (built-in, no extra install needed)**

```bash
# Create the environment
python -m venv .venv

# Activate — Linux / macOS
source .venv/bin/activate

# Activate — Windows (cmd.exe)
.venv\Scripts\activate.bat

# Activate — Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

**Option B — conda**

```bash
conda create -n socceranalytics python=3.10
conda activate socceranalytics
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Add your model and input video

1. Place your trained YOLOv8 model at `models/best.pt`.  
   *(See `training/football_training_yolo_v5.ipynb` to train your own, or use
   a pre-trained checkpoint from [Roboflow Universe](https://universe.roboflow.com/).)*

2. Place your input video at `input_videos/trimmed_live.mp4`  
   *(or update the path in `main.py`)*.

### 5 — Run the full analysis pipeline

```bash
python main.py
```

The annotated video will be saved to `output_videos/trimmed_live.mp4`.

### 6 — Run standalone video inference (raw YOLO only)

If you only want to run the YOLO detector on a video without the full
tracking/analytics pipeline, use `yolo_inference.py`:

```bash
python yolo_inference.py
```

By default it runs on `input_videos/08fd33_4.mp4`.  To use a different
source, edit the `model.predict(...)` call at the top of the file.

The script prints each detected bounding box and, because `save=True` is set,
Ultralytics also writes an annotated copy to `runs/detect/predict/`.

---

## Output Video

Running `main.py` produces a fully annotated video at
`output_videos/trimmed_live.mp4`.  Each frame contains:

| Annotation | Visual |
|---|---|
| **Player ground cursor** | Coloured ellipse at the feet of every player, coloured by team; player ID shown inside a small rectangle |
| **Ball triangle** | Green upward-pointing triangle above the ball |
| **Ball-holder indicator** | Red upward-pointing triangle above the player currently in possession |
| **Player interaction graph** | Semi-transparent lines connecting each player to their two nearest teammates, one graph per team rendered in the team's colour |
| **Ball possession overlay** | Bottom-right panel showing cumulative ball-control percentage for each team |
| **Camera movement overlay** | Top-left readout of estimated camera pan/tilt in pixels per frame |
| **Speed & distance overlay** | Per-player speed (km/h) and cumulative distance (m) rendered near each player |
| **Minimap** | Bottom-left corner top-down pitch view showing player/ball positions mapped via homography |

Example output frame (Team 1 = blue cursors / graph, Team 2 = red cursors / graph):

```
┌──────────────────────────────────────────────────────┐
│   [cam dx: 2  dy: -1]                                │
│                                                      │
│     ●━━━●    ▲(ball)                                 │
│    ╱     ╲                                           │
│   ●       ●━━●                                       │
│                                                      │
│   ○━━━○    ▲(ball-holder indicator)                  │
│    ╲  ╱                                              │
│     ○                                                │
│                                  ┌─────────────────┐ │
│                                  │Team 1 ctrl: 58% │ │
│                                  │Team 2 ctrl: 42% │ │
│                                  └─────────────────┘ │
└──────────────────────────────────────────────────────┘
  ● = Team 1 player (blue ellipse + graph)
  ○ = Team 2 player (red ellipse + graph)
  ▲ = triangle cursor
```

---

## Usage Details

### Stub caching

On the first run, tracking and camera-movement results are computed and cached
as pickle files in `stubs/`.  Subsequent runs load from the cache to save time.
To force recomputation, delete the `.pkl` files or set `read_from_stub=False`
in `main.py`.

### Standalone inference

See **Step 5** in [Getting Started](#getting-started) for the quick-start
command.  Additional detail: `yolo_inference.py` calls
`model.predict(..., save=True)`, which writes the annotated clip to
`runs/detect/predict/` via Ultralytics' built-in export, and also prints
every detected bounding box to stdout.

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
    └─► Minimap Overlay (homography → top-down 2-D pitch view)
    │
    ▼
Annotated Output Video
```

---

## Tracking Strategy — Detailed Documentation

This section explains in depth how the system detects, tracks, and identifies every object (players, referees, ball) from frame to frame.

---

### 1. Object Detection — YOLOv8

The first stage of the pipeline is a **fine-tuned YOLOv8** model (`models/soccer.onnx`) that runs on every frame to produce raw bounding boxes and class labels.

- **Classes detected:** `player`, `goalkeeper`, `referee`, `ball`.
- **Goalkeepers are remapped to `player`** at detection time so they are tracked exactly like outfield players.
- **Confidence threshold:** `0.1` (very permissive) to maximise recall; higher-confidence gates are applied downstream by the tracker.
- Detection runs **frame-by-frame** (batch size 1) via `model.track(…, persist=True)`, which lets Ultralytics keep Kalman-filter state between calls.

---

### 2. Multi-Object Tracking — BoT-SORT

After detection, raw bounding boxes are handed to the **BoT-SORT** tracker (configured in `botsort_football.yaml`).

BoT-SORT performs **two association passes** each frame using the Hungarian algorithm:

| Pass | Candidate pool | Gate |
|---|---|---|
| **First (high-conf)** | Detections with score ≥ `track_high_thresh` (0.5) | `match_thresh` 0.85 |
| **Second (rescue)** | Remaining unmatched detections with score ≥ `track_low_thresh` (0.1) | `proximity_thresh` 0.5 (IoU) |

Key configuration knobs:

| Parameter | Value | Purpose |
|---|---|---|
| `track_buffer` | 120 frames (~4 s at 30 fps) | How long a "lost" track is kept alive before being permanently deleted |
| `new_track_thresh` | 0.6 | Minimum confidence to **create** a brand-new track |
| `match_thresh` | 0.85 | Maximum Hungarian assignment cost (higher = more permissive) |
| `appearance_thresh` | 0.25 | Cosine distance gate for appearance embedding matching |
| `gmc_method` | `sparseOptFlow` | BoT-SORT's internal Global Motion Compensation — aligns predictions to camera movement before association |
| `with_reid` | `False` | BoT-SORT's built-in ReID is disabled; a custom ReID module is used instead (see §3) |

BoT-SORT outputs a **raw track ID** for each detection. These raw IDs can reset or jump after long occlusions, so the system adds its own stable-ID layer on top (see §4).

---

### 3. Appearance Re-Identification (ReID) — MobileNetV2

To survive long occlusions and camera cuts, the system maintains its own **ReID module** (`trackers/bot_sort_tracker.py : ReIDExtractor`).

**Backbone:** MobileNetV2 (ImageNet pre-trained), classification head removed. Only the convolutional feature extractor is kept.

**Embedding pipeline per detection:**
1. Crop the player region from the frame using the bounding box.
2. Resize to **128 × 64 px** (standard ReID input size).
3. Normalise with ImageNet mean/std.
4. Forward pass through `backbone.features` → adaptive average pooling → **1280-dimensional vector**.
5. L2-normalise the vector so cosine similarity equals dot product.

**Gallery:** The last 30 embeddings per track are stored in a rolling deque (`TrackHistory.embeddings`). The mean of the last 10 embeddings is used as the **gallery representative** when matching.

---

### 4. Stable ID Layer — Raw → Stable Mapping

BoT-SORT can reassign raw IDs after occlusions. To prevent this from appearing as a new player entering the scene, a **stable ID map** is maintained:

```
_id_map: {raw_botsort_id → stable_per_video_id}
```

When a **new raw ID** appears, the system:

1. Gathers all currently active `TrackHistory` entries as "lost" candidates.
2. Extracts a ReID embedding for the new detection.
3. Calls `match_player()` to compute a **composite score** against every candidate:

```
score = 0.6 × appearance_similarity   (cosine similarity in embedding space)
      + 0.3 × motion_similarity        (1 – normalised spatial distance)
      + 0.1 × bbox_similarity          (IoU of bounding boxes)
```

4. If the best score ≥ `REID_MATCH_THRESHOLD` (0.35), the new raw ID is mapped to the **existing stable ID** (re-identification).
5. Otherwise a fresh stable ID is minted (`_next_stable_id`).

**Hard cap:** The system never mints more than **22 stable IDs** (`MAX_STABLE_IDS = 22`, i.e. 11 players × 2 teams). When the cap is reached any new raw ID is force-assigned to the best available history match, and a warning is logged.

---

### 5. Velocity Filter and ID-Switch Detection

Every frame, for each active track, the system checks whether the new foot-position represents a **physically realistic displacement**:

```
max_allowed = max(ABS_MAX_SPEED, avg_speed × MAX_SPEED_FACTOR)
            = max(80 px/frame,   rolling_avg × 3.0)
```

If the displacement exceeds `max_allowed`, an **ID-switch suspect** entry is logged with:
- Frame index
- Track ID
- Gap since last observation (frames)
- Euclidean pixel distance of the jump
- Cosine appearance similarity between current and stored embeddings

This log is printed as warnings at runtime and allows post-hoc diagnosis of tracking quality.

---

### 6. Per-Team Stable IDs — `assign_team_player_ids()`

Once tracking is complete, a second ID assignment pass (`main.py : assign_team_player_ids`) maps the 22 stable IDs to **human-readable jersey-like IDs**:

- Team 1 players: `a1` … `a11`
- Team 2 players: `b1` … `b11`

**Three-pass algorithm:**

| Pass | Description |
|---|---|
| **Team vote** | For every player × every frame, count how many frames they were classified as team 1 vs team 2. The majority team wins (robust to per-frame colour-clustering noise). |
| **Primary assignment** | The first 11 unique players per team, ordered by their first appearance frame, receive IDs `a1`–`a11` or `b1`–`b11` in order. |
| **Recycling** | Extra players (beyond 11) inherit the ID of the spatially nearest same-team player whose ID has not yet been recycled. This handles brief ghost detections or tracking splits. |
| **Fallback** | Any player still without an ID is assigned the historically closest same-team ID using a **combined spatial + temporal score**: `score = euclidean_distance + frame_index_difference`. Prefix invariant is enforced: `a`-prefixed IDs are assigned only to team-1 players and `b`-prefixed IDs only to team-2 players. |

Strict invariants are checked after assignment:
- Every player with a known team must have a `team_player_id`.
- No `a`-prefixed ID may appear on a team-2 player (and vice versa). Violations trigger a `RuntimeWarning`.

The nearest-teammate ID (`nearest_teammate_id`) is also computed per frame as the Euclidean closest same-team player, used to draw the **interaction graph** overlay.

---

### 7. Camera Movement Compensation — Lucas-Kanade Optical Flow

The camera pans and tilts to follow the ball, so raw pixel positions are not comparable across frames. The `CameraMovementEstimator` removes this bias.

**Method:** Sparse Lucas-Kanade optical flow (`cv2.calcOpticalFlowPyrLK`).

1. On the **first frame**, good features to track are detected (`cv2.goodFeaturesToTrack`) in two fixed **border strips** (columns 0–20 and 900–1050). These regions correspond to the pitch sidelines and advertising hoardings — areas that move only when the camera moves, not when players run.
2. Each frame, the optical-flow displacement of those features is computed.
3. The **largest single-feature displacement** (dx, dy) is taken as the camera movement for that frame. Using the maximum rather than the mean makes the estimate robust to features that drift due to player shadows.
4. A minimum displacement threshold of **5 px** filters out sensor noise.
5. All player and ball foot positions are then corrected:

```
position_adjusted = (position_x − camera_dx, position_y − camera_dy)
```

The corrected positions are stored as `position_adjusted` in the tracks dictionary.

---

### 8. Perspective Transformation — Pixel → Real-World Metres

With camera-corrected pixel positions available, a **homography** maps 2-D image coordinates to top-down real-world metres (`view_transformer/view_transformer.py`).

**Four calibration points** (manually picked from the footage):

| Image pixel (x, y) | Real-world (m) |
|---|---|
| (110, 1035) | (0, 68) — bottom-left corner |
| (265, 275) | (0, 0) — top-left corner |
| (910, 260) | (23.32, 0) — top-right corner |
| (1640, 915) | (23.32, 68) — bottom-right corner |

`cv2.getPerspectiveTransform` computes the 3×3 homography matrix from these four correspondences. The transform covers a **23.32 m × 68 m** zone of the pitch (one side of the field).

Only positions **inside the calibration quadrilateral** (`cv2.pointPolygonTest ≥ 0`) are transformed. Points outside this region receive `position_transformed = None` and are excluded from speed/distance computations.

---

### 9. Speed and Distance Estimation

With real-world coordinates available, speed and distance are computed per player over a **sliding window**:

```
frame_window = 5 frames
frame_rate   = 24 fps  →  time_elapsed = 5 / 24 ≈ 0.208 s

distance_covered    = Euclidean(position_transformed[frame],
                                position_transformed[frame + 5])   [metres]
speed_m_s           = distance_covered / time_elapsed
speed_km_h          = speed_m_s × 3.6
cumulative_distance += distance_covered   (accumulated over the full video)
```

The same window-based method is applied to the **ball** to estimate ball speed and total ball distance. Speed and cumulative distance for each player are stored back in the tracks dictionary and displayed as an overlay on the output video.

---

### 10. Ball Interpolation

YOLO occasionally misses the ball for a few frames (motion blur, partial occlusion). Missing frames are filled by **linear interpolation** (`interpolate_ball_positions`):

1. Extract the bounding-box coordinates for all frames into a Pandas DataFrame.
2. Call `DataFrame.interpolate()` (linear by default).
3. Back-fill any leading `NaN` values with `bfill()`.

This produces a continuous ball trajectory with no missing frames.

---

### Tracking Data Flow Summary

```
Raw video frame
    │
    ▼ YOLOv8 (soccer.onnx, conf=0.1)
Raw bounding boxes + class labels
    │
    ▼ BoT-SORT (botsort_football.yaml)
Raw track IDs + bboxes  ←── Kalman filter prediction
    │
    ▼ Stable ID mapping (ReID gallery + composite score)
Stable player IDs 1–22
    │
    ▼ Velocity filter + ID-switch logging
Validated positions per stable ID
    │
    ▼ Team assignment (KMeans) + majority vote
team=1 or team=2 per player
    │
    ▼ assign_team_player_ids()  (3-pass algorithm)
team_player_id: a1–a11 / b1–b11
nearest_teammate_id per frame
    │
    ▼ Camera movement compensation (Lucas-Kanade)
position_adjusted (camera-corrected pixels)
    │
    ▼ Perspective transform (homography)
position_transformed (real-world metres)
    │
    ▼ Speed & distance estimation (5-frame window)
speed (km/h), cumulative_distance (m)
    │
    ▼ Annotation + minimap
Fully annotated output video
```

---

## Known Limitations

- The perspective transform vertices (`view_transformer.py`) are hard-coded for
  a specific camera angle and resolution. Adjust `pixel_vertices` for your own
  footage.  The minimap dimensions (`MiniMap.COURT_LENGTH` / `COURT_WIDTH`) must
  be kept in sync with `ViewTransformer` when you change these values.
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
