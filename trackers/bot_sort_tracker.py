"""
trackers/bot_sort_tracker.py
────────────────────────────
Football player tracker that combines Ultralytics YOLO + BoT-SORT with:

  • ReID embedding history (MobileNetV2 backbone)
  • Per-player position / embedding history
  • Velocity-based association filter
  • Composite match_player() scoring function
  • ID-switch suspect detection and logging

Drop-in replacement for the original ``Tracker`` class: exposes the same
``get_object_tracks``, ``add_position_to_tracks``, ``interpolate_ball_positions``
and ``draw_annotations`` interface expected by ``main.py``.
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from ultralytics import YOLO

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# ReID Feature Extractor
# ──────────────────────────────────────────────────────────────────────────────
class ReIDExtractor:
    """Lightweight appearance feature extractor using MobileNetV2.

    Produces a 1280-dimensional L2-normalised embedding from a BGR image crop.
    The model runs in evaluation mode with gradients disabled for speed.
    """

    EMBED_DIM: int = 1280  # MobileNetV2 features channel count

    def __init__(self, device: Optional[str] = None) -> None:
        self.device: str = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        )
        # Use only the convolutional feature extractor (drop the classifier)
        self.features = backbone.features.eval().to(self.device)
        self._pool = torch.nn.AdaptiveAvgPool2d(1)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),          # standard ReID input size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def extract(self, crop: np.ndarray) -> np.ndarray:
        """Return a normalised embedding for a BGR image crop.

        Returns a zero vector when the crop is invalid.
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return np.zeros(self.EMBED_DIM, dtype=np.float32)

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        feat = self.features(tensor)                    # (1, 1280, h', w')
        feat = self._pool(feat).flatten().cpu().numpy()  # (1280,)
        norm = np.linalg.norm(feat)
        return feat / (norm + 1e-8)


# ──────────────────────────────────────────────────────────────────────────────
# Per-track history container
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TrackHistory:
    """Stores positional and appearance history for one player track."""

    #: Foot positions (x, y) for the last ~120 frames
    positions: deque = field(default_factory=lambda: deque(maxlen=120))
    #: Appearance embeddings for the last 30 frames
    embeddings: deque = field(default_factory=lambda: deque(maxlen=30))
    #: Frame index when this track was last observed
    last_seen: int = -1
    #: Most recent bounding box [x1, y1, x2, y2]
    bbox: Optional[List[float]] = None


# ──────────────────────────────────────────────────────────────────────────────
# Similarity helpers
# ──────────────────────────────────────────────────────────────────────────────
def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _bbox_iou(b1: List[float], b2: List[float]) -> float:
    """Intersection-over-Union for two [x1, y1, x2, y2] boxes."""
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (area1 + area2 - inter + 1e-8)


def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


# ──────────────────────────────────────────────────────────────────────────────
# Main tracker class
# ──────────────────────────────────────────────────────────────────────────────
class FootballBotSortTracker:
    """Football tracker: YOLO + BoT-SORT + ReID + history + velocity filter.

    Parameters
    ----------
    model_path:
        Path to the YOLO model weights (``*.pt`` or ``*.onnx``).
    config_path:
        Path to the BoT-SORT YAML tracker configuration file.
        Defaults to ``'botsort_football.yaml'``.
    reid_device:
        Torch device for the ReID model (``'cuda'`` / ``'cpu'``).
        Auto-selected when ``None``.
    """

    # Number of frames a lost track is retained
    MAX_LOST_FRAMES: int = 120
    # Rolling window for velocity estimation
    VEL_WINDOW: int = 10
    # A position jump greater than ``MAX_SPEED_FACTOR × avg_speed`` is flagged
    MAX_SPEED_FACTOR: float = 3.0
    # Hard cap on pixel displacement per frame (tribune camera, zoomed out)
    ABS_MAX_SPEED: float = 80.0
    # Number of recent embeddings averaged to form the appearance gallery
    GALLERY_SIZE: int = 10
    # Minimum composite score to re-identify a lost player instead of creating
    # a new stable ID.  Score formula: 0.6*appearance + 0.3*motion + 0.1*iou.
    REID_MATCH_THRESHOLD: float = 0.35
    # Hard cap: 11 players × 2 teams.  A 23rd stable ID is never minted;
    # detections beyond this limit are force-assigned to the best history match.
    MAX_STABLE_IDS: int = 22

    def __init__(
        self,
        model_path: str,
        config_path: str = 'botsort_football.yaml',
        reid_device: Optional[str] = None,
    ) -> None:
        self.model = YOLO(model_path, task='detect')
        self.config_path = config_path
        self.reid = ReIDExtractor(device=reid_device)

        # player_history[stable_id] -> TrackHistory
        self.player_history: Dict[int, TrackHistory] = {}
        # Chronological list of suspected ID-switch events
        self._id_switch_log: List[dict] = []

        # Maps raw BoT-SORT track IDs → stable per-video player IDs
        self._id_map: Dict[int, int] = {}
        # Counter for the next stable player ID to assign
        self._next_stable_id: int = 1

    # ── History helpers ───────────────────────────────────────────────────────

    def _get_or_create_history(self, track_id: int) -> TrackHistory:
        if track_id not in self.player_history:
            self.player_history[track_id] = TrackHistory()
        return self.player_history[track_id]

    def _mean_embedding(self, track_id: int) -> np.ndarray:
        """Return the mean of the last ``GALLERY_SIZE`` stored embeddings."""
        hist = self.player_history.get(track_id)
        if hist is None or len(hist.embeddings) == 0:
            return np.zeros(ReIDExtractor.EMBED_DIM, dtype=np.float32)
        gallery = list(hist.embeddings)[-self.GALLERY_SIZE:]
        mean = np.mean(gallery, axis=0)
        norm = np.linalg.norm(mean)
        return mean / (norm + 1e-8)

    # ── Velocity helpers ──────────────────────────────────────────────────────

    def _estimate_velocity(self, track_id: int) -> float:
        """Mean pixel-per-frame speed over the last ``VEL_WINDOW`` positions."""
        hist = self.player_history.get(track_id)
        if hist is None or len(hist.positions) < 2:
            return 0.0
        recent = list(hist.positions)[-self.VEL_WINDOW:]
        speeds = [
            float(np.sqrt(
                (recent[i][0] - recent[i - 1][0]) ** 2 +
                (recent[i][1] - recent[i - 1][1]) ** 2
            ))
            for i in range(1, len(recent))
        ]
        return float(np.mean(speeds)) if speeds else 0.0

    def _is_velocity_valid(
        self,
        track_id: int,
        new_position: Tuple[float, float],
    ) -> bool:
        """Return ``False`` when the jump to *new_position* is unrealistically large.

        The maximum allowed displacement per frame is::

            max(ABS_MAX_SPEED, avg_speed × MAX_SPEED_FACTOR)
        """
        hist = self.player_history.get(track_id)
        if hist is None or len(hist.positions) == 0:
            return True
        last = hist.positions[-1]
        dist = float(np.sqrt(
            (new_position[0] - last[0]) ** 2 +
            (new_position[1] - last[1]) ** 2
        ))
        avg_speed = self._estimate_velocity(track_id)
        max_allowed = max(self.ABS_MAX_SPEED, avg_speed * self.MAX_SPEED_FACTOR)
        return dist <= max_allowed

    # ── Crop extraction ───────────────────────────────────────────────────────

    def _extract_crop(
        self,
        frame: np.ndarray,
        bbox: List[float],
    ) -> np.ndarray:
        """Safely crop a player region from *frame* given a [x1,y1,x2,y2] box."""
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return frame[y1:y2, x1:x2]

    # ── match_player ──────────────────────────────────────────────────────────

    def match_player(
        self,
        detection_bbox: List[float],
        detection_embedding: np.ndarray,
        active_tracks: Dict[int, TrackHistory],
    ) -> Tuple[Optional[int], float]:
        """Find the best matching active track for a new detection.

        Scoring formula::

            score = 0.6 × appearance_similarity
                  + 0.3 × motion_similarity
                  + 0.1 × bbox_similarity

        When two tracks compete for the same detection the caller should keep
        the track with the higher score and place the other in a "lost" state.

        Parameters
        ----------
        detection_bbox:
            Bounding box of the new detection [x1, y1, x2, y2].
        detection_embedding:
            ReID embedding of the new detection (1280-dim, L2-normalised).
        active_tracks:
            Mapping ``{track_id: TrackHistory}`` of currently active tracks.

        Returns
        -------
        (best_track_id, best_score) — ``best_track_id`` is ``None`` when
        *active_tracks* is empty.
        """
        best_id: Optional[int] = None
        best_score: float = -np.inf
        det_center = _bbox_center(detection_bbox)

        for tid, hist in active_tracks.items():
            if hist.bbox is None:
                continue

            # 1. Appearance similarity (cosine distance in embedding space)
            gallery_emb = self._mean_embedding(tid)
            app_sim = _cosine_similarity(detection_embedding, gallery_emb)

            # 2. Motion similarity: penalise large spatial jumps
            avg_speed = self._estimate_velocity(tid)
            prev_center = _bbox_center(hist.bbox)
            dist = float(np.sqrt(
                (det_center[0] - prev_center[0]) ** 2 +
                (det_center[1] - prev_center[1]) ** 2
            ))
            ref_speed = (avg_speed * self.MAX_SPEED_FACTOR
                         if avg_speed > 0 else self.ABS_MAX_SPEED)
            motion_sim = max(0.0, 1.0 - dist / (ref_speed + 1e-8))

            # 3. Bounding-box overlap
            bbox_sim = _bbox_iou(detection_bbox, hist.bbox)

            score = (0.6 * app_sim +
                     0.3 * motion_sim +
                     0.1 * bbox_sim)

            if score > best_score:
                best_score = score
                best_id = tid

        return best_id, best_score

    # ── History update (called every frame) ──────────────────────────────────

    def _update_history(
        self,
        frame_id: int,
        frame: np.ndarray,
        player_tracks: Dict[int, dict],
    ) -> None:
        """Update ``player_history`` and log any suspected ID switches."""
        for track_id, info in player_tracks.items():
            hist = self._get_or_create_history(track_id)
            bbox = info['bbox']
            pos = get_foot_position(bbox)

            # ── Velocity check → log suspected ID switch ─────────────────
            if hist.last_seen >= 0 and not self._is_velocity_valid(track_id, pos):
                prev_pos = hist.positions[-1]
                dist = float(np.sqrt(
                    (pos[0] - prev_pos[0]) ** 2 +
                    (pos[1] - prev_pos[1]) ** 2
                ))
                crop = self._extract_crop(frame, bbox)
                current_emb = self.reid.extract(crop)
                app_sim = _cosine_similarity(current_emb, self._mean_embedding(track_id))

                entry = {
                    'frame': frame_id,
                    'track_id': track_id,
                    'gap_frames': frame_id - hist.last_seen,
                    'distance': round(dist, 1),
                    'appearance_sim': round(app_sim, 3),
                }
                self._id_switch_log.append(entry)
                logger.warning(
                    "⚠  ID switch suspect | track_id=%d | gap=%d frames "
                    "| distance=%.1f px | appearance_sim=%.3f",
                    track_id,
                    entry['gap_frames'],
                    dist,
                    app_sim,
                )

            # ── Extract embedding and store ───────────────────────────────
            crop = self._extract_crop(frame, bbox)
            emb = self.reid.extract(crop)
            hist.embeddings.append(emb)
            hist.positions.append(pos)
            hist.last_seen = frame_id
            hist.bbox = bbox

    # ── Detection / tracking ──────────────────────────────────────────────────

    def detect_frames(self, frames: List[np.ndarray]) -> list:
        """Run YOLO + BoT-SORT on every frame and return raw Ultralytics results."""
        results: list = []
        batch_size = 1
        for i in range(0, len(frames), batch_size):
            batch = self.model.track(
                frames[i:i + batch_size],
                conf=0.1,
                tracker=self.config_path,
                persist=True,
                verbose=False,
            )
            results.extend(batch)
            if (i // batch_size + 1) % 50 == 0:
                print(
                    f"    [detect_frames] {i + batch_size}/{len(frames)} "
                    "frames processed."
                )
        print(f"    [detect_frames] Done — {len(results)} frames detected.")
        return results

    def get_object_tracks(
        self,
        frames: List[np.ndarray],
        read_from_stub: bool = False,
        stub_path: Optional[str] = None,
    ) -> dict:
        """Run tracking and return a structured tracks dictionary.

        The returned dict has the shape::

            {
                "players":  [{track_id: {"bbox": [x1,y1,x2,y2]}, …}, …],
                "referees": [{track_id: {"bbox": …}, …}, …],
                "ball":     [{1: {"bbox": …}}, …],
            }

        Loads from *stub_path* when available and *read_from_stub* is ``True``.
        Saves a new stub after tracking when *stub_path* is provided.
        """
        if read_from_stub and stub_path and os.path.exists(stub_path):
            print(f"    [get_object_tracks] Loading tracks from stub: '{stub_path}'")
            with open(stub_path, 'rb') as fh:
                tracks = pickle.load(fh)
            print("    [get_object_tracks] Stub loaded.")
            return tracks

        # Reset all per-video tracking state so successive videos are independent
        self._id_map = {}
        self._next_stable_id = 1
        self.player_history = {}
        self._id_switch_log = []

        print(
            f"    [get_object_tracks] No stub — running BoT-SORT on "
            f"{len(frames)} frames …"
        )
        detections = self.detect_frames(frames)

        tracks: dict = {"players": [], "referees": [], "ball": []}

        for frame_num, det in enumerate(detections):
            cls_names: dict = det.names
            cls_names_inv: dict = {v: k for k, v in cls_names.items()}

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            if det.boxes is None:
                continue

            # Stable IDs claimed so far in this frame (prevents two raw IDs
            # from both re-identifying the same lost player).
            current_stable_ids: set = set()

            for box in det.boxes:
                bbox = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                cls_name = cls_names.get(cls_id, '')

                # Remap goalkeeper → player so goalkeepers are tracked uniformly
                if cls_name == 'goalkeeper':
                    cls_id = cls_names_inv.get('player', cls_id)
                    cls_name = 'player'

                raw_id = int(box.id[0]) if box.id is not None else -1

                if cls_name == 'player' and raw_id >= 0:
                    if raw_id in self._id_map:
                        stable_id = self._id_map[raw_id]
                        # Guard: if this stable_id was already claimed this frame
                        # by a different raw ID, try to find a free slot.  When no
                        # free slot is available (cap reached), skip this detection
                        # entirely to avoid assigning the same ID to two players in
                        # the same frame.
                        if stable_id in current_stable_ids:
                            if self._next_stable_id <= self.MAX_STABLE_IDS:
                                stable_id = self._next_stable_id
                                self._next_stable_id += 1
                                self._id_map[raw_id] = stable_id
                            else:
                                # No free slot — omit rather than duplicate
                                logger.warning(
                                    "ID cap reached, duplicate suppressed: "
                                    "raw=%d → stable=%d already in use this frame",
                                    raw_id, stable_id,
                                )
                                continue
                    else:
                        # New raw ID — try to re-identify a lost player from
                        # the appearance / position history.
                        lost_tracks = {
                            sid: hist
                            for sid, hist in self.player_history.items()
                            if sid not in current_stable_ids
                            and hist.bbox is not None
                        }

                        cap_reached = self._next_stable_id > self.MAX_STABLE_IDS

                        if lost_tracks:
                            crop = self._extract_crop(frames[frame_num], bbox)
                            emb = self.reid.extract(crop)
                            best_id, best_score = self.match_player(
                                bbox, emb, lost_tracks
                            )
                        else:
                            best_id, best_score = None, -1.0

                        if best_id is not None and (
                            best_score >= self.REID_MATCH_THRESHOLD or cap_reached
                        ):
                            stable_id = best_id
                            if cap_reached and best_score < self.REID_MATCH_THRESHOLD:
                                logger.warning(
                                    "ID cap reached — force re-id: "
                                    "raw=%d → stable=%d (score=%.3f)",
                                    raw_id, stable_id, best_score,
                                )
                            else:
                                logger.info(
                                    "Re-identified lost player: "
                                    "raw=%d → stable=%d (score=%.3f)",
                                    raw_id, stable_id, best_score,
                                )
                        elif cap_reached:
                            # No re-id candidate available and cap reached.
                            # Skip rather than duplicate an existing ID.
                            logger.warning(
                                "ID cap reached with no free candidate — "
                                "suppressing raw=%d",
                                raw_id,
                            )
                            continue
                        else:
                            stable_id = self._next_stable_id
                            self._next_stable_id += 1

                        self._id_map[raw_id] = stable_id

                    current_stable_ids.add(stable_id)
                    tracks["players"][frame_num][stable_id] = {"bbox": bbox}

                elif cls_name == 'referee' and raw_id >= 0:
                    tracks["referees"][frame_num][raw_id] = {"bbox": bbox}
                elif cls_name == 'ball':
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

            # Update ReID / position history using stable IDs
            self._update_history(
                frame_num,
                frames[frame_num],
                tracks["players"][frame_num],
            )

        if stub_path:
            print(f"    [get_object_tracks] Saving stub: '{stub_path}'")
            with open(stub_path, 'wb') as fh:
                pickle.dump(tracks, fh)
            print("    [get_object_tracks] Stub saved.")

        total_switches = len(self._id_switch_log)
        print(
            f"    [get_object_tracks] Tracking complete — "
            f"{total_switches} suspected ID switches logged."
        )
        return tracks

    # ── Position helper (same interface as Tracker) ───────────────────────────

    def add_position_to_tracks(self, tracks: dict) -> None:
        """Attach a ``'position'`` key to every track entry in-place."""
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    position = (
                        get_center_of_bbox(bbox)
                        if object_type == 'ball'
                        else get_foot_position(bbox)
                    )
                    tracks[object_type][frame_num][track_id]['position'] = position

    # ── Ball interpolation (same interface as Tracker) ────────────────────────

    def interpolate_ball_positions(self, ball_positions: list) -> list:
        """Fill missing ball detections by linear interpolation."""
        raw = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df = pd.DataFrame(raw, columns=['x1', 'y1', 'x2', 'y2'])
        df = df.interpolate().bfill()
        return [{1: {"bbox": row}} for row in df.to_numpy().tolist()]

    # ── ID-switch log accessor ────────────────────────────────────────────────

    def get_id_switch_log(self) -> List[dict]:
        """Return the list of all logged ID-switch suspect events."""
        return list(self._id_switch_log)

    # ── Drawing helpers (identical interface to Tracker) ─────────────────────

    def draw_ellipse(
        self,
        frame: np.ndarray,
        bbox: List[float],
        color: tuple,
        track_id: Optional[int] = None,
    ) -> np.ndarray:
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        rw, rh = 40, 20
        x1r = x_center - rw // 2
        x2r = x_center + rw // 2
        y1r = y2 - rh // 2 + 15
        y2r = y2 + rh // 2 + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1r), int(y1r)),
                (int(x2r), int(y2r)),
                color,
                cv2.FILLED,
            )
            x1t = x1r + 12 - (10 if len(str(track_id)) > 2 else 0)
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1t), int(y1r + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )
        return frame

    def draw_traingle(
        self,
        frame: np.ndarray,
        bbox: List[float],
        color: tuple,
    ) -> np.ndarray:
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        pts = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
        cv2.drawContours(frame, [pts], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [pts], 0, (0, 0, 0), 2)
        return frame

    def draw_player_interaction_graph(
        self,
        frame: np.ndarray,
        player_dict: dict,
    ) -> np.ndarray:
        """Draw semi-transparent proximity lines between nearest teammates."""
        team_players: dict = {}
        for _, player in player_dict.items():
            team = player.get('team')
            if team is None:
                continue
            if team not in team_players:
                team_players[team] = []
            x_center, _ = get_center_of_bbox(player['bbox'])
            y2 = int(player['bbox'][3])
            color = player.get('team_color', (0, 0, 255))
            team_players[team].append({'pos': (x_center, y2), 'color': color})

        overlay = frame.copy()
        k_neighbors = 2

        for players in team_players.values():
            if len(players) < 2:
                continue
            positions = np.array([p['pos'] for p in players], dtype=np.float32)
            color = players[0]['color']
            for i in range(len(players)):
                diffs = positions - positions[i]
                dists = np.sqrt((diffs ** 2).sum(axis=1))
                dists[i] = np.inf
                for j in np.argsort(dists)[:k_neighbors]:
                    cv2.line(
                        overlay,
                        tuple(map(int, positions[i])),
                        tuple(map(int, positions[j])),
                        color,
                        2,
                        cv2.LINE_AA,
                    )

        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        return frame

    def draw_team_ball_control(
        self,
        frame: np.ndarray,
        frame_num: int,
        team_ball_control: np.ndarray,
    ) -> np.ndarray:
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        ctrl = team_ball_control[:frame_num + 1]
        t1 = int((ctrl == 1).sum())
        t2 = int((ctrl == 2).sum())
        total = t1 + t2
        p1 = t1 / total if total > 0 else 0.0
        p2 = t2 / total if total > 0 else 0.0

        cv2.putText(
            frame,
            f"Team 1 Ball Control: {p1 * 100:.2f}%",
            (1400, 900),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            frame,
            f"Team 2 Ball Control: {p2 * 100:.2f}%",
            (1400, 950),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3,
        )
        return frame

    def draw_annotations(
        self,
        video_frames: List[np.ndarray],
        tracks: dict,
        team_ball_control: np.ndarray,
    ) -> List[np.ndarray]:
        """Render all per-frame annotations and return the output frame list."""
        output: List[np.ndarray] = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Interaction graph drawn first so ellipses sit on top
            frame = self.draw_player_interaction_graph(frame, player_dict)

            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                display_id = player.get('team_player_id', track_id)
                frame = self.draw_ellipse(frame, player["bbox"], color, display_id)
                if player.get('has_ball', False):
                    frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            for _, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            output.append(frame)

        return output
