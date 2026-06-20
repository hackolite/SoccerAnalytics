"""
socceranalytics/modules/calibration.py
────────────────────────────────────────
Camera-calibration / view-transformation module.

Converts camera-compensated pixel positions to real-world pitch coordinates
(metres) using a homography estimated from four manually labelled pitch
corners (the existing hardcoded ``ViewTransformer``).

Future improvement: replace with an automatic keypoint-based calibration
backend (TVCalib / nbjw_calib style) as done in sn-gamestate.

Configuration keys (configs/modules/calibration.yaml):
  use_prev_homography – reuse last valid homography when transform fails
                        (matches sn-gamestate's temporal smoothing flag)
  pixel_vertices      – list of 4 [x, y] pixel corners (optional override)
  court_length        – pitch segment length in metres (default: 23.32)
  court_width         – pitch segment width in metres (default: 68.0)

Outputs:
  ``position_transformed`` on every detection: [x_m, y_m] in metres.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional

import cv2
import numpy as np

_REPO_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _REPO_ROOT)

from ..state import GameState
from .base import BaseModule


class CalibrationModule(BaseModule):
    """Perspective transform: pixel → real-world pitch coordinates (metres)."""

    name = 'calibration'
    input_keys = ['position_adjusted']
    output_keys = ['position_transformed']

    # Default hardcoded corners – same values used by the original ViewTransformer
    _DEFAULT_PIXEL_VERTICES = [[110, 1035], [265, 275], [910, 260], [1640, 915]]

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        court_length: float = float(cfg.get('court_length', 23.32))
        court_width: float = float(cfg.get('court_width', 68.0))
        self.use_prev_homography: bool = bool(cfg.get('use_prev_homography', True))
        self._last_valid_H: Optional[np.ndarray] = None

        pv_raw = cfg.get('pixel_vertices', self._DEFAULT_PIXEL_VERTICES)
        pixel_vertices = np.array(pv_raw, dtype=np.float32)
        target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width],
        ], dtype=np.float32)

        self._H = cv2.getPerspectiveTransform(pixel_vertices, target_vertices)
        self._pixel_vertices = pixel_vertices

    def _transform_point(self, point: tuple) -> Optional[List[float]]:
        """Map a single (x, y) pixel point to real-world metres.

        Returns ``None`` when the point lies outside the defined quadrilateral
        and ``use_prev_homography`` is False (strict mode).
        """
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self._pixel_vertices, p, False) >= 0
        if not is_inside:
            if self.use_prev_homography and self._last_valid_H is not None:
                H = self._last_valid_H
            else:
                return None
        else:
            H = self._H
            self._last_valid_H = H

        pt = np.array([[point]], dtype=np.float32)
        tp = cv2.perspectiveTransform(pt, H)
        return tp.reshape(-1, 2)[0].tolist()

    def process(self, state: GameState, video_frames: list) -> GameState:
        tracks = state.to_tracks()
        for obj_tracks in tracks.values():
            for frame in obj_tracks:
                for track_id, info in frame.items():
                    pos = info.get('position_adjusted')
                    if pos is None:
                        continue
                    pt = self._transform_point(np.array(pos, dtype=np.float64))
                    info['position_transformed'] = pt
        return state
