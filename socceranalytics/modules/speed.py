"""
socceranalytics/modules/speed.py
──────────────────────────────────
Speed and distance estimation module.

Wraps the existing ``SpeedAndDistance_Estimator``.

Configuration keys (configs/modules/speed.yaml):
  frame_window – number of frames per speed window (default: 5)
  frame_rate   – video frame rate used for time computation (default: 24)
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _REPO_ROOT)

from ..state import GameState
from .base import BaseModule


class SpeedDistanceModule(BaseModule):
    """Compute per-player speed (km/h) and cumulative distance (m)."""

    name = 'speed'
    input_keys = ['position_transformed']
    output_keys = ['speed', 'distance']

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self._estimator = None

    def _get_estimator(self):
        if self._estimator is None:
            from speed_and_distance_estimator import SpeedAndDistance_Estimator  # noqa: E402
            est = SpeedAndDistance_Estimator()
            est.frame_window = int(self.cfg.get('frame_window', 5))
            est.frame_rate = int(self.cfg.get('frame_rate', 24))
            self._estimator = est
        return self._estimator

    def process(self, state: GameState, video_frames: list) -> GameState:
        tracks = state.to_tracks()
        self._get_estimator().add_speed_and_distance_to_tracks(tracks)
        return state
