"""
socceranalytics/modules/camera.py
──────────────────────────────────
Camera-movement estimation module.

Wraps the existing Lucas-Kanade optical-flow estimator in the BaseModule
interface.  Adds ``position_adjusted`` to every tracked object and stores
the per-frame ``[dx, dy]`` displacement in ``state.camera_movement``.

Configuration keys (configs/modules/camera.yaml):
  stub_path         – cache file path (optional)
  minimum_distance  – minimum pixel displacement to register as camera move
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _REPO_ROOT)

from ..state import GameState
from .base import BaseModule


class CameraMovementModule(BaseModule):
    """Estimate inter-frame camera motion and compensate all positions."""

    name = 'camera'
    input_keys = ['players', 'referees', 'ball']
    output_keys = ['camera_movement', 'position_adjusted']

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.stub_path: str | None = cfg.get('stub_path')

    def process(self, state: GameState, video_frames: list) -> GameState:
        from camera_movement_estimator import CameraMovementEstimator  # noqa: E402
        estimator = CameraMovementEstimator(video_frames[0])
        camera_movement = estimator.get_camera_movement(
            video_frames,
            read_from_stub=self.stub_path is not None,
            stub_path=self.stub_path,
        )
        tracks = state.to_tracks()
        estimator.add_adjust_positions_to_tracks(tracks, camera_movement)
        state.camera_movement = camera_movement
        return state
