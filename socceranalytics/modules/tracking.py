"""
socceranalytics/modules/tracking.py
────────────────────────────────────
Object detection + BoT-SORT tracking module.

Wraps the existing ``FootballBotSortTracker`` (which uses YOLO + BoT-SORT +
MobileNetV2 ReID) in the ``BaseModule`` interface.  Detection results populate
``state.players``, ``state.referees``, and ``state.ball``.

Configuration keys (from configs/modules/tracking.yaml):
  model_path      – path to the ONNX/PT model (default: models/soccer.onnx)
  config_path     – BoT-SORT YAML config (default: botsort_football.yaml)
  stub_path       – if set, cache/reload tracks from a pickle stub
  conf_threshold  – detection confidence (default: 0.1)
  detect_referees – also track referees (default: true)
  detect_gk       – also track goalkeepers (default: true)
"""
from __future__ import annotations

import os
import sys

# Ensure the repo root is on the path so we can import the existing tracker
_REPO_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _REPO_ROOT)

from ..state import GameState
from .base import BaseModule


class TrackingModule(BaseModule):
    """Detection + BoT-SORT tracking for players, referees, and ball."""

    name = 'tracking'
    input_keys: list = []
    output_keys = ['players', 'referees', 'ball']

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        # Tracker is instantiated lazily in process() so that the package
        # can be imported without ultralytics/torch installed.
        self._tracker = None
        self.stub_path: str | None = cfg.get('stub_path')
        self.detect_referees: bool = bool(cfg.get('detect_referees', True))
        self.detect_gk: bool = bool(cfg.get('detect_gk', True))

    def _get_tracker(self):
        if self._tracker is None:
            from trackers import FootballBotSortTracker  # noqa: E402
            model_path = self.cfg.get('model_path', 'models/soccer.onnx')
            config_path = self.cfg.get('config_path', 'botsort_football.yaml')
            self._tracker = FootballBotSortTracker(model_path, config_path=config_path)
        return self._tracker

    def process(self, state: GameState, video_frames: list) -> GameState:
        tracker = self._get_tracker()
        stub = self.stub_path
        tracks = tracker.get_object_tracks(
            video_frames,
            read_from_stub=stub is not None,
            stub_path=stub,
        )

        # Optionally drop goalkeeper / referee tracks
        if not self.detect_gk:
            for frame in tracks.get('players', []):
                for tid in list(frame.keys()):
                    if frame[tid].get('is_goalkeeper'):
                        del frame[tid]

        if not self.detect_referees:
            tracks['referees'] = [{} for _ in tracks.get('referees', [])]

        state.players = tracks.get('players', [])
        state.referees = tracks.get('referees', [])
        state.ball = tracks.get('ball', [])

        tracker.add_position_to_tracks(tracks)
        return state
