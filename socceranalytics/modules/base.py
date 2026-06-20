"""
socceranalytics/modules/base.py
────────────────────────────────
Abstract base class for all pipeline modules, mirroring sn-gamestate's
TrackLab module hierarchy.

Each module declares:
  - ``name``            – unique identifier used in the YAML pipeline list
  - ``input_keys``      – GameState fields this module reads
  - ``output_keys``     – GameState fields this module writes
  - ``process(state)``  – the single entry-point called by Pipeline

Three concrete base classes follow the sn-gamestate pattern:
  - ``ImageLevelModule``    – operates frame-by-frame (detection, calibration)
  - ``DetectionLevelModule``– operates on individual bounding-box crops (OCR, ReID)
  - ``VideoLevelModule``    – operates on the full video at once (team clustering)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..state import GameState


class BaseModule(ABC):
    """Abstract base for all pipeline stages."""

    name: str = ''
    input_keys: List[str] = []
    output_keys: List[str] = []

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    @abstractmethod
    def process(self, state: GameState, video_frames: list) -> GameState:
        """Run this module and update *state* in place.

        Parameters
        ----------
        state:
            The shared :class:`~socceranalytics.state.GameState`.
        video_frames:
            Raw BGR video frames as a list of ``np.ndarray``.

        Returns
        -------
        The same *state* object (mutated in place and returned for chaining).
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class ImageLevelModule(BaseModule):
    """Processes each video frame independently (detection, calibration…)."""

    @abstractmethod
    def process_frame(self, frame, frame_num: int, state: GameState) -> None:
        """Update *state* with data derived from a single *frame*."""

    def process(self, state: GameState, video_frames: list) -> GameState:
        for frame_num, frame in enumerate(video_frames):
            self.process_frame(frame, frame_num, state)
        return state


class DetectionLevelModule(BaseModule):
    """Processes individual bounding-box crops (OCR, ReID…)."""

    @abstractmethod
    def process_detection(self, crop, track_id: int, frame_num: int,
                          state: GameState) -> None:
        """Update *state* for a single player crop."""

    def process(self, state: GameState, video_frames: list) -> GameState:
        for frame_num, frame in enumerate(video_frames):
            for track_id, det in list(state.players[frame_num].items()):
                bbox = det.get('bbox', [])
                if len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                # Guard against out-of-bounds crops
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2]
                self.process_detection(crop, track_id, frame_num, state)
        return state


class VideoLevelModule(BaseModule):
    """Processes the full video at once (team clustering, tracklet voting…)."""

    @abstractmethod
    def process(self, state: GameState, video_frames: list) -> GameState:
        """Update *state* using all frames at once."""
