"""
socceranalytics/state.py
────────────────────────
Central data-bus that flows through every pipeline module.

Inspired by sn-gamestate's ``TrackerState``:  all per-frame detection data
lives in a single ``GameState`` object so that each module reads what it
needs and writes its outputs back without ad-hoc dict passing.

Internal storage stays dict-based (backward-compatible with the original
``tracks`` format) but is wrapped behind a typed interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Per-detection record (one slot per tracked object per frame)
# ──────────────────────────────────────────────────────────────────────────────
DETECTION_DEFAULTS: dict = dict(
    bbox=[],
    role='player',           # 'player' | 'goalkeeper' | 'referee' | 'ball'
    team=None,               # 1 | 2
    team_side=None,          # 'left' | 'right'
    jersey_number=None,      # str  e.g. '10'
    position=None,           # (x, y) pixel – foot / centre
    position_adjusted=None,  # camera-compensated pixel
    position_transformed=None,  # real-world metres (homography)
    speed=None,              # km/h
    distance=None,           # cumulative metres
    team_player_id=None,     # 'a1'–'a11' / 'b1'–'b11'
    nearest_teammate_id=None,
    has_ball=False,
    team_color=None,
    is_goalkeeper=False,
)


# ──────────────────────────────────────────────────────────────────────────────
# GameState – the central data bus
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class GameState:
    """All tracking data for one processed video clip.

    Attributes
    ----------
    players, referees, ball:
        Per-frame detection lists – same layout as the legacy ``tracks`` dict.
        Each element is ``{track_id: {field: value, ...}}``.
    camera_movement:
        ``[[dx, dy], ...]`` one entry per video frame.
    team_ball_control:
        ``[team_id, ...]`` – which team holds the ball on each frame (1, 2 or 0).
    video_path:
        Source video path (informational).
    """

    players: List[Dict[int, dict]] = field(default_factory=list)
    referees: List[Dict[int, dict]] = field(default_factory=list)
    ball: List[Dict[int, dict]] = field(default_factory=list)

    camera_movement: List[List[float]] = field(default_factory=list)
    team_ball_control: List[int] = field(default_factory=list)

    video_path: str = ''

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def to_tracks(self) -> dict:
        """Return the legacy ``tracks`` dict expected by drawing utilities."""
        return {
            'players': self.players,
            'referees': self.referees,
            'ball': self.ball,
        }

    @classmethod
    def from_tracks(cls, tracks: dict, video_path: str = '') -> 'GameState':
        """Wrap a legacy tracks dict in a GameState."""
        return cls(
            players=tracks.get('players', []),
            referees=tracks.get('referees', []),
            ball=tracks.get('ball', []),
            video_path=video_path,
        )

    @property
    def n_frames(self) -> int:
        return len(self.players)

    def iter_player_frames(self):
        """Yield ``(frame_num, {track_id: det_dict})`` for every frame."""
        yield from enumerate(self.players)

    def get_player(self, frame_num: int, track_id: int) -> Optional[dict]:
        if frame_num < len(self.players):
            return self.players[frame_num].get(track_id)
        return None

    def set_player_attr(self, frame_num: int, track_id: int, **kwargs) -> None:
        """Write one or more attributes into a player's dict."""
        if frame_num < len(self.players) and track_id in self.players[frame_num]:
            self.players[frame_num][track_id].update(kwargs)

    def all_player_ids(self) -> set:
        """Return every unique player track_id seen across all frames."""
        ids: set = set()
        for frame in self.players:
            ids.update(frame.keys())
        return ids
