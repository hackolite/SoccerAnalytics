"""
socceranalytics/modules/ball.py
─────────────────────────────────
Ball interpolation + possession-assignment module.

Wraps the existing ``PlayerBallAssigner`` and ball-position interpolation
logic from the original ``Tracker`` class.

Configuration keys (configs/modules/ball.yaml):
  max_player_ball_distance – max pixel distance to consider a player as
                             holding the ball (default: 70)
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _REPO_ROOT)

from ..state import GameState
from .base import BaseModule


class BallModule(BaseModule):
    """Interpolate missing ball detections and assign ball possession."""

    name = 'ball'
    input_keys = ['ball', 'players']
    output_keys = ['team_ball_control', 'has_ball']

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self._assigner = None
        max_dist: int = int(cfg.get('max_player_ball_distance', 70))
        self._max_dist = max_dist

    def _get_assigner(self):
        if self._assigner is None:
            from player_ball_assigner import PlayerBallAssigner  # noqa: E402
            self._assigner = PlayerBallAssigner()
            self._assigner.max_player_ball_distance = self._max_dist
        return self._assigner

    @staticmethod
    def _interpolate_ball(ball_frames: list) -> list:
        """Fill in missing ball bboxes using pandas linear interpolation."""
        import pandas as pd
        raw = [x.get(1, {}).get('bbox', []) for x in ball_frames]
        df = pd.DataFrame(raw, columns=['x1', 'y1', 'x2', 'y2'])
        df = df.interpolate().bfill()
        return [{1: {'bbox': row}} for row in df.to_numpy().tolist()]

    def process(self, state: GameState, video_frames: list) -> GameState:
        # 1. Interpolate missing ball positions
        state.ball = self._interpolate_ball(state.ball)

        # 2. Assign ball possession per frame
        team_ball_control: list = []
        for frame_num, player_track in enumerate(state.players):
            if not state.ball[frame_num]:
                team_ball_control.append(
                    team_ball_control[-1] if team_ball_control else 0
                )
                continue
            ball_bbox = state.ball[frame_num][1]['bbox']
            assigned = self._get_assigner().assign_ball_to_player(
                player_track, ball_bbox
            )
            if assigned != -1:
                state.players[frame_num][assigned]['has_ball'] = True
                team_ball_control.append(
                    state.players[frame_num][assigned].get('team', 0)
                )
            else:
                team_ball_control.append(
                    team_ball_control[-1] if team_ball_control else 0
                )

        state.team_ball_control = team_ball_control
        return state
