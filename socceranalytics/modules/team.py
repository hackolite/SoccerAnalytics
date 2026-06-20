"""
socceranalytics/modules/team.py
─────────────────────────────────
Team assignment module (VideoLevelModule).

Stage 1 – ``TeamClusteringModule``:
  Global KMeans(2) on per-player averaged jersey colours → team 1 / 2.
  Mirrors the ``TrackletTeamClustering`` stage in sn-gamestate but operates
  on colour features rather than ReID embeddings (no heavy backbone needed).

Stage 2 – ``TeamSideLabelingModule``:
  Labels each team as 'left' or 'right' based on their mean pitch x-coordinate.
  Matches sn-gamestate's ``TrackletTeamSideLabeling`` stage.

Configuration keys (configs/modules/team.yaml):
  n_sample_frames – frames to sample for KMeans (default: 50)
  n_init          – KMeans n_init (default: 10)
"""
from __future__ import annotations

import os
import sys
from typing import Dict

import numpy as np
from sklearn.cluster import KMeans

_REPO_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _REPO_ROOT)

from ..state import GameState
from .base import VideoLevelModule


class TeamClusteringModule(VideoLevelModule):
    """Global KMeans team assignment (jersey colours).

    Improved over the original per-frame approach: one KMeans fit on
    50 sampled frames gives stable, noise-robust team assignments
    (same approach as sn-gamestate's tracklet-level clustering).
    """

    name = 'team_clustering'
    input_keys = ['players']
    output_keys = ['team', 'team_color']

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.n_sample_frames: int = int(cfg.get('n_sample_frames', 50))
        self._assigner = None

    def _get_assigner(self):
        if self._assigner is None:
            from team_assigner import TeamAssigner  # noqa: E402
            self._assigner = TeamAssigner()
        return self._assigner

    def process(self, state: GameState, video_frames: list) -> GameState:
        print("  [TeamClustering] Running global KMeans team assignment...")
        assigner = self._get_assigner()
        team_map = assigner.assign_teams_global(
            video_frames, state.players
        )

        for frame_num, player_track in enumerate(state.players):
            for player_id, track in player_track.items():
                team = team_map.get(
                    player_id,
                    assigner.get_player_team(
                        video_frames[frame_num], track['bbox'], player_id
                    ),
                )
                track['team'] = team
                track['team_color'] = assigner.team_colors.get(team, (128, 128, 128))

        print("  [TeamClustering] Done.")
        return state


class TeamSideLabelingModule(VideoLevelModule):
    """Label each team as 'left' or 'right' based on pitch position.

    Mirrors sn-gamestate's ``TrackletTeamSideLabeling``:
      - Compute mean real-world x-coordinate per team (all frames, all players)
      - Assign 'left' to the team with the smaller mean x, 'right' to the other
      - Goalkeepers are assigned based on their own x relative to pitch centre

    Requires ``position_transformed`` to be populated by CalibrationModule.
    """

    name = 'team_side'
    input_keys = ['team', 'position_transformed']
    output_keys = ['team_side']

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)

    def process(self, state: GameState, video_frames: list) -> GameState:
        # Gather mean x per team
        team_x: Dict[int, list] = {}
        for frame in state.players:
            for _, det in frame.items():
                team = det.get('team')
                pos = det.get('position_transformed')
                if team is None or pos is None:
                    continue
                team_x.setdefault(team, []).append(pos[0])

        if len(team_x) < 2:
            return state

        mean_x = {t: float(np.mean(xs)) for t, xs in team_x.items()}
        sorted_teams = sorted(mean_x, key=mean_x.__getitem__)
        side_map = {sorted_teams[0]: 'left', sorted_teams[1]: 'right'}

        # Compute overall pitch x range for goalkeeper assignment
        all_x = [x for xs in team_x.values() for x in xs]
        pitch_x_mid = float(np.mean(all_x)) if all_x else 0.0

        for frame in state.players:
            for _, det in frame.items():
                team = det.get('team')
                pos = det.get('position_transformed')
                if team is None:
                    continue
                if det.get('is_goalkeeper') and pos is not None:
                    # Goalkeeper: assign side by their own x position
                    det['team_side'] = 'left' if pos[0] < pitch_x_mid else 'right'
                else:
                    det['team_side'] = side_map.get(team)

        print(f"  [TeamSideLabeling] team {sorted_teams[0]} → left, "
              f"team {sorted_teams[1]} → right (mean x: {mean_x})")
        return state
