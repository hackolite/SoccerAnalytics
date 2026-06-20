"""
socceranalytics/modules/visualization.py
──────────────────────────────────────────
All frame-annotation drawing code consolidated in one module.

Includes:
  - Player ellipses + jersey-number / team-player-id labels
  - Ball triangle
  - Team-ball-control HUD (bottom-right)
  - Camera-movement HUD (top-left)
  - Speed & distance overlays per player
  - Minimap (top-down pitch view, bottom-left)
  - Player interaction graph (nearest-teammate lines)

Configuration keys (configs/modules/visualization.yaml):
  show_interaction_graph – draw nearest-teammate lines (default: true)
  show_minimap           – draw top-down minimap (default: true)
  show_camera_movement   – draw camera movement HUD (default: true)
  show_speed             – draw speed/distance per player (default: true)
  show_jersey_number     – show jersey numbers if available (default: true)
"""
from __future__ import annotations

import os
import sys
from typing import List

import numpy as np

_REPO_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _REPO_ROOT)

from ..state import GameState
from .base import BaseModule


class VisualizationModule(BaseModule):
    """Draw all visual overlays onto video frames."""

    name = 'visualization'
    input_keys = ['players', 'ball', 'camera_movement', 'team_ball_control']
    output_keys: list = []  # mutates frames in place

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.show_interaction_graph: bool = bool(cfg.get('show_interaction_graph', True))
        self.show_minimap: bool = bool(cfg.get('show_minimap', True))
        self.show_camera_movement: bool = bool(cfg.get('show_camera_movement', True))
        self.show_speed: bool = bool(cfg.get('show_speed', True))
        self.show_jersey_number: bool = bool(cfg.get('show_jersey_number', True))

    def process(
        self,
        state: GameState,
        video_frames: list,
        return_frames: bool = True,
    ) -> list:
        """Draw all overlays and return the annotated frame list."""
        # Import drawing helpers lazily to avoid circular-import issues
        from trackers import FootballBotSortTracker
        from trackers.tracker import Tracker
        from camera_movement_estimator import CameraMovementEstimator
        from speed_and_distance_estimator import SpeedAndDistance_Estimator
        from minimap import MiniMap

        tracks = state.to_tracks()
        team_ball_control = np.array(state.team_ball_control)

        # --- Player / ball / team-ball-control annotations ---
        import cv2
        from utils import get_center_of_bbox, get_bbox_width, get_foot_position

        output_frames: list = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks['players'][frame_num]
            ball_dict = tracks['ball'][frame_num]

            # Draw players
            for track_id, player in player_dict.items():
                color = player.get('team_color', (0, 0, 255))

                # Label: prefer jersey number, then team_player_id
                label = None
                if self.show_jersey_number:
                    label = player.get('jersey_number') or player.get('team_player_id')
                else:
                    label = player.get('team_player_id')

                frame = _draw_ellipse(frame, player['bbox'], color, label)
                if player.get('has_ball', False):
                    frame = _draw_triangle(frame, player['bbox'], (0, 0, 255))

            # Draw ball
            for _, ball in ball_dict.items():
                frame = _draw_triangle(frame, ball['bbox'], (0, 255, 0))

            # Team ball control HUD
            frame = _draw_team_ball_control(frame, frame_num, team_ball_control)

            output_frames.append(frame)

        # --- Interaction graph ---
        if self.show_interaction_graph:
            for frame_num, frame in enumerate(output_frames):
                _draw_player_interaction_graph(frame, tracks['players'][frame_num])

        # --- Camera movement HUD ---
        if self.show_camera_movement and state.camera_movement:
            dummy_first = video_frames[0] if video_frames else None
            if dummy_first is not None:
                est = CameraMovementEstimator(dummy_first)
                output_frames = est.draw_camera_movement(
                    output_frames, state.camera_movement
                )

        # --- Speed & distance overlay ---
        if self.show_speed:
            estimator = SpeedAndDistance_Estimator()
            output_frames = estimator.draw_speed_and_distance(output_frames, tracks)

        # --- Minimap ---
        if self.show_minimap:
            minimap = MiniMap()
            output_frames = minimap.draw_minimap(output_frames, tracks)

        return output_frames


# ──────────────────────────────────────────────────────────────────────────────
# Low-level drawing helpers (mirror tracker.py originals)
# ──────────────────────────────────────────────────────────────────────────────

def _draw_ellipse(frame, bbox, color, track_id=None):
    import cv2
    y2 = int(bbox[3])
    x1, y1, x2, _ = bbox
    x_center = int((x1 + x2) / 2)
    width = int(x2 - x1)

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
    if track_id is not None:
        rect_w, rect_h = 40, 20
        x1_r = x_center - rect_w // 2
        x2_r = x_center + rect_w // 2
        y1_r = (y2 - rect_h // 2) + 15
        y2_r = (y2 + rect_h // 2) + 15
        cv2.rectangle(frame, (x1_r, y1_r), (x2_r, y2_r), color, cv2.FILLED)
        x1_t = x1_r + 12
        if len(str(track_id)) > 2:
            x1_t -= 10
        cv2.putText(
            frame, f"{track_id}",
            (int(x1_t), int(y1_r + 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
        )
    return frame


def _draw_triangle(frame, bbox, color):
    import cv2
    import numpy as np
    y = int(bbox[1])
    x1, _, x2, _ = bbox
    x = int((x1 + x2) / 2)
    pts = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
    cv2.drawContours(frame, [pts], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [pts], 0, (0, 0, 0), 2)
    return frame


def _draw_team_ball_control(frame, frame_num, team_ball_control):
    import cv2
    overlay = frame.copy()
    cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    ctrl = team_ball_control[:frame_num + 1]
    t1 = np.sum(ctrl == 1)
    t2 = np.sum(ctrl == 2)
    total = t1 + t2
    p1 = t1 / total if total > 0 else 0
    p2 = t2 / total if total > 0 else 0

    cv2.putText(frame, f"Team 1 Ball Control: {p1*100:.2f}%",
                (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(frame, f"Team 2 Ball Control: {p2*100:.2f}%",
                (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    return frame


def _draw_player_interaction_graph(frame, player_dict: dict):
    import cv2
    team_players: dict = {}
    for pid, player in player_dict.items():
        team = player.get('team')
        tpid = player.get('team_player_id')
        if team is None or tpid is None:
            continue
        x1, _, x2, y2 = player['bbox']
        x_center = int((x1 + x2) / 2)
        color = player.get('team_color', (0, 0, 255))
        nearest = player.get('nearest_teammate_id')
        team_players.setdefault(team, []).append({
            'pid': pid, 'tpid': tpid,
            'pos': (x_center, int(y2)), 'color': color, 'nearest': nearest,
        })

    overlay = frame.copy()
    for players in team_players.values():
        if len(players) < 2:
            continue
        tpid_to_pos = {p['tpid']: p['pos'] for p in players}
        color = players[0]['color']
        drawn = set()
        for p in players:
            nearest = p['nearest']
            if nearest is None or nearest not in tpid_to_pos:
                continue
            pair = tuple(sorted((p['tpid'], nearest)))
            pt1 = tuple(map(int, p['pos']))
            pt2 = tuple(map(int, tpid_to_pos[nearest]))
            cv2.line(overlay, pt1, pt2, color, 2, lineType=cv2.LINE_AA)
            if pair not in drawn:
                mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                cv2.putText(overlay, str(nearest), mid,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(overlay, str(nearest), mid,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                drawn.add(pair)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
