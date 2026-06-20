"""
socceranalytics/modules/analytics.py
──────────────────────────────────────
Advanced tactical-analytics modules for the sn-gamestate pipeline.

Modules
-------
FormationAnalyticsModule
    Detects the outfield formation of each team per frame by binning
    player real-world x-positions into depth lines (defenders /
    midfielders / forwards) and producing a canonical label such as
    "4-3-3" or "4-4-2".  The label is stored on every outfield player
    dict under the key ``'formation'``.

PressureIndexModule
    Computes a per-player *pressure index* — the number of opponents
    within ``pressure_radius`` metres (default 5 m) — and a team-level
    *compactness* score (inverse of the standard deviation of player
    positions).  Results are stored on every player dict under the keys
    ``'pressure_index'`` and ``'team_compactness'``.

Both modules extend ``VideoLevelModule`` and follow the sn-gamestate
BaseModule interface: they read from ``state.players`` and write new
keys back to the same dicts, leaving all other state untouched.

Configuration keys
------------------
FormationAnalyticsModule:
  n_lines         – number of depth lines to split players into (default: 3)
  min_players     – minimum outfield players required per team to
                    compute a formation label (default: 5)

PressureIndexModule:
  pressure_radius – opponent proximity threshold in metres (default: 5.0)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..state import GameState
from .base import VideoLevelModule


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _split_into_lines(x_values: List[float], n_lines: int) -> List[int]:
    """Bin a sorted list of x-values into *n_lines* depth lines.

    Uses equal-width bins spanning the full x-range of the group.
    Returns the count of players in each bin (deepest first).

    Parameters
    ----------
    x_values:
        Real-world x-coordinates of all outfield players on one team in
        one frame, in any order.
    n_lines:
        Number of lines to split into (e.g. 3 for def/mid/fwd).

    Returns
    -------
    list of int
        Player counts per bin, ordered from smallest x to largest x.
    """
    if not x_values:
        return [0] * n_lines
    xs = sorted(x_values)
    x_min, x_max = xs[0], xs[-1]
    span = x_max - x_min or 1.0
    bin_width = span / n_lines
    counts = [0] * n_lines
    for x in xs:
        idx = min(int((x - x_min) / bin_width), n_lines - 1)
        counts[idx] += 1
    return counts


def _formation_label(counts: List[int]) -> str:
    """Convert a list of per-line player counts to a formation string.

    The goalkeeper (already excluded from *counts*) is not shown.
    Leading/trailing zeros are stripped.

    Examples
    --------
    >>> _formation_label([4, 3, 3])
    '4-3-3'
    >>> _formation_label([0, 4, 4, 2])
    '4-4-2'
    """
    # Strip leading zeros (artefacts of sparse frames)
    while counts and counts[0] == 0:
        counts = counts[1:]
    while counts and counts[-1] == 0:
        counts = counts[:-1]
    return '-'.join(str(c) for c in counts) if counts else '?'


# ──────────────────────────────────────────────────────────────────────────────
# Formation detection
# ──────────────────────────────────────────────────────────────────────────────

class FormationAnalyticsModule(VideoLevelModule):
    """Detect and annotate the tactical formation of each team per frame.

    Requires ``position_transformed`` (real-world coordinates in metres,
    produced by ``CalibrationModule``) to be present in the player dicts.
    Frames where fewer than ``min_players`` outfield players of a team are
    visible receive a ``'formation'`` value of ``'?'``.

    The computed label is written to every visible outfield player dict
    under the key ``'formation'``.
    """

    name = 'formation'
    input_keys = ['position_transformed', 'team']
    output_keys = ['formation']

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.n_lines: int = int(cfg.get('n_lines', 3))
        self.min_players: int = int(cfg.get('min_players', 5))

    def process(self, state: GameState, video_frames: list) -> GameState:
        print("  [FormationAnalytics] Computing per-frame formation labels…")
        n_labeled = 0

        for frame_num, player_frame in enumerate(state.players):
            # Collect outfield player positions per team
            team_positions: Dict[int, List[Tuple[int, float]]] = {}
            for track_id, det in player_frame.items():
                if det.get('is_goalkeeper', False):
                    continue
                team = det.get('team')
                pos = det.get('position_transformed')
                if team is None or pos is None:
                    continue
                team_positions.setdefault(team, []).append((track_id, float(pos[0])))

            # Compute formation per team and annotate each player
            team_labels: Dict[int, str] = {}
            for team, id_x_pairs in team_positions.items():
                if len(id_x_pairs) < self.min_players:
                    team_labels[team] = '?'
                    continue
                x_values = [x for _, x in id_x_pairs]
                counts = _split_into_lines(x_values, self.n_lines)
                team_labels[team] = _formation_label(counts)

            for track_id, det in player_frame.items():
                team = det.get('team')
                if team in team_labels:
                    det['formation'] = team_labels[team]
                    n_labeled += 1

        print(f"  [FormationAnalytics] Done — {n_labeled} player-frame labels written.")
        return state


# ──────────────────────────────────────────────────────────────────────────────
# Pressure index
# ──────────────────────────────────────────────────────────────────────────────

class PressureIndexModule(VideoLevelModule):
    """Compute per-player pressure index and per-team compactness score.

    For every frame, for every player *p*:

    * ``pressure_index`` (int): number of *opponents* whose
      ``position_transformed`` is within ``pressure_radius`` metres of *p*.
      A high value means *p* is surrounded by many opponents — useful for
      identifying pressing situations and high-pressure zones.

    * ``team_compactness`` (float): inverse of the mean pairwise distance
      between all visible outfield players of the same team.  High value →
      compact / well-organised defensive block; low value → spread out /
      attacking shape.

    Requires ``position_transformed`` (metres) from ``CalibrationModule``.
    Players without a valid ``position_transformed`` are skipped.
    """

    name = 'pressure'
    input_keys = ['position_transformed', 'team']
    output_keys = ['pressure_index', 'team_compactness']

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.pressure_radius: float = float(cfg.get('pressure_radius', 5.0))

    def process(self, state: GameState, video_frames: list) -> GameState:
        print("  [PressureIndex] Computing pressure and compactness per frame…")

        for player_frame in state.players:
            # Group positions by team
            team_positions: Dict[int, List[Tuple[int, float, float]]] = {}
            for track_id, det in player_frame.items():
                pos = det.get('position_transformed')
                team = det.get('team')
                if pos is None or team is None:
                    continue
                team_positions.setdefault(team, []).append(
                    (track_id, float(pos[0]), float(pos[1]))
                )

            if not team_positions:
                continue

            all_teams = list(team_positions.keys())

            # ── Pressure index ────────────────────────────────────────────
            r2 = self.pressure_radius ** 2
            for team, own_players in team_positions.items():
                opponents = [
                    (ox, oy)
                    for t, opp_list in team_positions.items()
                    if t != team
                    for _, ox, oy in opp_list
                ]
                for track_id, px, py in own_players:
                    count = sum(
                        1 for ox, oy in opponents
                        if (px - ox) ** 2 + (py - oy) ** 2 <= r2
                    )
                    if track_id in player_frame:
                        player_frame[track_id]['pressure_index'] = count

            # ── Team compactness ─────────────────────────────────────────
            for team, own_players in team_positions.items():
                # Exclude goalkeepers from compactness (they're always spread out)
                outfield = [
                    (x, y) for tid, x, y in own_players
                    if not player_frame.get(tid, {}).get('is_goalkeeper', False)
                ]
                if len(outfield) < 2:
                    compactness = 0.0
                else:
                    xs = np.array([x for x, _ in outfield])
                    ys = np.array([y for _, y in outfield])
                    mean_pairwise = _mean_pairwise_dist(xs, ys)
                    compactness = 1.0 / (mean_pairwise + 1e-6)

                for tid, _, _ in own_players:
                    if tid in player_frame:
                        player_frame[tid]['team_compactness'] = round(compactness, 4)

        print("  [PressureIndex] Done.")
        return state


def _mean_pairwise_dist(xs: np.ndarray, ys: np.ndarray) -> float:
    """Mean Euclidean pairwise distance between N points (O(N²))."""
    n = len(xs)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += math.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
            count += 1
    return total / count if count else 0.0
