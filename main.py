import os
import glob
import warnings

from utils import read_video, save_video
from trackers import FootballBotSortTracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from minimap import MiniMap

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')


def assign_team_player_ids(tracks):
    """Assign stable per-team IDs to each player and compute nearest teammate ID.

    Each team gets its own ID counter that resets from 1, with a letter prefix:
    team 1 → 'a1' … 'a10', team 2 → 'b1' … 'b10'.  The mapping is built in
    first-appearance order and kept consistent across all frames.  Two new keys
    are added to every player entry:
      - 'team_player_id'     : str 'a1'-'a10' or 'b1'-'b10'
      - 'nearest_teammate_id': str of the spatially closest teammate, or None

    Strict invariants enforced at every stage:
      - 'a'-prefixed IDs are only ever assigned to team-1 players.
      - 'b'-prefixed IDs are only ever assigned to team-2 players.
      - No two tracker IDs receive the same team_player_id unless all 10 IDs
        of a team have already been claimed (last-resort duplicate allowed only
        when the pool is fully exhausted).

    Team membership is determined by majority vote across all frames in which
    a player appears, making the assignment robust to per-frame noise from the
    colour-clustering team assigner.

    When the team pool is not yet exhausted, extra players inherit the ID of
    the nearest available same-team player (spatial distance from average
    position across all frames).  When every ID in the pool has been claimed,
    the extra player is assigned the ID of the spatially+temporally closest
    same-team player (duplicate allowed as last resort, strictly within the
    same team).
    """
    TEAM_PREFIX = {1: 'a', 2: 'b'}

    # --- Preliminary pass: collect team votes and full position history ---
    player_team_votes = {}  # {player_id: {team: frame_count}}
    position_history = {}   # {player_id: [pos, ...]}
    first_frame = {}        # {player_id: frame_index} for first-appearance ordering
    goalkeeper_ids = set()  # player IDs flagged as goalkeepers

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, player_info in player_track.items():
            if player_id not in first_frame:
                first_frame[player_id] = frame_num
            if player_info.get('is_goalkeeper', False):
                goalkeeper_ids.add(player_id)
            team = player_info.get('team')
            if team is not None:
                votes = player_team_votes.setdefault(player_id, {})
                votes[team] = votes.get(team, 0) + 1
            pos = player_info.get('position_adjusted') or player_info.get('position')
            if pos is not None:
                position_history.setdefault(player_id, []).append(pos)

    # Derive stable team from majority vote across all frames.
    # Using all frames (not just the first) makes the mapping robust to
    # per-frame mis-classifications by the colour-clustering team assigner.
    player_teams = {}
    for player_id, votes in player_team_votes.items():
        player_teams[player_id] = max(votes, key=votes.get)

    def avg_pos(player_id):
        """Return the average (x, y) position across all recorded frames."""
        hist = position_history.get(player_id)
        if not hist:
            return None
        return (sum(p[0] for p in hist) / len(hist),
                sum(p[1] for p in hist) / len(hist))

    # --- Assign goalkeeper IDs first ('0'), one per team ---
    # Goalkeepers are excluded from the outfield 10-player pool so they
    # never displace an outfield player ID.  All goalkeepers receive the
    # special ID '0' regardless of which team they belong to.
    team_player_id_map = {}
    gk_assigned_teams: set = set()  # tracks which teams already have a GK ID

    for player_id in sorted(goalkeeper_ids, key=lambda p: first_frame.get(p, 0)):
        team = player_teams.get(player_id)
        if team is None or team in gk_assigned_teams:
            continue
        team_player_id_map[player_id] = '0'
        gk_assigned_teams.add(team)

    # --- Assign fresh IDs to the first 10 unique outfield players per team ---
    team_counters = {1: 0, 2: 0}

    for player_id in sorted(first_frame, key=first_frame.__getitem__):
        if player_id in team_player_id_map:
            continue
        if player_id in goalkeeper_ids:
            continue
        team = player_teams.get(player_id)
        if team not in team_counters:
            continue
        if team_counters[team] >= 10:
            continue
        team_counters[team] += 1
        prefix = TEAM_PREFIX.get(team, str(team))
        team_player_id_map[player_id] = f"{prefix}{team_counters[team]}"

    def _expected_prefix(team):
        """Return the single letter that all IDs for *team* must start with."""
        return TEAM_PREFIX.get(team, '')

    # --- Recycle IDs for extra outfield players using average position history ---
    # Each existing ID may be recycled at most once (no-duplicate rule).
    # Safety: only IDs whose prefix matches the player's team are considered.
    # Goalkeeper IDs ('ag' / 'bg') are never recycled to outfield players.
    recycled_ids = set()

    for player_id in sorted(first_frame, key=first_frame.__getitem__):
        if player_id in team_player_id_map:
            continue
        if player_id in goalkeeper_ids:
            continue
        team = player_teams.get(player_id)
        if team is None:
            continue
        pos = avg_pos(player_id)
        if pos is None:
            continue

        expected_prefix = _expected_prefix(team)
        best_id = None
        min_dist = float('inf')
        for existing_pid, existing_tpid in team_player_id_map.items():
            # Skip goalkeeper IDs ('0') — they must not be recycled to outfield players.
            if existing_tpid == '0':
                continue
            # Strict team check — 'a' IDs with team-1 players, 'b' with team-2.
            if player_teams.get(existing_pid) != team:
                continue
            if not existing_tpid.startswith(expected_prefix):
                continue
            if existing_tpid in recycled_ids:
                continue
            existing_pos = avg_pos(existing_pid)
            if existing_pos is None:
                continue
            dist = ((pos[0] - existing_pos[0]) ** 2 + (pos[1] - existing_pos[1]) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                best_id = existing_tpid

        if best_id is not None:
            team_player_id_map[player_id] = best_id
            recycled_ids.add(best_id)

    # --- Fallback: players still without ID after recycling ---
    # Strict invariants:
    #   1. 'a'-prefixed IDs → team 1 only; 'b'-prefixed IDs → team 2 only.
    #      Cross-team assignment is never performed; prefix is validated
    #      explicitly on every candidate before selection.
    #   2. Duplicates are avoided by tracking which IDs have already been
    #      claimed in this pass.  When the closest candidate's ID is already
    #      taken, the next-closest same-team candidate is tried instead.
    #   3. Only when *all* same-team IDs are exhausted is a duplicate allowed
    #      (last resort), still strictly within the same team.
    #   4. Players whose team is unknown are skipped — no ID is assigned.

    def _closest_from_history(player_id, restrict_team, excluded_ids=None):
        """Return the team_player_id of the historically closest eligible player.

        Combines Euclidean spatial distance (pixels, using average position)
        with temporal distance (frame-index difference) into a single score.
        Candidates whose team_player_id is in *excluded_ids* are skipped.
        Goalkeeper IDs (ending with 'g') are never returned as candidates.
        Prefix is validated so that only IDs matching *restrict_team*'s
        expected letter are considered (no cross-team contamination).
        Returns None when no eligible candidate exists in *team_player_id_map*.
        """
        pos = avg_pos(player_id)
        frame_idx = first_frame[player_id]
        expected_prefix = _expected_prefix(restrict_team) if restrict_team is not None else None
        best_id = None
        min_score = float('inf')
        for existing_pid, existing_tpid in team_player_id_map.items():
            # Never recycle goalkeeper IDs ('0') to outfield players.
            if existing_tpid == '0':
                continue
            if restrict_team is not None and player_teams.get(existing_pid) != restrict_team:
                continue
            # Explicit prefix guard — ensures 'a' IDs stay with team 1 and
            # 'b' IDs stay with team 2, even in the fallback pass.
            if expected_prefix and not existing_tpid.startswith(expected_prefix):
                continue
            if excluded_ids and existing_tpid in excluded_ids:
                continue
            existing_pos = avg_pos(existing_pid)
            existing_frame = first_frame.get(existing_pid, 0)
            if pos is not None and existing_pos is not None:
                spatial = ((pos[0] - existing_pos[0]) ** 2 +
                           (pos[1] - existing_pos[1]) ** 2) ** 0.5
            else:
                spatial = 0.0
            temporal = abs(frame_idx - existing_frame)
            score = spatial + temporal
            if score < min_score:
                min_score = score
                best_id = existing_tpid
        return best_id

    # Track which IDs have already been claimed in the fallback pass (per team).
    fallback_used_ids: dict[int, set] = {}

    for player_id in sorted(first_frame, key=first_frame.__getitem__):
        if player_id in team_player_id_map:
            continue
        if player_id in goalkeeper_ids:
            continue
        team = player_teams.get(player_id)
        if team is None:
            # Unknown team: skip — never assign a cross-team or arbitrary ID.
            continue

        used = fallback_used_ids.setdefault(team, set())

        # Try to find the closest same-team ID that is not yet claimed.
        best_id = _closest_from_history(player_id, restrict_team=team,
                                        excluded_ids=used)
        if best_id is None:
            # All same-team IDs are exhausted: allow a duplicate as last resort,
            # but still strictly within the same team (no cross-team assignment).
            best_id = _closest_from_history(player_id, restrict_team=team,
                                            excluded_ids=None)

        if best_id is not None:
            team_player_id_map[player_id] = best_id
            used.add(best_id)

    # --- Rigorous invariant checks ---
    # 1. Every tracked player with a known team MUST have a team_player_id.
    missing_ids = [pid for pid in first_frame
                   if pid not in team_player_id_map and player_teams.get(pid) is not None]
    if missing_ids:
        warnings.warn(
            f"assign_team_player_ids: {len(missing_ids)} player(s) still without "
            f"a team_player_id after all assignment passes "
            f"(first 10: {missing_ids[:10]}{'...' if len(missing_ids) > 10 else ''}). "
            "This should not happen — check team assignment and tracking data.",
            RuntimeWarning,
            stacklevel=2,
        )

    # 2. Prefix-team consistency: 'a' IDs must belong to team 1 only,
    #    'b' IDs must belong to team 2 only.  Goalkeeper IDs ('0') are
    #    exempt from the prefix check.  Any other violation is a bug.
    cross_team = [
        (pid, tpid, player_teams.get(pid))
        for pid, tpid in team_player_id_map.items()
        if player_teams.get(pid) is not None
        and tpid != '0'
        and not tpid.startswith(_expected_prefix(player_teams[pid]))
    ]
    if cross_team:
        warnings.warn(
            f"assign_team_player_ids: {len(cross_team)} cross-team ID assignment(s) "
            f"detected (first 5: {cross_team[:5]}). "
            "'a'-prefixed IDs should only appear for team-1 players and "
            "'b'-prefixed IDs for team-2 players.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Apply IDs and compute nearest teammate for every frame
    for frame_num, player_track in enumerate(tracks['players']):
        # Build per-team lookup: {team: [(player_id, tpid, pos), ...]}
        team_groups = {}
        for player_id, player_info in player_track.items():
            team = player_info.get('team')
            tpid = team_player_id_map.get(player_id)
            if team is None or tpid is None:
                continue
            pos = player_info.get('position_adjusted') or player_info.get('position')
            if pos is None:
                continue
            team_groups.setdefault(team, []).append((player_id, tpid, pos))

        for player_id, player_info in player_track.items():
            tpid = team_player_id_map.get(player_id)
            if tpid is None:
                continue
            tracks['players'][frame_num][player_id]['team_player_id'] = tpid

            team = player_info.get('team')
            pos = player_info.get('position_adjusted') or player_info.get('position')
            teammates = team_groups.get(team, [])

            nearest_tpid = None
            if pos is not None and len(teammates) >= 2:
                min_dist = float('inf')
                for tid, t_tpid, t_pos in teammates:
                    if tid == player_id:
                        continue
                    dist = ((pos[0] - t_pos[0]) ** 2 + (pos[1] - t_pos[1]) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        nearest_tpid = t_tpid

            tracks['players'][frame_num][player_id]['nearest_teammate_id'] = nearest_tpid


def process_video(video_path, tracker):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join('output_videos', os.path.basename(video_path))
    track_stub = os.path.join('stubs', f'{video_name}_track_stubs.pkl')
    camera_stub = os.path.join('stubs', f'{video_name}_camera_movement_stub.pkl')

    print(f"\n=== Processing: {video_path} ===")

    # Read Video
    print(f"  [1/8] Reading video frames from '{video_path}'...")
    video_frames = read_video(video_path)
    print(f"        -> {len(video_frames)} frames loaded.")

    print(f"  [2/8] Running object tracking...")
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path=track_stub)
    # Get object positions
    tracker.add_position_to_tracks(tracks)
    print(f"        -> Tracking complete.")

    # Camera movement estimator
    print(f"  [3/8] Estimating camera movement...")
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path=camera_stub,
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    print(f"        -> Camera movement estimated for {len(camera_movement_per_frame)} frames.")

    # View Transformer
    print(f"  [4/8] Applying view transformation...")
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    print(f"        -> View transformation applied.")

    # Interpolate Ball Positions
    print(f"  [5/8] Interpolating ball positions...")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    print(f"        -> Ball positions interpolated.")

    # Speed and distance estimator
    print(f"  [6/8] Computing speed and distance...")
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    print(f"        -> Speed and distance computed.")

    # Assign Player Teams
    print(f"  [7/8] Assigning player teams...")
    team_assigner = TeamAssigner()

    # Global KMeans(2) on per-player averaged jersey colors: collect colors
    # from 50 evenly-spaced frames, average per player, cluster → definitive
    # team map.  Goalkeepers are excluded from the KMeans fit and assigned
    # to a team via spatial proximity.  This is more robust than per-frame
    # prediction because transient detection artefacts (occlusions, bad crops)
    # are smoothed out by averaging.
    team_map = team_assigner.assign_teams_global(video_frames, tracks['players'])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            # Use global map; fall back to per-frame prediction for players
            # that were not seen during sampling (rare edge case).
            team = team_map.get(
                player_id,
                team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id),
            )
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    print(f"        -> Teams assigned.")

    # Assign per-team IDs (1-10) and nearest teammate ID
    assign_team_player_ids(tracks)

    # Assign Ball Acquisition
    print(f"  [8/8] Assigning ball possession per frame...")
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
    team_ball_control = np.array(team_ball_control)
    print(f"        -> Ball possession assigned for {len(team_ball_control)} frames.")

    # Draw output
    print(f"  [draw] Drawing annotations...")
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    print(f"         Drawing camera movement overlay...")
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    print(f"         Drawing speed and distance overlay...")
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)
    print(f"         Drawing minimap overlay...")
    minimap = MiniMap()
    output_video_frames = minimap.draw_minimap(output_video_frames, tracks)
    print(f"         All overlays drawn.")

    # Save video
    os.makedirs('output_videos', exist_ok=True)
    print(f"  [save] Saving output video to '{output_path}'...")
    save_video(output_video_frames, output_path)
    print(f"         -> Saved to: {output_path}")


def main():
    os.makedirs('stubs', exist_ok=True)

    video_files = sorted([
        f for f in glob.glob('input_videos/*')
        if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
    ])

    if not video_files:
        print("No video files found in 'input_videos/'. Supported formats:", VIDEO_EXTENSIONS)
        return

    print(f"Found {len(video_files)} video(s) to process: {video_files}")

    # Initialize tracker once and reuse across videos
    tracker = FootballBotSortTracker('models/soccer.onnx', config_path='botsort_football.yaml')

    for video_path in video_files:
        process_video(video_path, tracker)

    print("\nAll videos processed successfully.")


if __name__ == '__main__':
    main()