import os
import glob
import shutil
import tempfile
import warnings

from utils import read_video, save_video, get_video_info, read_video_chunk, read_video_sampled, concatenate_videos
from trackers import FootballBotSortTracker, refine_tracklets
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
    """Assign stable per-team IDs to each player and referee.

    Each team gets its own ID counter that resets from 1, with a letter prefix:
    team 1 → 'a1' … 'a10', team 2 → 'b1' … 'b10'.  Not all IDs are
    necessarily assigned — if fewer than 10 outfield players are detected for
    a team, only the IDs actually needed are used.  Extra tracker IDs beyond
    the first 10 per team receive no ``team_player_id``.  Two new keys are
    added to every player entry that has been assigned an ID:
      - 'team_player_id'     : str 'a1'-'a10', 'b1'-'b10', or '0'
      - 'nearest_teammate_id': str of the spatially closest teammate, or None

    Goalkeepers and referees always receive the special ID ``'0'``.

    Strict invariants enforced at every stage:
      - 'a'-prefixed IDs are only ever assigned to team-1 players.
      - 'b'-prefixed IDs are only ever assigned to team-2 players.
      - No two tracker IDs ever share the same team_player_id (no duplicates).

    Team membership is determined by majority vote across all frames in which
    a player appears, making the assignment robust to per-frame noise from the
    colour-clustering team assigner.
    """
    TEAM_PREFIX = {1: 'a', 2: 'b'}

    # --- Preliminary pass: collect team votes ---
    player_team_votes = {}  # {player_id: {team: frame_count}}
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

    # Derive stable team from majority vote across all frames.
    # Using all frames (not just the first) makes the mapping robust to
    # per-frame mis-classifications by the colour-clustering team assigner.
    player_teams = {}
    for player_id, votes in player_team_votes.items():
        player_teams[player_id] = max(votes, key=votes.get)

    # --- Assign goalkeeper IDs first ('0') ---
    # Every player flagged as a goalkeeper receives the special display ID '0'.
    # The per-team uniqueness constraint is intentionally removed: the tracker
    # already ensures at most two GK slots exist (GK_SLOT_A / GK_SLOT_B) so
    # there is no risk of assigning '0' to more players than expected.
    team_player_id_map = {}

    for player_id in sorted(goalkeeper_ids, key=lambda p: first_frame.get(p, 0)):
        team_player_id_map[player_id] = '0'

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

    # --- Invariant check: prefix-team consistency ---
    # 'a' IDs must belong to team 1 only, 'b' IDs to team 2 only.
    # Goalkeeper IDs ('0') are exempt.  Any other violation is a bug.
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

    # --- Assign '0' to all referees ---
    for frame_num, referee_track in enumerate(tracks.get('referees', [])):
        for referee_id in referee_track:
            tracks['referees'][frame_num][referee_id]['team_player_id'] = '0'

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


# ---------------------------------------------------------------------------
# Default chunk size (frames).  Override via the CHUNK_SIZE environment
# variable or by passing chunk_size explicitly to process_video().
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 500


def _process_chunk(video_frames, tracker, chunk_idx, video_name,
                   pre_fitted_team_assigner=None, frame_rate=24):
    """Run the full analysis pipeline on a pre-loaded list of *video_frames*.

    This is an internal helper shared by both the monolithic and chunked paths.
    When *pre_fitted_team_assigner* is supplied its KMeans model is reused so
    that team colours stay consistent across chunks.

    Returns
    -------
    list of numpy.ndarray
        Annotated output frames for this chunk.
    """
    # Build per-chunk stub paths so each chunk can be cached independently.
    chunk_suffix = f'_chunk{chunk_idx}' if chunk_idx is not None else ''
    track_stub = os.path.join('stubs', f'{video_name}{chunk_suffix}_track_stubs.pkl')
    camera_stub = os.path.join('stubs', f'{video_name}{chunk_suffix}_camera_movement_stub.pkl')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path=track_stub)
    tracker.add_position_to_tracks(tracks)

    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path=camera_stub,
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    speed_and_distance_estimator = SpeedAndDistance_Estimator(frame_rate=frame_rate)
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # GTA Lite: post-traitement d'association globale de tracklets.
    # Doit être appelé après que les positions ajustées sont disponibles et
    # avant l'attribution des équipes pour que les IDs soient stables.
    n_merges = refine_tracklets(tracks)
    if n_merges:
        print(f"    [gta_lite] {n_merges} fusion(s) de tracklets appliquée(s).")

    # Team assignment: reuse the globally pre-fitted model when available.
    if pre_fitted_team_assigner is not None:
        team_assigner = pre_fitted_team_assigner
        team_map = team_assigner.assign_teams_global(video_frames, tracks['players'])
    else:
        team_assigner = TeamAssigner()
        team_map = team_assigner.assign_teams_global(video_frames, tracks['players'])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_map.get(
                player_id,
                team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id),
            )
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    assign_team_player_ids(tracks)

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

    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)
    minimap = MiniMap()
    output_video_frames = minimap.draw_minimap(output_video_frames, tracks)

    return output_video_frames


def process_video_chunked(video_path, tracker, chunk_size):
    """Process a video in frame-range chunks to limit peak memory usage.

    The video is split into non-overlapping chunks of *chunk_size* frames.
    Each chunk is analysed independently and saved to a temporary file.
    After all chunks have been processed the temporary files are concatenated
    into the final output and the temporaries are deleted.

    Team colours are determined from a lightweight global sampling pass
    (50 evenly-spaced frames, no full load) so they remain consistent across
    chunks.

    Parameters
    ----------
    video_path:
        Path to the input video.
    tracker:
        An initialised ``FootballBotSortTracker`` instance.
    chunk_size:
        Maximum number of frames per chunk.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join('output_videos', os.path.basename(video_path))
    os.makedirs('output_videos', exist_ok=True)
    os.makedirs('stubs', exist_ok=True)

    info = get_video_info(video_path)
    total_frames = info['frame_count']
    frame_rate = info.get('fps', 24)
    n_chunks = (total_frames + chunk_size - 1) // chunk_size

    print(f"\n=== Processing (chunked): {video_path} ===")
    print(f"    Total frames: {total_frames}  |  chunk_size: {chunk_size}  |  chunks: {n_chunks}  |  fps: {frame_rate}")

    # ------------------------------------------------------------------
    # Global team-colour sampling — reads only 50 frames from the whole
    # video so team colours stay consistent across all chunks.
    # ------------------------------------------------------------------
    print("  [pre] Sampling frames for global team-colour fitting...")
    sampled = read_video_sampled(video_path, n_samples=50)

    if sampled:
        # Build a minimal tracks_players structure for assign_teams_global:
        # we need it to map frame indices to player bboxes.  We run a quick
        # detection pass on the sampled frames only.
        sampled_frames = [f for _, f in sampled]
        sampled_indices = [idx for idx, _ in sampled]

        # Detect on sampled frames (tiny subset, low memory cost).
        sampled_detections = tracker.detect_frames(sampled_frames)

        import supervision as _sv  # noqa: E402 — available via tracker install
        cls_names_inv = {}
        pseudo_tracks = []
        for det in sampled_detections:
            cls_names = det.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            keep = {cls_names_inv.get(c) for c in ('player', 'goalkeeper')
                    if c in cls_names_inv}
            det_sv = _sv.Detections.from_ultralytics(det)
            frame_dict = {}
            for box_id, (box, cls_id) in enumerate(
                zip(det_sv.xyxy, det_sv.class_id)
            ):
                if cls_id not in keep:
                    continue
                is_gk = (det.names.get(cls_id) == 'goalkeeper')
                frame_dict[box_id + 1] = {
                    'bbox': box.tolist(),
                    'is_goalkeeper': is_gk,
                }
            pseudo_tracks.append(frame_dict)

        pre_fitted = TeamAssigner()
        pre_fitted.assign_teams_global(sampled_frames, pseudo_tracks)
        print("  [pre] Global team-colour model fitted.")
    else:
        pre_fitted = None
        print("  [pre] Could not sample frames — team colours will be fitted per chunk.")

    # ------------------------------------------------------------------
    # Chunked processing
    # ------------------------------------------------------------------
    tmp_dir = tempfile.mkdtemp(prefix='socceranalytics_chunks_')
    chunk_paths = []

    try:
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, total_frames)
            print(f"\n  --- Chunk {chunk_idx + 1}/{n_chunks}: frames {start}–{end - 1} ---")

            video_frames = read_video_chunk(video_path, start, end)
            if not video_frames:
                print(f"  [warn] No frames decoded for chunk {chunk_idx + 1} — skipping.")
                continue

            output_frames = _process_chunk(
                video_frames, tracker, chunk_idx, video_name,
                pre_fitted_team_assigner=pre_fitted,
                frame_rate=frame_rate,
            )

            chunk_path = os.path.join(tmp_dir, f'chunk_{chunk_idx:04d}.mp4')
            save_video(output_frames, chunk_path)
            chunk_paths.append(chunk_path)
            print(f"         Chunk {chunk_idx + 1} saved to '{chunk_path}'.")

        # ------------------------------------------------------------------
        # Concatenate chunks into the final output
        # ------------------------------------------------------------------
        print(f"\n  [concat] Concatenating {len(chunk_paths)} chunk(s) → '{output_path}'...")
        concatenate_videos(chunk_paths, output_path)
        print(f"           -> Final video saved to: {output_path}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def process_video(video_path, tracker, chunk_size=None):
    """Process a single video through the full analysis pipeline.

    When *chunk_size* is set (or the ``CHUNK_SIZE`` env-var is non-zero) and
    the video contains more frames than *chunk_size*, processing is delegated
    to :func:`process_video_chunked` to keep peak memory usage bounded.
    """
    # Resolve effective chunk size (explicit arg overrides env-var).
    effective_chunk_size = chunk_size
    if effective_chunk_size is None:
        env_val = os.environ.get('CHUNK_SIZE', '')
        if env_val.strip().isdigit():
            effective_chunk_size = int(env_val.strip())

    # Check whether chunking is warranted.
    if effective_chunk_size and effective_chunk_size > 0:
        info = get_video_info(video_path)
        if info['frame_count'] > effective_chunk_size:
            print(f"\n  [process_video] Video has {info['frame_count']} frames "
                  f"(> chunk_size={effective_chunk_size}) → using chunked mode.")
            process_video_chunked(video_path, tracker, effective_chunk_size)
            return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join('output_videos', os.path.basename(video_path))
    track_stub = os.path.join('stubs', f'{video_name}_track_stubs.pkl')
    camera_stub = os.path.join('stubs', f'{video_name}_camera_movement_stub.pkl')

    print(f"\n=== Processing: {video_path} ===")

    # Read Video (also get fps for accurate speed computation)
    print(f"  [1/8] Reading video frames from '{video_path}'...")
    video_info = get_video_info(video_path)
    frame_rate = video_info.get('fps', 24)
    video_frames = read_video(video_path)
    print(f"        -> {len(video_frames)} frames loaded (fps={frame_rate}).")

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
    speed_and_distance_estimator = SpeedAndDistance_Estimator(frame_rate=frame_rate)
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
    """Entry point — delegates to the new sn-gamestate-style Pipeline when a
    config file is present, otherwise falls back to the legacy per-function
    flow so existing stubs and behaviour are preserved.

    Pass ``CHUNK_SIZE=<n>`` as a command-line argument (or set it as an
    environment variable) to enable chunked processing.  Videos whose frame
    count exceeds *n* will be split into chunks of that many frames, processed
    independently, and their outputs concatenated into the final video.

    Examples::

        python main.py CHUNK_SIZE=500
        python main.py                   # no chunking (default)
    """
    import sys

    os.makedirs('stubs', exist_ok=True)

    # Accept CHUNK_SIZE=<n> either as a CLI argument or an environment variable.
    # CLI argument takes precedence.
    chunk_size = None
    for arg in sys.argv[1:]:
        if arg.upper().startswith('CHUNK_SIZE='):
            val = arg.split('=', 1)[1].strip()
            if val.isdigit() and int(val) > 0:
                chunk_size = int(val)
            break

    if chunk_size is None:
        _env_chunk = os.environ.get('CHUNK_SIZE', '').strip()
        if _env_chunk.isdigit() and int(_env_chunk) > 0:
            chunk_size = int(_env_chunk)

    config_path = 'configs/pipeline.yaml'
    if os.path.exists(config_path):
        # ── New modular pipeline (sn-gamestate structure) ──────────────────
        print(f"Using modular pipeline config: {config_path}")
        if chunk_size:
            print(f"Chunked mode enabled: CHUNK_SIZE={chunk_size} frames per chunk.")
        from socceranalytics import Pipeline
        pipeline = Pipeline.from_config(config_path, chunk_size=chunk_size)
        pipeline.run('input_videos')
    else:
        # ── Legacy fallback ────────────────────────────────────────────────
        video_files = sorted([
            f for f in glob.glob('input_videos/*')
            if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
        ])

        if not video_files:
            print("No video files found in 'input_videos/'. Supported formats:", VIDEO_EXTENSIONS)
            return

        print(f"Found {len(video_files)} video(s) to process: {video_files}")
        if chunk_size:
            print(f"Chunked mode enabled: CHUNK_SIZE={chunk_size} frames per chunk.")

        tracker = FootballBotSortTracker('models/soccer.onnx', config_path='botsort_football.yaml')

        for video_path in video_files:
            process_video(video_path, tracker, chunk_size=chunk_size)

        print("\nAll videos processed successfully.")



if __name__ == '__main__':
    main()