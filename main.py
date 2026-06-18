import os
import glob

from utils import read_video, save_video
from trackers import Tracker
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
    """Assign stable per-team IDs (1-11) to each player and compute nearest teammate ID.

    Each team gets its own ID counter that resets from 1.  The mapping is built
    in first-appearance order and kept consistent across all frames.  Two new
    keys are added to every player entry:
      - 'team_player_id'     : int 1-11 (per-team sequential ID)
      - 'nearest_teammate_id': int 1-11 of the spatially closest teammate, or None
    """
    # Build stable per-team ID mapping {original_track_id: team_local_id}
    team_player_id_map = {}
    team_counters = {1: 0, 2: 0}

    for player_track in tracks['players']:
        for player_id, player_info in player_track.items():
            if player_id in team_player_id_map:
                continue
            team = player_info.get('team')
            if team not in team_counters:
                continue
            if team_counters[team] >= 11:
                continue
            team_counters[team] += 1
            team_player_id_map[player_id] = team_counters[team]

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
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    print(f"        -> Teams assigned.")

    # Assign per-team IDs (1-11) and nearest teammate ID
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
    tracker = Tracker('models/soccer.onnx')

    for video_path in video_files:
        process_video(video_path, tracker)

    print("\nAll videos processed successfully.")


if __name__ == '__main__':
    main()