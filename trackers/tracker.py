from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path, task='detect')
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        batch_size=1 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
            if (i // batch_size + 1) % 50 == 0:
                print(f"    [detect_frames] {i + batch_size}/{len(frames)} frames processed.")
        print(f"    [detect_frames] Done — {len(detections)} frames detected.")
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            print(f"    [get_object_tracks] Loading tracks from stub: '{stub_path}'")
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            print(f"    [get_object_tracks] Stub loaded.")
            return tracks

        print(f"    [get_object_tracks] No stub found — running detection on {len(frames)} frames...")
        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Keep only players and the ball — skip goalkeepers and referees
            # as their jersey colours differ too much from outfield players
            # and would corrupt team assignment.
            keep_classes = {cls_names_inv.get(c) for c in ('player', 'ball')
                            if c in cls_names_inv}
            import numpy as _np
            mask = _np.isin(detection_supervision.class_id, list(keep_classes))
            detection_supervision = detection_supervision[mask]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            print(f"    [get_object_tracks] Saving tracks to stub: '{stub_path}'")
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)
            print(f"    [get_object_tracks] Stub saved.")

        return tracks
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if len(str(track_id)) > 2:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_player_interaction_graph(self, frame, player_dict):
        """Draw a proximity line to the nearest teammate for each player.

        A semi-transparent line is drawn from each player to their single
        nearest teammate using the team colour.  The nearest teammate's
        team-local ID (1-11) is shown at the mid-point of the line.
        """
        team_players = {}
        for pid, player in player_dict.items():
            team = player.get('team')
            if team is None:
                continue
            # Only include players that received a pool ID (a1-a11 / b1-b11).
            # Players without one would leak raw tracker IDs into the graph.
            tpid = player.get('team_player_id')
            if tpid is None:
                continue
            if team not in team_players:
                team_players[team] = []
            x_center, _ = get_center_of_bbox(player['bbox'])
            y2 = int(player['bbox'][3])
            color = player.get('team_color', (0, 0, 255))
            nearest = player.get('nearest_teammate_id')
            team_players[team].append({
                'pid': pid,
                'tpid': tpid,
                'pos': (x_center, y2),
                'color': color,
                'nearest': nearest,
            })

        overlay = frame.copy()

        for players in team_players.values():
            if len(players) < 2:
                continue
            # Build a map from tpid → pos for label lookup
            tpid_to_pos = {p['tpid']: p['pos'] for p in players}
            color = players[0]['color']

            drawn_pairs = set()
            for p in players:
                nearest_tpid = p['nearest']
                if nearest_tpid is None or nearest_tpid not in tpid_to_pos:
                    continue
                pair = tuple(sorted((p['tpid'], nearest_tpid)))
                pt1 = tuple(map(int, p['pos']))
                pt2 = tuple(map(int, tpid_to_pos[nearest_tpid]))
                cv2.line(overlay, pt1, pt2, color, 2, lineType=cv2.LINE_AA)

                if pair not in drawn_pairs:
                    mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                    cv2.putText(overlay, str(nearest_tpid), mid,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(overlay, str(nearest_tpid), mid,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    drawn_pairs.add(pair)

        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        total_frames = team_1_num_frames + team_2_num_frames
        team_1 = team_1_num_frames / total_frames if total_frames > 0 else 0
        team_2 = team_2_num_frames / total_frames if total_frames > 0 else 0

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame

    def draw_annotations(self,video_frames, tracks,team_ball_control):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                # Only display IDs from the fixed pool (a1-a11, b1-b11).
                # If a player has no pool ID (unknown team, or pool already full),
                # draw the ellipse without any label rather than leaking the raw
                # tracker ID (e.g. 227, 228).
                display_id = player.get('team_player_id')
                frame = self.draw_ellipse(frame, player["bbox"],color, display_id)

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255))

            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))


            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames