import cv2
import sys 
sys.path.append('../')
from utils import measure_distance ,get_foot_position

class SpeedAndDistance_Estimator():
    def __init__(self):
        self.frame_window=5
        self.frame_rate=24
    
    def add_speed_and_distance_to_tracks(self,tracks):
        print(f"    [SpeedAndDistance_Estimator] Computing speed and distance for players...")
        total_distance= {}

        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue 
            number_of_frames = len(object_tracks)
            for frame_num in range(0,number_of_frames, self.frame_window):
                last_frame = min(frame_num+self.frame_window,number_of_frames-1 )

                for track_id,_ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    if start_position is None or end_position is None:
                        continue
                    
                    distance_covered = measure_distance(start_position,end_position)
                    time_elapsed = (last_frame-frame_num)/self.frame_rate
                    speed_meteres_per_second = distance_covered/time_elapsed
                    speed_km_per_hour = speed_meteres_per_second*3.6

                    if object not in total_distance:
                        total_distance[object]= {}
                    
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    
                    total_distance[object][track_id] += distance_covered

                    for frame_num_batch in range(frame_num,last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]

        # Ball distance and speed
        print(f"    [SpeedAndDistance_Estimator] Computing speed and distance for ball...")
        total_ball_distance = 0
        ball_tracks = tracks.get("ball", [])
        number_of_frames = len(ball_tracks)
        for frame_num in range(0, number_of_frames, self.frame_window):
            last_frame = min(frame_num + self.frame_window, number_of_frames - 1)
            if 1 not in ball_tracks[frame_num] or 1 not in ball_tracks[last_frame]:
                continue
            start_position = ball_tracks[frame_num][1].get('position_transformed')
            end_position = ball_tracks[last_frame][1].get('position_transformed')
            if start_position is None or end_position is None:
                continue
            distance_covered = measure_distance(start_position, end_position)
            time_elapsed = (last_frame - frame_num) / self.frame_rate
            speed_km_per_hour = (distance_covered / time_elapsed) * 3.6
            total_ball_distance += distance_covered
            for frame_num_batch in range(frame_num, last_frame):
                if 1 not in tracks["ball"][frame_num_batch]:
                    continue
                tracks["ball"][frame_num_batch][1]['speed'] = speed_km_per_hour
                tracks["ball"][frame_num_batch][1]['distance'] = total_ball_distance

    def draw_speed_and_distance(self,frames,tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees":
                    continue 
                for _, track_info in object_tracks[frame_num].items():
                   if "speed" in track_info:
                       speed = track_info.get('speed',None)
                       distance = track_info.get('distance',None)
                       if speed is None or distance is None:
                           continue
                       
                       bbox = track_info['bbox']
                       position = get_foot_position(bbox)
                       position = list(position)
                       position[1]+=40

                       position = tuple(map(int,position))
                       cv2.putText(frame, f"{speed:.2f} km/h",position,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
                       cv2.putText(frame, f"{distance:.2f} m",(position[0],position[1]+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)

            # Ball distance overlay (fixed HUD, top-right corner)
            ball_tracks = tracks.get("ball", [])
            if frame_num < len(ball_tracks):
                ball_info = ball_tracks[frame_num].get(1, {})
                ball_dist = ball_info.get('distance')
                ball_speed = ball_info.get('speed')
                if ball_dist is not None:
                    fh, fw = frame.shape[:2]
                    box_w, box_h = 300, 70
                    x0 = fw - box_w - 10
                    y0 = 10
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), -1)
                    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                    cv2.putText(frame, f"Ball dist: {ball_dist:.1f} m",
                                (x0 + 10, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    if ball_speed is not None:
                        cv2.putText(frame, f"Ball speed: {ball_speed:.1f} km/h",
                                    (x0 + 10, y0 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            output_frames.append(frame)
        
        return output_frames