import sys 
sys.path.append('../')
from utils import measure_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70
    
    def assign_ball_to_player(self, players, ball_bbox):
        # Use the bottom-centre of the ball bounding box as the contact point.
        # The ball rolls at ground level, so comparing its foot position to the
        # players' foot positions gives a more accurate proximity metric than
        # using the geometric centre (which sits above the ground for aerial shots).
        ball_cx = (ball_bbox[0] + ball_bbox[2]) / 2.0
        ball_bottom = ball_bbox[3]
        ball_position = (ball_cx, ball_bottom)

        miniumum_distance = 99999
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    assigned_player = player_id

        return assigned_player