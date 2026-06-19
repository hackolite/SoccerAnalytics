from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
    
    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        top_half_image = image[0:int(image.shape[0]/2),:]

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels forr each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_teams_global(self, frames, tracks_players, sample_every=10):
        """Assign teams to all players with a single global KMeans(2) fit.

        For each stable player ID, jersey colors are collected from every
        ``sample_every``-th frame.  The colors are averaged per player to
        produce one representative RGB vector that is robust to partial
        occlusions and detection noise.  A single KMeans(n_clusters=2) model
        is then fitted on those averaged vectors so every player is assigned
        to the cluster (team) that best matches their typical jersey color.

        This is more reliable than per-frame prediction with caching because:
          - Averaging suppresses noisy single-frame crops.
          - All players are clustered together, so the two centroids always
            correspond to the two teams rather than background colors.

        Returns
        -------
        dict
            ``{player_id: team}`` where team is 1 or 2.  Also updates
            ``self.team_colors``, ``self.kmeans``, and
            ``self.player_team_dict`` so that ``get_player_team`` falls back
            correctly for any player not seen during sampling.
        """
        # Collect per-player color samples from evenly-spaced frames.
        color_samples: dict[int, list] = {}
        sampled_indices = range(0, len(frames), max(1, sample_every))
        for frame_idx in sampled_indices:
            frame = frames[frame_idx]
            for player_id, player_info in tracks_players[frame_idx].items():
                bbox = player_info['bbox']
                try:
                    color = self.get_player_color(frame, bbox)
                    color_samples.setdefault(player_id, []).append(color)
                except Exception:
                    continue

        if len(color_samples) < 2:
            print("    [TeamAssigner] Not enough players sampled for global KMeans — skipping.")
            return {}

        # Average colors per player → one representative vector each.
        player_ids = sorted(color_samples.keys())
        avg_colors = np.array([np.mean(color_samples[pid], axis=0) for pid in player_ids])

        print(f"    [TeamAssigner] Global KMeans(2) on {len(player_ids)} players "
              f"(sampled from {len(list(sampled_indices))} frames)...")

        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=0)
        labels = kmeans.fit_predict(avg_colors)

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

        # Build and cache the global player → team map (label 0 → team 1, etc.).
        team_map = {pid: int(label) + 1 for pid, label in zip(player_ids, labels)}
        self.player_team_dict.update(team_map)

        n1 = sum(1 for t in team_map.values() if t == 1)
        n2 = sum(1 for t in team_map.values() if t == 2)
        print(f"    [TeamAssigner] Assignment complete: team 1 → {n1} players, "
              f"team 2 → {n2} players.")
        print(f"    [TeamAssigner] Team colors: team1={self.team_colors[1]}, "
              f"team2={self.team_colors[2]}")

        return team_map

    def get_player_team(self,frame,player_bbox,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame,player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id+=1

        if player_id ==91:
            team_id=1

        self.player_team_dict[player_id] = team_id

        return team_id
