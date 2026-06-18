import cv2
import numpy as np


class MiniMap:
    """Renders a top-down 2-D minimap of the pitch and overlays it on each
    video frame.

    Player/ball positions come from ``tracks[object][frame_num][id]
    ['position_transformed']``, which are already in real-world metres as
    produced by :class:`view_transformer.ViewTransformer` via OpenCV's
    ``getPerspectiveTransform`` (homography).

    Coordinate convention (matches ViewTransformer):
        x  in  [0, COURT_LENGTH]  — along the pitch (short axis of the crop)
        y  in  [0, COURT_WIDTH]   — across the pitch

    The minimap is drawn landscape:  horizontal axis = pitch width (y),
    vertical axis = pitch length (x).
    """

    # Real-world dimensions — must match ViewTransformer
    COURT_LENGTH: float = 23.32   # metres  (x axis)
    COURT_WIDTH: float = 68.0     # metres  (y axis)

    # Minimap canvas size (pixels)
    MINIMAP_W: int = 280          # maps to COURT_WIDTH
    MINIMAP_H: int = int(280 * COURT_LENGTH / COURT_WIDTH)  # ≈ 96 px

    PADDING: int = 8              # inner padding around pitch lines
    MARGIN: int = 15              # distance from frame edge

    # Colours (BGR)
    _PITCH_BG = (34, 139, 34)
    _LINE_COLOR = (255, 255, 255)
    _BALL_COLOR = (0, 255, 255)
    _REFEREE_COLOR = (0, 165, 255)

    def __init__(self) -> None:
        inner_w = self.MINIMAP_W - 2 * self.PADDING
        inner_h = self.MINIMAP_H - 2 * self.PADDING
        self._scale_x = inner_w / self.COURT_WIDTH    # px per metre (y → px_x)
        self._scale_y = inner_h / self.COURT_LENGTH   # px per metre (x → px_y)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _world_to_px(self, x_world: float, y_world: float):
        """Map world coords (metres) to minimap pixel coords."""
        px = self.PADDING + int(y_world * self._scale_x)
        py = self.PADDING + int(x_world * self._scale_y)
        return px, py

    def _blank_pitch(self) -> np.ndarray:
        canvas = np.full(
            (self.MINIMAP_H, self.MINIMAP_W, 3),
            self._PITCH_BG,
            dtype=np.uint8,
        )
        # Outer boundary
        cv2.rectangle(
            canvas,
            (self.PADDING, self.PADDING),
            (self.MINIMAP_W - self.PADDING, self.MINIMAP_H - self.PADDING),
            self._LINE_COLOR,
            1,
        )
        # Centre line (vertical, halfway along x axis)
        cx = self._world_to_px(self.COURT_LENGTH / 2, 0)[1]
        cv2.line(
            canvas,
            (self.PADDING, cx),
            (self.MINIMAP_W - self.PADDING, cx),
            self._LINE_COLOR,
            1,
        )
        return canvas

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw_frame_minimap(
        self,
        frame: np.ndarray,
        players: dict,
        referees: dict,
        ball: dict,
    ) -> np.ndarray:
        """Overlay the minimap for a single frame onto *frame* (in-place).

        Parameters
        ----------
        frame:     H×W×3 BGR video frame.
        players:   ``tracks['players'][frame_num]``
        referees:  ``tracks['referees'][frame_num]``
        ball:      ``tracks['ball'][frame_num]``
        """
        minimap = self._blank_pitch()

        # --- Players ---
        for _, info in players.items():
            pos = info.get("position_transformed")
            if pos is None:
                continue
            color = info.get("team_color", (200, 200, 200))
            px, py = self._world_to_px(pos[0], pos[1])
            cv2.circle(minimap, (px, py), 4, color, -1)
            cv2.circle(minimap, (px, py), 4, (0, 0, 0), 1)

        # --- Referees ---
        for _, info in referees.items():
            pos = info.get("position_transformed")
            if pos is None:
                continue
            px, py = self._world_to_px(pos[0], pos[1])
            cv2.circle(minimap, (px, py), 4, self._REFEREE_COLOR, -1)
            cv2.circle(minimap, (px, py), 4, (0, 0, 0), 1)

        # --- Ball ---
        for _, info in ball.items():
            pos = info.get("position_transformed")
            if pos is None:
                continue
            px, py = self._world_to_px(pos[0], pos[1])
            cv2.circle(minimap, (px, py), 5, self._BALL_COLOR, -1)
            cv2.circle(minimap, (px, py), 5, (0, 0, 0), 1)

        # --- Overlay on frame (bottom-left corner) ---
        fh, fw = frame.shape[:2]
        y0 = fh - self.MINIMAP_H - self.MARGIN
        x0 = self.MARGIN
        roi = frame[y0 : y0 + self.MINIMAP_H, x0 : x0 + self.MINIMAP_W]
        cv2.addWeighted(minimap, 0.75, roi, 0.25, 0, roi)
        frame[y0 : y0 + self.MINIMAP_H, x0 : x0 + self.MINIMAP_W] = roi

        # Border around minimap
        cv2.rectangle(
            frame,
            (x0, y0),
            (x0 + self.MINIMAP_W, y0 + self.MINIMAP_H),
            (255, 255, 255),
            1,
        )

        return frame

    def draw_minimap(
        self,
        frames: list,
        tracks: dict,
    ) -> list:
        """Apply :meth:`draw_frame_minimap` to every frame in *frames*.

        Parameters
        ----------
        frames: list of H×W×3 BGR numpy arrays.
        tracks: the full tracking dict produced by Tracker / ViewTransformer.

        Returns
        -------
        The same list with minimap overlays added in-place.
        """
        players_all = tracks.get("players", [])
        referees_all = tracks.get("referees", [])
        ball_all = tracks.get("ball", [])

        for frame_num, frame in enumerate(frames):
            players = players_all[frame_num] if frame_num < len(players_all) else {}
            referees = referees_all[frame_num] if frame_num < len(referees_all) else {}
            ball = ball_all[frame_num] if frame_num < len(ball_all) else {}
            self.draw_frame_minimap(frame, players, referees, ball)

        return frames
