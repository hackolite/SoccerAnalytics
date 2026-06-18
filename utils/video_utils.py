import os
import cv2


def read_video(video_path):
    """Read all frames from a video file and return them as a list.

    Raises
    ------
    FileNotFoundError
        If *video_path* does not exist on disk.
    RuntimeError
        If OpenCV cannot open the file, or if no frames could be decoded.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video file not found: '{video_path}'. "
            "Place your input video at that path or update the path in main.py."
        )
    print(f"    [read_video] Opening '{video_path}'...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV could not open video file: '{video_path}'. "
            "The file may be corrupt or in an unsupported format."
        )
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(
            f"No frames could be decoded from '{video_path}'. "
            "The file may be empty or corrupt."
        )
    print(f"    [read_video] {len(frames)} frames read successfully.")
    return frames


def save_video(output_video_frames, output_video_path):
    """Save a list of frames as a video file (mp4)."""
    if not output_video_frames:
        raise ValueError("No frames to save.")
    print(f"    [save_video] Writing {len(output_video_frames)} frames to '{output_video_path}'...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = output_video_frames[0].shape[:2]
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))
    for frame in output_video_frames:
        out.write(frame)
    out.release()
    print(f"    [save_video] Video saved successfully.")
