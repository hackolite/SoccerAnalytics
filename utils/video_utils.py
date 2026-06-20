import os
import numpy as np
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


def get_video_info(video_path):
    """Return metadata for a video file without loading all frames.

    Parameters
    ----------
    video_path:
        Path to the video file.

    Returns
    -------
    dict
        Keys: ``frame_count`` (int), ``fps`` (float), ``width`` (int),
        ``height`` (int).

    Raises
    ------
    FileNotFoundError
        If *video_path* does not exist on disk.
    RuntimeError
        If OpenCV cannot open the file.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video file not found: '{video_path}'."
        )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV could not open video file: '{video_path}'."
        )
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"frame_count": frame_count, "fps": fps, "width": width, "height": height}


def read_video_chunk(video_path, start_frame, end_frame):
    """Read a contiguous range of frames from a video file.

    Parameters
    ----------
    video_path:
        Path to the video file.
    start_frame:
        Index of the first frame to read (inclusive, 0-based).
    end_frame:
        Index of the last frame to read (exclusive).

    Returns
    -------
    list of numpy.ndarray
        The decoded frames.

    Raises
    ------
    FileNotFoundError
        If *video_path* does not exist on disk.
    RuntimeError
        If OpenCV cannot open the file or seeking fails.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video file not found: '{video_path}'."
        )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV could not open video file: '{video_path}'."
        )
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"    [read_video_chunk] Frames {start_frame}–{start_frame + len(frames) - 1} "
          f"({len(frames)} frames) read from '{video_path}'.")
    return frames


def read_video_sampled(video_path, n_samples):
    """Read *n_samples* evenly-spaced frames without loading the entire video.

    Useful for lightweight global operations such as team-colour clustering
    before committing to a full (or chunked) processing pass.

    Parameters
    ----------
    video_path:
        Path to the video file.
    n_samples:
        Number of frames to sample.

    Returns
    -------
    list of tuple (int, numpy.ndarray)
        Each element is ``(original_frame_index, frame)``.
    """
    info = get_video_info(video_path)
    total = info["frame_count"]
    indices = [
        int(i)
        for i in np.linspace(0, total - 1, min(n_samples, total))
    ]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV could not open video file: '{video_path}'."
        )
    result = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            result.append((idx, frame))
    cap.release()
    print(f"    [read_video_sampled] Sampled {len(result)} frames from '{video_path}'.")
    return result


def concatenate_videos(input_paths, output_path):
    """Concatenate multiple video files into a single output file.

    All input videos must share the same frame dimensions.  The frame rate
    from the first file is used for the output.

    Parameters
    ----------
    input_paths:
        Ordered list of paths to the video chunks to concatenate.
    output_path:
        Destination path for the merged video.

    Raises
    ------
    ValueError
        If *input_paths* is empty.
    RuntimeError
        If any input file cannot be opened.
    """
    if not input_paths:
        raise ValueError("No input video paths provided for concatenation.")

    print(f"    [concatenate_videos] Merging {len(input_paths)} chunk(s) → '{output_path}'...")

    # Determine output dimensions from the first file.
    info = get_video_info(input_paths[0])
    fps = info["fps"]
    width = info["width"]
    height = info["height"]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = 0
    for chunk_path in input_paths:
        cap = cv2.VideoCapture(chunk_path)
        if not cap.isOpened():
            raise RuntimeError(
                f"OpenCV could not open chunk file: '{chunk_path}'."
            )
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            total_frames += 1
        cap.release()

    out.release()
    print(f"    [concatenate_videos] Done — {total_frames} frames written to '{output_path}'.")


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
