#!/usr/bin/env python3
"""
football_tracker.py
───────────────────
Standalone example: run BoT-SORT + ReID tracking on a football video
(default: ``input_videos/football.mp4``) using ``FootballBotSortTracker``.

Usage
-----
    python football_tracker.py
    python football_tracker.py --video input_videos/match.mp4 --model models/soccer.onnx

The script produces an annotated output video in ``output_videos/`` and
prints a summary of any suspected ID-switch events at the end.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Allow running from the repo root without installation
sys.path.insert(0, os.path.dirname(__file__))

from trackers.bot_sort_tracker import FootballBotSortTracker
from utils import read_video, save_video


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Football player tracker — YOLO + BoT-SORT + ReID"
    )
    parser.add_argument(
        '--video',
        default=os.path.join('input_videos', 'football.mp4'),
        help="Path to the input football video.",
    )
    parser.add_argument(
        '--model',
        default=os.path.join('models', 'soccer.onnx'),
        help="Path to the YOLO model weights (.pt or .onnx).",
    )
    parser.add_argument(
        '--config',
        default='botsort_football.yaml',
        help="Path to the BoT-SORT YAML configuration file.",
    )
    parser.add_argument(
        '--output',
        default=None,
        help=(
            "Path for the annotated output video. "
            "Defaults to output_videos/<input_name>_botsort.mp4."
        ),
    )
    parser.add_argument(
        '--no-stub',
        action='store_true',
        help="Force re-tracking even if a stub file already exists.",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = _parse_args()

    video_path = args.video
    model_path = args.model
    config_path = args.config

    video_stem = os.path.splitext(os.path.basename(video_path))[0]

    output_path = args.output or os.path.join(
        'output_videos', f"{video_stem}_botsort.mp4"
    )
    stub_path = os.path.join('stubs', f"{video_stem}_botsort_tracks.pkl")

    os.makedirs('output_videos', exist_ok=True)
    os.makedirs('stubs', exist_ok=True)

    # ── Validation ────────────────────────────────────────────────────────────
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        print("Place your football video at the path above and re-run.")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)

    if not os.path.exists(config_path):
        print(f"[ERROR] BoT-SORT config not found: {config_path}")
        sys.exit(1)

    # ── Initialise tracker ────────────────────────────────────────────────────
    print("[1/5] Initialising FootballBotSortTracker …")
    tracker = FootballBotSortTracker(
        model_path=model_path,
        config_path=config_path,
    )
    print(f"      Model     : {model_path}")
    print(f"      Tracker   : {config_path}")
    print(f"      ReID dev. : {tracker.reid.device}")

    # ── Load video frames ─────────────────────────────────────────────────────
    print(f"[2/5] Reading frames from '{video_path}' …")
    frames = read_video(video_path)
    print(f"      {len(frames)} frames loaded.")

    # ── Run tracking ──────────────────────────────────────────────────────────
    print("[3/5] Running BoT-SORT + ReID tracking …")
    tracks = tracker.get_object_tracks(
        frames,
        read_from_stub=not args.no_stub,
        stub_path=stub_path,
    )
    tracker.add_position_to_tracks(tracks)

    # ── Interpolate ball positions ────────────────────────────────────────────
    print("[4/5] Interpolating ball positions …")
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    # ── Render annotated video ────────────────────────────────────────────────
    print(f"[5/5] Rendering annotated video → '{output_path}' …")
    # Dummy ball-control array: full pipeline requires TeamAssigner (see main.py)
    dummy_ball_control = np.ones(len(frames), dtype=int)
    output_frames = tracker.draw_annotations(frames, tracks, dummy_ball_control)
    save_video(output_frames, output_path)
    print(f"      Saved → {output_path}")

    # ── ID-switch report ──────────────────────────────────────────────────────
    log = tracker.get_id_switch_log()
    sep = '─' * 62
    print(f"\n{sep}")
    print(f"  ID-switch suspects: {len(log)}")

    if log:
        header = f"  {'Frame':>6}  {'TrackID':>8}  {'Gap':>5}  {'Dist (px)':>9}  {'AppSim':>7}"
        print(header)
        print(f"  {'─'*6}  {'─'*8}  {'─'*5}  {'─'*9}  {'─'*7}")
        for entry in log[:25]:
            print(
                f"  {entry['frame']:>6}  "
                f"{entry['track_id']:>8}  "
                f"{entry['gap_frames']:>5}  "
                f"{entry['distance']:>9.1f}  "
                f"{entry['appearance_sim']:>7.3f}"
            )
        if len(log) > 25:
            print(f"  … and {len(log) - 25} more. "
                  "Access full log via tracker.get_id_switch_log().")

    print(f"{sep}\n")


if __name__ == '__main__':
    main()
