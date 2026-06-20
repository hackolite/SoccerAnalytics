"""
socceranalytics/pipeline.py
────────────────────────────
Pipeline orchestrator — mirrors sn-gamestate's TrackLab pipeline engine.

The ``Pipeline`` class:
  1. Loads module configs from a YAML file (``configs/pipeline.yaml``).
  2. Instantiates each module in declaration order.
  3. Runs them sequentially, passing the shared ``GameState`` through.

Each module is identified by its ``name`` key in the YAML ``pipeline`` list.
Additional per-module parameters live under a top-level dict keyed by module
name (same pattern as sn-gamestate's Hydra defaults list).

Example YAML layout::

    pipeline:
      - tracking
      - camera
      - calibration
      - ball_interpolation
      - speed
      - team_clustering
      - team_side
      - jersey
      - team_player_ids
      - visualization

    tracking:
      model_path: models/soccer.onnx
      config_path: botsort_football.yaml
      stub_path: stubs/{video_name}_track_stubs.pkl

    jersey:
      enabled: true
      min_confidence: 0.4
      vote_across_track: true
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

_REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _REPO_ROOT)

from utils import read_video, save_video, get_video_info, read_video_chunk, concatenate_videos  # noqa: E402

from .state import GameState
from .modules import (
    TrackingModule,
    CameraMovementModule,
    CalibrationModule,
    TeamClusteringModule,
    TeamSideLabelingModule,
    JerseyOCRModule,
    BallModule,
    SpeedDistanceModule,
    VisualizationModule,
)

# Registry: module name → class
_MODULE_REGISTRY: Dict[str, Any] = {
    'tracking':         TrackingModule,
    'camera':           CameraMovementModule,
    'calibration':      CalibrationModule,
    'ball':             BallModule,
    'speed':            SpeedDistanceModule,
    'team_clustering':  TeamClusteringModule,
    'team_side':        TeamSideLabelingModule,
    'jersey':           JerseyOCRModule,
    'visualization':    VisualizationModule,
    # Alias used by ball interpolation stage
    'ball_interpolation': BallModule,
}


class Pipeline:
    """Run the full analysis pipeline on one or more video files.

    Parameters
    ----------
    modules:
        Ordered list of instantiated module objects.
    cfg:
        Full config dict (used for runtime resolution of per-video paths).
    chunk_size:
        Maximum number of video frames to hold in RAM at once.  When the
        video contains more frames than this value the pipeline processes it
        in non-overlapping chunks (each chunk independently analysed) and
        concatenates the output files.  ``None`` (the default) disables
        chunking and loads the whole video into memory — suitable only for
        short clips.  For full matches set e.g. ``chunk_size=500`` or pass
        ``CHUNK_SIZE=500`` on the command line / as an environment variable.
    """

    def __init__(self, modules: list, cfg: dict, chunk_size: Optional[int] = None) -> None:
        self.modules = modules
        self.cfg = cfg
        self.chunk_size = chunk_size

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str, chunk_size: Optional[int] = None) -> 'Pipeline':
        """Build a Pipeline from a YAML config file."""
        with open(config_path, 'r') as fh:
            cfg: dict = yaml.safe_load(fh)

        pipeline_names: List[str] = cfg.get('pipeline', [])
        modules: list = []
        for name in pipeline_names:
            cls_type = _MODULE_REGISTRY.get(name)
            if cls_type is None:
                warnings.warn(
                    f"Pipeline: unknown module '{name}' — skipping.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            module_cfg: dict = cfg.get(name, {})
            modules.append(cls_type(module_cfg))
            print(f"  [Pipeline] Registered module: {name}")

        return cls(modules=modules, cfg=cfg, chunk_size=chunk_size)

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def process_video(self, video_path: str) -> None:
        """Run the full pipeline on a single video file.

        When ``self.chunk_size`` is set and the video has more frames than
        that value, the video is split into non-overlapping chunks.  Each
        chunk is processed independently and the annotated outputs are
        concatenated into the final file.

        Saves an annotated output video to ``output_videos/``.
        """
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join('output_videos', os.path.basename(video_path))
        os.makedirs('output_videos', exist_ok=True)
        os.makedirs('stubs', exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Processing: {video_path}")
        print(f"{'='*60}")

        # Resolve stub path templates once per video
        self._resolve_stub_paths(video_name)

        # Determine whether chunked processing is needed
        info = get_video_info(video_path)
        total_frames = info['frame_count']

        if self.chunk_size and self.chunk_size > 0 and total_frames > self.chunk_size:
            print(f"  [chunked] Video has {total_frames} frames "
                  f"(> chunk_size={self.chunk_size}) — processing in chunks.")
            self._process_video_chunked(video_path, output_path, total_frames)
            return

        # Warn when the video is large enough to risk a MemoryError
        # (rough heuristic: > 5 000 frames at 1080p ≈ >30 GB RAM).
        if total_frames > 5_000 and (not self.chunk_size or self.chunk_size <= 0):
            print(
                f"  [warn] Video has {total_frames} frames.  Loading all frames "
                "into RAM may exhaust memory.\n"
                "         Re-run with a chunk size to process in batches:\n"
                "             python main.py CHUNK_SIZE=500\n"
                "         or set the environment variable:  CHUNK_SIZE=500"
            )

        # Load frames
        print("  [1] Reading video frames…")
        video_frames = read_video(video_path)
        print(f"       → {len(video_frames)} frames loaded.")

        output_frames = self._run_pipeline(video_frames, video_path)

        # Save
        print(f"  [save] Saving to '{output_path}'…")
        save_video(output_frames, output_path)
        print(f"         → Saved.")

    def _run_pipeline(self, video_frames: list, video_path: str) -> list:
        """Run all modules on *video_frames* and return annotated frames."""
        # Initialise state
        state = GameState(video_path=video_path)

        # Run modules
        for i, module in enumerate(self.modules, 1):
            print(f"  [{i+1}] Running module: {module.name}…")
            module.process(state, video_frames)

        # Apply stable per-team IDs
        print("  [post] Assigning team-player IDs…")
        _assign_team_player_ids(state)

        # Draw final output
        viz_module = next(
            (m for m in self.modules if m.name == 'visualization'), None
        )
        if viz_module is not None:
            print("  [viz] Drawing annotations…")
            output_frames = viz_module.process(state, video_frames)
        else:
            output_frames = video_frames

        return output_frames

    def _process_video_chunked(
        self, video_path: str, output_path: str, total_frames: int
    ) -> None:
        """Process *video_path* in chunks of ``self.chunk_size`` frames.

        Each chunk is annotated independently and saved to a temporary file.
        All temporary files are then concatenated into *output_path* and
        removed.
        """
        chunk_size = self.chunk_size
        n_chunks = (total_frames + chunk_size - 1) // chunk_size
        print(f"  [chunked] Total frames: {total_frames}  |  "
              f"chunk_size: {chunk_size}  |  chunks: {n_chunks}")

        tmp_dir = tempfile.mkdtemp(prefix='socceranalytics_chunks_')
        chunk_paths: List[str] = []

        try:
            for chunk_idx in range(n_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, total_frames)
                print(f"\n  --- Chunk {chunk_idx + 1}/{n_chunks}: "
                      f"frames {start}–{end - 1} ---")

                video_frames = read_video_chunk(video_path, start, end)
                if not video_frames:
                    print(f"  [warn] No frames decoded for chunk "
                          f"{chunk_idx + 1} — skipping.")
                    continue

                output_frames = self._run_pipeline(video_frames, video_path)

                chunk_path = os.path.join(
                    tmp_dir, f'chunk_{chunk_idx:04d}.mp4'
                )
                save_video(output_frames, chunk_path)
                chunk_paths.append(chunk_path)
                print(f"         Chunk {chunk_idx + 1} saved to "
                      f"'{chunk_path}'.")

            if not chunk_paths:
                raise RuntimeError("No chunks were processed successfully.")

            print(f"\n  [concat] Concatenating {len(chunk_paths)} "
                  f"chunk(s) → '{output_path}'…")
            concatenate_videos(chunk_paths, output_path)
            print(f"           → Final video saved to: {output_path}")

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def run(self, video_dir_or_path: str = 'input_videos') -> None:
        """Process one file or every video in a directory."""
        import glob as _glob

        VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')
        if os.path.isfile(video_dir_or_path):
            video_files = [video_dir_or_path]
        else:
            video_files = sorted([
                f for f in _glob.glob(os.path.join(video_dir_or_path, '*'))
                if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
            ])

        if not video_files:
            print(f"No video files found in '{video_dir_or_path}'.")
            return

        print(f"Found {len(video_files)} video(s) to process.")
        for vp in video_files:
            self.process_video(vp)
        print("\nAll videos processed successfully.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_stub_paths(self, video_name: str) -> None:
        """Replace ``{video_name}`` placeholders in stub_path configs."""
        for module in self.modules:
            stub = module.cfg.get('stub_path')
            if stub and '{video_name}' in stub:
                resolved = stub.replace('{video_name}', video_name)
                module.cfg['stub_path'] = resolved
                # Also propagate to sub-objects that already read the value
                if hasattr(module, 'stub_path'):
                    module.stub_path = resolved


# ──────────────────────────────────────────────────────────────────────────────
# Stable per-team ID assignment (unchanged from main.py)
# ──────────────────────────────────────────────────────────────────────────────

def _assign_team_player_ids(state: GameState) -> None:
    """Wrap the ``assign_team_player_ids`` function from ``main.py``."""
    # Import from the top-level main module to avoid duplication
    import importlib.util, types

    spec = importlib.util.spec_from_file_location(
        '_main_module',
        os.path.join(_REPO_ROOT, 'main.py'),
    )
    if spec is None or spec.loader is None:
        warnings.warn("Could not import assign_team_player_ids from main.py",
                      RuntimeWarning, stacklevel=2)
        return

    _main = types.ModuleType('_main_module')
    try:
        spec.loader.exec_module(_main)  # type: ignore[attr-defined]
        _main.assign_team_player_ids(state.to_tracks())
    except Exception as exc:
        warnings.warn(
            f"assign_team_player_ids failed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
