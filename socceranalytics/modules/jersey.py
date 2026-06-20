"""
socceranalytics/modules/jersey.py
───────────────────────────────────
Jersey-number OCR module (DetectionLevelModule).

Uses EasyOCR to read jersey numbers from player crops, then applies a
tracklet-level majority vote to produce a stable, noise-robust jersey number
for each player.

This mirrors the ``jersey_number_detect`` + ``tracklet_agg`` stages in
sn-gamestate, but uses EasyOCR (no mmocr/mmdet heavy dependency).

Configuration keys (configs/modules/jersey.yaml):
  enabled           – set to false to skip OCR entirely (default: true)
  lang              – EasyOCR language list (default: ['en'])
  min_confidence    – minimum confidence to accept an OCR result (default: 0.4)
  top_half_only     – only crop the top half of the bbox (default: true)
  vote_across_track – use tracklet-level majority vote (default: true)
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from ..state import GameState
from .base import BaseModule


def _extract_jersey(text: str) -> Optional[str]:
    """Extract a 1-2 digit jersey number from raw OCR text."""
    digits = re.sub(r'[^0-9]', '', text)
    if 1 <= len(digits) <= 2:
        return digits
    return None


class JerseyOCRModule(BaseModule):
    """Read jersey numbers from player crops using EasyOCR + tracklet voting."""

    name = 'jersey'
    input_keys = ['players']
    output_keys = ['jersey_number']

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.enabled: bool = bool(cfg.get('enabled', True))
        self.min_confidence: float = float(cfg.get('min_confidence', 0.4))
        self.top_half_only: bool = bool(cfg.get('top_half_only', True))
        self.vote_across_track: bool = bool(cfg.get('vote_across_track', True))
        self._reader = None
        if self.enabled:
            lang = cfg.get('lang', ['en'])
            self._init_reader(lang)

    def _init_reader(self, lang: list) -> None:
        try:
            import easyocr  # type: ignore
            self._reader = easyocr.Reader(lang, verbose=False)
            print("  [JerseyOCR] EasyOCR reader initialised.")
        except ImportError:
            print("  [JerseyOCR] WARNING: easyocr not installed — jersey OCR disabled. "
                  "Install with: pip install easyocr")
            self.enabled = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_crop(self, crop) -> Optional[Tuple[str, float]]:
        """Return (jersey_number, confidence) for a single image crop.

        Returns ``None`` when no valid number is found.
        """
        if self._reader is None:
            return None
        import numpy as np
        if crop is None or crop.size == 0:
            return None
        # Optionally restrict to top half of the bounding box
        if self.top_half_only:
            crop = crop[: crop.shape[0] // 2, :]
        if crop.shape[0] < 5 or crop.shape[1] < 5:
            return None
        try:
            results = self._reader.readtext(crop, detail=1)
        except Exception:
            return None
        best_jn: Optional[str] = None
        best_conf = 0.0
        for (_bbox, text, conf) in results:
            jn = _extract_jersey(text)
            if jn is not None and conf >= self.min_confidence and conf > best_conf:
                best_jn = jn
                best_conf = conf
        return (best_jn, best_conf) if best_jn is not None else None

    def _vote_jersey_number(
        self, candidates: List[Tuple[str, float]]
    ) -> Optional[str]:
        """Return the jersey number with the highest total confidence."""
        if not candidates:
            return None
        # Weighted vote: accumulate confidence per candidate value
        scores: Dict[str, float] = {}
        for jn, conf in candidates:
            scores[jn] = scores.get(jn, 0.0) + conf
        return max(scores, key=scores.__getitem__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, state: GameState, video_frames: list) -> GameState:
        if not self.enabled:
            return state

        import numpy as np

        print(f"  [JerseyOCR] Reading jersey numbers on {state.n_frames} frames…")

        # Per-tracklet candidate list: {track_id: [(jn, conf), ...]}
        tracklet_candidates: Dict[int, List[Tuple[str, float]]] = {}

        for frame_num, frame in enumerate(video_frames):
            h, w = frame.shape[:2]
            for track_id, det in state.players[frame_num].items():
                if det.get('role') not in (None, 'player', 'goalkeeper'):
                    continue
                bbox = det.get('bbox', [])
                if len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2]
                result = self._read_crop(crop)
                if result is not None:
                    tracklet_candidates.setdefault(track_id, []).append(result)
                    # Also write the per-frame result
                    det['jersey_number'] = result[0]

        if self.vote_across_track:
            # Overwrite per-frame results with the tracklet vote
            voted: Dict[int, Optional[str]] = {
                tid: self._vote_jersey_number(cands)
                for tid, cands in tracklet_candidates.items()
            }
            for frame in state.players:
                for track_id, det in frame.items():
                    jn = voted.get(track_id)
                    if jn is not None:
                        det['jersey_number'] = jn

        assigned = sum(1 for tid in tracklet_candidates if tracklet_candidates[tid])
        print(f"  [JerseyOCR] Jersey numbers read for {assigned} unique players.")
        return state
