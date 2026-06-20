"""
socceranalytics
===============
Modular soccer video analysis package structured after sn-gamestate.

Usage::

    from socceranalytics import Pipeline

    pipeline = Pipeline.from_config('configs/pipeline.yaml')
    pipeline.run('input_videos/match.mp4')
"""

from .pipeline import Pipeline
from .state import GameState

__all__ = ['Pipeline', 'GameState']
