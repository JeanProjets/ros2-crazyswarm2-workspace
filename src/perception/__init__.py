"""
Perception module for Scenario 4: Occlusion-robust tracking system.
"""

from .robust_tracker import RobustKalmanTracker, TrackState
from .occlusion_filter import EdgeClipper
from .map_projector import MapProjector
from .reid_manager import TargetFingerprint

__all__ = [
    'RobustKalmanTracker',
    'TrackState',
    'EdgeClipper',
    'MapProjector',
    'TargetFingerprint',
]
