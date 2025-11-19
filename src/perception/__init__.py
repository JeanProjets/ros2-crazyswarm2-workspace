"""
Perception module for Scenario 3: Mobile Target Tracking.

This module provides high-speed tracking, state estimation, and fallback
scanning capabilities for intercepting moving targets.
"""

from .fast_tracker import ROITracker, Detection, TrackingState
from .state_estimator import KalmanFilter1D, VelocityCalculator, TargetState
from .zone_scanner import MotionScanner, MovingBlob, ScannerMode
from .camera_control import MotionExposureControl, ExposureMode, ExposureSettings

__all__ = [
    # Fast Tracker
    'ROITracker',
    'Detection',
    'TrackingState',

    # State Estimator
    'KalmanFilter1D',
    'VelocityCalculator',
    'TargetState',

    # Zone Scanner
    'MotionScanner',
    'MovingBlob',
    'ScannerMode',

    # Camera Control
    'MotionExposureControl',
    'ExposureMode',
    'ExposureSettings',
]
