"""
Perception Module for Scenario 2 - Vision System

This package provides computer vision capabilities for drone target detection
and precision guidance in corner scenarios.

Modules:
    - long_range_detector: Long-range target detection with digital zoom
    - clutter_filter: Background clutter rejection for cage environments
    - visual_servo: Precision visual servoing for terminal guidance
    - vision_state_manager: Power and resource management for AI Deck

Author: Agent 3 (Vision Developer)
Scenario: 2 - Corner Target Detection
"""

from .long_range_detector import LongRangeDetector, BoundingBox, DetectionMode
from .clutter_filter import ClutterRejection, BboxHistory
from .visual_servo import PrecisionGuidance, VisualError, ApproachPhase
from .vision_state_manager import VisionLifecycle, VisionMode, VisionConfig

__all__ = [
    'LongRangeDetector',
    'BoundingBox',
    'DetectionMode',
    'ClutterRejection',
    'BboxHistory',
    'PrecisionGuidance',
    'VisualError',
    'ApproachPhase',
    'VisionLifecycle',
    'VisionMode',
    'VisionConfig',
]

__version__ = '1.0.0'
__author__ = 'Agent 3 (Vision Developer)'
__scenario__ = 'Scenario 2 - Corner Target Detection'
