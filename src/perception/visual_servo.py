"""
Visual Servoing for Scenario 2 - Precision Guidance System

This module provides real-time error corrections for the flight controller
during corner approach where OptiTrack may be unreliable.

Author: Agent 3 (Vision Developer)
Scenario: 2 - Corner Target Precision Approach
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum


class ApproachPhase(Enum):
    """Flight approach phases."""
    TRANSIT = "transit"
    LONG_RANGE = "long_range"
    PRECISION_APPROACH = "precision_approach"
    TERMINAL = "terminal"


@dataclass
class VisualError:
    """Visual servoing error output."""
    err_x: float  # Normalized X error [-1, 1]
    err_y: float  # Normalized Y error [-1, 1]
    err_yaw: float  # Yaw correction [-1, 1]
    distance: float  # Estimated distance to target
    confidence: float  # Error estimate confidence
    visual_error_norm: float  # Overall error magnitude


class PrecisionGuidance:
    """
    Visual servoing system for precision guidance.

    Provides real-time alignment corrections when approaching corner target.
    Operates at 30Hz+ during PRECISION_APPROACH phase.
    """

    def __init__(self, image_width: int = 160, image_height: int = 120):
        """
        Initialize precision guidance system.

        Args:
            image_width: Camera image width in pixels
            image_height: Camera image height in pixels
        """
        self.image_width = image_width
        self.image_height = image_height

        # Image center (ideal target position)
        self.center_x = image_width / 2.0
        self.center_y = image_height / 2.0

        # Distance estimation parameters
        self.real_width_mm = 92.0  # Crazyflie width
        self.focal_length_px = 120.0  # Calibrated focal length

        # Control parameters
        self.target_update_rate = 30.0  # Hz
        self.min_confidence = 0.5

        # Error history for smoothing
        self.error_history_x = []
        self.error_history_y = []
        self.max_history = 5

        # Safety parameters
        self.max_error_magnitude = 1.0
        self.target_lost_timeout = 0.5  # seconds

        # Timing
        self.last_detection_time = 0.0
        self.current_phase = ApproachPhase.TRANSIT

    def calculate_centering_error(self, bbox: Tuple[float, float, float, float],
                                   image_center: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
        """
        Calculate centering error for visual servoing.

        Args:
            bbox: Bounding box (x, y, width, height) where x,y is center
            image_center: Optional custom image center (defaults to frame center)

        Returns:
            Tuple of (err_x, err_y) normalized to [-1, 1]
        """
        bbox_x, bbox_y, bbox_w, bbox_h = bbox

        # Use custom center or default
        if image_center is None:
            target_x = self.center_x
            target_y = self.center_y
        else:
            target_x, target_y = image_center

        # Calculate pixel errors
        pixel_err_x = bbox_x - target_x
        pixel_err_y = bbox_y - target_y

        # Normalize to [-1, 1] based on image dimensions
        # Positive err_x means target is to the right (drone should move right)
        # Positive err_y means target is below center (drone should move down)
        err_x = pixel_err_x / target_x
        err_y = pixel_err_y / target_y

        # Clamp to valid range
        err_x = np.clip(err_x, -1.0, 1.0)
        err_y = np.clip(err_y, -1.0, 1.0)

        return err_x, err_y

    def calculate_yaw_correction(self, bbox_center_x: float) -> float:
        """
        Calculate yaw correction to keep target centered.

        Args:
            bbox_center_x: Bounding box center X coordinate

        Returns:
            Yaw correction from -1.0 (rotate left) to 1.0 (rotate right)
        """
        # Error from center
        error = (bbox_center_x - self.center_x) / self.center_x

        # Clamp to valid range
        yaw_correction = np.clip(error, -1.0, 1.0)

        return yaw_correction

    def estimate_distance_to_impact(self, bbox_width: float) -> float:
        """
        Estimate distance to target using pinhole camera model.

        Args:
            bbox_width: Bounding box width in pixels

        Returns:
            Estimated distance in meters
        """
        if bbox_width <= 0:
            return float('inf')

        # Pinhole camera model: d = (real_width * focal_length) / pixel_width
        dist_mm = (self.real_width_mm * self.focal_length_px) / bbox_width

        return dist_mm / 1000.0  # Convert to meters

    def smooth_error(self, err_x: float, err_y: float) -> Tuple[float, float]:
        """
        Apply temporal smoothing to error signals.

        Args:
            err_x: Raw X error
            err_y: Raw Y error

        Returns:
            Smoothed (err_x, err_y)
        """
        # Add to history
        self.error_history_x.append(err_x)
        self.error_history_y.append(err_y)

        # Maintain max history length
        if len(self.error_history_x) > self.max_history:
            self.error_history_x.pop(0)
        if len(self.error_history_y) > self.max_history:
            self.error_history_y.pop(0)

        # Calculate moving average
        smoothed_x = np.mean(self.error_history_x)
        smoothed_y = np.mean(self.error_history_y)

        return smoothed_x, smoothed_y

    def compute_visual_error(self, bbox: Tuple[float, float, float, float],
                            current_time: float,
                            phase: ApproachPhase = ApproachPhase.PRECISION_APPROACH) -> VisualError:
        """
        Compute complete visual servoing error.

        Args:
            bbox: Bounding box (x, y, width, height)
            current_time: Current timestamp
            phase: Current approach phase

        Returns:
            VisualError object with all error components
        """
        self.current_phase = phase
        self.last_detection_time = current_time

        bbox_x, bbox_y, bbox_w, bbox_h = bbox

        # Calculate centering errors
        err_x, err_y = self.calculate_centering_error(bbox)

        # Calculate yaw correction
        err_yaw = self.calculate_yaw_correction(bbox_x)

        # Estimate distance
        distance = self.estimate_distance_to_impact(bbox_w)

        # Apply smoothing during precision approach
        if phase == ApproachPhase.PRECISION_APPROACH or phase == ApproachPhase.TERMINAL:
            err_x, err_y = self.smooth_error(err_x, err_y)

        # Calculate overall error magnitude
        visual_error_norm = np.sqrt(err_x**2 + err_y**2)

        # Confidence based on detection quality
        # Higher confidence when bbox is well-formed and centered
        confidence = self._calculate_confidence(bbox, visual_error_norm)

        return VisualError(
            err_x=err_x,
            err_y=err_y,
            err_yaw=err_yaw,
            distance=distance,
            confidence=confidence,
            visual_error_norm=visual_error_norm
        )

    def _calculate_confidence(self, bbox: Tuple[float, float, float, float],
                              error_norm: float) -> float:
        """
        Calculate confidence in error estimate.

        Args:
            bbox: Bounding box
            error_norm: Error magnitude

        Returns:
            Confidence score [0, 1]
        """
        bbox_x, bbox_y, bbox_w, bbox_h = bbox

        # Check bbox validity
        if bbox_w <= 0 or bbox_h <= 0:
            return 0.0

        # Confidence decreases with error magnitude
        error_confidence = 1.0 - min(error_norm, 1.0)

        # Confidence increases with bbox size (closer = more reliable)
        size_confidence = min(bbox_w / 50.0, 1.0)

        # Combined confidence
        confidence = (error_confidence + size_confidence) / 2.0

        return np.clip(confidence, 0.0, 1.0)

    def check_target_lost(self, current_time: float) -> bool:
        """
        Check if target has been lost for too long.

        Args:
            current_time: Current timestamp

        Returns:
            True if target lost beyond timeout threshold
        """
        time_since_detection = current_time - self.last_detection_time

        return time_since_detection > self.target_lost_timeout

    def should_send_hover_immediate(self, current_time: float,
                                    phase: ApproachPhase) -> bool:
        """
        Determine if HOVER_IMMEDIATE flag should be sent.

        Args:
            current_time: Current timestamp
            phase: Current approach phase

        Returns:
            True if drone should hover immediately
        """
        # Only critical during precision approach
        if phase not in [ApproachPhase.PRECISION_APPROACH, ApproachPhase.TERMINAL]:
            return False

        # Check if target lost
        return self.check_target_lost(current_time)

    def get_control_output(self, bbox: Optional[Tuple[float, float, float, float]],
                          current_time: float,
                          phase: ApproachPhase = ApproachPhase.PRECISION_APPROACH) -> Dict:
        """
        Get complete control output for flight controller.

        Args:
            bbox: Detected bounding box (None if no detection)
            current_time: Current timestamp
            phase: Current approach phase

        Returns:
            Dictionary with control parameters
        """
        if bbox is None:
            # No detection
            return {
                'error': None,
                'hover_immediate': self.should_send_hover_immediate(current_time, phase),
                'target_lost': True,
                'time_since_detection': current_time - self.last_detection_time
            }

        # Compute visual error
        error = self.compute_visual_error(bbox, current_time, phase)

        return {
            'error': error,
            'hover_immediate': False,
            'target_lost': False,
            'time_since_detection': 0.0,
            'target_relative_coords': self._compute_relative_coords(error)
        }

    def _compute_relative_coords(self, error: VisualError) -> Tuple[float, float, float]:
        """
        Compute target relative coordinates for controller.

        Args:
            error: Visual error

        Returns:
            Relative (x, y, z) coordinates
        """
        # Simplified computation
        # In full implementation, this would use distance and angle
        # to compute 3D relative position

        # For now, use error directly as relative offset
        # Scale by distance estimate
        scale = error.distance

        rel_x = error.err_x * scale
        rel_y = error.err_y * scale
        rel_z = 0.0  # Assume level flight

        return (rel_x, rel_y, rel_z)

    def reset(self):
        """Reset visual servoing state."""
        self.error_history_x.clear()
        self.error_history_y.clear()
        self.last_detection_time = 0.0
        self.current_phase = ApproachPhase.TRANSIT

    def set_camera_parameters(self, focal_length_px: float, real_width_mm: float):
        """
        Update camera calibration parameters.

        Args:
            focal_length_px: Calibrated focal length in pixels
            real_width_mm: Real target width in millimeters
        """
        self.focal_length_px = focal_length_px
        self.real_width_mm = real_width_mm
