"""
Clutter Filter for Scenario 2 - Background Rejection System

This module implements logic to distinguish the drone from cage walls and corners.
Designed to reject false positives from cage mesh, poles, and structural elements.

Author: Agent 3 (Vision Developer)
Scenario: 2 - Corner Target Detection with Wall Rejection
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class BboxHistory:
    """History entry for bounding box tracking."""
    x: float
    y: float
    width: float
    height: float
    timestamp: float


class ClutterRejection:
    """
    Background clutter rejection for distinguishing target from cage structures.

    Features:
    - Stationarity checking for static objects
    - Linear structure filtering (poles, mesh lines)
    - Geometric impossibility checks
    - Parallax-based depth estimation
    """

    def __init__(self, image_width: int = 160, image_height: int = 120):
        """
        Initialize clutter rejection system.

        Args:
            image_width: Camera image width
            image_height: Camera image height
        """
        self.image_width = image_width
        self.image_height = image_height

        # Tracking parameters
        self.bbox_history = deque(maxlen=10)
        self.max_position_variance = 15.0  # pixels
        self.min_blob_circularity = 0.4  # Drone is blob-like, not linear

        # Cage boundaries (safety margins in pixels from image edges)
        self.wall_margin_px = 20

        # Linear feature detection parameters
        self.line_strength_threshold = 0.7
        self.max_line_overlap_ratio = 0.5

    def check_stationarity(self, bbox_history: List[BboxHistory]) -> bool:
        """
        Check if detected object is stationary (likely background).

        Args:
            bbox_history: List of historical bounding box detections

        Returns:
            True if object appears stationary (and should be rejected)
        """
        if len(bbox_history) < 3:
            return False  # Not enough data

        # Extract positions
        positions = np.array([[bbox.x, bbox.y] for bbox in bbox_history])

        # Calculate variance
        variance = np.var(positions, axis=0)
        total_variance = np.sum(variance)

        # If variance is very low, object is stationary
        # However, in our scenario, the TARGET is stationary, so this needs context
        # We use this with optical flow: if WE are moving but object appears stationary
        # in the world frame, it's likely the target (not the cage)

        # For now, return False as stationary objects could be our target
        return False

    def calculate_circularity(self, bbox_width: float, bbox_height: float) -> float:
        """
        Calculate circularity/compactness of bounding box.

        A square/circular object (drone) has circularity close to 1.
        A long thin object (pole) has circularity close to 0.

        Args:
            bbox_width: Bounding box width
            bbox_height: Bounding box height

        Returns:
            Circularity score [0, 1]
        """
        if bbox_width <= 0 or bbox_height <= 0:
            return 0.0

        # Aspect ratio approach
        aspect_ratio = min(bbox_width, bbox_height) / max(bbox_width, bbox_height)

        return aspect_ratio

    def filter_linear_structures(self, image: np.ndarray, bbox: Tuple[float, float, float, float]) -> bool:
        """
        Filter out linear structures like cage mesh and poles.

        Args:
            image: Input greyscale image
            bbox: Bounding box (x, y, width, height)

        Returns:
            True if detection should be REJECTED (is a linear structure)
        """
        x, y, w, h = bbox

        # Check circularity first
        circularity = self.calculate_circularity(w, h)

        if circularity < self.min_blob_circularity:
            # Very elongated - likely a pole or mesh line
            return True

        # Extract ROI
        x1 = int(max(0, x - w/2))
        x2 = int(min(self.image_width, x + w/2))
        y1 = int(max(0, y - h/2))
        y2 = int(min(self.image_height, y + h/2))

        if x2 <= x1 or y2 <= y1:
            return True  # Invalid bbox

        roi = image[y1:y2, x1:x2]

        # Detect strong linear features using gradient analysis
        has_strong_lines = self._detect_linear_features(roi)

        return has_strong_lines

    def _detect_linear_features(self, roi: np.ndarray) -> bool:
        """
        Detect if ROI contains strong linear features.

        Args:
            roi: Region of interest

        Returns:
            True if strong linear features detected
        """
        if roi.size == 0 or roi.shape[0] < 3 or roi.shape[1] < 3:
            return False

        # Calculate gradients
        # Simplified implementation using numpy (in production, use optimized version)
        grad_y = np.abs(np.diff(roi, axis=0))
        grad_x = np.abs(np.diff(roi, axis=1))

        # Check for dominant horizontal or vertical edges
        # Strong horizontal lines will have high grad_y variance
        # Strong vertical lines will have high grad_x variance

        h_variance = np.var(np.sum(grad_y, axis=1)) if grad_y.size > 0 else 0
        v_variance = np.var(np.sum(grad_x, axis=0)) if grad_x.size > 0 else 0

        total_variance = np.var(roi)

        if total_variance == 0:
            return False

        # Normalized line strength
        line_strength = (h_variance + v_variance) / (total_variance + 1e-6)

        # If line features dominate, reject
        return line_strength > self.line_strength_threshold

    def check_geometric_validity(self, bbox: Tuple[float, float, float, float],
                                  drone_position: Optional[Tuple[float, float, float]] = None) -> bool:
        """
        Check if detection is geometrically valid.

        Args:
            bbox: Bounding box (x, y, width, height)
            drone_position: Current drone position (x, y, z) in meters (optional)

        Returns:
            True if geometrically valid, False if impossible
        """
        x, y, w, h = bbox

        # Check if detection is too close to image boundaries
        # (likely wall or cage structure)
        if x < self.wall_margin_px or x > (self.image_width - self.wall_margin_px):
            return False

        if y < self.wall_margin_px or y > (self.image_height - self.wall_margin_px):
            return False

        # If drone position is known, check depth consistency
        if drone_position is not None:
            drone_x, drone_y, drone_z = drone_position

            # For Scenario 2, target is at (9.5, 0.5, 5)
            # If drone is at x > 9.5, detection cannot be valid
            # (can't see target behind us)

            # This requires odometry integration - placeholder for now
            pass

        # Check if bbox size is reasonable
        if w < 5 or h < 5:  # Too small
            return False

        if w > self.image_width * 0.8 or h > self.image_height * 0.8:  # Too large
            return False

        return True

    def calculate_parallax_score(self, bbox_history: List[BboxHistory],
                                  camera_motion: Optional[np.ndarray] = None) -> float:
        """
        Calculate parallax score to distinguish foreground from background.

        The target is at a specific distance while cage walls are at boundary.
        As camera moves, parallax effects differ.

        Args:
            bbox_history: Historical detections
            camera_motion: Camera motion vector (optional)

        Returns:
            Parallax score (higher = more likely to be target)
        """
        if len(bbox_history) < 3:
            return 0.5  # Neutral score

        # Calculate motion of detection across frames
        positions = np.array([[bbox.x, bbox.y] for bbox in bbox_history])
        motion_vectors = np.diff(positions, axis=0)

        # Calculate motion consistency
        motion_variance = np.var(motion_vectors, axis=0)
        motion_magnitude = np.mean(np.linalg.norm(motion_vectors, axis=1))

        # If camera_motion is known, we can correlate
        # For now, use motion consistency as a proxy
        # Consistent motion = likely target
        # Erratic motion = likely noise/background

        if motion_magnitude < 2.0:
            # Very little motion - could be distant target or background
            return 0.6

        # Smooth motion is good
        consistency_score = 1.0 / (1.0 + np.sum(motion_variance))

        return min(1.0, consistency_score)

    def should_reject(self, image: np.ndarray,
                      bbox: Tuple[float, float, float, float],
                      bbox_history: Optional[List[BboxHistory]] = None,
                      drone_position: Optional[Tuple[float, float, float]] = None) -> bool:
        """
        Determine if detection should be rejected as clutter.

        Args:
            image: Input image
            bbox: Bounding box (x, y, width, height)
            bbox_history: Historical detections (optional)
            drone_position: Current drone position (optional)

        Returns:
            True if detection should be REJECTED
        """
        # Check geometric validity
        if not self.check_geometric_validity(bbox, drone_position):
            return True

        # Check for linear structures (poles, mesh)
        if self.filter_linear_structures(image, bbox):
            return True

        # If we have history, check parallax
        if bbox_history and len(bbox_history) >= 3:
            parallax_score = self.calculate_parallax_score(bbox_history)

            if parallax_score < 0.3:
                # Low parallax score - likely background
                return True

        return False

    def add_detection_to_history(self, bbox: Tuple[float, float, float, float],
                                  timestamp: float):
        """
        Add detection to tracking history.

        Args:
            bbox: Bounding box (x, y, width, height)
            timestamp: Detection timestamp
        """
        x, y, w, h = bbox
        entry = BboxHistory(x=x, y=y, width=w, height=h, timestamp=timestamp)
        self.bbox_history.append(entry)

    def get_history_list(self) -> List[BboxHistory]:
        """
        Get current tracking history.

        Returns:
            List of historical detections
        """
        return list(self.bbox_history)

    def reset_history(self):
        """Reset tracking history."""
        self.bbox_history.clear()
