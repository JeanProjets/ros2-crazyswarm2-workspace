"""
Partial Occlusion Handler for Scenario 4.

This module detects when a target is partially occluded by an obstacle edge
and prevents bad distance estimates from clipped bounding boxes.
"""

from typing import Optional, Tuple
import numpy as np


class EdgeClipper:
    """
    Detects and handles partial occlusion of targets.

    When a target moves behind a wall edge:
    - Bounding box width decreases
    - Distance calculation (d ∝ 1/w) spikes to infinity
    - Aspect ratio changes drastically (becomes vertical sliver)

    This class flags such measurements as UNRELIABLE and provides corrected estimates.
    """

    def __init__(
        self,
        normal_aspect_ratio: float = 1.0,
        min_aspect_ratio: float = 0.6,
        max_aspect_ratio: float = 1.67,
        aspect_ratio_tolerance: float = 0.3
    ):
        """
        Args:
            normal_aspect_ratio: Expected aspect ratio of target (Crazyflie ~1.0)
            min_aspect_ratio: Below this, bbox is considered a vertical sliver
            max_aspect_ratio: Above this, bbox is considered a horizontal sliver
            aspect_ratio_tolerance: Tolerance for aspect ratio changes
        """
        self.normal_aspect_ratio = normal_aspect_ratio
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.aspect_ratio_tolerance = aspect_ratio_tolerance

        # Cache for last known good measurements
        self._last_good_distance = None
        self._last_good_bbox = None
        self._aspect_ratio_history = []
        self._max_history_length = 10

    def check_partial_occlusion(
        self,
        bbox: np.ndarray,
        depth_map_proxy: Optional[np.ndarray] = None
    ) -> bool:
        """
        Check if bounding box shows signs of partial occlusion.

        Args:
            bbox: Bounding box [x, y, w, h]
            depth_map_proxy: Optional depth map for edge detection (can be None)

        Returns:
            True if bbox appears to be partially occluded
        """
        w, h = bbox[2], bbox[3]

        if h == 0:
            return True  # Degenerate bbox

        aspect_ratio = w / h

        # Track aspect ratio history
        self._aspect_ratio_history.append(aspect_ratio)
        if len(self._aspect_ratio_history) > self._max_history_length:
            self._aspect_ratio_history.pop(0)

        # Check 1: Aspect ratio out of bounds (sliver detection)
        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            return True

        # Check 2: Drastic aspect ratio change from history
        if len(self._aspect_ratio_history) >= 3:
            recent_avg = np.mean(self._aspect_ratio_history[-3:])
            if abs(aspect_ratio - recent_avg) > self.aspect_ratio_tolerance:
                return True

        # Check 3: Depth map edge detection (if available)
        if depth_map_proxy is not None:
            if self._check_edge_proximity(bbox, depth_map_proxy):
                return True

        return False

    def _check_edge_proximity(self, bbox: np.ndarray, depth_map: np.ndarray) -> bool:
        """
        Check if bbox edge touches an obstacle region in depth map.

        Args:
            bbox: [x, y, w, h]
            depth_map: 2D array representing obstacle regions

        Returns:
            True if bbox touches obstacle edge
        """
        x, y, w, h = bbox.astype(int)

        # Check if any edge of bbox is near a depth discontinuity
        # (Simple heuristic: if bbox touches edge of depth map)
        map_h, map_w = depth_map.shape

        # Check if bbox is at image edges (likely clipped)
        if x <= 2 or y <= 2 or (x + w) >= (map_w - 2) or (y + h) >= (map_h - 2):
            return True

        return False

    def is_bad_depth_measurement(self, bbox: np.ndarray) -> bool:
        """
        Detect if bbox geometry indicates unreliable depth measurement.

        Standard Crazyflie is roughly square (AR ~1.0).
        If AR drops below 0.6, we're seeing a sliver -> don't use for distance.

        Args:
            bbox: [x, y, w, h]

        Returns:
            True if depth measurement should not be trusted
        """
        w, h = bbox[2], bbox[3]

        if h == 0:
            return True

        aspect_ratio = w / h

        # Sliver detection (vertical or horizontal)
        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            return True

        return False

    def correct_distance_estimate(
        self,
        bbox_width: float,
        visible_ratio: float,
        focal_length: float = 1.0,
        known_width: float = 0.1
    ) -> Tuple[float, bool]:
        """
        Correct distance estimate accounting for partial occlusion.

        Distance formula: d = (focal_length * known_width) / bbox_width
        Problem: If bbox_width decreases due to clipping, d spikes.

        Args:
            bbox_width: Current bounding box width in pixels
            visible_ratio: Estimated fraction of target visible (0-1)
            focal_length: Camera focal length
            known_width: Known physical width of target (meters)

        Returns:
            (corrected_distance, is_reliable) tuple
        """
        # If bbox width is very small or zero, measurement is unreliable
        if bbox_width < 5:  # Minimum reasonable pixel width
            if self._last_good_distance is not None:
                return self._last_good_distance, False
            else:
                return 1.0, False  # Default distance

        # Calculate raw distance
        raw_distance = (focal_length * known_width) / bbox_width

        # If visible ratio is low (partial occlusion), correct the estimate
        if visible_ratio < 0.7:
            # Bbox is clipped, so apparent width is smaller than actual
            # Correct by scaling up the width
            corrected_width = bbox_width / max(visible_ratio, 0.3)
            corrected_distance = (focal_length * known_width) / corrected_width

            # Cache as last good distance
            self._last_good_distance = corrected_distance
            return corrected_distance, False  # Not fully reliable

        # Measurement seems good
        self._last_good_distance = raw_distance
        return raw_distance, True

    def get_locked_distance(self) -> Optional[float]:
        """
        Get last known good distance (for use during full occlusion).

        Returns:
            Last good distance or None
        """
        return self._last_good_distance

    def reset_history(self):
        """Reset aspect ratio history (e.g., when track is lost)."""
        self._aspect_ratio_history = []
        self._last_good_distance = None
        self._last_good_bbox = None


class MeasurementQualityFilter:
    """
    Filter that decides whether to use a measurement or rely on prediction.

    During partial occlusion:
    - Flag measurement as UNRELIABLE
    - Lock distance to last known good value
    - Rely on Kalman prediction until full shape returns
    """

    def __init__(self, edge_clipper: EdgeClipper):
        """
        Args:
            edge_clipper: EdgeClipper instance for occlusion detection
        """
        self.edge_clipper = edge_clipper

    def filter_measurement(
        self,
        bbox: np.ndarray,
        use_for_update: bool = True
    ) -> Tuple[np.ndarray, bool]:
        """
        Filter a measurement and decide if it should be used for Kalman update.

        Args:
            bbox: Bounding box [x, y, w, h]
            use_for_update: Whether this measurement should be used

        Returns:
            (filtered_bbox, is_reliable) tuple
        """
        # Check for partial occlusion
        is_occluded = self.edge_clipper.check_partial_occlusion(bbox)

        if is_occluded:
            # Measurement is unreliable, use last good bbox if available
            if self.edge_clipper._last_good_bbox is not None:
                # Return last good bbox, mark as unreliable
                return self.edge_clipper._last_good_bbox, False
            else:
                # No history, use current but mark unreliable
                return bbox, False

        # Measurement is good, cache it
        self.edge_clipper._last_good_bbox = bbox.copy()
        return bbox, True

    def should_skip_measurement(self, bbox: np.ndarray) -> bool:
        """
        Determine if measurement should be skipped entirely (use prediction only).

        Args:
            bbox: Bounding box [x, y, w, h]

        Returns:
            True if measurement should be skipped
        """
        # Skip if bbox shows signs of severe clipping
        if self.edge_clipper.is_bad_depth_measurement(bbox):
            return True

        return False
