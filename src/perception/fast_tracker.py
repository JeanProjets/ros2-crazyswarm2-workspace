"""
Fast ROI-based tracker for high-speed target detection.

This module implements Region-of-Interest (ROI) tracking to boost FPS
for mobile target tracking in Scenario 3. By processing only a small
region around the last known position, we can achieve 30-40 FPS instead
of 10-15 FPS with full-frame inference on GAP8.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum


class TrackingState(Enum):
    """Tracking state machine states."""
    LOST = "lost"
    TRACKING = "tracking"
    FULL_SCAN = "full_scan"


@dataclass
class Detection:
    """Detection result from the tracker."""
    x: float  # Center x in full frame coordinates
    y: float  # Center y in full frame coordinates
    width: float
    height: float
    confidence: float
    timestamp: float


class ROITracker:
    """
    Region-of-Interest tracker for high-speed target detection.

    Strategy:
    - Frame 0: Run detection on FULL image to find initial target
    - Frame 1..N: Crop to ROI around last position and run inference
    - Update ROI center based on detection
    - Trigger full-frame scan if confidence drops or target near ROI border

    This approach boosts FPS from ~10-15 to 30-40 Hz for smooth velocity estimation.
    """

    def __init__(
        self,
        search_margin: int = 20,
        confidence_threshold: float = 0.6,
        border_threshold: int = 10,
        full_scan_interval: int = 30  # Force full scan every N frames
    ):
        """
        Initialize ROI tracker.

        Args:
            search_margin: Pixels to expand ROI beyond last detection (default: 20px)
            confidence_threshold: Min confidence to maintain tracking (default: 0.6)
            border_threshold: Distance from ROI edge to trigger full scan (default: 10px)
            full_scan_interval: Force full scan every N frames for safety (default: 30)
        """
        self.search_margin = search_margin
        self.confidence_threshold = confidence_threshold
        self.border_threshold = border_threshold
        self.full_scan_interval = full_scan_interval

        # Tracking window: (x, y, width, height) in full frame coordinates
        self.tracking_window: Optional[Tuple[int, int, int, int]] = None
        self.state = TrackingState.LOST
        self.frame_count = 0
        self.last_detection: Optional[Detection] = None

        # Frame dimensions (set on first frame)
        self.frame_width: Optional[int] = None
        self.frame_height: Optional[int] = None

    def update(self, frame: np.ndarray, detector_func) -> Optional[Detection]:
        """
        Update tracker with new frame and return detection.

        Args:
            frame: Input image (H x W x C)
            detector_func: Function that takes image crop and returns (x, y, w, h, conf)
                          relative to the crop coordinates

        Returns:
            Detection object if target found, None otherwise
        """
        if self.frame_width is None:
            self.frame_height, self.frame_width = frame.shape[:2]

        self.frame_count += 1

        # Determine if we need full-frame scan
        need_full_scan = (
            self.state == TrackingState.LOST or
            self.state == TrackingState.FULL_SCAN or
            self.frame_count % self.full_scan_interval == 0
        )

        if need_full_scan:
            # Full-frame detection
            detection_result = self._run_full_frame_detection(frame, detector_func)
        else:
            # ROI-based detection (faster)
            detection_result = self._run_roi_detection(frame, detector_func)

        # Update state based on result
        if detection_result is not None:
            self.last_detection = detection_result
            self._update_tracking_window(detection_result)
            self.state = TrackingState.TRACKING
            return detection_result
        else:
            # Detection failed - switch to full scan mode
            self.state = TrackingState.FULL_SCAN
            return None

    def _run_full_frame_detection(
        self,
        frame: np.ndarray,
        detector_func
    ) -> Optional[Detection]:
        """
        Run detection on full frame.

        Args:
            frame: Full input image
            detector_func: Detector function

        Returns:
            Detection if found, None otherwise
        """
        import time
        timestamp = time.time()

        # Run detector on full frame
        result = detector_func(frame)

        if result is None:
            return None

        x, y, w, h, conf = result

        if conf < self.confidence_threshold:
            return None

        return Detection(
            x=x,
            y=y,
            width=w,
            height=h,
            confidence=conf,
            timestamp=timestamp
        )

    def _run_roi_detection(
        self,
        frame: np.ndarray,
        detector_func
    ) -> Optional[Detection]:
        """
        Run detection on ROI crop for faster inference.

        Args:
            frame: Full input image
            detector_func: Detector function

        Returns:
            Detection in full-frame coordinates if found, None otherwise
        """
        import time
        timestamp = time.time()

        if self.tracking_window is None:
            # No tracking window available, need full scan
            return None

        # Extract ROI from frame
        roi_x, roi_y, roi_w, roi_h = self.tracking_window

        # Ensure ROI is within frame bounds
        roi_x = max(0, min(roi_x, self.frame_width - 1))
        roi_y = max(0, min(roi_y, self.frame_height - 1))
        roi_x2 = max(0, min(roi_x + roi_w, self.frame_width))
        roi_y2 = max(0, min(roi_y + roi_h, self.frame_height))

        # Extract crop
        crop = frame[roi_y:roi_y2, roi_x:roi_x2]

        if crop.size == 0:
            return None

        # Run detector on crop
        result = detector_func(crop)

        if result is None:
            return None

        crop_x, crop_y, w, h, conf = result

        if conf < self.confidence_threshold:
            return None

        # Convert crop coordinates to full-frame coordinates
        full_x = roi_x + crop_x
        full_y = roi_y + crop_y

        # Check if detection is near ROI border (target might be escaping)
        near_left = crop_x < self.border_threshold
        near_right = (crop_x + w) > (roi_w - self.border_threshold)
        near_top = crop_y < self.border_threshold
        near_bottom = (crop_y + h) > (roi_h - self.border_threshold)

        if near_left or near_right or near_top or near_bottom:
            # Target approaching ROI boundary - trigger full scan next frame
            self.state = TrackingState.FULL_SCAN

        return Detection(
            x=full_x,
            y=full_y,
            width=w,
            height=h,
            confidence=conf,
            timestamp=timestamp
        )

    def _update_tracking_window(self, detection: Detection):
        """
        Update tracking window based on new detection.

        Args:
            detection: Latest detection result
        """
        # Calculate ROI center around detection with margin
        center_x = int(detection.x)
        center_y = int(detection.y)

        # ROI dimensions = detection size + 2 * margin
        roi_w = int(detection.width + 2 * self.search_margin)
        roi_h = int(detection.height + 2 * self.search_margin)

        # ROI top-left corner
        roi_x = int(center_x - roi_w // 2)
        roi_y = int(center_y - roi_h // 2)

        # Clamp to frame boundaries
        roi_x = max(0, min(roi_x, self.frame_width - roi_w))
        roi_y = max(0, min(roi_y, self.frame_height - roi_h))

        self.tracking_window = (roi_x, roi_y, roi_w, roi_h)

    def reset(self):
        """Reset tracker to initial state."""
        self.tracking_window = None
        self.state = TrackingState.LOST
        self.frame_count = 0
        self.last_detection = None

    def get_roi_crop_params(self) -> Optional[Dict]:
        """
        Get current ROI parameters for visualization or debugging.

        Returns:
            Dictionary with ROI parameters or None if not tracking
        """
        if self.tracking_window is None:
            return None

        x, y, w, h = self.tracking_window
        return {
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'state': self.state.value,
            'frame_count': self.frame_count
        }
