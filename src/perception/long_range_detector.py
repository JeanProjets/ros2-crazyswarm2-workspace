"""
Long Range Detector for Scenario 2 - Vision System

This module implements detection logic optimized for smaller objects at distance.
Designed for AI Deck 1.1 (GAP8) with 160x120 Greyscale camera.

Author: Agent 3 (Vision Developer)
Scenario: 2 - Corner Target Detection
"""

import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum


@dataclass
class BoundingBox:
    """Represents a detected target bounding box."""
    x: float  # Center x coordinate
    y: float  # Center y coordinate
    width: float
    height: float
    confidence: float

    def to_dict(self) -> Dict:
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'confidence': self.confidence
        }


class DetectionMode(Enum):
    """Detection modes for different ranges."""
    FULL_FRAME = "full_frame"
    ROI_SCAN = "roi_scan"
    DIGITAL_ZOOM = "digital_zoom"


class LongRangeDetector:
    """
    Long-range target detector optimized for small object detection.

    Features:
    - Digital zoom/ROI scanning for distant targets
    - Multi-frame tracking for verification
    - Quantized model support for GAP8
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the long-range detector.

        Args:
            model_path: Path to quantized model (optional for mocking)
        """
        self.model_path = model_path
        self.model = None
        self.confidence_threshold = 0.5  # Lower for long-range
        self.image_width = 160
        self.image_height = 120
        self.detection_history = []
        self.max_history = 5

        # ROI scanning parameters
        self.roi_overlap = 0.2  # 20% overlap between scan windows
        self.roi_size = (80, 60)  # Half-resolution scanning window

        # Distance estimation parameters
        self.real_width_mm = 92.0  # Crazyflie width
        self.focal_length_px = 120.0  # Calibrated focal length

        if model_path:
            self.model = self.load_model_v2_corner()

    def load_model_v2_corner(self) -> object:
        """
        Load quantized model optimized for corner detection.

        Returns:
            Quantized model object (mocked for now)
        """
        # Mock implementation for environments without GAP8
        # In production, this would load a TFLite or GAP8-optimized model
        class MockModel:
            def __init__(self):
                self.quantized = True
                self.input_shape = (120, 160, 1)

            def predict(self, image: np.ndarray) -> Tuple[BoundingBox, float]:
                """Mock prediction - simulates detection."""
                # Simulate a detection in the center with some noise
                h, w = image.shape[:2]
                center_x = w // 2 + np.random.randint(-10, 10)
                center_y = h // 2 + np.random.randint(-10, 10)
                bbox_width = np.random.randint(20, 40)
                bbox_height = np.random.randint(15, 30)
                confidence = 0.7 + np.random.random() * 0.2

                bbox = BoundingBox(
                    x=center_x,
                    y=center_y,
                    width=bbox_width,
                    height=bbox_height,
                    confidence=confidence
                )
                return bbox, confidence

        return MockModel()

    def preprocess_zoom(self, image: np.ndarray, zoom_factor: float = 1.0) -> np.ndarray:
        """
        Apply digital zoom to the input image.

        Args:
            image: Input greyscale image (120x160)
            zoom_factor: Zoom factor (1.0 = no zoom, 2.0 = 2x zoom)

        Returns:
            Cropped and resized image
        """
        if zoom_factor <= 1.0:
            return image

        h, w = image.shape[:2]

        # Calculate crop dimensions
        crop_h = int(h / zoom_factor)
        crop_w = int(w / zoom_factor)

        # Calculate crop center
        center_y = h // 2
        center_x = w // 2

        # Extract ROI
        y1 = max(0, center_y - crop_h // 2)
        y2 = min(h, center_y + crop_h // 2)
        x1 = max(0, center_x - crop_w // 2)
        x2 = min(w, center_x + crop_w // 2)

        cropped = image[y1:y2, x1:x2]

        # Resize back to original size for model input
        # In production, use cv2.resize or hardware-accelerated resize
        from scipy.ndimage import zoom as scipy_zoom
        zoom_y = h / cropped.shape[0]
        zoom_x = w / cropped.shape[1]
        resized = scipy_zoom(cropped, (zoom_y, zoom_x), order=1)

        return resized

    def scan_with_roi(self, image: np.ndarray) -> Optional[BoundingBox]:
        """
        Perform sliding window ROI scan for small targets.

        Args:
            image: Input greyscale image

        Returns:
            Best detected bounding box or None
        """
        h, w = image.shape[:2]
        roi_h, roi_w = self.roi_size

        # Calculate step sizes with overlap
        step_h = int(roi_h * (1 - self.roi_overlap))
        step_w = int(roi_w * (1 - self.roi_overlap))

        best_detection = None
        best_confidence = 0.0

        # Sliding window scan
        for y in range(0, h - roi_h + 1, step_h):
            for x in range(0, w - roi_w + 1, step_w):
                roi = image[y:y+roi_h, x:x+roi_w]

                # Resize ROI to full frame size for model
                roi_resized = self._resize_roi(roi, (h, w))

                # Run detection on ROI
                if self.model:
                    bbox, confidence = self.model.predict(roi_resized)

                    if confidence > best_confidence and confidence > self.confidence_threshold:
                        # Adjust bbox coordinates to original frame
                        bbox.x = x + bbox.x * (roi_w / w)
                        bbox.y = y + bbox.y * (roi_h / h)
                        bbox.width = bbox.width * (roi_w / w)
                        bbox.height = bbox.height * (roi_h / h)

                        best_detection = bbox
                        best_confidence = confidence

        return best_detection

    def _resize_roi(self, roi: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resize ROI to target shape."""
        from scipy.ndimage import zoom as scipy_zoom
        h, w = roi.shape[:2]
        target_h, target_w = target_shape

        zoom_y = target_h / h
        zoom_x = target_w / w

        return scipy_zoom(roi, (zoom_y, zoom_x), order=1)

    def detect(self, image: np.ndarray, mode: DetectionMode = DetectionMode.FULL_FRAME) -> Optional[BoundingBox]:
        """
        Detect target in the image.

        Args:
            image: Input greyscale image (120x160)
            mode: Detection mode to use

        Returns:
            Detected bounding box or None
        """
        if image is None or image.size == 0:
            return None

        # Ensure correct shape
        if len(image.shape) == 3:
            image = image[:, :, 0]  # Take first channel if color

        detection = None

        if mode == DetectionMode.FULL_FRAME:
            # Standard full-frame detection
            if self.model:
                detection, confidence = self.model.predict(image)
                if confidence < self.confidence_threshold:
                    detection = None

        elif mode == DetectionMode.ROI_SCAN:
            # Sliding window ROI scan
            detection = self.scan_with_roi(image)

        elif mode == DetectionMode.DIGITAL_ZOOM:
            # Try with digital zoom
            zoomed = self.preprocess_zoom(image, zoom_factor=2.0)
            if self.model:
                detection, confidence = self.model.predict(zoomed)
                if confidence < self.confidence_threshold:
                    detection = None

        # Add to history for multi-frame tracking
        if detection:
            self.detection_history.append(detection)
            if len(self.detection_history) > self.max_history:
                self.detection_history.pop(0)

        return detection

    def verify_with_tracking(self, current_detection: Optional[BoundingBox]) -> bool:
        """
        Verify detection using multi-frame tracking.

        Args:
            current_detection: Current frame detection

        Returns:
            True if detection is verified across multiple frames
        """
        if current_detection is None:
            return False

        if len(self.detection_history) < 2:
            return False

        # Check if recent detections are consistent
        recent_detections = self.detection_history[-3:]

        # Calculate position variance
        positions = np.array([[d.x, d.y] for d in recent_detections])
        variance = np.var(positions, axis=0)

        # Low variance indicates stable tracking
        max_variance = 10.0  # pixels
        return np.all(variance < max_variance)

    def estimate_distance(self, bbox: BoundingBox) -> float:
        """
        Estimate distance to target using pinhole camera model.

        Args:
            bbox: Detected bounding box

        Returns:
            Estimated distance in meters
        """
        if bbox.width <= 0:
            return float('inf')

        # d = (real_width_mm * focal_length_px) / width_px
        dist_mm = (self.real_width_mm * self.focal_length_px) / bbox.width
        return dist_mm / 1000.0  # Convert to meters

    def detect_with_distance(self, image: np.ndarray, mode: DetectionMode = DetectionMode.FULL_FRAME) -> Optional[Tuple[BoundingBox, float]]:
        """
        Detect target and estimate distance.

        Args:
            image: Input greyscale image
            mode: Detection mode

        Returns:
            Tuple of (BoundingBox, distance) or None
        """
        detection = self.detect(image, mode)

        if detection is None:
            return None

        distance = self.estimate_distance(detection)

        return (detection, distance)

    def reset_tracking(self):
        """Reset detection history."""
        self.detection_history.clear()
