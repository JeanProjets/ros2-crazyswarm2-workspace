"""
Target Detection System for drone identification and tracking.

Implements optimized detection algorithms for GAP8 processor with
blob detection, distance estimation, and target classification.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import time
import logging


@dataclass
class Detection:
    """
    Detection result for a single target.

    Attributes:
        bbox: Bounding box (x, y, width, height) in pixels
        confidence: Detection confidence score (0-1)
        drone_type: Classification ('hostile', 'friendly', 'unknown')
        estimated_distance: Distance to target in meters
        timestamp: Detection timestamp
        frame_id: Frame number
    """
    bbox: Tuple[int, int, int, int]
    confidence: float
    drone_type: str
    estimated_distance: float
    timestamp: float
    frame_id: int = 0


class TargetDetector:
    """
    Target detection and tracking for drone swarm.

    Optimized for GAP8 processor constraints with efficient blob detection,
    size-based distance estimation, and simple classification.
    """

    def __init__(self, drone_id: str):
        """
        Initialize target detector.

        Args:
            drone_id: Unique identifier for the drone
        """
        self.drone_id = drone_id
        self.logger = logging.getLogger(f"TargetDetector_{drone_id}")

        # Detection parameters
        self.confidence_threshold = 0.7
        self.min_blob_size = 10  # pixels
        self.max_blob_size = 80  # pixels
        self.max_detection_range = 3.0  # meters

        # Camera parameters (Himax HM01B0)
        self.img_width = 160
        self.img_height = 120
        self.focal_length = 120  # pixels (approximate)
        self.drone_width_mm = 92  # Crazyflie dimensions

        # Tracking state
        self.last_detection: Optional[Detection] = None
        self.detection_history: List[Detection] = []
        self.max_history = 30  # Keep last 30 detections

        # Performance tracking
        self.frame_count = 0
        self.detection_count = 0

        self.logger.info(f"Target detector initialized for {drone_id}")

    def detect_drone(self, frame: np.ndarray) -> List[Detection]:
        """
        Main detection pipeline optimized for GAP8.

        Args:
            frame: Input grayscale image (H x W)

        Returns:
            List of detections found in frame
        """
        self.frame_count += 1
        detections = []

        try:
            # Preprocessing (keep simple for GAP8)
            processed = self.preprocess_frame(frame)

            # Find blob candidates
            blobs = self.find_blobs_fast(processed)

            # Classify each blob
            for blob in blobs:
                if self.is_drone_blob(blob, frame):
                    detection = self.create_detection(blob, frame)
                    if detection.confidence >= self.confidence_threshold:
                        detections.append(detection)
                        self.detection_count += 1

            # Update tracking
            if detections:
                self.last_detection = detections[0]
                self.detection_history.append(detections[0])

                # Maintain history size
                if len(self.detection_history) > self.max_history:
                    self.detection_history.pop(0)

            return detections

        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return []

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Minimal preprocessing for speed.

        Args:
            frame: Input grayscale frame

        Returns:
            Processed binary frame
        """
        try:
            # Histogram equalization for contrast (simple version)
            frame_eq = self._simple_histogram_equalization(frame)

            # Gaussian blur simulation (3x3) - use simple averaging
            frame_blur = self._fast_blur(frame_eq)

            # Thresholding for blob detection
            # Use adaptive threshold based on image statistics
            threshold = int(np.mean(frame_blur) + np.std(frame_blur))
            threshold = max(50, min(200, threshold))  # Clamp to reasonable range

            binary = np.where(frame_blur > threshold, 255, 0).astype(np.uint8)

            return binary

        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            # Return simple threshold as fallback
            return np.where(frame > 100, 255, 0).astype(np.uint8)

    def find_blobs_fast(self, binary_image: np.ndarray) -> List[dict]:
        """
        Fast blob detection without OpenCV (optimized for GAP8).

        Uses connected component analysis with optimized algorithm.

        Args:
            binary_image: Binary input image

        Returns:
            List of blob dictionaries with bbox and area
        """
        blobs = []
        h, w = binary_image.shape
        visited = np.zeros((h, w), dtype=bool)

        def flood_fill(start_y, start_x):
            """Simple flood fill to find connected component."""
            stack = [(start_y, start_x)]
            points = []

            while stack:
                y, x = stack.pop()

                if (y < 0 or y >= h or x < 0 or x >= w or
                    visited[y, x] or binary_image[y, x] == 0):
                    continue

                visited[y, x] = True
                points.append((y, x))

                # 4-connectivity
                stack.append((y+1, x))
                stack.append((y-1, x))
                stack.append((y, x+1))
                stack.append((y, x-1))

            return points

        # Find all blobs
        for y in range(h):
            for x in range(w):
                if binary_image[y, x] > 0 and not visited[y, x]:
                    points = flood_fill(y, x)

                    if len(points) >= self.min_blob_size:
                        # Calculate bounding box
                        ys = [p[0] for p in points]
                        xs = [p[1] for p in points]

                        bbox_x = min(xs)
                        bbox_y = min(ys)
                        bbox_w = max(xs) - bbox_x + 1
                        bbox_h = max(ys) - bbox_y + 1

                        # Filter by size
                        if (bbox_w >= self.min_blob_size and
                            bbox_h >= self.min_blob_size and
                            bbox_w <= self.max_blob_size and
                            bbox_h <= self.max_blob_size):

                            blobs.append({
                                'bbox': (bbox_x, bbox_y, bbox_w, bbox_h),
                                'area': len(points),
                                'points': points
                            })

        return blobs

    def is_drone_blob(self, blob: dict, frame: np.ndarray) -> bool:
        """
        Check if blob is likely a drone.

        Args:
            blob: Blob dictionary with bbox and area
            frame: Original frame for additional checks

        Returns:
            True if blob characteristics match drone
        """
        bbox = blob['bbox']
        area = blob['area']
        x, y, w, h = bbox

        # Size check - drones should be roughly square
        aspect_ratio = w / h if h > 0 else 0
        if not (0.5 <= aspect_ratio <= 2.0):
            return False

        # Area check - should be compact
        bbox_area = w * h
        if bbox_area == 0:
            return False

        compactness = area / bbox_area
        if compactness < 0.3:  # Too sparse
            return False

        # Position check - not at edges (likely noise)
        if x < 5 or y < 5 or x + w > self.img_width - 5 or y + h > self.img_height - 5:
            return False

        return True

    def create_detection(self, blob: dict, frame: np.ndarray) -> Detection:
        """
        Create Detection object from blob.

        Args:
            blob: Blob dictionary
            frame: Original frame

        Returns:
            Detection object
        """
        bbox = blob['bbox']
        area = blob['area']

        # Estimate distance from blob size
        distance = self.estimate_distance(bbox[2], bbox[3])

        # Classify drone type
        drone_type = self.classify_drone_type(blob, frame)

        # Calculate confidence based on multiple factors
        confidence = self._calculate_confidence(blob, distance)

        return Detection(
            bbox=bbox,
            confidence=confidence,
            drone_type=drone_type,
            estimated_distance=distance,
            timestamp=time.time(),
            frame_id=self.frame_count
        )

    def classify_drone_type(self, blob: dict, frame: np.ndarray) -> str:
        """
        Classify drone as hostile, friendly, or unknown.

        For Scenario 1: Simple heuristic-based classification.

        Args:
            blob: Blob dictionary
            frame: Original frame

        Returns:
            Drone type string
        """
        # For Scenario 1, we assume any detected drone is hostile
        # More sophisticated classification would analyze:
        # - LED patterns
        # - Motion patterns
        # - Communication signals

        x, y, w, h = blob['bbox']

        # Check average brightness in blob region (LED indicator)
        roi = frame[y:y+h, x:x+w]
        avg_brightness = np.mean(roi)

        # Simple heuristic: very bright = friendly (LED), dim = hostile
        if avg_brightness > 180:
            return 'friendly'
        elif avg_brightness > 100:
            return 'unknown'
        else:
            return 'hostile'

    def estimate_distance(
        self,
        bbox_width: int,
        bbox_height: int,
        known_drone_size: float = 0.092
    ) -> float:
        """
        Estimate distance using pinhole camera model.

        distance = (known_width_m * focal_length_px) / bbox_width_px

        Args:
            bbox_width: Bounding box width in pixels
            bbox_height: Bounding box height in pixels
            known_drone_size: Actual drone size in meters (default: 92mm)

        Returns:
            Estimated distance in meters
        """
        if bbox_width == 0:
            return self.max_detection_range

        # Use average of width and height for better estimate
        bbox_size = (bbox_width + bbox_height) / 2.0

        # Pinhole camera formula
        distance = (known_drone_size * self.focal_length) / bbox_size

        # Clamp to reasonable range
        distance = max(0.3, min(self.max_detection_range, distance))

        return distance

    def track_target(
        self,
        previous_detection: Detection,
        current_frame: np.ndarray
    ) -> Optional[Detection]:
        """
        Track target from previous detection.

        Uses simple tracking based on position prediction.

        Args:
            previous_detection: Last known detection
            current_frame: Current frame

        Returns:
            Updated detection or None if tracking lost
        """
        if previous_detection is None:
            return None

        # Predict search region based on previous position
        prev_bbox = previous_detection.bbox
        search_margin = 20  # pixels

        # Expand search region
        x = max(0, prev_bbox[0] - search_margin)
        y = max(0, prev_bbox[1] - search_margin)
        w = min(self.img_width - x, prev_bbox[2] + 2 * search_margin)
        h = min(self.img_height - y, prev_bbox[3] + 2 * search_margin)

        # Extract search region
        search_roi = current_frame[y:y+h, x:x+w]

        # Run detection on ROI
        detections = self.detect_drone(search_roi)

        if detections:
            # Adjust bbox coordinates back to full frame
            det = detections[0]
            adjusted_bbox = (
                det.bbox[0] + x,
                det.bbox[1] + y,
                det.bbox[2],
                det.bbox[3]
            )
            det.bbox = adjusted_bbox
            return det

        return None

    def _calculate_confidence(self, blob: dict, distance: float) -> float:
        """
        Calculate detection confidence score.

        Args:
            blob: Blob dictionary
            distance: Estimated distance

        Returns:
            Confidence score (0-1)
        """
        confidence = 0.5  # Base confidence

        # Size-based confidence
        x, y, w, h = blob['bbox']
        size_score = min(1.0, (w * h) / 400.0)  # Normalize by typical size
        confidence += 0.2 * size_score

        # Distance-based confidence (closer = more confident)
        if distance < 1.0:
            confidence += 0.2
        elif distance < 2.0:
            confidence += 0.1

        # Compactness-based confidence
        area = blob['area']
        bbox_area = w * h
        if bbox_area > 0:
            compactness = area / bbox_area
            confidence += 0.1 * compactness

        return min(1.0, confidence)

    def _simple_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Simple histogram equalization for contrast enhancement.

        Args:
            image: Input grayscale image

        Returns:
            Equalized image
        """
        # Calculate histogram
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])

        # Calculate cumulative distribution function
        cdf = hist.cumsum()
        cdf_normalized = cdf * 255 / cdf[-1]

        # Use linear interpolation to find new pixel values
        equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)
        equalized = equalized.reshape(image.shape).astype(np.uint8)

        return equalized

    def _fast_blur(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Fast blur using simple box filter (optimized for GAP8).

        Args:
            image: Input image
            kernel_size: Blur kernel size (default: 3)

        Returns:
            Blurred image
        """
        # Simple box filter implementation
        h, w = image.shape
        blurred = np.zeros_like(image)
        margin = kernel_size // 2

        for y in range(margin, h - margin):
            for x in range(margin, w - margin):
                region = image[y-margin:y+margin+1, x-margin:x+margin+1]
                blurred[y, x] = np.mean(region)

        return blurred.astype(np.uint8)

    def get_detection_stats(self) -> dict:
        """
        Get detection statistics.

        Returns:
            Dictionary with detection statistics
        """
        return {
            'total_frames': self.frame_count,
            'total_detections': self.detection_count,
            'detection_rate': self.detection_count / max(1, self.frame_count),
            'history_size': len(self.detection_history)
        }
