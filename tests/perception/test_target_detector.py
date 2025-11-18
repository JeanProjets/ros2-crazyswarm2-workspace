"""
Tests for target detection system.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'src'))

import pytest
import numpy as np
from perception.target_detector import TargetDetector, Detection


class TestTargetDetector:
    """Test suite for TargetDetector."""

    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return TargetDetector(drone_id='cf1')

    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.drone_id == 'cf1'
        assert detector.confidence_threshold == 0.7
        assert detector.frame_count == 0
        assert detector.detection_count == 0

    def test_detect_drone_empty_frame(self, detector):
        """Test detection on empty frame."""
        frame = np.zeros((120, 160), dtype=np.uint8)
        detections = detector.detect_drone(frame)
        assert isinstance(detections, list)
        assert detector.frame_count == 1

    def test_detect_drone_with_target(self, detector):
        """Test detection with target present."""
        # Create frame with bright blob in center
        frame = np.random.randint(20, 60, (120, 160), dtype=np.uint8)
        frame[50:70, 70:90] = 200  # Add target blob

        detections = detector.detect_drone(frame)
        assert isinstance(detections, list)

    def test_preprocess_frame(self, detector):
        """Test frame preprocessing."""
        frame = np.random.randint(0, 255, (120, 160), dtype=np.uint8)
        processed = detector.preprocess_frame(frame)
        assert processed.shape == frame.shape
        assert processed.dtype == np.uint8

    def test_find_blobs_fast(self, detector):
        """Test fast blob detection."""
        # Create binary image with blob
        binary = np.zeros((120, 160), dtype=np.uint8)
        binary[50:70, 70:90] = 255

        blobs = detector.find_blobs_fast(binary)
        assert isinstance(blobs, list)

    def test_is_drone_blob_valid(self, detector):
        """Test valid drone blob check."""
        blob = {
            'bbox': (70, 50, 20, 20),
            'area': 300,
            'points': [(i, j) for i in range(50, 70) for j in range(70, 90)]
        }
        frame = np.zeros((120, 160), dtype=np.uint8)

        result = detector.is_drone_blob(blob, frame)
        assert isinstance(result, bool)

    def test_is_drone_blob_invalid_aspect(self, detector):
        """Test blob with invalid aspect ratio."""
        blob = {
            'bbox': (70, 50, 5, 40),  # Very tall
            'area': 150,
            'points': []
        }
        frame = np.zeros((120, 160), dtype=np.uint8)

        result = detector.is_drone_blob(blob, frame)
        assert result is False

    def test_estimate_distance(self, detector):
        """Test distance estimation."""
        distance = detector.estimate_distance(bbox_width=30, bbox_height=30)
        assert isinstance(distance, float)
        assert 0.3 <= distance <= 3.0

    def test_estimate_distance_zero_width(self, detector):
        """Test distance estimation with zero width."""
        distance = detector.estimate_distance(bbox_width=0, bbox_height=20)
        assert distance == detector.max_detection_range

    def test_classify_drone_type(self, detector):
        """Test drone type classification."""
        blob = {'bbox': (70, 50, 20, 20), 'area': 300}
        frame = np.zeros((120, 160), dtype=np.uint8)
        frame[50:70, 70:90] = 150

        drone_type = detector.classify_drone_type(blob, frame)
        assert drone_type in ['hostile', 'friendly', 'unknown']

    def test_create_detection(self, detector):
        """Test detection object creation."""
        blob = {
            'bbox': (70, 50, 20, 20),
            'area': 300,
            'points': []
        }
        frame = np.zeros((120, 160), dtype=np.uint8)

        detection = detector.create_detection(blob, frame)
        assert isinstance(detection, Detection)
        assert detection.bbox == (70, 50, 20, 20)
        assert 0 <= detection.confidence <= 1.0
        assert detection.estimated_distance > 0

    def test_track_target_no_previous(self, detector):
        """Test tracking without previous detection."""
        frame = np.zeros((120, 160), dtype=np.uint8)
        result = detector.track_target(None, frame)
        assert result is None

    def test_get_detection_stats(self, detector):
        """Test getting detection statistics."""
        # Run some detections
        frame = np.zeros((120, 160), dtype=np.uint8)
        detector.detect_drone(frame)
        detector.detect_drone(frame)

        stats = detector.get_detection_stats()
        assert 'total_frames' in stats
        assert 'total_detections' in stats
        assert stats['total_frames'] == 2

    def test_histogram_equalization(self, detector):
        """Test histogram equalization."""
        image = np.random.randint(50, 150, (120, 160), dtype=np.uint8)
        equalized = detector._simple_histogram_equalization(image)
        assert equalized.shape == image.shape
        assert equalized.dtype == np.uint8

    def test_fast_blur(self, detector):
        """Test fast blur."""
        image = np.random.randint(0, 255, (120, 160), dtype=np.uint8)
        blurred = detector._fast_blur(image, kernel_size=3)
        assert blurred.shape == image.shape
        assert blurred.dtype == np.uint8


class TestDetection:
    """Test Detection dataclass."""

    def test_detection_creation(self):
        """Test creating Detection object."""
        det = Detection(
            bbox=(10, 20, 30, 40),
            confidence=0.85,
            drone_type='hostile',
            estimated_distance=1.5,
            timestamp=12345.0,
            frame_id=100
        )

        assert det.bbox == (10, 20, 30, 40)
        assert det.confidence == 0.85
        assert det.drone_type == 'hostile'
        assert det.estimated_distance == 1.5
        assert det.frame_id == 100
