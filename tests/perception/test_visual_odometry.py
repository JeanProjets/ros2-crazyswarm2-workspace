"""
Tests for visual positioning and odometry.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'src'))

import pytest
import numpy as np
from perception.visual_odometry import (
    VisualPositionEstimator,
    CameraCalibration,
    CameraCalibrationData
)
from perception.target_detector import Detection
import time


class TestCameraCalibration:
    """Test suite for CameraCalibration."""

    @pytest.fixture
    def calibration(self):
        """Create calibration instance."""
        return CameraCalibration(drone_id='cf1')

    def test_initialization(self, calibration):
        """Test calibration initialization."""
        assert calibration.drone_id == 'cf1'
        assert calibration.calib.focal_length_px == 120.0

    def test_pixel_to_world(self, calibration):
        """Test pixel to world coordinate conversion."""
        pixel_coords = (80, 60)  # Image center
        depth = 1.5

        world_coords = calibration.pixel_to_world(pixel_coords, depth)
        assert isinstance(world_coords, tuple)
        assert len(world_coords) == 3
        assert world_coords[2] == depth

    def test_world_to_pixel(self, calibration):
        """Test world to pixel coordinate conversion."""
        world_coords = (0.0, 0.0, 1.5)
        pixel_coords = calibration.world_to_pixel(world_coords)
        assert isinstance(pixel_coords, tuple)
        assert len(pixel_coords) == 2

    def test_world_to_pixel_zero_depth(self, calibration):
        """Test conversion with zero depth."""
        world_coords = (1.0, 1.0, 0.0)
        pixel_coords = calibration.world_to_pixel(world_coords)
        assert pixel_coords == (0, 0)

    def test_calculate_drone_bearing(self, calibration):
        """Test bearing calculation."""
        bbox_center = (100, 60)  # Right of center
        bearing = calibration.calculate_drone_bearing(bbox_center)
        assert isinstance(bearing, float)
        assert -90 <= bearing <= 90

    def test_calculate_drone_bearing_center(self, calibration):
        """Test bearing at image center."""
        bbox_center = (80, 60)  # Center
        bearing = calibration.calculate_drone_bearing(bbox_center)
        assert abs(bearing) < 1.0  # Should be close to zero

    def test_get_calibration_data(self, calibration):
        """Test getting calibration data."""
        calib_data = calibration.get_calibration_data()
        assert isinstance(calib_data, CameraCalibrationData)


class TestVisualPositionEstimator:
    """Test suite for VisualPositionEstimator."""

    @pytest.fixture
    def estimator(self):
        """Create estimator instance."""
        return VisualPositionEstimator(drone_id='cf1')

    def test_initialization(self, estimator):
        """Test estimator initialization."""
        assert estimator.drone_id == 'cf1'
        assert len(estimator.position_history) == 0

    def test_estimate_relative_position(self, estimator):
        """Test relative position estimation."""
        target_bbox = (70, 50, 20, 20)
        camera_params = {'focal_length_px': 120.0}
        estimated_distance = 1.5

        rel_pos = estimator.estimate_relative_position(
            target_bbox, camera_params, estimated_distance
        )
        assert isinstance(rel_pos, tuple)
        assert len(rel_pos) == 3

    def test_calculate_approach_vector(self, estimator):
        """Test approach vector calculation."""
        current_pos = (1.0, 1.0, 4.0)
        target_pos = (7.5, 3.0, 5.0)

        vector = estimator.calculate_approach_vector(current_pos, target_pos)
        assert isinstance(vector, np.ndarray)
        assert len(vector) == 3

        # Check normalization
        norm = np.linalg.norm(vector)
        assert abs(norm - 1.0) < 0.01

    def test_estimate_target_velocity_insufficient_data(self, estimator):
        """Test velocity estimation with insufficient data."""
        detections = []
        velocity = estimator.estimate_target_velocity(detections)
        assert velocity == (0.0, 0.0, 0.0)

    def test_estimate_target_velocity(self, estimator):
        """Test velocity estimation with detections."""
        det1 = Detection(
            bbox=(70, 50, 20, 20),
            confidence=0.8,
            drone_type='hostile',
            estimated_distance=1.5,
            timestamp=time.time(),
            frame_id=1
        )

        det2 = Detection(
            bbox=(75, 52, 20, 20),
            confidence=0.8,
            drone_type='hostile',
            estimated_distance=1.4,
            timestamp=time.time() + 0.1,
            frame_id=2
        )

        velocity = estimator.estimate_target_velocity([det1, det2])
        assert isinstance(velocity, tuple)
        assert len(velocity) == 3

    def test_validate_with_optitrack_agree(self, estimator):
        """Test validation when positions agree."""
        visual_pos = (7.5, 3.0, 5.0)
        optitrack_pos = (7.4, 3.1, 5.0)

        result = estimator.validate_with_optitrack(visual_pos, optitrack_pos)
        assert result is True

    def test_validate_with_optitrack_disagree(self, estimator):
        """Test validation when positions disagree."""
        visual_pos = (7.5, 3.0, 5.0)
        optitrack_pos = (5.0, 1.0, 3.0)

        result = estimator.validate_with_optitrack(visual_pos, optitrack_pos)
        assert result is False

    def test_transform_to_world_frame(self, estimator):
        """Test camera to world frame transformation."""
        camera_frame_pos = (0.0, 0.0, 1.5)
        drone_world_pos = (2.5, 2.5, 4.0)
        drone_yaw = 0.0

        world_pos = estimator.transform_to_world_frame(
            camera_frame_pos, drone_world_pos, drone_yaw
        )
        assert isinstance(world_pos, tuple)
        assert len(world_pos) == 3

    def test_transform_to_world_frame_with_rotation(self, estimator):
        """Test transformation with yaw rotation."""
        camera_frame_pos = (1.0, 0.0, 0.0)
        drone_world_pos = (0.0, 0.0, 0.0)
        drone_yaw = 90.0  # 90 degree rotation

        world_pos = estimator.transform_to_world_frame(
            camera_frame_pos, drone_world_pos, drone_yaw
        )
        # After 90° rotation, x becomes y
        assert abs(world_pos[1] - 1.0) < 0.1

    def test_estimate_world_position(self, estimator):
        """Test world position estimation."""
        detection = Detection(
            bbox=(70, 50, 20, 20),
            confidence=0.8,
            drone_type='hostile',
            estimated_distance=1.5,
            timestamp=time.time(),
            frame_id=1
        )

        drone_world_pos = (2.5, 2.5, 4.0)
        drone_yaw = 0.0

        world_pos = estimator.estimate_world_position(
            detection, drone_world_pos, drone_yaw
        )
        assert isinstance(world_pos, tuple)
        assert len(world_pos) == 3
        assert len(estimator.position_history) == 1

    def test_get_smoothed_position_insufficient_data(self, estimator):
        """Test smoothed position with insufficient data."""
        result = estimator.get_smoothed_position(window_size=5)
        assert result is None

    def test_get_smoothed_position(self, estimator):
        """Test smoothed position calculation."""
        # Add some positions to history
        for i in range(10):
            estimator.position_history.append((7.5 + i*0.1, 3.0, 5.0))

        smoothed = estimator.get_smoothed_position(window_size=5)
        assert smoothed is not None
        assert isinstance(smoothed, tuple)
        assert len(smoothed) == 3
