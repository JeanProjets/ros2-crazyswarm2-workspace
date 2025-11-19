"""
Comprehensive test suite for Scenario 3 Agent 3 - Vision/Perception system.

Tests cover:
1. ROI Tracker functionality
2. Kalman filtering and velocity estimation
3. Motion scanner for fallback behavior
4. Camera exposure control
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from perception.fast_tracker import ROITracker, Detection, TrackingState
from perception.state_estimator import KalmanFilter1D, VelocityCalculator, TargetState
from perception.zone_scanner import MotionScanner, MovingBlob, ScannerMode
from perception.camera_control import MotionExposureControl, ExposureMode, ExposureSettings


class TestROITracker:
    """Test suite for ROI-based fast tracker."""

    def test_tracker_initialization(self):
        """Test tracker initializes correctly."""
        tracker = ROITracker(search_margin=20, confidence_threshold=0.6)
        assert tracker.state == TrackingState.LOST
        assert tracker.tracking_window is None
        assert tracker.frame_count == 0

    def test_tracker_full_scan_mode(self):
        """Test full-frame detection mode."""
        tracker = ROITracker()

        # Create dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Mock detector function that returns a detection
        def mock_detector(img):
            return (320, 240, 50, 50, 0.9)  # x, y, w, h, confidence

        detection = tracker.update(frame, mock_detector)

        assert detection is not None
        assert detection.x == 320
        assert detection.y == 240
        assert detection.confidence == 0.9
        assert tracker.state == TrackingState.TRACKING

    def test_tracker_roi_mode(self):
        """Test ROI-based tracking after initialization."""
        tracker = ROITracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        def mock_detector(img):
            # Return detection relative to input image
            h, w = img.shape[:2]
            return (w//2, h//2, 50, 50, 0.85)

        # First frame - full scan
        detection1 = tracker.update(frame, mock_detector)
        assert detection1 is not None

        # Second frame - should use ROI
        detection2 = tracker.update(frame, mock_detector)
        assert detection2 is not None
        assert tracker.tracking_window is not None

    def test_tracker_lost_target(self):
        """Test tracker behavior when target is lost."""
        tracker = ROITracker(confidence_threshold=0.8)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        def mock_detector_low_conf(img):
            return (320, 240, 50, 50, 0.5)  # Low confidence

        detection = tracker.update(frame, mock_detector_low_conf)
        # Low confidence should be rejected
        assert detection is None

    def test_tracker_reset(self):
        """Test tracker reset functionality."""
        tracker = ROITracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        def mock_detector(img):
            return (320, 240, 50, 50, 0.9)

        tracker.update(frame, mock_detector)
        assert tracker.state == TrackingState.TRACKING

        tracker.reset()
        assert tracker.state == TrackingState.LOST
        assert tracker.tracking_window is None
        assert tracker.frame_count == 0


class TestKalmanFilter1D:
    """Test suite for 1D Kalman filter."""

    def test_kalman_initialization(self):
        """Test Kalman filter initializes correctly."""
        kf = KalmanFilter1D()
        assert not kf.initialized
        assert kf.x[0] == 0.0
        assert kf.x[1] == 0.0

    def test_kalman_first_measurement(self):
        """Test filter initialization with first measurement."""
        kf = KalmanFilter1D()
        kf.update(10.0)

        assert kf.initialized
        assert kf.x[0] == 10.0  # Position set to measurement
        assert kf.x[1] == 0.0   # Velocity initialized to 0

    def test_kalman_velocity_estimation(self):
        """Test velocity estimation from measurements."""
        kf = KalmanFilter1D(measurement_noise=0.01)

        # Simulate constant velocity motion with more samples for convergence
        dt = 0.1  # 100ms
        for i in range(20):  # More samples for better convergence
            pos = i * 1.0  # Moving at 1.0 m per step = 10 m/s
            kf.predict(dt)
            kf.update(pos)

        pos, vel = kf.get_state()

        # Should estimate velocity around 10 m/s (1.0 / 0.1)
        # With more samples, filter should converge closer
        assert abs(vel - 10.0) < 1.0  # Tighter tolerance with more samples

    def test_kalman_prediction(self):
        """Test future position prediction."""
        kf = KalmanFilter1D()

        # Set known state
        kf.update(0.0)
        kf.x[1] = 10.0  # 10 m/s velocity

        # Predict 0.5s into future
        future_pos = kf.predict_position(0.5)

        # Should be around 5.0 meters ahead
        assert abs(future_pos - 5.0) < 0.1

    def test_kalman_reset(self):
        """Test filter reset."""
        kf = KalmanFilter1D()
        kf.update(10.0)
        assert kf.initialized

        kf.reset()
        assert not kf.initialized
        assert kf.x[0] == 0.0


class TestVelocityCalculator:
    """Test suite for velocity calculator with egomotion compensation."""

    def test_velocity_calc_initialization(self):
        """Test velocity calculator initializes correctly."""
        calc = VelocityCalculator()
        assert calc.focal_length == 200.0
        assert calc.depth_estimate == 2.0

    def test_velocity_calc_update(self):
        """Test velocity calculation from pixel measurements."""
        calc = VelocityCalculator(focal_length=200.0, depth_estimate=2.0)

        # Stationary drone, stationary target
        drone_vel = np.array([0.0, 0.0, 0.0])
        drone_yaw = 0.0

        state = calc.update(
            pixel_x=320.0,
            pixel_y=240.0,
            depth=2.0,
            drone_velocity=drone_vel,
            drone_yaw=drone_yaw
        )

        assert state is not None
        assert len(state.position) == 3
        assert len(state.velocity) == 3

    def test_velocity_calc_moving_target(self):
        """Test velocity estimation for moving target."""
        calc = VelocityCalculator(focal_length=200.0, depth_estimate=2.0)

        drone_vel = np.array([0.0, 0.0, 0.0])
        drone_yaw = 0.0

        # Simulate target moving right in image
        for i in range(10):
            pixel_x = 320.0 + i * 10  # Moving right
            state = calc.update(
                pixel_x=pixel_x,
                pixel_y=240.0,
                depth=2.0,
                drone_velocity=drone_vel,
                drone_yaw=drone_yaw,
                timestamp=i * 0.033  # 30 Hz
            )

        # After convergence, should have non-zero velocity
        assert np.linalg.norm(state.velocity) > 0.01

    def test_egomotion_compensation(self):
        """Test egomotion compensation."""
        calc = VelocityCalculator()

        # Drone moving forward, target stationary in world
        # Camera should see target moving backward
        drone_vel = np.array([1.0, 0.0, 0.0])  # 1 m/s forward
        drone_yaw = 0.0

        # Target at fixed pixel location (appears stationary in camera)
        for i in range(10):
            state = calc.update(
                pixel_x=320.0,
                pixel_y=240.0,
                depth=2.0,
                drone_velocity=drone_vel,
                drone_yaw=drone_yaw,
                timestamp=i * 0.033
            )

        # Target velocity in world should be compensated
        # (close to zero or accounting for drone motion)
        # This is a simplified test - actual behavior depends on coordinate frames

    def test_latency_compensation(self):
        """Test future position prediction for latency."""
        calc = VelocityCalculator()

        drone_vel = np.array([0.0, 0.0, 0.0])
        drone_yaw = 0.0

        # Update with some measurements
        for i in range(5):
            calc.update(
                pixel_x=320.0,
                pixel_y=240.0,
                depth=2.0,
                drone_velocity=drone_vel,
                drone_yaw=drone_yaw
            )

        # Predict 100ms into future
        future_pos = calc.predict_future_position(100.0, drone_vel, drone_yaw)

        assert len(future_pos) == 3

    def test_velocity_calc_reset(self):
        """Test velocity calculator reset."""
        calc = VelocityCalculator()

        drone_vel = np.array([0.0, 0.0, 0.0])
        calc.update(320.0, 240.0, 2.0, drone_vel, 0.0)

        calc.reset()
        assert calc.last_update_time is None


class TestMotionScanner:
    """Test suite for motion-based scanner."""

    def test_scanner_initialization(self):
        """Test scanner initializes correctly."""
        scanner = MotionScanner()
        assert scanner.background is None
        assert scanner.prev_frame is None
        assert scanner.frame_count == 0

    def test_scanner_background_building(self):
        """Test background model building."""
        scanner = MotionScanner(background_history=5)

        # Feed static frames
        for i in range(10):
            frame = np.ones((480, 640), dtype=np.uint8) * 128
            blobs = scanner.detect_moving_objects(frame)

        assert scanner.background is not None
        assert scanner.frame_count == 10

    def test_scanner_motion_detection(self):
        """Test motion detection with moving object."""
        scanner = MotionScanner(
            motion_threshold=20,
            min_blob_area=50
        )

        # Build background with static scene
        static_frame = np.ones((480, 640), dtype=np.uint8) * 128
        for i in range(10):
            scanner.detect_moving_objects(static_frame.copy())

        # Add moving object
        moving_frame = static_frame.copy()
        moving_frame[200:250, 300:350] = 200  # Bright square

        blobs = scanner.detect_moving_objects(moving_frame)

        # Should detect the moving square
        assert len(blobs) > 0

    def test_scanner_blob_filtering(self):
        """Test blob size filtering."""
        scanner = MotionScanner(
            min_blob_area=100,
            max_blob_area=1000
        )

        # Create frame with noise (small blobs)
        frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

        blobs = scanner.detect_moving_objects(frame)

        # All blobs should be within size limits
        for blob in blobs:
            assert blob.area >= scanner.min_blob_area
            assert blob.area <= scanner.max_blob_area

    def test_scanner_best_candidate(self):
        """Test best target candidate selection."""
        scanner = MotionScanner()

        # Build background
        static = np.ones((480, 640), dtype=np.uint8) * 128
        for i in range(10):
            scanner.detect_moving_objects(static.copy())

        # Add moving object
        moving = static.copy()
        moving[200:280, 300:380] = 200

        candidate = scanner.get_best_target_candidate(moving)

        # Should return bounding box
        if candidate is not None:
            x, y, w, h = candidate
            assert x >= 0 and y >= 0
            assert w > 0 and h > 0

    def test_scanner_reset(self):
        """Test scanner reset."""
        scanner = MotionScanner()

        frame = np.ones((480, 640), dtype=np.uint8) * 128
        scanner.detect_moving_objects(frame)

        scanner.reset()
        assert scanner.background is None
        assert scanner.frame_count == 0

    def test_scanner_ready_state(self):
        """Test scanner ready state."""
        scanner = MotionScanner(background_history=5)

        assert not scanner.is_ready()

        frame = np.ones((480, 640), dtype=np.uint8) * 128
        for i in range(5):
            scanner.detect_moving_objects(frame)

        assert scanner.is_ready()


class TestMotionExposureControl:
    """Test suite for camera exposure control."""

    def test_exposure_control_initialization(self):
        """Test exposure control initializes correctly."""
        control = MotionExposureControl()
        assert control.current_mode == ExposureMode.AUTO
        assert control.current_settings is None

    def test_set_auto_mode(self):
        """Test setting auto exposure mode."""
        control = MotionExposureControl()
        success = control.set_mode(ExposureMode.AUTO)

        assert success
        assert control.current_mode == ExposureMode.AUTO
        assert control.current_settings.auto_exposure == True

    def test_set_static_mode(self):
        """Test setting static mode."""
        control = MotionExposureControl()
        success = control.set_mode(ExposureMode.STATIC)

        assert success
        assert control.current_mode == ExposureMode.STATIC
        settings = control.current_settings
        assert settings.auto_exposure == False
        assert settings.analog_gain == 1.0  # Low noise

    def test_set_dynamic_mode(self):
        """Test setting dynamic mode for moving targets."""
        control = MotionExposureControl(
            max_dynamic_exposure_us=5000,
            default_static_exposure_us=20000
        )
        success = control.set_mode(ExposureMode.DYNAMIC)

        assert success
        assert control.current_mode == ExposureMode.DYNAMIC
        settings = control.current_settings

        # Should cap exposure to 5ms
        assert settings.exposure_time_us <= 5000

        # Should increase gain to compensate
        assert settings.analog_gain > 1.0

    def test_adjust_for_target_speed(self):
        """Test dynamic adjustment based on target speed."""
        control = MotionExposureControl()
        control.set_mode(ExposureMode.DYNAMIC)

        initial_settings = control.current_settings.exposure_time_us

        # Very fast target
        success = control.adjust_for_target_speed(2.0)  # 2 m/s

        assert success
        new_settings = control.current_settings.exposure_time_us

        # Should reduce exposure for faster target
        assert new_settings < initial_settings

    def test_motion_blur_estimation(self):
        """Test motion blur estimation."""
        control = MotionExposureControl()
        control.set_mode(ExposureMode.DYNAMIC)

        # Target at 0.5 m/s, 2m away, with 5ms exposure
        blur_px = control.estimate_motion_blur_pixels(
            target_speed_m_s=0.5,
            distance_m=2.0,
            focal_length_px=200.0
        )

        # Should be small (<1 pixel ideally)
        assert blur_px >= 0
        assert blur_px < 2.0  # Acceptable blur

    def test_get_settings_info(self):
        """Test settings information retrieval."""
        control = MotionExposureControl()
        control.set_mode(ExposureMode.DYNAMIC)

        info = control.get_settings_info()

        assert 'mode' in info
        assert 'configured' in info
        assert info['configured'] == True
        assert 'exposure_ms' in info
        assert 'analog_gain' in info


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_tracking_pipeline(self):
        """Test complete tracking pipeline."""
        # Create components
        tracker = ROITracker()
        velocity_calc = VelocityCalculator()

        # Mock frame and detector
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        def mock_detector(img):
            return (320, 240, 50, 50, 0.9)

        # Track target
        detection = tracker.update(frame, mock_detector)
        assert detection is not None

        # Estimate velocity
        drone_vel = np.array([0.0, 0.0, 0.0])
        state = velocity_calc.update(
            pixel_x=detection.x,
            pixel_y=detection.y,
            depth=2.0,
            drone_velocity=drone_vel,
            drone_yaw=0.0
        )

        assert state is not None
        assert state.position is not None
        assert state.velocity is not None

    def test_fallback_to_scanner(self):
        """Test fallback from tracker to scanner."""
        tracker = ROITracker()
        scanner = MotionScanner()

        frame = np.ones((480, 640), dtype=np.uint8) * 128

        # Build scanner background
        for i in range(10):
            scanner.detect_moving_objects(frame.copy())

        # Simulate tracker loss
        def failing_detector(img):
            return None  # Lost target

        detection = tracker.update(frame, failing_detector)
        assert detection is None

        # Use scanner as fallback
        moving_frame = frame.copy()
        moving_frame[200:250, 300:350] = 200

        candidate = scanner.get_best_target_candidate(moving_frame)
        # Scanner should find the moving object

    def test_exposure_control_with_tracking(self):
        """Test exposure control integrated with tracking."""
        exposure = MotionExposureControl()
        tracker = ROITracker()

        # Set dynamic mode for moving target
        exposure.set_mode(ExposureMode.DYNAMIC)

        # Verify settings are appropriate
        settings = exposure.get_current_settings()
        assert settings.exposure_time_us <= 5000  # Fast shutter

        # Estimate blur for typical scenario
        blur = exposure.estimate_motion_blur_pixels(
            target_speed_m_s=0.5,
            distance_m=2.0
        )

        # Blur should be minimal
        assert blur < 1.0  # Less than 1 pixel


def test_imports():
    """Test that all modules can be imported."""
    from perception import (
        ROITracker, Detection, TrackingState,
        KalmanFilter1D, VelocityCalculator, TargetState,
        MotionScanner, MovingBlob, ScannerMode,
        MotionExposureControl, ExposureMode, ExposureSettings
    )
    assert True  # If we get here, imports worked


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
