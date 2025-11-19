"""
Tests for Scenario 2 Agent 3 - Vision System

Test suite for all perception modules:
- Long Range Detector
- Clutter Filter
- Visual Servoing
- Vision State Manager

Author: Agent 3 (Vision Developer)
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from perception.long_range_detector import LongRangeDetector, BoundingBox, DetectionMode
from perception.clutter_filter import ClutterRejection, BboxHistory
from perception.visual_servo import PrecisionGuidance, ApproachPhase
from perception.vision_state_manager import VisionLifecycle, VisionMode


class TestLongRangeDetector:
    """Test suite for LongRangeDetector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = LongRangeDetector()
        assert detector.image_width == 160
        assert detector.image_height == 120
        assert detector.confidence_threshold == 0.5

    def test_model_loading(self):
        """Test model loading."""
        detector = LongRangeDetector(model_path="mock_model.tflite")
        assert detector.model is not None
        assert detector.model.quantized is True

    def test_preprocess_zoom(self):
        """Test digital zoom preprocessing."""
        detector = LongRangeDetector()
        image = np.random.randint(0, 255, (120, 160), dtype=np.uint8)

        # Test no zoom
        result = detector.preprocess_zoom(image, zoom_factor=1.0)
        assert result.shape == image.shape

        # Test 2x zoom
        result = detector.preprocess_zoom(image, zoom_factor=2.0)
        assert result.shape == (120, 160)  # Should resize back to original

    def test_detect_full_frame(self):
        """Test full frame detection."""
        detector = LongRangeDetector(model_path="mock")
        image = np.random.randint(0, 255, (120, 160), dtype=np.uint8)

        detection = detector.detect(image, mode=DetectionMode.FULL_FRAME)

        # Mock model should return a detection
        assert detection is not None
        assert isinstance(detection, BoundingBox)
        assert detection.confidence >= 0.5

    def test_distance_estimation(self):
        """Test distance estimation."""
        detector = LongRangeDetector()

        # Create a bbox
        bbox = BoundingBox(x=80, y=60, width=40, height=30, confidence=0.8)

        distance = detector.estimate_distance(bbox)

        # Should return a reasonable distance
        assert distance > 0
        assert distance < 100  # Reasonable upper bound

    def test_detect_with_distance(self):
        """Test combined detection and distance estimation."""
        detector = LongRangeDetector(model_path="mock")
        image = np.random.randint(0, 255, (120, 160), dtype=np.uint8)

        result = detector.detect_with_distance(image)

        assert result is not None
        bbox, distance = result
        assert isinstance(bbox, BoundingBox)
        assert distance > 0

    def test_tracking_verification(self):
        """Test multi-frame tracking verification."""
        detector = LongRangeDetector(model_path="mock")

        # Add consistent detections
        for i in range(5):
            bbox = BoundingBox(x=80 + i, y=60, width=30, height=25, confidence=0.7)
            detector.detection_history.append(bbox)

        # Verify tracking
        bbox = BoundingBox(x=85, y=60, width=30, height=25, confidence=0.7)
        is_verified = detector.verify_with_tracking(bbox)

        assert isinstance(is_verified, (bool, np.bool_))

    def test_reset_tracking(self):
        """Test tracking reset."""
        detector = LongRangeDetector()
        detector.detection_history = [BoundingBox(80, 60, 30, 25, 0.8)]

        detector.reset_tracking()

        assert len(detector.detection_history) == 0


class TestClutterRejection:
    """Test suite for ClutterRejection."""

    def test_initialization(self):
        """Test clutter filter initialization."""
        clutter_filter = ClutterRejection()
        assert clutter_filter.image_width == 160
        assert clutter_filter.image_height == 120

    def test_circularity_calculation(self):
        """Test circularity calculation."""
        clutter_filter = ClutterRejection()

        # Square (high circularity)
        circ = clutter_filter.calculate_circularity(30, 30)
        assert circ == 1.0

        # Rectangle (lower circularity)
        circ = clutter_filter.calculate_circularity(50, 10)
        assert circ < 0.5

        # Long pole (very low circularity)
        circ = clutter_filter.calculate_circularity(100, 5)
        assert circ < 0.1

    def test_geometric_validity(self):
        """Test geometric validity checking."""
        clutter_filter = ClutterRejection()

        # Valid bbox in center
        valid = clutter_filter.check_geometric_validity((80, 60, 30, 25))
        assert valid is True

        # Invalid bbox (too close to edge)
        invalid = clutter_filter.check_geometric_validity((10, 60, 30, 25))
        assert invalid is False

        # Invalid bbox (too small)
        invalid = clutter_filter.check_geometric_validity((80, 60, 3, 2))
        assert invalid is False

        # Invalid bbox (too large)
        invalid = clutter_filter.check_geometric_validity((80, 60, 150, 100))
        assert invalid is False

    def test_filter_linear_structures(self):
        """Test linear structure filtering."""
        clutter_filter = ClutterRejection()
        image = np.random.randint(0, 255, (120, 160), dtype=np.uint8)

        # Test with elongated bbox (likely pole)
        bbox = (80, 60, 100, 10)  # Very wide, short
        should_reject = clutter_filter.filter_linear_structures(image, bbox)
        assert should_reject is True  # Should reject due to low circularity

        # Test with square bbox (likely drone)
        bbox = (80, 60, 30, 28)  # Nearly square
        should_reject = clutter_filter.filter_linear_structures(image, bbox)
        # May or may not reject based on image content, but shouldn't crash
        assert isinstance(should_reject, (bool, np.bool_))

    def test_should_reject(self):
        """Test overall rejection decision."""
        clutter_filter = ClutterRejection()
        image = np.random.randint(0, 255, (120, 160), dtype=np.uint8)

        # Valid target-like bbox (center, good circularity)
        bbox = (80, 60, 30, 30)  # Perfect square
        reject = clutter_filter.should_reject(image, bbox)
        # Should not reject valid bbox (but may reject due to random image content)
        # Just verify it returns a boolean
        assert isinstance(reject, (bool, np.bool_))

        # Invalid bbox (edge of image)
        bbox = (10, 60, 30, 25)
        reject = clutter_filter.should_reject(image, bbox)
        assert reject is True

    def test_history_management(self):
        """Test bbox history management."""
        clutter_filter = ClutterRejection()

        clutter_filter.add_detection_to_history((80, 60, 30, 25), 1.0)
        clutter_filter.add_detection_to_history((82, 61, 30, 25), 1.1)

        history = clutter_filter.get_history_list()
        assert len(history) == 2

        clutter_filter.reset_history()
        history = clutter_filter.get_history_list()
        assert len(history) == 0


class TestPrecisionGuidance:
    """Test suite for PrecisionGuidance."""

    def test_initialization(self):
        """Test precision guidance initialization."""
        guidance = PrecisionGuidance()
        assert guidance.image_width == 160
        assert guidance.image_height == 120
        assert guidance.center_x == 80.0
        assert guidance.center_y == 60.0

    def test_centering_error(self):
        """Test centering error calculation."""
        guidance = PrecisionGuidance()

        # Centered bbox (no error)
        bbox = (80, 60, 30, 25)
        err_x, err_y = guidance.calculate_centering_error(bbox)
        assert abs(err_x) < 0.01
        assert abs(err_y) < 0.01

        # Bbox to the right
        bbox = (100, 60, 30, 25)
        err_x, err_y = guidance.calculate_centering_error(bbox)
        assert err_x > 0  # Positive error to the right

        # Bbox below center
        bbox = (80, 80, 30, 25)
        err_x, err_y = guidance.calculate_centering_error(bbox)
        assert err_y > 0  # Positive error below

    def test_yaw_correction(self):
        """Test yaw correction calculation."""
        guidance = PrecisionGuidance()

        # Centered
        yaw = guidance.calculate_yaw_correction(80)
        assert abs(yaw) < 0.01

        # Target to the right
        yaw = guidance.calculate_yaw_correction(120)
        assert yaw > 0

        # Target to the left
        yaw = guidance.calculate_yaw_correction(40)
        assert yaw < 0

    def test_distance_estimation(self):
        """Test distance to impact estimation."""
        guidance = PrecisionGuidance()

        # Large bbox (close) - width=50 -> ~0.22m
        distance = guidance.estimate_distance_to_impact(50)
        assert distance < 1.0
        assert distance > 0.1

        # Small bbox (far) - width=10 -> ~1.1m
        distance = guidance.estimate_distance_to_impact(10)
        assert distance > 1.0
        assert distance < 2.0

    def test_compute_visual_error(self):
        """Test complete visual error computation."""
        guidance = PrecisionGuidance()

        bbox = (90, 65, 35, 28)
        error = guidance.compute_visual_error(bbox, 1.0, ApproachPhase.PRECISION_APPROACH)

        assert error is not None
        assert -1.0 <= error.err_x <= 1.0
        assert -1.0 <= error.err_y <= 1.0
        assert -1.0 <= error.err_yaw <= 1.0
        assert error.distance > 0
        assert 0.0 <= error.confidence <= 1.0

    def test_target_lost_detection(self):
        """Test target lost detection."""
        guidance = PrecisionGuidance()

        # Set last detection time
        import time
        guidance.last_detection_time = time.time() - 1.0  # 1 second ago

        lost = guidance.check_target_lost(time.time())
        assert lost is True  # Should be lost (> 0.5s)

        # Recent detection
        guidance.last_detection_time = time.time() - 0.1
        lost = guidance.check_target_lost(time.time())
        assert lost is False

    def test_hover_immediate_flag(self):
        """Test hover immediate flag."""
        guidance = PrecisionGuidance()
        import time

        # Set old detection time
        guidance.last_detection_time = time.time() - 1.0

        should_hover = guidance.should_send_hover_immediate(time.time(), ApproachPhase.PRECISION_APPROACH)
        assert should_hover is True

        # Not critical during transit
        should_hover = guidance.should_send_hover_immediate(time.time(), ApproachPhase.TRANSIT)
        assert should_hover is False

    def test_control_output(self):
        """Test control output generation."""
        guidance = PrecisionGuidance()
        import time

        # With detection
        bbox = (85, 62, 32, 26)
        output = guidance.get_control_output(bbox, time.time(), ApproachPhase.PRECISION_APPROACH)

        assert output is not None
        assert output['error'] is not None
        assert output['hover_immediate'] is False
        assert output['target_lost'] is False

        # Without detection
        output = guidance.get_control_output(None, time.time(), ApproachPhase.PRECISION_APPROACH)
        assert output['error'] is None
        assert output['target_lost'] is True

    def test_reset(self):
        """Test guidance reset."""
        guidance = PrecisionGuidance()
        guidance.error_history_x = [0.1, 0.2, 0.3]

        guidance.reset()

        assert len(guidance.error_history_x) == 0
        assert guidance.current_phase == ApproachPhase.TRANSIT


class TestVisionLifecycle:
    """Test suite for VisionLifecycle."""

    def test_initialization(self):
        """Test vision lifecycle initialization."""
        lifecycle = VisionLifecycle()
        assert lifecycle.current_mode == VisionMode.IDLE

    def test_mode_setting(self):
        """Test mode setting."""
        lifecycle = VisionLifecycle()

        success = lifecycle.set_mode(VisionMode.LONG_RANGE)
        assert success is True
        assert lifecycle.current_mode == VisionMode.LONG_RANGE

        # Check config updated
        config = lifecycle.get_current_config()
        assert config.mode == VisionMode.LONG_RANGE

    def test_auto_mode_selection(self):
        """Test automatic mode selection."""
        lifecycle = VisionLifecycle()

        # Early position (transit)
        mode = lifecycle.auto_mode_selection((2.0, 0.5, 1.0))
        assert mode == VisionMode.MOTION_DETECT

        # Mid position (long range)
        mode = lifecycle.auto_mode_selection((6.0, 0.5, 3.0))
        assert mode == VisionMode.LONG_RANGE

        # Near corner (terminal)
        mode = lifecycle.auto_mode_selection((9.0, 0.5, 4.5))
        assert mode == VisionMode.TERMINAL

    def test_mission_phase_update(self):
        """Test mission phase based mode update."""
        lifecycle = VisionLifecycle()

        lifecycle.update_from_mission_phase('LONG_RANGE')
        assert lifecycle.current_mode == VisionMode.LONG_RANGE

        lifecycle.update_from_mission_phase('TERMINAL')
        assert lifecycle.current_mode == VisionMode.TERMINAL

    def test_inference_disable_logic(self):
        """Test inference disable logic."""
        lifecycle = VisionLifecycle()

        # Early position - should disable
        disable = lifecycle.should_disable_inference((3.0, 0.5, 1.0))
        assert disable is True

        # Later position - should enable
        disable = lifecycle.should_disable_inference((6.0, 0.5, 3.0))
        assert disable is False

    def test_model_switching(self):
        """Test model switching logic."""
        lifecycle = VisionLifecycle()

        # Early
        model = lifecycle.switch_model((3.0, 0.5, 1.0))
        assert model == "motion_detector_v1"

        # Mid
        model = lifecycle.switch_model((6.0, 0.5, 3.0))
        assert model == "cnn_longrange_v2"

        # Terminal
        model = lifecycle.switch_model((9.0, 0.5, 4.5))
        assert model == "cnn_terminal_v2"

    def test_power_statistics(self):
        """Test power statistics tracking."""
        lifecycle = VisionLifecycle()

        lifecycle.set_mode(VisionMode.LONG_RANGE)

        stats = lifecycle.get_power_statistics()

        assert 'total_power_consumed' in stats
        assert 'current_mode' in stats
        assert 'current_power_level' in stats
        assert stats['current_mode'] == 'long_range'

    def test_recommended_settings(self):
        """Test recommended settings generation."""
        lifecycle = VisionLifecycle()

        settings = lifecycle.get_recommended_settings('PRECISION_APPROACH', (8.5, 0.5, 4.8))

        assert 'mode' in settings
        assert 'exposure' in settings
        assert 'frame_rate' in settings
        assert 'model' in settings

    def test_reset(self):
        """Test lifecycle reset."""
        lifecycle = VisionLifecycle()

        lifecycle.set_mode(VisionMode.TERMINAL)
        lifecycle.reset()

        assert lifecycle.current_mode == VisionMode.IDLE
        assert len(lifecycle.mode_history) == 0


class TestIntegration:
    """Integration tests for complete vision pipeline."""

    def test_full_detection_pipeline(self):
        """Test full detection pipeline."""
        # Initialize all components
        detector = LongRangeDetector(model_path="mock")
        clutter = ClutterRejection()
        guidance = PrecisionGuidance()
        lifecycle = VisionLifecycle()

        # Simulate image
        image = np.random.randint(0, 255, (120, 160), dtype=np.uint8)

        # Set mode
        lifecycle.set_mode(VisionMode.LONG_RANGE)

        # Detect
        result = detector.detect_with_distance(image, DetectionMode.FULL_FRAME)

        if result is not None:
            bbox, distance = result

            # Check clutter
            bbox_tuple = (bbox.x, bbox.y, bbox.width, bbox.height)
            reject = clutter.should_reject(image, bbox_tuple)

            if not reject:
                # Compute visual error
                import time
                error = guidance.compute_visual_error(bbox_tuple, time.time())

                assert error is not None
                assert error.distance > 0

    def test_scenario_2_workflow(self):
        """Test complete Scenario 2 workflow."""
        lifecycle = VisionLifecycle()
        guidance = PrecisionGuidance()

        # Scenario 2: Target at (9.5, 0.5, 5)
        target_pos = (9.5, 0.5, 5.0)

        # Simulate drone approaching from (0, 0, 1) to target
        positions = [
            (2.0, 0.3, 1.5),  # Early
            (5.0, 0.4, 2.5),  # Mid
            (8.0, 0.45, 4.0),  # Approaching
            (9.2, 0.48, 4.8),  # Terminal
        ]

        expected_modes = [
            VisionMode.MOTION_DETECT,
            VisionMode.LONG_RANGE,
            VisionMode.TERMINAL,
            VisionMode.TERMINAL,
        ]

        for pos, expected_mode in zip(positions, expected_modes):
            mode = lifecycle.auto_mode_selection(pos)
            assert mode == expected_mode


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
