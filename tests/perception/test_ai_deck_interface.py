"""
Tests for AI Deck interface and GAP8 processor.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'src'))

import pytest
import numpy as np
from perception.ai_deck_interface import AIDeckCamera, GAP8Processor, CameraParams


class TestAIDeckCamera:
    """Test suite for AIDeckCamera."""

    @pytest.fixture
    def camera(self):
        """Create camera instance."""
        return AIDeckCamera(drone_id='cf1')

    def test_initialization(self, camera):
        """Test camera initialization."""
        assert camera.drone_id == 'cf1'
        assert not camera.is_initialized
        assert not camera.is_streaming
        assert camera.frame_count == 0

    def test_initialize_camera_success(self, camera):
        """Test successful camera initialization."""
        result = camera.initialize_camera(resolution=(160, 120), fps=30)
        assert result is True
        assert camera.is_initialized is True
        assert camera.params.resolution == (160, 120)
        assert camera.params.fps == 30

    def test_initialize_camera_invalid_resolution(self, camera):
        """Test camera initialization with invalid resolution."""
        result = camera.initialize_camera(resolution=(200, 200), fps=30)
        assert result is False

    def test_initialize_camera_high_fps(self, camera):
        """Test camera initialization with high FPS (should cap)."""
        result = camera.initialize_camera(resolution=(160, 120), fps=100)
        assert result is True
        assert camera.params.fps == 60  # Capped at 60

    def test_capture_frame_not_initialized(self, camera):
        """Test frame capture without initialization."""
        frame = camera.capture_frame()
        assert frame is None

    def test_capture_frame_success(self, camera):
        """Test successful frame capture."""
        camera.initialize_camera()
        frame = camera.capture_frame()
        assert frame is not None
        assert frame.shape == (120, 160)
        assert frame.dtype == np.uint8
        assert camera.frame_count == 1

    def test_get_camera_params(self, camera):
        """Test getting camera parameters."""
        camera.initialize_camera()
        params = camera.get_camera_params()
        assert 'resolution' in params
        assert 'fov_horizontal' in params
        assert 'focal_length_px' in params
        assert params['is_monochrome'] is True

    def test_adjust_exposure_auto(self, camera):
        """Test auto exposure adjustment."""
        result = camera.adjust_exposure(auto=True)
        assert result is True
        assert camera.auto_exposure is True

    def test_adjust_exposure_manual(self, camera):
        """Test manual exposure adjustment."""
        result = camera.adjust_exposure(auto=False, value=150)
        assert result is True
        assert camera.auto_exposure is False
        assert camera.exposure_value == 150

    def test_adjust_exposure_invalid_value(self, camera):
        """Test exposure adjustment with invalid value."""
        result = camera.adjust_exposure(auto=False, value=300)
        assert result is False

    def test_stream_to_gap8_not_initialized(self, camera):
        """Test streaming without initialization."""
        result = camera.stream_to_gap8(enable=True)
        assert result is False

    def test_stream_to_gap8_success(self, camera):
        """Test successful streaming enable."""
        camera.initialize_camera()
        result = camera.stream_to_gap8(enable=True)
        assert result is True
        assert camera.is_streaming is True

    def test_generate_mock_frame(self, camera):
        """Test mock frame generation."""
        frame = camera._generate_mock_frame(160, 120)
        assert frame.shape == (120, 160)
        assert frame.dtype == np.uint8


class TestGAP8Processor:
    """Test suite for GAP8Processor."""

    @pytest.fixture
    def processor(self):
        """Create GAP8 processor instance."""
        return GAP8Processor(drone_id='cf1')

    def test_initialization(self, processor):
        """Test processor initialization."""
        assert processor.drone_id == 'cf1'
        assert not processor.model_loaded
        assert processor.max_model_size_kb == 400
        assert processor.max_inference_time_ms == 100

    def test_load_model_success(self, processor):
        """Test successful model loading."""
        result = processor.load_model('model.tflite.mock')
        assert result is True
        assert processor.model_loaded is True

    def test_run_inference_no_model(self, processor):
        """Test inference without loaded model."""
        image = np.zeros((120, 160), dtype=np.uint8)
        result = processor.run_inference(image)
        assert 'detections' in result
        assert len(result['detections']) == 0

    def test_run_inference_success(self, processor):
        """Test successful inference."""
        processor.load_model('model.tflite.mock')

        # Create image with bright center (simulated target)
        image = np.zeros((120, 160), dtype=np.uint8)
        image[40:80, 60:100] = 200

        result = processor.run_inference(image)
        assert 'detections' in result
        assert 'inference_time' in result
        assert result['inference_time'] > 0

    def test_get_inference_time(self, processor):
        """Test getting inference time."""
        processor.load_model('model.tflite.mock')
        image = np.zeros((120, 160), dtype=np.uint8)
        processor.run_inference(image)

        inference_time = processor.get_inference_time()
        assert inference_time > 0

    def test_optimize_for_gap8(self, processor):
        """Test model optimization."""
        optimized_path = processor.optimize_for_gap8('model.tflite')
        assert optimized_path is not None
        assert '_gap8' in optimized_path
