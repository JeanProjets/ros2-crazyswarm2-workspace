"""
AI Deck camera interface and GAP8 processor integration.

This module provides interfaces for the AI Deck 1.1 hardware including
camera control and GAP8 processor communication for real-time inference.
"""
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class CameraParams:
    """Camera calibration and specification parameters."""
    resolution: Tuple[int, int] = (160, 120)
    fps: int = 30
    fov_horizontal: float = 60.0  # degrees
    fov_vertical: float = 45.0    # degrees
    focal_length_px: float = 120.0
    is_monochrome: bool = True
    effective_range_min: float = 0.3  # meters
    effective_range_max: float = 3.0  # meters


class AIDeckCamera:
    """
    Interface for AI Deck 1.1 camera (Himax HM01B0).

    Provides camera control and frame capture for vision processing.
    Hardware specs: 160x120 monochrome, 60 FPS max.
    """

    def __init__(self, drone_id: str):
        """
        Initialize AI Deck camera.

        Args:
            drone_id: Unique identifier for the drone
        """
        self.drone_id = drone_id
        self.logger = logging.getLogger(f"AIDeckCamera_{drone_id}")

        # Camera state
        self.is_initialized = False
        self.is_streaming = False
        self.frame_count = 0

        # Camera parameters
        self.params = CameraParams()

        # Mock hardware state
        self.exposure_value = 128
        self.auto_exposure = True

        self.logger.info(f"AI Deck camera created for {drone_id}")

    def initialize_camera(
        self,
        resolution: Tuple[int, int] = (160, 120),
        fps: int = 30
    ) -> bool:
        """
        Initialize camera hardware.

        Args:
            resolution: Camera resolution (width, height)
            fps: Target frames per second

        Returns:
            True if initialization successful
        """
        try:
            self.logger.info(f"Initializing camera: {resolution} @ {fps} FPS")

            # Validate parameters
            if resolution[0] > 160 or resolution[1] > 120:
                self.logger.error("Resolution exceeds AI Deck maximum (160x120)")
                return False

            if fps > 60:
                self.logger.warning("FPS capped at 60 (hardware limit)")
                fps = 60

            # Set parameters
            self.params.resolution = resolution
            self.params.fps = fps

            # Mock hardware initialization
            time.sleep(0.1)  # Simulate initialization delay

            self.is_initialized = True
            self.logger.info("Camera initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from camera.

        Returns:
            Frame as numpy array (H x W) or None if capture failed
        """
        if not self.is_initialized:
            self.logger.warning("Camera not initialized")
            return None

        try:
            # Mock frame capture - generate synthetic frame
            width, height = self.params.resolution
            frame = self._generate_mock_frame(width, height)

            self.frame_count += 1
            return frame

        except Exception as e:
            self.logger.error(f"Frame capture failed: {e}")
            return None

    def get_camera_params(self) -> Dict:
        """
        Get camera calibration parameters.

        Returns:
            Dictionary of camera parameters
        """
        return {
            'resolution': self.params.resolution,
            'fps': self.params.fps,
            'fov_horizontal': self.params.fov_horizontal,
            'fov_vertical': self.params.fov_vertical,
            'focal_length_px': self.params.focal_length_px,
            'is_monochrome': self.params.is_monochrome,
            'effective_range': (self.params.effective_range_min,
                               self.params.effective_range_max)
        }

    def adjust_exposure(
        self,
        auto: bool = True,
        value: Optional[int] = None
    ) -> bool:
        """
        Adjust camera exposure settings.

        Args:
            auto: Enable auto-exposure
            value: Manual exposure value (0-255) if auto=False

        Returns:
            True if adjustment successful
        """
        try:
            self.auto_exposure = auto

            if not auto and value is not None:
                if 0 <= value <= 255:
                    self.exposure_value = value
                    self.logger.info(f"Exposure set to {value}")
                else:
                    self.logger.warning("Exposure value must be 0-255")
                    return False
            elif auto:
                self.logger.info("Auto-exposure enabled")

            return True

        except Exception as e:
            self.logger.error(f"Exposure adjustment failed: {e}")
            return False

    def stream_to_gap8(self, enable: bool = True) -> bool:
        """
        Enable/disable streaming to GAP8 processor.

        Args:
            enable: Enable streaming

        Returns:
            True if successful
        """
        try:
            if enable and not self.is_initialized:
                self.logger.error("Camera must be initialized before streaming")
                return False

            self.is_streaming = enable
            status = "enabled" if enable else "disabled"
            self.logger.info(f"GAP8 streaming {status}")
            return True

        except Exception as e:
            self.logger.error(f"Streaming control failed: {e}")
            return False

    def _generate_mock_frame(self, width: int, height: int) -> np.ndarray:
        """
        Generate synthetic camera frame for testing.

        Args:
            width: Frame width
            height: Frame height

        Returns:
            Synthetic grayscale frame
        """
        # Create base frame with noise
        frame = np.random.randint(20, 80, (height, width), dtype=np.uint8)

        # Add synthetic "target" blob in center area
        # Simulates a drone at ~1.5m distance
        if self.frame_count % 60 < 45:  # Target visible 75% of time
            cx, cy = width // 2, height // 2
            blob_size = 15  # Approximate size for drone at 1.5m

            y_start = max(0, cy - blob_size // 2)
            y_end = min(height, cy + blob_size // 2)
            x_start = max(0, cx - blob_size // 2)
            x_end = min(width, cx + blob_size // 2)

            frame[y_start:y_end, x_start:x_end] = 200

        return frame

    def get_frame_count(self) -> int:
        """Get total frames captured."""
        return self.frame_count


class GAP8Processor:
    """
    Interface for GAP8 processor on AI Deck.

    Handles model loading and inference execution on the 8-core GAP8 chip.
    Constraints: INT8 models only, max 400KB, target <100ms inference.
    """

    def __init__(self, drone_id: str):
        """
        Initialize GAP8 processor.

        Args:
            drone_id: Unique identifier for the drone
        """
        self.drone_id = drone_id
        self.logger = logging.getLogger(f"GAP8_{drone_id}")

        # Processor state
        self.model_loaded = False
        self.model_path = None

        # Performance tracking
        self.last_inference_time = 0.0
        self.inference_count = 0

        # Hardware constraints
        self.max_model_size_kb = 400
        self.max_inference_time_ms = 100

        self.logger.info(f"GAP8 processor initialized for {drone_id}")

    def load_model(self, model_path: str) -> bool:
        """
        Load TensorFlow Lite model onto GAP8.

        Args:
            model_path: Path to .tflite model file

        Returns:
            True if model loaded successfully
        """
        try:
            self.logger.info(f"Loading model: {model_path}")

            # Mock model loading - check file exists and size
            # In real implementation, this would load to GAP8 memory
            import os

            if not os.path.exists(model_path) and not model_path.endswith('.mock'):
                self.logger.warning(f"Model file not found (using mock): {model_path}")

            # Simulate model load time
            time.sleep(0.2)

            self.model_path = model_path
            self.model_loaded = True
            self.logger.info("Model loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            return False

    def run_inference(self, image: np.ndarray) -> Dict:
        """
        Run inference on GAP8 processor.

        Args:
            image: Input image (grayscale, H x W)

        Returns:
            Dictionary with detection results
        """
        if not self.model_loaded:
            self.logger.warning("No model loaded")
            return {'detections': [], 'inference_time': 0.0}

        try:
            start_time = time.time()

            # Mock inference - in real implementation, this runs on GAP8
            # Simulate processing time (optimized for GAP8)
            time.sleep(0.05)  # 50ms inference

            # Generate mock detection results
            detections = self._mock_inference(image)

            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            self.last_inference_time = inference_time
            self.inference_count += 1

            return {
                'detections': detections,
                'inference_time': inference_time,
                'timestamp': time.time()
            }

        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            return {'detections': [], 'inference_time': 0.0}

    def get_inference_time(self) -> float:
        """
        Get last inference time in milliseconds.

        Returns:
            Inference time in ms
        """
        return self.last_inference_time

    def optimize_for_gap8(self, model_path: str) -> Optional[str]:
        """
        Optimize model for GAP8 deployment.

        Args:
            model_path: Path to original model

        Returns:
            Path to optimized model or None if optimization failed
        """
        try:
            self.logger.info(f"Optimizing model for GAP8: {model_path}")

            # Mock optimization process
            # Real implementation would:
            # 1. Quantize to INT8
            # 2. Prune unused operations
            # 3. Optimize for GAP8 instruction set

            optimized_path = model_path.replace('.tflite', '_gap8.tflite')

            self.logger.info(f"Model optimized: {optimized_path}")
            return optimized_path

        except Exception as e:
            self.logger.error(f"Model optimization failed: {e}")
            return None

    def _mock_inference(self, image: np.ndarray) -> list:
        """
        Generate mock inference results for testing.

        Args:
            image: Input image

        Returns:
            List of mock detections
        """
        # Check for bright blob in center (simulated target)
        h, w = image.shape
        center_region = image[h//3:2*h//3, w//3:2*w//3]

        if np.mean(center_region) > 150:
            # Target detected
            return [{
                'bbox': [w//2 - 15, h//2 - 15, 30, 30],
                'confidence': 0.85,
                'class': 'drone'
            }]

        return []
