"""
Camera exposure control for motion scenarios.

This module provides smart exposure control to prevent motion blur when
tracking moving targets. For Scenario 3, we prioritize sharp edges over
low noise by using fast shutter speeds and increased gain.
"""

from enum import Enum
from typing import Optional, Dict
from dataclasses import dataclass


class ExposureMode(Enum):
    """Camera exposure modes."""
    AUTO = "auto"           # Standard auto-exposure
    STATIC = "static"       # Optimized for static targets
    DYNAMIC = "dynamic"     # Optimized for moving targets (Scenario 3)
    MANUAL = "manual"       # Manual control


@dataclass
class ExposureSettings:
    """Camera exposure settings."""
    exposure_time_us: int   # Exposure time in microseconds
    analog_gain: float      # Analog gain multiplier
    digital_gain: float     # Digital gain multiplier
    auto_exposure: bool     # Auto-exposure enabled


class MotionExposureControl:
    """
    Smart exposure control for motion scenarios.

    For moving targets (Scenario 3), we need to prevent motion blur which
    would degrade detection performance. Strategy:
    - Cap exposure time to <5ms to freeze motion
    - Increase analog gain to compensate for reduced light
    - Accept higher noise (noise is better than blur for detection)
    """

    def __init__(
        self,
        camera_interface=None,  # Mock-friendly: actual camera interface
        max_dynamic_exposure_us: int = 5000,  # 5ms max for motion
        min_exposure_us: int = 100,           # 0.1ms minimum
        max_analog_gain: float = 16.0,        # Max gain before too noisy
        default_static_exposure_us: int = 20000  # 20ms for static targets
    ):
        """
        Initialize exposure controller.

        Args:
            camera_interface: Hardware camera interface (can be None for mock)
            max_dynamic_exposure_us: Max exposure in microseconds for moving targets
            min_exposure_us: Minimum exposure in microseconds
            max_analog_gain: Maximum analog gain to use
            default_static_exposure_us: Default exposure for static scenarios
        """
        self.camera = camera_interface
        self.max_dynamic_exposure_us = max_dynamic_exposure_us
        self.min_exposure_us = min_exposure_us
        self.max_analog_gain = max_analog_gain
        self.default_static_exposure_us = default_static_exposure_us

        self.current_mode = ExposureMode.AUTO
        self.current_settings: Optional[ExposureSettings] = None

    def set_mode(self, mode: ExposureMode) -> bool:
        """
        Set camera exposure mode.

        Args:
            mode: Desired exposure mode

        Returns:
            True if mode was successfully set
        """
        self.current_mode = mode

        if mode == ExposureMode.AUTO:
            return self._configure_auto_mode()
        elif mode == ExposureMode.STATIC:
            return self._configure_static_mode()
        elif mode == ExposureMode.DYNAMIC:
            return self._configure_dynamic_mode()
        elif mode == ExposureMode.MANUAL:
            # Manual mode - don't change settings
            return True

        return False

    def _configure_auto_mode(self) -> bool:
        """
        Configure standard auto-exposure mode.

        Returns:
            True if successful
        """
        settings = ExposureSettings(
            exposure_time_us=0,  # Auto
            analog_gain=1.0,      # Auto
            digital_gain=1.0,     # Auto
            auto_exposure=True
        )

        return self._apply_settings(settings)

    def _configure_static_mode(self) -> bool:
        """
        Configure exposure for static target scenarios.

        Optimizes for low noise and good image quality.

        Returns:
            True if successful
        """
        settings = ExposureSettings(
            exposure_time_us=self.default_static_exposure_us,
            analog_gain=1.0,      # Minimal gain for low noise
            digital_gain=1.0,
            auto_exposure=False
        )

        return self._apply_settings(settings)

    def _configure_dynamic_mode(self) -> bool:
        """
        Configure exposure for moving target scenarios (Scenario 3).

        Strategy:
        - Cap exposure to <5ms to prevent motion blur
        - Increase analog gain to compensate for brightness
        - Noise is acceptable; blur is not

        Returns:
            True if successful
        """
        # Fast shutter to freeze motion
        exposure_time = self.max_dynamic_exposure_us

        # Calculate required gain to maintain brightness
        # Assuming auto-exposure would use 20ms, we're now using 5ms
        # Need ~4x gain to compensate
        brightness_compensation = self.default_static_exposure_us / exposure_time
        analog_gain = min(brightness_compensation, self.max_analog_gain)

        settings = ExposureSettings(
            exposure_time_us=exposure_time,
            analog_gain=analog_gain,
            digital_gain=1.0,
            auto_exposure=False
        )

        return self._apply_settings(settings)

    def _apply_settings(self, settings: ExposureSettings) -> bool:
        """
        Apply exposure settings to camera hardware.

        Args:
            settings: Exposure settings to apply

        Returns:
            True if successfully applied
        """
        self.current_settings = settings

        # If no camera interface (mock mode), just store settings
        if self.camera is None:
            return True

        # Apply to actual hardware
        try:
            if hasattr(self.camera, 'set_exposure'):
                self.camera.set_exposure(settings.exposure_time_us)

            if hasattr(self.camera, 'set_analog_gain'):
                self.camera.set_analog_gain(settings.analog_gain)

            if hasattr(self.camera, 'set_digital_gain'):
                self.camera.set_digital_gain(settings.digital_gain)

            if hasattr(self.camera, 'set_auto_exposure'):
                self.camera.set_auto_exposure(settings.auto_exposure)

            return True

        except Exception as e:
            print(f"Failed to apply camera settings: {e}")
            return False

    def adjust_for_target_speed(self, speed_m_s: float) -> bool:
        """
        Dynamically adjust exposure based on target speed.

        For very fast targets, we might need even shorter exposure.

        Args:
            speed_m_s: Target speed in meters per second

        Returns:
            True if settings were adjusted
        """
        if self.current_mode != ExposureMode.DYNAMIC:
            return False

        # Calculate motion blur distance during exposure
        # At 0.5 m/s and 5ms exposure: blur = 0.5 * 0.005 = 2.5mm
        # At typical 2m distance, this is ~0.25 pixels with AI Deck
        # Acceptable for most cases

        # For very fast targets (>1 m/s), reduce exposure further
        if speed_m_s > 1.0:
            # Reduce exposure proportionally
            desired_exposure = int(self.max_dynamic_exposure_us / (speed_m_s / 0.5))
            desired_exposure = max(self.min_exposure_us, desired_exposure)

            # Increase gain to compensate
            gain_factor = self.max_dynamic_exposure_us / desired_exposure
            desired_gain = min(gain_factor * 4.0, self.max_analog_gain)

            settings = ExposureSettings(
                exposure_time_us=desired_exposure,
                analog_gain=desired_gain,
                digital_gain=1.0,
                auto_exposure=False
            )

            return self._apply_settings(settings)

        return False

    def get_current_settings(self) -> Optional[ExposureSettings]:
        """
        Get current exposure settings.

        Returns:
            Current exposure settings or None if not configured
        """
        return self.current_settings

    def get_mode(self) -> ExposureMode:
        """
        Get current exposure mode.

        Returns:
            Current exposure mode
        """
        return self.current_mode

    def estimate_motion_blur_pixels(
        self,
        target_speed_m_s: float,
        distance_m: float,
        focal_length_px: float = 200.0
    ) -> float:
        """
        Estimate motion blur in pixels for current settings.

        Args:
            target_speed_m_s: Target speed in m/s
            distance_m: Distance to target in meters
            focal_length_px: Camera focal length in pixels

        Returns:
            Estimated motion blur in pixels
        """
        if self.current_settings is None:
            return 0.0

        # Motion during exposure time
        exposure_s = self.current_settings.exposure_time_us / 1e6
        motion_distance_m = target_speed_m_s * exposure_s

        # Project to image plane
        # blur_pixels = (motion_distance / distance) * focal_length
        if distance_m > 0:
            blur_px = (motion_distance_m / distance_m) * focal_length_px
        else:
            blur_px = 0.0

        return blur_px

    def get_settings_info(self) -> Dict:
        """
        Get human-readable settings information for logging/debugging.

        Returns:
            Dictionary with settings information
        """
        if self.current_settings is None:
            return {
                'mode': self.current_mode.value,
                'configured': False
            }

        return {
            'mode': self.current_mode.value,
            'configured': True,
            'exposure_ms': self.current_settings.exposure_time_us / 1000.0,
            'analog_gain': self.current_settings.analog_gain,
            'digital_gain': self.current_settings.digital_gain,
            'auto_exposure': self.current_settings.auto_exposure
        }
