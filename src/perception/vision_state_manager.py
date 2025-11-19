"""
Vision State Manager for Scenario 2 - Power & Resource Management

This module manages AI Deck resources to optimize battery and bandwidth usage
during the long flight to the corner target.

Author: Agent 3 (Vision Developer)
Scenario: 2 - Corner Target with Power Optimization
"""

import time
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass


class VisionMode(Enum):
    """Vision processing modes for different flight phases."""
    IDLE = "idle"  # Camera off, no processing
    LONG_RANGE = "long_range"  # High exposure, full FOV, long-range detection
    TERMINAL = "terminal"  # Low exposure, ROI processing, precision guidance
    MOTION_DETECT = "motion_detect"  # Lightweight motion detection only


@dataclass
class VisionConfig:
    """Configuration for vision processing mode."""
    mode: VisionMode
    exposure: float  # Camera exposure setting
    use_full_fov: bool  # Use full field of view or ROI
    inference_enabled: bool  # Enable neural network inference
    frame_rate: int  # Target frame rate
    power_level: float  # Power consumption estimate (0-1)


class VisionLifecycle:
    """
    Manages AI Deck lifecycle and resource allocation.

    Features:
    - Dynamic mode switching based on mission phase
    - Power optimization for long-range flights
    - Bandwidth management
    - Camera parameter adaptation
    """

    def __init__(self):
        """Initialize vision lifecycle manager."""
        # Mode transition history
        self.mode_history = []
        self.max_history = 20

        # Power tracking
        self.total_power_consumed = 0.0
        self.mode_start_time = time.time()

        # Camera parameters by mode (must be defined before _get_config_for_mode call)
        self.mode_configs = {
            VisionMode.IDLE: VisionConfig(
                mode=VisionMode.IDLE,
                exposure=0.0,
                use_full_fov=False,
                inference_enabled=False,
                frame_rate=0,
                power_level=0.0
            ),
            VisionMode.LONG_RANGE: VisionConfig(
                mode=VisionMode.LONG_RANGE,
                exposure=0.8,  # High exposure for distant targets
                use_full_fov=True,
                inference_enabled=True,
                frame_rate=15,  # Moderate frame rate
                power_level=0.6
            ),
            VisionMode.TERMINAL: VisionConfig(
                mode=VisionMode.TERMINAL,
                exposure=0.3,  # Low exposure to prevent LED washout
                use_full_fov=False,  # ROI processing
                inference_enabled=True,
                frame_rate=30,  # High frame rate for precision
                power_level=0.9
            ),
            VisionMode.MOTION_DETECT: VisionConfig(
                mode=VisionMode.MOTION_DETECT,
                exposure=0.5,
                use_full_fov=True,
                inference_enabled=False,  # No CNN, just motion detection
                frame_rate=10,
                power_level=0.3
            )
        }

        # Transition rules based on drone position
        self.position_thresholds = {
            'long_range_start': 3.0,  # Start long-range mode at X > 3.0m
            'terminal_start': 8.0,  # Start terminal mode at X > 8.0m
            'transit_end': 5.0  # End transit mode at X > 5.0m
        }

        # Initialize current mode and config (after mode_configs is defined)
        self.current_mode = VisionMode.IDLE
        self.current_config = self._get_config_for_mode(VisionMode.IDLE)

    def _get_config_for_mode(self, mode: VisionMode) -> VisionConfig:
        """
        Get configuration for specified mode.

        Args:
            mode: Vision mode

        Returns:
            VisionConfig for the mode
        """
        return self.mode_configs.get(mode, self.mode_configs[VisionMode.IDLE])

    def set_mode(self, mode: VisionMode) -> bool:
        """
        Set vision processing mode.

        Args:
            mode: Target vision mode

        Returns:
            True if mode change successful
        """
        if mode == self.current_mode:
            return True  # Already in this mode

        # Record power consumption from previous mode
        self._update_power_consumption()

        # Add to history
        self.mode_history.append({
            'from': self.current_mode,
            'to': mode,
            'timestamp': time.time()
        })

        if len(self.mode_history) > self.max_history:
            self.mode_history.pop(0)

        # Update mode
        self.current_mode = mode
        self.current_config = self._get_config_for_mode(mode)
        self.mode_start_time = time.time()

        # Apply mode-specific settings
        self._apply_mode_settings()

        return True

    def _apply_mode_settings(self):
        """Apply camera and processing settings for current mode."""
        config = self.current_config

        # In production, this would configure the AI Deck hardware
        # For now, just log the settings

        settings = {
            'exposure': config.exposure,
            'full_fov': config.use_full_fov,
            'inference': config.inference_enabled,
            'fps': config.frame_rate
        }

        # Mock hardware configuration
        # In production: ai_deck.configure(settings)
        pass

    def _update_power_consumption(self):
        """Update total power consumption tracking."""
        elapsed_time = time.time() - self.mode_start_time
        power_consumed = self.current_config.power_level * elapsed_time

        self.total_power_consumed += power_consumed

    def auto_mode_selection(self, drone_position: tuple) -> VisionMode:
        """
        Automatically select vision mode based on drone position.

        Args:
            drone_position: Current drone position (x, y, z)

        Returns:
            Recommended vision mode
        """
        x, y, z = drone_position

        # For Scenario 2, target is at (9.5, 0.5, 5)
        # Adjust mode based on X position (distance to corner)

        if x < self.position_thresholds['transit_end']:
            # Early transit - save power
            return VisionMode.MOTION_DETECT

        elif x < self.position_thresholds['terminal_start']:
            # Approaching - enable long-range detection
            return VisionMode.LONG_RANGE

        else:
            # Close to corner - precision mode
            return VisionMode.TERMINAL

    def update_from_mission_phase(self, mission_phase: str, drone_position: Optional[tuple] = None):
        """
        Update vision mode based on mission phase.

        Args:
            mission_phase: Current mission phase string
            drone_position: Optional drone position for auto-selection
        """
        phase_mode_map = {
            'IDLE': VisionMode.IDLE,
            'TAKEOFF': VisionMode.IDLE,
            'TRANSIT': VisionMode.MOTION_DETECT,
            'LONG_RANGE': VisionMode.LONG_RANGE,
            'PRECISION_APPROACH': VisionMode.TERMINAL,
            'TERMINAL': VisionMode.TERMINAL,
            'IMPACT': VisionMode.TERMINAL,
            'LANDING': VisionMode.IDLE
        }

        # Get mode from phase
        target_mode = phase_mode_map.get(mission_phase, VisionMode.LONG_RANGE)

        # Override with position-based if available
        if drone_position is not None and mission_phase in ['TRANSIT', 'LONG_RANGE', 'PRECISION_APPROACH']:
            target_mode = self.auto_mode_selection(drone_position)

        # Set the mode
        self.set_mode(target_mode)

    def should_disable_inference(self, drone_position: tuple) -> bool:
        """
        Determine if neural network inference should be disabled.

        Args:
            drone_position: Current drone position (x, y, z)

        Returns:
            True if inference should be disabled to save power
        """
        x, y, z = drone_position

        # Disable inference during initial transit (X < 5.0m)
        if x < self.position_thresholds['transit_end']:
            return True

        return False

    def get_current_config(self) -> VisionConfig:
        """
        Get current vision configuration.

        Returns:
            Current VisionConfig
        """
        return self.current_config

    def get_power_statistics(self) -> Dict[str, Any]:
        """
        Get power consumption statistics.

        Returns:
            Dictionary with power statistics
        """
        # Update current consumption
        self._update_power_consumption()

        return {
            'total_power_consumed': self.total_power_consumed,
            'current_mode': self.current_mode.value,
            'current_power_level': self.current_config.power_level,
            'time_in_mode': time.time() - self.mode_start_time
        }

    def get_mode_history(self) -> list:
        """
        Get mode transition history.

        Returns:
            List of mode transitions
        """
        return self.mode_history.copy()

    def optimize_for_battery(self) -> VisionMode:
        """
        Get battery-optimized vision mode.

        Returns:
            Most power-efficient mode for current state
        """
        # Use motion detection instead of full CNN when possible
        if self.current_mode == VisionMode.LONG_RANGE:
            return VisionMode.MOTION_DETECT

        return self.current_mode

    def switch_model(self, drone_position: tuple) -> str:
        """
        Determine which detection model to use.

        Args:
            drone_position: Current drone position (x, y, z)

        Returns:
            Model identifier string
        """
        x, y, z = drone_position

        if x < self.position_thresholds['transit_end']:
            # Use lightweight motion detector
            return "motion_detector_v1"

        elif x < self.position_thresholds['terminal_start']:
            # Use long-range CNN
            return "cnn_longrange_v2"

        else:
            # Use precision terminal CNN
            return "cnn_terminal_v2"

    def reset(self):
        """Reset vision lifecycle state."""
        self._update_power_consumption()
        self.current_mode = VisionMode.IDLE
        self.current_config = self._get_config_for_mode(VisionMode.IDLE)
        self.mode_history.clear()
        self.total_power_consumed = 0.0
        self.mode_start_time = time.time()

    def get_recommended_settings(self, mission_phase: str, drone_position: tuple) -> Dict[str, Any]:
        """
        Get recommended vision settings for current situation.

        Args:
            mission_phase: Current mission phase
            drone_position: Current drone position (x, y, z)

        Returns:
            Dictionary with recommended settings
        """
        # Update mode based on phase and position
        self.update_from_mission_phase(mission_phase, drone_position)

        config = self.current_config

        return {
            'mode': config.mode.value,
            'exposure': config.exposure,
            'use_full_fov': config.use_full_fov,
            'inference_enabled': config.inference_enabled,
            'frame_rate': config.frame_rate,
            'power_level': config.power_level,
            'model': self.switch_model(drone_position),
            'disable_inference': self.should_disable_inference(drone_position)
        }
