"""
Base Drone Controller for Crazyflie swarm control.

This module provides high-level control interface for individual Crazyflie drones,
including state management, safety checks, and basic flight operations.
"""
import numpy as np
import logging
from enum import Enum
from typing import Tuple, Optional

try:
    from pycrazyswarm import Crazyswarm
except ImportError:
    from .pycrazyswarm_mock import Crazyswarm


class DroneState(Enum):
    """Drone operational states."""
    IDLE = "idle"
    TAKING_OFF = "taking_off"
    FLYING = "flying"
    LANDING = "landing"
    EMERGENCY = "emergency"


class DroneController:
    """
    High-level controller for a single Crazyflie drone.

    Provides safety-checked control methods and state management for
    autonomous flight operations.
    """

    def __init__(
        self,
        drone_id: str,
        crazyswarm: Crazyswarm,
        cage_bounds: Optional[Tuple[float, float, float]] = None,
        min_battery: float = 20.0
    ):
        """
        Initialize drone controller for a specific Crazyflie.

        Args:
            drone_id: Unique identifier for the drone (e.g., 'cf1')
            crazyswarm: Crazyswarm instance for hardware communication
            cage_bounds: Maximum bounds (x, y, z) for safety checks
            min_battery: Minimum battery percentage before emergency landing
        """
        self.drone_id = drone_id
        self.cf = crazyswarm.allcfs.crazyfliesByName[drone_id]
        self.state = DroneState.IDLE
        self.logger = logging.getLogger(f"DroneController_{drone_id}")

        # Safety parameters
        self.cage_bounds = cage_bounds or (10.0, 6.0, 8.0)
        self.min_battery = min_battery

        # Current position tracking
        self._current_position = np.array([0.0, 0.0, 0.0])

        self.logger.info(f"Initialized controller for {drone_id}")

    def takeoff(self, height: float, duration: float = 2.0) -> bool:
        """
        Command drone to take off to specified height.

        Args:
            height: Target height in meters
            duration: Time to complete takeoff in seconds

        Returns:
            True if takeoff command successful, False otherwise
        """
        try:
            # Safety checks
            if not self._check_battery():
                self.logger.error("Insufficient battery for takeoff")
                return False

            if height > self.cage_bounds[2]:
                self.logger.error(f"Height {height}m exceeds cage limit {self.cage_bounds[2]}m")
                return False

            self.logger.info(f"Taking off to {height}m")
            self.state = DroneState.TAKING_OFF

            self.cf.takeoff(targetHeight=height, duration=duration)

            self.state = DroneState.FLYING
            self._current_position[2] = height

            return True

        except Exception as e:
            self.logger.error(f"Takeoff failed: {e}")
            self.state = DroneState.EMERGENCY
            return False

    def land(self, duration: float = 2.0) -> bool:
        """
        Command drone to land.

        Args:
            duration: Time to complete landing in seconds

        Returns:
            True if landing command successful, False otherwise
        """
        try:
            self.logger.info("Landing")
            self.state = DroneState.LANDING

            self.cf.land(targetHeight=0.0, duration=duration)

            self.state = DroneState.IDLE
            self._current_position[2] = 0.0

            return True

        except Exception as e:
            self.logger.error(f"Landing failed: {e}")
            self.state = DroneState.EMERGENCY
            return False

    def go_to(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float = 0.0,
        duration: float = 3.0
    ) -> bool:
        """
        Command drone to fly to specified position.

        Args:
            x: Target x position in meters
            y: Target y position in meters
            z: Target z position in meters
            yaw: Target yaw angle in degrees
            duration: Time to complete movement in seconds

        Returns:
            True if command successful, False otherwise
        """
        try:
            # Safety checks
            if not self._check_battery():
                self.logger.error("Insufficient battery for movement")
                return False

            if not self._check_position_bounds(x, y, z):
                self.logger.error(f"Target position ({x}, {y}, {z}) out of bounds")
                return False

            target_position = np.array([x, y, z])
            self.logger.info(f"Moving to ({x:.2f}, {y:.2f}, {z:.2f})")

            self.cf.goTo(goal=target_position, yaw=yaw, duration=duration)

            self._current_position = target_position

            return True

        except Exception as e:
            self.logger.error(f"Go to failed: {e}")
            self.state = DroneState.EMERGENCY
            return False

    def get_position(self) -> Tuple[float, float, float]:
        """
        Get current drone position.

        Returns:
            Tuple of (x, y, z) coordinates in meters
        """
        try:
            # Try to get real position from hardware
            position = self.cf.position()
            self._current_position = position
            return tuple(position)
        except Exception as e:
            self.logger.warning(f"Could not get position from hardware: {e}")
            # Return cached position
            return tuple(self._current_position)

    def get_battery_percentage(self) -> float:
        """
        Get current battery level.

        Returns:
            Battery percentage (0-100)
        """
        try:
            battery = self.cf.getBatteryLevel()
            return float(battery)
        except Exception as e:
            self.logger.error(f"Could not get battery level: {e}")
            return 0.0

    def _check_battery(self) -> bool:
        """
        Check if battery level is sufficient.

        Returns:
            True if battery is above minimum threshold
        """
        battery = self.get_battery_percentage()
        if battery < self.min_battery:
            self.logger.warning(f"Low battery: {battery:.1f}%")
            return False
        return True

    def _check_position_bounds(self, x: float, y: float, z: float) -> bool:
        """
        Check if position is within cage bounds.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate

        Returns:
            True if position is within bounds
        """
        if not (0 <= x <= self.cage_bounds[0]):
            return False
        if not (0 <= y <= self.cage_bounds[1]):
            return False
        if not (0 <= z <= self.cage_bounds[2]):
            return False
        return True

    def emergency_stop(self) -> None:
        """
        Emergency stop and land immediately.
        """
        self.logger.warning("EMERGENCY STOP")
        self.state = DroneState.EMERGENCY
        try:
            self.land(duration=1.0)
            # Maintain emergency state after landing
            self.state = DroneState.EMERGENCY
        except Exception as e:
            self.logger.error(f"Emergency landing failed: {e}")

    def get_state(self) -> DroneState:
        """
        Get current drone state.

        Returns:
            Current DroneState
        """
        return self.state

    def is_ready(self) -> bool:
        """
        Check if drone is ready for flight operations.

        Returns:
            True if drone is in good state for flight
        """
        return (
            self.state in [DroneState.IDLE, DroneState.FLYING] and
            self._check_battery()
        )
