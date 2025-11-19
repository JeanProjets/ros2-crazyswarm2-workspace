"""
Safe Drone Controller for Scenario 2.

This module extends the base DroneController with strict boundary enforcement
and battery management for precision flight near cage boundaries.
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any
from src.core.drone_controller import DroneController


class SafeDroneController(DroneController):
    """
    Extended drone controller with strict boundary enforcement for Scenario 2.

    Key Features:
    - Automatic position clamping to prevent wall collisions
    - Battery voltage monitoring with critical threshold alerts
    - Precision hover capabilities for tight spaces
    - Enhanced logging for safety-critical operations
    """

    # Safety thresholds
    BATTERY_WARNING_VOLTAGE = 3.5  # Return to Home immediately
    BATTERY_CRITICAL_VOLTAGE = 3.4  # Emergency landing

    def __init__(self, drone_id: str, crazyswarm=None, config: Dict[str, Any] = None):
        """
        Initialize the safe drone controller.

        Args:
            drone_id: Unique identifier for the drone
            crazyswarm: Crazyswarm object (optional, for real hardware)
            config: Configuration dictionary containing safety_bounds
        """
        super().__init__(drone_id, crazyswarm)

        # Extract safety bounds from config
        if config and 'safety_bounds' in config:
            self.bounds = config['safety_bounds']
        else:
            # Default bounds for Scenario 2
            self.bounds = {
                'x_min': 0.3,
                'x_max': 9.7,
                'y_min': 0.3,
                'y_max': 5.7,
                'z_min': 0.2,
                'z_max': 5.8
            }

        self.logger.info(f"Initialized SafeDroneController with bounds: X[{self.bounds['x_min']}, {self.bounds['x_max']}], "
                        f"Y[{self.bounds['y_min']}, {self.bounds['y_max']}], Z[{self.bounds['z_min']}, {self.bounds['z_max']}]")

        self.clamp_count = 0  # Track how many times positions were clamped

    def clamp_position(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Ensure coordinates are within cage safety margins.

        This is the most critical safety function. It prevents drones from
        being commanded to positions that could result in wall collisions.

        Args:
            x: Desired x coordinate
            y: Desired y coordinate
            z: Desired z coordinate

        Returns:
            Tuple of (safe_x, safe_y, safe_z) clamped to safety bounds
        """
        safe_x = np.clip(x, self.bounds['x_min'], self.bounds['x_max'])
        safe_y = np.clip(y, self.bounds['y_min'], self.bounds['y_max'])
        safe_z = np.clip(z, self.bounds['z_min'], self.bounds['z_max'])

        # Log warning if clamping occurs
        if safe_x != x or safe_y != y or safe_z != z:
            self.clamp_count += 1
            self.logger.warning(
                f"⚠️  POSITION CLAMPED (#{self.clamp_count}): "
                f"({x:.2f}, {y:.2f}, {z:.2f}) -> ({safe_x:.2f}, {safe_y:.2f}, {safe_z:.2f})"
            )

        return safe_x, safe_y, safe_z

    def clamped_navigate(self, x: float, y: float, z: float, yaw: float = 0.0, duration: float = 2.0):
        """
        Navigate to a position with automatic safety clamping.

        This method should be used instead of go_to() for all high-level commands
        to ensure safety bounds are respected.

        Args:
            x: Target x coordinate
            y: Target y coordinate
            z: Target z coordinate
            yaw: Target yaw angle in radians
            duration: Duration to reach target in seconds
        """
        # Check battery before navigation
        voltage = self.get_battery_voltage()
        if voltage < self.BATTERY_CRITICAL_VOLTAGE:
            self.logger.error(f"🔋 CRITICAL BATTERY ({voltage:.2f}V) - Navigation aborted!")
            return
        elif voltage < self.BATTERY_WARNING_VOLTAGE:
            self.logger.warning(f"🔋 LOW BATTERY ({voltage:.2f}V) - Return to Home recommended!")

        # Clamp position to safe bounds
        safe_x, safe_y, safe_z = self.clamp_position(x, y, z)

        # Execute navigation using base class method
        self.go_to(safe_x, safe_y, safe_z, yaw, duration)

    def precision_hover(self, height: float, duration: float = 5.0):
        """
        Hover with tighter control for station keeping near boundaries.

        This method is designed for scenarios where the drone needs to maintain
        position near walls or in tight spaces (like Scenario 2's corner target).

        Args:
            height: Target hover height in meters
            duration: Duration to maintain hover in seconds
        """
        # Clamp height to safe bounds
        _, _, safe_height = self.clamp_position(self.state.x, self.state.y, height)

        self.logger.info(f"🎯 Precision hover at height {safe_height:.2f}m for {duration}s")

        # In a real implementation, this would use tighter PID gains or reduced velocity limits
        # For now, we use the base hover method
        self.state.z = safe_height
        self.hover(duration)

    def get_battery_voltage(self) -> float:
        """
        Get battery voltage with warning thresholds.

        Returns:
            Battery voltage in volts
        """
        voltage = super().get_battery_voltage()

        # Check against thresholds
        if voltage < self.BATTERY_CRITICAL_VOLTAGE:
            self.logger.error(f"🔋 CRITICAL BATTERY: {voltage:.2f}V (Threshold: {self.BATTERY_CRITICAL_VOLTAGE}V)")
        elif voltage < self.BATTERY_WARNING_VOLTAGE:
            self.logger.warning(f"🔋 LOW BATTERY: {voltage:.2f}V (Threshold: {self.BATTERY_WARNING_VOLTAGE}V)")

        return voltage

    def check_battery_status(self) -> str:
        """
        Check battery status and return a status string.

        Returns:
            Status string: 'CRITICAL', 'WARNING', or 'OK'
        """
        voltage = self.get_battery_voltage()

        if voltage < self.BATTERY_CRITICAL_VOLTAGE:
            return 'CRITICAL'
        elif voltage < self.BATTERY_WARNING_VOLTAGE:
            return 'WARNING'
        else:
            return 'OK'

    def is_position_safe(self, x: float, y: float, z: float) -> bool:
        """
        Check if a position is within safe bounds without clamping.

        Args:
            x: X coordinate to check
            y: Y coordinate to check
            z: Z coordinate to check

        Returns:
            True if position is safe, False otherwise
        """
        return (
            self.bounds['x_min'] <= x <= self.bounds['x_max'] and
            self.bounds['y_min'] <= y <= self.bounds['y_max'] and
            self.bounds['z_min'] <= z <= self.bounds['z_max']
        )

    def get_distance_to_boundary(self, x: float, y: float) -> float:
        """
        Calculate minimum distance to any boundary wall.

        Useful for determining how close the drone is to danger zones.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Minimum distance to any wall in meters
        """
        dist_to_x_min = x - self.bounds['x_min']
        dist_to_x_max = self.bounds['x_max'] - x
        dist_to_y_min = y - self.bounds['y_min']
        dist_to_y_max = self.bounds['y_max'] - y

        return min(dist_to_x_min, dist_to_x_max, dist_to_y_min, dist_to_y_max)

    def get_safety_stats(self) -> Dict[str, Any]:
        """
        Get safety statistics for this drone.

        Returns:
            Dictionary containing safety metrics
        """
        return {
            'drone_id': self.drone_id,
            'clamp_count': self.clamp_count,
            'current_position': self.get_position(),
            'battery_voltage': self.get_battery_voltage(),
            'battery_status': self.check_battery_status(),
            'is_flying': self.state.is_flying,
            'distance_to_boundary': self.get_distance_to_boundary(self.state.x, self.state.y)
        }
