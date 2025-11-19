"""
Base Drone Controller for Crazyflie drones.

This module provides the foundational controller class for interacting with
Crazyflie drones. It serves as a mock implementation that can be extended
for simulation and testing purposes.
"""

import logging
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class DroneState:
    """Represents the current state of a drone."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    battery_voltage: float = 4.2  # Default full charge for LiPo
    is_flying: bool = False


class DroneController:
    """
    Base controller for Crazyflie drones.

    This is a mock implementation that can be used for testing and simulation.
    In a real deployment, this would interface with the Crazyswarm library.
    """

    def __init__(self, drone_id: str, crazyswarm=None):
        """
        Initialize the drone controller.

        Args:
            drone_id: Unique identifier for the drone (e.g., 'cf1', 'cf2')
            crazyswarm: Crazyswarm object (optional, for real hardware)
        """
        self.drone_id = drone_id
        self.crazyswarm = crazyswarm
        self.logger = logging.getLogger(f"DroneController.{drone_id}")
        self.state = DroneState()

        # Configure logging
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'[%(asctime)s] [{drone_id}] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def takeoff(self, target_height: float, duration: float = 2.0):
        """
        Command the drone to take off to a target height.

        Args:
            target_height: Target height in meters
            duration: Duration of the takeoff maneuver in seconds
        """
        self.logger.info(f"Taking off to {target_height}m in {duration}s")
        self.state.z = target_height
        self.state.is_flying = True

    def land(self, target_height: float = 0.0, duration: float = 2.0):
        """
        Command the drone to land.

        Args:
            target_height: Target landing height in meters (usually 0.0)
            duration: Duration of the landing maneuver in seconds
        """
        self.logger.info(f"Landing to {target_height}m in {duration}s")
        self.state.z = target_height
        self.state.is_flying = False

    def go_to(self, x: float, y: float, z: float, yaw: float, duration: float = 2.0):
        """
        Command the drone to fly to a specific position.

        Args:
            x: Target x coordinate in meters
            y: Target y coordinate in meters
            z: Target z coordinate in meters
            yaw: Target yaw angle in radians
            duration: Duration to reach the target in seconds
        """
        self.logger.info(f"Going to ({x:.2f}, {y:.2f}, {z:.2f}) yaw={yaw:.2f} in {duration}s")
        self.state.x = x
        self.state.y = y
        self.state.z = z
        self.state.yaw = yaw

    def hover(self, duration: float):
        """
        Command the drone to hover at current position.

        Args:
            duration: Duration to hover in seconds
        """
        self.logger.info(f"Hovering for {duration}s at ({self.state.x:.2f}, {self.state.y:.2f}, {self.state.z:.2f})")

    def get_position(self) -> Tuple[float, float, float]:
        """
        Get the current position of the drone.

        Returns:
            Tuple of (x, y, z) coordinates in meters
        """
        return (self.state.x, self.state.y, self.state.z)

    def get_battery_voltage(self) -> float:
        """
        Get the current battery voltage.

        Returns:
            Battery voltage in volts
        """
        # Simulate battery drain (very simple model)
        if self.state.is_flying:
            self.state.battery_voltage = max(3.0, self.state.battery_voltage - 0.001)
        return self.state.battery_voltage

    def emergency_stop(self):
        """
        Emergency stop - immediately cut motors.
        WARNING: This will cause the drone to fall!
        """
        self.logger.warning("EMERGENCY STOP ACTIVATED!")
        self.state.is_flying = False
        self.state.z = 0.0
