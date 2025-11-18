"""
Mock implementation of pycrazyswarm for testing without hardware.
This allows the code to run in environments where crazyswarm2 is not available.
"""
import numpy as np
from typing import Dict, Optional
import logging


class MockCrazyflie:
    """Mock Crazyflie drone for simulation."""

    def __init__(self, drone_id: str, initial_position: tuple = (0, 0, 0)):
        self.id = drone_id
        self._position = np.array(initial_position, dtype=float)
        self.battery = 100.0
        self.is_flying = False
        self.logger = logging.getLogger(f"MockCF_{drone_id}")

    def takeoff(self, targetHeight: float, duration: float):
        """Simulate takeoff."""
        self.logger.info(f"Taking off to {targetHeight}m in {duration}s")
        self._position[2] = targetHeight
        self.is_flying = True
        self.battery -= 1.0

    def land(self, targetHeight: float, duration: float):
        """Simulate landing."""
        self.logger.info(f"Landing to {targetHeight}m in {duration}s")
        self._position[2] = targetHeight
        self.is_flying = False
        self.battery -= 0.5

    def goTo(self, goal: np.ndarray, yaw: float, duration: float):
        """Simulate movement to target position."""
        self.logger.info(f"Going to {goal} with yaw {yaw} in {duration}s")
        self._position = np.array(goal, dtype=float)
        self.battery -= 0.5

    def position(self) -> np.ndarray:
        """Get current position."""
        return self._position.copy()

    def getBatteryLevel(self) -> float:
        """Get battery percentage."""
        return max(0.0, self.battery)


class MockCrazyfliesByName:
    """Container for drones indexed by name."""

    def __init__(self):
        self._drones: Dict[str, MockCrazyflie] = {}

    def __getitem__(self, name: str) -> MockCrazyflie:
        if name not in self._drones:
            self._drones[name] = MockCrazyflie(name)
        return self._drones[name]

    def __setitem__(self, name: str, drone: MockCrazyflie):
        self._drones[name] = drone


class MockAllCfs:
    """Mock collection of all Crazyflies."""

    def __init__(self):
        self.crazyfliesByName = MockCrazyfliesByName()


class Crazyswarm:
    """Mock Crazyswarm controller."""

    def __init__(self):
        self.allcfs = MockAllCfs()
        self.timeHelper = None  # Not needed for basic mock

        # Pre-populate common drone IDs
        for drone_id in ['cf1', 'cf2', 'cf3', 'cf4']:
            self.allcfs.crazyfliesByName[drone_id] = MockCrazyflie(drone_id)


# Try to import real pycrazyswarm, fall back to mock if not available
try:
    from pycrazyswarm import Crazyswarm as RealCrazyswarm
    USING_MOCK = False
except ImportError:
    RealCrazyswarm = Crazyswarm
    USING_MOCK = True
    logging.warning("pycrazyswarm not found, using mock implementation")
