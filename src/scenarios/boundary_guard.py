"""
Boundary Guard System for Scenario 2

This module implements a safety monitor that runs parallel to the mission
to prevent drones from colliding with cage walls during corner operations.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """Types of boundary violations"""
    NONE = "none"
    HARD_LIMIT = "hard_limit"
    SOFT_LIMIT = "soft_limit"
    PREDICTED = "predicted"


@dataclass
class Telemetry:
    """Drone telemetry data"""
    drone_id: str
    x: float
    y: float
    z: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0


class GeofenceMonitor:
    """
    Monitor that ensures drones stay within safe boundaries.

    The physical cage is 10.0m x 10.0m, but we use conservative limits
    to prevent collisions during corner operations in Scenario 2.
    """

    def __init__(self):
        """Initialize the geofence monitor with hard and soft limits"""
        # Physical cage is 10.0/0.0, we use conservative limits
        self.hard_limits = {
            'x_max': 9.8,
            'x_min': 0.2,
            'y_max': 9.8,
            'y_min': 0.2,
            'z_max': 5.5,
            'z_min': 0.1
        }

        # Soft limits are the target position for corner operations
        self.soft_limits = {
            'x_max': 9.5,
            'x_min': 0.5,
            'y_max': 9.5,
            'y_min': 0.5,
            'z_max': 5.0,
            'z_min': 0.5
        }

        # Prediction time window (seconds)
        self.prediction_window = 0.5

        # Violation callbacks
        self.violation_callbacks: List[callable] = []

    def add_violation_callback(self, callback: callable):
        """
        Add a callback to be called when a violation is detected

        Args:
            callback: Function to call with (drone_id, violation_type, details)
        """
        self.violation_callbacks.append(callback)

    def check_bounds(self, position: Dict[str, float], limit_type: str = 'hard') -> ViolationType:
        """
        Check if a position is within bounds

        Args:
            position: Dict with 'x', 'y', 'z' keys
            limit_type: 'hard' or 'soft' limits to check against

        Returns:
            ViolationType indicating if and how bounds were violated
        """
        limits = self.hard_limits if limit_type == 'hard' else self.soft_limits

        for axis in ['x', 'y', 'z']:
            if position.get(axis, 0.0) > limits[f'{axis}_max']:
                return ViolationType.HARD_LIMIT if limit_type == 'hard' else ViolationType.SOFT_LIMIT
            if position.get(axis, 0.0) < limits[f'{axis}_min']:
                return ViolationType.HARD_LIMIT if limit_type == 'hard' else ViolationType.SOFT_LIMIT

        return ViolationType.NONE

    def predict_violation(self, position: Dict[str, float], velocity: Dict[str, float]) -> bool:
        """
        Predict if current trajectory will cause a violation

        Projects the drone's position forward by prediction_window seconds
        and checks if it would violate hard limits.

        Args:
            position: Current position dict with 'x', 'y', 'z'
            velocity: Current velocity dict with 'x', 'y', 'z'

        Returns:
            True if a collision is predicted within prediction_window
        """
        # Project position forward
        future_position = {
            'x': position.get('x', 0.0) + velocity.get('x', 0.0) * self.prediction_window,
            'y': position.get('y', 0.0) + velocity.get('y', 0.0) * self.prediction_window,
            'z': position.get('z', 0.0) + velocity.get('z', 0.0) * self.prediction_window
        }

        # Check if future position violates hard limits
        violation = self.check_bounds(future_position, 'hard')
        return violation != ViolationType.NONE

    def check_swarm_bounds(self, telemetry_list: List[Telemetry]) -> Dict[str, Dict]:
        """
        Check bounds for entire swarm

        Args:
            telemetry_list: List of Telemetry objects for all drones

        Returns:
            Dict mapping drone_id to violation info, or empty dict if all safe
        """
        violations = {}

        for telem in telemetry_list:
            position = {'x': telem.x, 'y': telem.y, 'z': telem.z}
            velocity = {'x': telem.vx, 'y': telem.vy, 'z': telem.vz}

            # Check current position against hard limits
            hard_violation = self.check_bounds(position, 'hard')
            if hard_violation != ViolationType.NONE:
                violation_info = {
                    'type': hard_violation,
                    'position': position,
                    'severity': 'CRITICAL'
                }
                violations[telem.drone_id] = violation_info
                self._trigger_callbacks(telem.drone_id, hard_violation, violation_info)
                logger.error(f"CRITICAL: Drone {telem.drone_id} violated hard limits at {position}")
                continue

            # Check for predicted violations
            if self.predict_violation(position, velocity):
                future_pos = {
                    'x': position['x'] + velocity['x'] * self.prediction_window,
                    'y': position['y'] + velocity['y'] * self.prediction_window,
                    'z': position['z'] + velocity['z'] * self.prediction_window
                }
                violation_info = {
                    'type': ViolationType.PREDICTED,
                    'position': position,
                    'velocity': velocity,
                    'predicted_position': future_pos,
                    'time_to_impact': self.prediction_window,
                    'severity': 'HIGH'
                }
                violations[telem.drone_id] = violation_info
                self._trigger_callbacks(telem.drone_id, ViolationType.PREDICTED, violation_info)
                logger.warning(f"WARNING: Drone {telem.drone_id} will hit wall in {self.prediction_window}s")

        return violations

    def _trigger_callbacks(self, drone_id: str, violation_type: ViolationType, details: Dict):
        """Trigger all registered violation callbacks"""
        for callback in self.violation_callbacks:
            try:
                callback(drone_id, violation_type, details)
            except Exception as e:
                logger.error(f"Error in violation callback: {e}")


class SafetyOverride:
    """
    Safety override system that can halt mission execution

    This class monitors the swarm and can trigger emergency stops
    when boundary violations are detected.
    """

    def __init__(self, swarm_manager=None):
        """
        Initialize safety override

        Args:
            swarm_manager: Reference to the swarm manager (mocked if None)
        """
        self.swarm = swarm_manager
        self.geofence = GeofenceMonitor()
        self.emergency_stop_active = False
        self.violation_history: List[Dict] = []

        # Register callback
        self.geofence.add_violation_callback(self._handle_violation)

    def _handle_violation(self, drone_id: str, violation_type: ViolationType, details: Dict):
        """
        Handle a boundary violation

        Args:
            drone_id: ID of the violating drone
            violation_type: Type of violation
            details: Additional violation details
        """
        # Log the violation
        self.violation_history.append({
            'drone_id': drone_id,
            'type': violation_type,
            'details': details
        })

        # Take action based on severity
        if details.get('severity') == 'CRITICAL':
            self.trigger_emergency_stop(drone_id, "HARD_LIMIT_VIOLATION")
        elif violation_type == ViolationType.PREDICTED:
            self.trigger_emergency_stop(drone_id, "WALL_COLLISION_PREDICTED")

    def trigger_emergency_stop(self, drone_id: str, reason: str):
        """
        Trigger emergency stop for a drone

        Args:
            drone_id: ID of drone to stop
            reason: Reason for emergency stop
        """
        self.emergency_stop_active = True
        logger.critical(f"EMERGENCY STOP: Drone {drone_id} - Reason: {reason}")

        # If we have a swarm manager, send stop command
        if self.swarm and hasattr(self.swarm, 'emergency_stop'):
            self.swarm.emergency_stop(drone_id, reason)
        else:
            # Mock behavior for testing
            logger.info(f"Mock: Would send EMERGENCY_BRAKE to drone {drone_id}")

    def monitor_loop(self, telemetry_list: List[Telemetry]) -> Dict[str, Dict]:
        """
        Main monitoring loop - checks all drones for violations

        Args:
            telemetry_list: List of current telemetry for all drones

        Returns:
            Dict of violations (empty if all safe)
        """
        if not telemetry_list:
            return {}

        violations = self.geofence.check_swarm_bounds(telemetry_list)
        return violations

    def reset_emergency_stop(self):
        """Reset emergency stop flag"""
        self.emergency_stop_active = False
        logger.info("Emergency stop reset")

    def get_violation_summary(self) -> Dict:
        """Get summary of all violations"""
        return {
            'total_violations': len(self.violation_history),
            'critical_violations': sum(1 for v in self.violation_history
                                      if v['details'].get('severity') == 'CRITICAL'),
            'predicted_violations': sum(1 for v in self.violation_history
                                       if v['type'] == ViolationType.PREDICTED),
            'history': self.violation_history
        }
