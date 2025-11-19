"""
Target Reacquisition Behavior for Scenario 4
Handles target tracking when target is temporarily occluded by obstacles
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum


class SearchState(Enum):
    """States for target reacquisition"""
    TRACKING = 1  # Target is visible
    PREDICTING = 2  # Target just lost, using prediction
    SEARCHING = 3  # Active search maneuver
    REACQUIRED = 4  # Target found again


@dataclass
class TargetState:
    """Represents known target state"""
    position: np.ndarray
    velocity: np.ndarray
    timestamp: float
    confidence: float


class OcclusionHandler:
    """
    Handles target reacquisition when target disappears behind obstacles.

    Implements predictive tracking to "cut corners" and intercept
    the target where it emerges from occlusion.
    """

    def __init__(self, confidence_threshold: float = 0.4,
                 prediction_time_limit: float = 3.0,
                 min_velocity_threshold: float = 0.1):
        """
        Initialize OcclusionHandler.

        Args:
            confidence_threshold: Minimum confidence to consider target "locked"
            prediction_time_limit: Max time to rely on prediction (seconds)
            min_velocity_threshold: Minimum target velocity to predict (m/s)
        """
        self.confidence_threshold = confidence_threshold
        self.prediction_time_limit = prediction_time_limit
        self.min_velocity_threshold = min_velocity_threshold

        self.state = SearchState.TRACKING
        self.last_known_target: Optional[TargetState] = None
        self.occlusion_start_time: Optional[float] = None
        self.predicted_emergence_point: Optional[np.ndarray] = None

    def update_target_observation(self, position: np.ndarray,
                                  velocity: np.ndarray,
                                  timestamp: float,
                                  confidence: float) -> None:
        """
        Update with new target observation from vision system.

        Args:
            position: Target position [x, y, z]
            velocity: Target velocity [vx, vy, vz]
            timestamp: Current time (seconds)
            confidence: Detection confidence [0, 1]
        """
        if confidence >= self.confidence_threshold:
            # Good detection - update state
            self.last_known_target = TargetState(
                position=position.copy(),
                velocity=velocity.copy(),
                timestamp=timestamp,
                confidence=confidence
            )

            # Reset occlusion tracking
            if self.state != SearchState.TRACKING:
                self.state = SearchState.REACQUIRED

            self.state = SearchState.TRACKING
            self.occlusion_start_time = None

        else:
            # Lost or weak detection
            if self.state == SearchState.TRACKING:
                # Just lost target
                self.state = SearchState.PREDICTING
                self.occlusion_start_time = timestamp

    def predict_emergence_point(self, current_time: float,
                               grid_map: Optional['GridMap'] = None) -> Optional[np.ndarray]:
        """
        Predict where target will emerge from behind obstacle.

        Uses last known position and velocity to project target motion
        and find where the trajectory exits the obstacle shadow.

        Args:
            current_time: Current time (seconds)
            grid_map: Optional GridMap for ray-casting

        Returns:
            Predicted emergence point, or None if cannot predict
        """
        if self.last_known_target is None:
            return None

        # Check if velocity is significant enough to predict
        vel_mag = np.linalg.norm(self.last_known_target.velocity)
        if vel_mag < self.min_velocity_threshold:
            # Target might have stopped - return last known position
            return self.last_known_target.position

        # Time since last observation
        if self.occlusion_start_time is None:
            dt = 0.0
        else:
            dt = current_time - self.occlusion_start_time

        # Extrapolate position based on constant velocity model
        predicted_pos = (self.last_known_target.position +
                        self.last_known_target.velocity * dt)

        if grid_map is not None:
            # Use map to find emergence point
            # Ray-cast along velocity direction to find edge of obstacle
            predicted_pos = self._find_emergence_with_map(
                self.last_known_target.position,
                self.last_known_target.velocity,
                dt,
                grid_map
            )

        self.predicted_emergence_point = predicted_pos
        return predicted_pos

    def _find_emergence_with_map(self, start_pos: np.ndarray,
                                 velocity: np.ndarray,
                                 dt: float,
                                 grid_map: 'GridMap') -> np.ndarray:
        """
        Use grid map to find where target emerges from obstacle.

        Ray-casts along velocity vector to find boundary of obstacle.

        Args:
            start_pos: Last known position
            velocity: Last known velocity
            dt: Time since occlusion started
            grid_map: GridMap for collision checking

        Returns:
            Predicted emergence point
        """
        # Simple ray-casting
        vel_mag = np.linalg.norm(velocity)
        if vel_mag < 1e-6:
            return start_pos

        vel_unit = velocity / vel_mag

        # Cast ray forward in steps
        step_size = 0.1  # meters
        max_steps = int((vel_mag * (dt + 2.0)) / step_size)  # Look ahead

        current_pos = start_pos
        last_free_pos = start_pos

        for _ in range(max_steps):
            current_pos = current_pos + vel_unit * step_size

            if not grid_map.is_collision(current_pos):
                # Found free space - this is likely emergence point
                return current_pos
            else:
                # Still in obstacle, keep going
                pass

        # Fallback to simple extrapolation
        return start_pos + velocity * dt

    def execute_search_maneuver(self, drone_pos: np.ndarray,
                               current_time: float,
                               max_speed: float = 1.0) -> Tuple[np.ndarray, str]:
        """
        Execute search maneuver when target is lost.

        Strategy:
        1. If recently lost - fly to predicted emergence point
        2. If lost for too long - execute spiral search pattern

        Args:
            drone_pos: Current drone position [x, y, z]
            current_time: Current time (seconds)
            max_speed: Maximum velocity (m/s)

        Returns:
            Tuple of (command_velocity, maneuver_description)
        """
        if self.last_known_target is None:
            return np.zeros(3), "No target data - hovering"

        # Predict where target should be
        predicted_pos = self.predict_emergence_point(current_time)

        if predicted_pos is None:
            return np.zeros(3), "Cannot predict - hovering"

        # Check if we've been searching too long
        if self.occlusion_start_time is not None:
            search_duration = current_time - self.occlusion_start_time

            if search_duration > self.prediction_time_limit:
                # Prediction is stale - execute search pattern
                self.state = SearchState.SEARCHING
                return self._execute_spiral_search(drone_pos, current_time, max_speed)

        # Fly towards predicted emergence point
        error = predicted_pos - drone_pos
        error_mag = np.linalg.norm(error)

        if error_mag < 1e-6:
            # At predicted point - hover and scan
            return np.zeros(3), "At emergence point - scanning"

        # Move towards emergence point
        cmd_vel = (error / error_mag) * max_speed

        # Continue with target's velocity to "cut the corner"
        if self.last_known_target.velocity is not None:
            feedforward_gain = 0.4
            cmd_vel = cmd_vel + feedforward_gain * self.last_known_target.velocity

            # Limit speed
            cmd_mag = np.linalg.norm(cmd_vel)
            if cmd_mag > max_speed:
                cmd_vel = (cmd_vel / cmd_mag) * max_speed

        return cmd_vel, "Pursuing to emergence point"

    def _execute_spiral_search(self, drone_pos: np.ndarray,
                              current_time: float,
                              max_speed: float) -> Tuple[np.ndarray, str]:
        """
        Execute spiral search pattern around last known position.

        Args:
            drone_pos: Current drone position
            current_time: Current time
            max_speed: Maximum velocity

        Returns:
            Tuple of (command_velocity, description)
        """
        if self.last_known_target is None:
            return np.zeros(3), "No search reference"

        # Simple spiral around last known position
        # Use time to parameterize spiral
        search_time = current_time - (self.occlusion_start_time or current_time)

        # Spiral parameters
        spiral_radius = 0.5 + 0.2 * search_time  # Expanding radius
        angular_freq = 0.5  # rad/s

        # Spiral center is last known position
        center = self.last_known_target.position

        # Calculate spiral point
        angle = angular_freq * search_time
        offset = np.array([
            spiral_radius * np.cos(angle),
            spiral_radius * np.sin(angle),
            0.0  # Keep same altitude
        ])

        target_point = center + offset

        # Move towards spiral point
        error = target_point - drone_pos
        error_mag = np.linalg.norm(error)

        if error_mag < 1e-6:
            return np.zeros(3), "Spiral search - at waypoint"

        cmd_vel = (error / error_mag) * max_speed
        return cmd_vel, "Executing spiral search"

    def is_target_lost(self) -> bool:
        """Check if target is currently lost"""
        return self.state in [SearchState.PREDICTING, SearchState.SEARCHING]

    def get_state(self) -> SearchState:
        """Get current search state"""
        return self.state

    def should_continue_mission(self) -> bool:
        """
        Determine if mission should continue despite lost target.

        Key principle: DO NOT STOP when target is lost.
        Stopping ensures you lose the race.

        Returns:
            True - always continue pursuing
        """
        return True  # Never give up!
