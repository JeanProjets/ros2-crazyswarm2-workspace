"""
Dynamic Tracking Controller for Scenario 3 - Mobile Target Pursuit

This module implements predictive tracking with velocity feedforward control
for pursuing moving targets. It uses lead pursuit logic to intercept targets
by predicting their future positions based on current velocity.
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from scipy.signal import butter, filtfilt


@dataclass
class TargetState:
    """Represents the state of a tracked target"""
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    timestamp: float

    def __post_init__(self):
        self.position = np.array(self.position, dtype=float)
        self.velocity = np.array(self.velocity, dtype=float)


class VelocityFilter:
    """Low-pass filter for smoothing velocity estimates"""

    def __init__(self, cutoff_freq: float = 2.0, sample_rate: float = 10.0, order: int = 2):
        """
        Initialize velocity filter

        Args:
            cutoff_freq: Cutoff frequency in Hz
            sample_rate: Sample rate in Hz
            order: Filter order
        """
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        self.order = order
        self.history: List[np.ndarray] = []
        self.max_history = 10

    def filter(self, velocity: np.ndarray) -> np.ndarray:
        """
        Apply low-pass filter to velocity estimate

        Args:
            velocity: Current velocity measurement

        Returns:
            Filtered velocity
        """
        self.history.append(np.array(velocity))

        # Keep only recent history
        if len(self.history) > self.max_history:
            self.history.pop(0)

        # Need at least 3 samples for filtering
        if len(self.history) < 3:
            return velocity

        # Apply Butterworth filter
        try:
            nyquist = self.sample_rate / 2
            normal_cutoff = self.cutoff_freq / nyquist
            b, a = butter(self.order, normal_cutoff, btype='low', analog=False)

            # Stack history for filtering
            history_array = np.array(self.history)
            filtered = filtfilt(b, a, history_array, axis=0)

            return filtered[-1]
        except Exception:
            # Fallback to simple moving average
            return np.mean(self.history[-3:], axis=0)


class DroneController:
    """Base class for drone controllers (mocked for testing)"""

    def __init__(self, drone_id: str):
        self.drone_id = drone_id
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])

    def get_position(self) -> np.ndarray:
        """Get current drone position"""
        return self.position.copy()

    def get_velocity(self) -> np.ndarray:
        """Get current drone velocity"""
        return self.velocity.copy()

    def cmd_velocity_world(self, vx: float, vy: float, vz: float, yaw_rate: float):
        """
        Send velocity command in world frame

        Args:
            vx: Velocity in x direction (m/s)
            vy: Velocity in y direction (m/s)
            vz: Velocity in z direction (m/s)
            yaw_rate: Yaw rotation rate (rad/s)
        """
        # Mock implementation - in real system, this would send to Crazyflie
        self.velocity = np.array([vx, vy, vz])

    def cmd_position(self, x: float, y: float, z: float, yaw: float):
        """
        Send position command

        Args:
            x: Target x position (m)
            y: Target y position (m)
            z: Target z position (m)
            yaw: Target yaw angle (rad)
        """
        # Mock implementation
        pass


class DynamicTracker(DroneController):
    """
    Dynamic tracking controller for pursuing moving targets

    Implements predictive tracking with velocity feedforward to maintain
    pursuit of moving targets without lag or oscillation.
    """

    def __init__(
        self,
        drone_id: str,
        max_velocity: float = 2.0,
        lookahead_gain: float = 1.0,
        kp_position: float = 1.5,
        kd_velocity: float = 0.5,
        min_lookahead: float = 0.1,
        max_lookahead: float = 1.0
    ):
        """
        Initialize dynamic tracker

        Args:
            drone_id: Unique identifier for this drone
            max_velocity: Maximum allowed velocity (m/s)
            lookahead_gain: Gain for lookahead time calculation
            kp_position: Proportional gain for position error
            kd_velocity: Derivative gain for velocity matching
            min_lookahead: Minimum lookahead time (s)
            max_lookahead: Maximum lookahead time (s)
        """
        super().__init__(drone_id)

        self.max_velocity = max_velocity
        self.lookahead_gain = lookahead_gain
        self.kp_position = kp_position
        self.kd_velocity = kd_velocity
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead

        self.target_state: Optional[TargetState] = None
        self.velocity_filter = VelocityFilter()

    def update_target_state(
        self,
        position: Tuple[float, float, float],
        velocity: Tuple[float, float, float],
        timestamp: float = 0.0
    ):
        """
        Update the tracked target's state

        Args:
            position: Target position (x, y, z)
            velocity: Target velocity (vx, vy, vz)
            timestamp: Timestamp of measurement
        """
        # Filter velocity to reduce jitter
        filtered_vel = self.velocity_filter.filter(np.array(velocity))

        self.target_state = TargetState(
            position=position,
            velocity=filtered_vel,
            timestamp=timestamp
        )

    def compute_intercept_vector(
        self,
        drone_pos: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray
    ) -> np.ndarray:
        """
        Compute velocity command to intercept moving target using lead pursuit

        Args:
            drone_pos: Current drone position [x, y, z]
            target_pos: Current target position [x, y, z]
            target_vel: Current target velocity [vx, vy, vz]

        Returns:
            Commanded velocity vector [vx, vy, vz]
        """
        # 1. Calculate vector to current target position
        error_vector = target_pos - drone_pos
        distance = np.linalg.norm(error_vector)

        # 2. Estimate time to intercept
        # Use drone's max speed for time calculation
        if distance < 0.01:
            # Already at target
            return target_vel * self.kd_velocity

        time_to_go = distance / (self.max_velocity * self.lookahead_gain)

        # 3. Clip lookahead time to prevent overprediction
        lookahead_time = np.clip(time_to_go, self.min_lookahead, self.max_lookahead)

        # 4. Predict future target position (lead point)
        future_target_pos = target_pos + (target_vel * lookahead_time)

        # 5. Calculate command vector to lead point
        cmd_vector = future_target_pos - drone_pos

        # 6. Proportional control to lead point + velocity feedforward
        if np.linalg.norm(cmd_vector) > 0:
            # Position error term (proportional)
            vel_from_position = (cmd_vector / distance) * self.kp_position * distance

            # Velocity matching term (feedforward)
            vel_from_target = target_vel * self.kd_velocity

            # Combine both terms
            cmd_velocity = vel_from_position + vel_from_target
        else:
            # Just match target velocity if already at lead point
            cmd_velocity = target_vel * self.kd_velocity

        # 7. Apply velocity limits
        cmd_speed = np.linalg.norm(cmd_velocity)
        if cmd_speed > self.max_velocity:
            cmd_velocity = (cmd_velocity / cmd_speed) * self.max_velocity

        return cmd_velocity

    def match_velocity_hover(self, target_vel: np.ndarray) -> np.ndarray:
        """
        Compute velocity command to hover while matching target velocity

        This is used during the jamming phase where the drone needs to
        maintain relative position while both drone and target are moving.

        Args:
            target_vel: Target velocity to match [vx, vy, vz]

        Returns:
            Commanded velocity [vx, vy, vz]
        """
        # Cap the matched velocity to safety limits
        cmd_velocity = np.array(target_vel)
        cmd_speed = np.linalg.norm(cmd_velocity)

        if cmd_speed > self.max_velocity:
            cmd_velocity = (cmd_velocity / cmd_speed) * self.max_velocity

        return cmd_velocity

    def track_target(self, yaw_rate: float = 0.0) -> bool:
        """
        Execute one tracking update cycle

        Args:
            yaw_rate: Desired yaw rate (rad/s)

        Returns:
            True if tracking successfully, False if no target
        """
        if self.target_state is None:
            return False

        # Compute intercept velocity
        cmd_vel = self.compute_intercept_vector(
            self.get_position(),
            self.target_state.position,
            self.target_state.velocity
        )

        # Send velocity command
        self.cmd_velocity_world(
            cmd_vel[0],
            cmd_vel[1],
            cmd_vel[2],
            yaw_rate
        )

        return True


class TrackingController:
    """
    Legacy interface for tracking controller

    Provides standalone tracking functions without drone controller base.
    """

    def __init__(self, max_velocity: float = 2.0):
        """
        Initialize tracking controller

        Args:
            max_velocity: Maximum velocity limit (m/s)
        """
        self.max_velocity = max_velocity

    def compute_lead_pursuit(
        self,
        drone_pos: Tuple[float, float, float],
        target_pos: Tuple[float, float, float],
        target_vel: Tuple[float, float, float],
        speed_gain: float = 1.0
    ) -> np.ndarray:
        """
        Calculate velocity command to intercept moving target

        Args:
            drone_pos: Current drone position (x, y, z)
            target_pos: Current target position (x, y, z)
            target_vel: Current target velocity (vx, vy, vz)
            speed_gain: Speed multiplier for pursuit

        Returns:
            Command velocity [vx, vy, vz]
        """
        # 1. Distance to target
        dist_vector = np.array(target_pos) - np.array(drone_pos)
        dist = np.linalg.norm(dist_vector)

        if dist < 0.01:
            return np.array([0.0, 0.0, 0.0])

        # 2. Estimate time to intercept
        time_to_go = dist / speed_gain if speed_gain > 0 else 1.0

        # 3. Predict future target position
        # Cap prediction time to avoid overshooting
        pred_time = np.clip(time_to_go, 0.1, 1.0)
        future_target = np.array(target_pos) + (np.array(target_vel) * pred_time)

        # 4. Calculate command vector
        cmd_vector = future_target - np.array(drone_pos)

        # Normalize and scale
        if np.linalg.norm(cmd_vector) > 0:
            cmd_vel = (cmd_vector / np.linalg.norm(cmd_vector)) * speed_gain
        else:
            cmd_vel = np.array([0.0, 0.0, 0.0])

        # Apply velocity limits
        cmd_speed = np.linalg.norm(cmd_vel)
        if cmd_speed > self.max_velocity:
            cmd_vel = (cmd_vel / cmd_speed) * self.max_velocity

        return cmd_vel
