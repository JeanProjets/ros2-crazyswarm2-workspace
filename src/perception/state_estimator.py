"""
State estimation and velocity calculation for mobile target tracking.

This module provides Kalman filtering and velocity estimation to convert
jittery pixel detections into smooth 3D position and velocity estimates.
Critical for Scenario 3 where the target is moving and we need accurate
velocity estimates for interception.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class TargetState:
    """Complete target state estimate."""
    position: np.ndarray  # [x, y, z] in meters (world frame)
    velocity: np.ndarray  # [vx, vy, vz] in m/s (world frame)
    confidence: float
    timestamp: float


class KalmanFilter1D:
    """
    1D Kalman filter for smoothing position measurements and estimating velocity.

    State vector: [position, velocity]
    Measurement: [position]

    This filter smooths jittery bounding box measurements and predicts
    the next position to compensate for processing latency.
    """

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        initial_velocity_uncertainty: float = 1.0
    ):
        """
        Initialize 1D Kalman filter.

        Args:
            process_noise: Process noise covariance (Q)
            measurement_noise: Measurement noise covariance (R)
            initial_velocity_uncertainty: Initial velocity uncertainty
        """
        # State: [position, velocity]
        self.x = np.array([0.0, 0.0])

        # Covariance matrix
        self.P = np.array([
            [1.0, 0.0],
            [0.0, initial_velocity_uncertainty]
        ])

        # Process noise covariance
        self.Q = np.array([
            [process_noise, 0.0],
            [0.0, process_noise]
        ])

        # Measurement noise covariance
        self.R = np.array([[measurement_noise]])

        # Measurement matrix (we only measure position)
        self.H = np.array([[1.0, 0.0]])

        self.initialized = False

    def predict(self, dt: float):
        """
        Prediction step.

        Args:
            dt: Time delta since last update (seconds)
        """
        # State transition matrix
        F = np.array([
            [1.0, dt],
            [0.0, 1.0]
        ])

        # Predict state
        self.x = F @ self.x

        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement: float):
        """
        Update step with new measurement.

        Args:
            measurement: Measured position
        """
        if not self.initialized:
            # Initialize state with first measurement
            self.x[0] = measurement
            self.x[1] = 0.0
            self.initialized = True
            return

        # Innovation (measurement residual)
        y = measurement - (self.H @ self.x)[0]

        # Innovation covariance
        S = (self.H @ self.P @ self.H.T + self.R)[0, 0]

        # Kalman gain
        K = (self.P @ self.H.T) / S

        # Update state
        self.x = self.x + K.flatten() * y

        # Update covariance
        I = np.eye(2)
        self.P = (I - np.outer(K, self.H)) @ self.P

    def get_state(self) -> Tuple[float, float]:
        """
        Get current state estimate.

        Returns:
            Tuple of (position, velocity)
        """
        return self.x[0], self.x[1]

    def predict_position(self, dt: float) -> float:
        """
        Predict position at future time (for latency compensation).

        Args:
            dt: Time into the future (seconds)

        Returns:
            Predicted position
        """
        return self.x[0] + self.x[1] * dt

    def reset(self):
        """Reset filter to initial state."""
        self.x = np.array([0.0, 0.0])
        self.P = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.initialized = False


class VelocityCalculator:
    """
    Calculate 3D target velocity with egomotion compensation.

    Converts pixel-space velocity to metric velocity and compensates for
    drone motion (egomotion) to obtain target velocity in world frame.
    """

    def __init__(
        self,
        focal_length: float = 200.0,  # pixels (typical for AI Deck)
        depth_estimate: float = 2.0    # meters (initial depth estimate)
    ):
        """
        Initialize velocity calculator.

        Args:
            focal_length: Camera focal length in pixels
            depth_estimate: Initial depth estimate in meters
        """
        self.focal_length = focal_length
        self.depth_estimate = depth_estimate

        # Kalman filters for X, Y, Z coordinates (in camera frame)
        self.kf_x = KalmanFilter1D(process_noise=0.01, measurement_noise=0.1)
        self.kf_y = KalmanFilter1D(process_noise=0.01, measurement_noise=0.1)
        self.kf_z = KalmanFilter1D(process_noise=0.01, measurement_noise=0.05)

        self.last_update_time: Optional[float] = None

    def update(
        self,
        pixel_x: float,
        pixel_y: float,
        depth: float,
        drone_velocity: np.ndarray,
        drone_yaw: float,
        timestamp: Optional[float] = None
    ) -> TargetState:
        """
        Update velocity estimate with new measurement.

        Args:
            pixel_x: Target x position in pixels (image frame)
            pixel_y: Target y position in pixels (image frame)
            depth: Estimated depth to target in meters
            drone_velocity: Drone velocity [vx, vy, vz] in world frame (m/s)
            drone_yaw: Drone yaw angle in radians
            timestamp: Measurement timestamp (uses current time if None)

        Returns:
            TargetState with position and velocity in world frame
        """
        if timestamp is None:
            timestamp = time.time()

        # Calculate dt
        if self.last_update_time is None:
            dt = 0.033  # Assume 30 Hz for first frame
        else:
            dt = max(0.001, timestamp - self.last_update_time)  # Min 1ms

        self.last_update_time = timestamp

        # Update depth estimate (can use more sophisticated depth estimation)
        self.depth_estimate = depth

        # Convert pixel coordinates to metric coordinates in camera frame
        # Camera frame: X=right, Y=down, Z=forward
        metric_x = (pixel_x * depth) / self.focal_length
        metric_y = (pixel_y * depth) / self.focal_length
        metric_z = depth

        # Predict (time update)
        self.kf_x.predict(dt)
        self.kf_y.predict(dt)
        self.kf_z.predict(dt)

        # Update (measurement update)
        self.kf_x.update(metric_x)
        self.kf_y.update(metric_y)
        self.kf_z.update(metric_z)

        # Get state estimates (position and velocity in camera frame)
        pos_x, vel_x = self.kf_x.get_state()
        pos_y, vel_y = self.kf_y.get_state()
        pos_z, vel_z = self.kf_z.get_state()

        # Camera-relative velocity
        vel_camera = np.array([vel_x, vel_y, vel_z])

        # Calculate target velocity in world frame with egomotion compensation
        vel_world = self._calculate_target_velocity_world(
            vel_camera,
            drone_velocity,
            drone_yaw
        )

        # Convert position to world frame
        pos_camera = np.array([pos_x, pos_y, pos_z])
        pos_world = self._rotate_camera_to_world(pos_camera, drone_yaw)

        return TargetState(
            position=pos_world,
            velocity=vel_world,
            confidence=1.0,  # Could be based on covariance or detection confidence
            timestamp=timestamp
        )

    def _calculate_target_velocity_world(
        self,
        vel_rel_camera: np.ndarray,
        vel_drone_world: np.ndarray,
        yaw_drone: float
    ) -> np.ndarray:
        """
        Decouple target motion from drone motion (egomotion compensation).

        If the drone moves right at 1 m/s and the target is stationary,
        the camera sees the target moving left at 1 m/s. We need to add
        the drone velocity to get the true target velocity in world frame.

        Args:
            vel_rel_camera: Velocity of target as seen by camera (camera frame)
            vel_drone_world: Velocity of drone (world frame)
            yaw_drone: Drone yaw angle in radians

        Returns:
            Target velocity in world frame
        """
        # Rotate camera-relative velocity to world frame
        vel_rel_world = self._rotate_camera_to_world(vel_rel_camera, yaw_drone)

        # Add drone velocity to compensate for egomotion
        # Target_vel_world = Target_vel_rel_world + Drone_vel_world
        vel_target = vel_rel_world + vel_drone_world

        return vel_target

    def _rotate_camera_to_world(
        self,
        vec_camera: np.ndarray,
        yaw: float
    ) -> np.ndarray:
        """
        Rotate vector from camera frame to world frame.

        Camera frame: X=right, Y=down, Z=forward
        World frame: X=forward, Y=left, Z=up (standard NED-like frame)

        Args:
            vec_camera: Vector in camera frame [x, y, z]
            yaw: Drone yaw angle in radians

        Returns:
            Vector in world frame
        """
        # Rotation matrix for yaw rotation around Z axis
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        # First, transform from camera frame to body frame
        # Camera: X=right, Y=down, Z=forward
        # Body: X=forward, Y=right, Z=down (standard)
        # Transformation: body_x = camera_z, body_y = camera_x, body_z = camera_y
        body_x = vec_camera[2]
        body_y = vec_camera[0]
        body_z = vec_camera[1]

        # Then rotate from body frame to world frame using yaw
        world_x = cos_yaw * body_x - sin_yaw * body_y
        world_y = sin_yaw * body_x + cos_yaw * body_y
        world_z = body_z

        return np.array([world_x, world_y, world_z])

    def predict_future_position(
        self,
        latency_ms: float,
        drone_velocity: np.ndarray,
        drone_yaw: float
    ) -> np.ndarray:
        """
        Predict target position compensating for processing latency.

        Args:
            latency_ms: Processing latency in milliseconds
            drone_velocity: Drone velocity in world frame
            drone_yaw: Drone yaw angle in radians

        Returns:
            Predicted position in world frame
        """
        dt = latency_ms / 1000.0

        # Get current filtered positions
        pos_x = self.kf_x.predict_position(dt)
        pos_y = self.kf_y.predict_position(dt)
        pos_z = self.kf_z.predict_position(dt)

        # Convert to world frame
        pos_camera = np.array([pos_x, pos_y, pos_z])
        pos_world = self._rotate_camera_to_world(pos_camera, drone_yaw)

        return pos_world

    def reset(self):
        """Reset all filters to initial state."""
        self.kf_x.reset()
        self.kf_y.reset()
        self.kf_z.reset()
        self.last_update_time = None
