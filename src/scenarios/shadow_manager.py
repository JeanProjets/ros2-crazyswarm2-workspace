"""
Shadow Hunter Logic for Scenario 4
Handles targets disappearing behind walls and calculates emergence points
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class ObstacleInfo:
    """Information about an obstacle"""
    center: np.ndarray
    size: np.ndarray  # [width, height]
    bounds: Tuple[float, float, float, float]  # [x_min, x_max, y_min, y_max]


class OcclusionStrategy:
    """
    Strategy for handling occlusions and predicting emergence points
    """

    def __init__(self, obstacle_map=None):
        """
        Initialize occlusion strategy

        Args:
            obstacle_map: Map containing obstacle information
        """
        self.obstacle_map = obstacle_map
        self.max_projection_time = 3.0  # seconds
        self.emergence_point_cache = {}

    def calculate_emergence_point(self, target_last_pos: np.ndarray,
                                  target_velocity: np.ndarray,
                                  obstacle_map=None) -> Optional[np.ndarray]:
        """
        Calculate where target will emerge from obstacle shadow

        This function projects the target's velocity vector through obstacles
        to find where it will reappear in open space. This allows the drone
        to intercept rather than chase.

        Args:
            target_last_pos: Last known target position [x, y, z]
            target_velocity: Last known target velocity [vx, vy, vz]
            obstacle_map: Obstacle map (optional, uses self.obstacle_map if None)

        Returns:
            Predicted emergence point [x, y, z] or None if cannot predict
        """
        if obstacle_map is None:
            obstacle_map = self.obstacle_map

        if obstacle_map is None:
            # No map - cannot predict emergence
            return None

        velocity_mag = np.linalg.norm(target_velocity[:2])  # 2D velocity
        if velocity_mag < 0.1:
            # Target not moving - no emergence prediction
            return None

        # Project velocity vector forward in time
        direction = target_velocity / velocity_mag
        max_distance = velocity_mag * self.max_projection_time

        # Raycast along velocity vector
        num_samples = int(max_distance / 0.1) + 1  # 10cm resolution

        in_obstacle = obstacle_map.is_collision(target_last_pos[0], target_last_pos[1])
        emergence_point = None

        for i in range(num_samples):
            sample_dist = (i / num_samples) * max_distance
            sample_pos = target_last_pos + direction * sample_dist

            is_collision = obstacle_map.is_collision(sample_pos[0], sample_pos[1])

            # Detect transition from obstacle to free space
            if in_obstacle and not is_collision:
                emergence_point = sample_pos
                break

            in_obstacle = is_collision

        return emergence_point

    def find_interception_point(self, drone_pos: np.ndarray,
                               target_last_pos: np.ndarray,
                               target_velocity: np.ndarray,
                               obstacle_map=None) -> Optional[np.ndarray]:
        """
        Find optimal interception point when target is occluded

        Instead of chasing the "tail" (last seen position), fly to where
        the target will be (the emergence point).

        Args:
            drone_pos: Current drone position [x, y, z]
            target_last_pos: Last known target position
            target_velocity: Last known target velocity
            obstacle_map: Obstacle map

        Returns:
            Interception point or None
        """
        emergence_point = self.calculate_emergence_point(
            target_last_pos, target_velocity, obstacle_map
        )

        if emergence_point is None:
            # Fallback: fly to last known position
            return target_last_pos

        return emergence_point

    def get_obstacle_blocking_los(self, drone_pos: np.ndarray,
                                  target_pos: np.ndarray,
                                  obstacle_map=None) -> Optional[ObstacleInfo]:
        """
        Identify which obstacle is blocking line of sight

        Args:
            drone_pos: Drone position
            target_pos: Target position
            obstacle_map: Obstacle map

        Returns:
            ObstacleInfo of blocking obstacle or None
        """
        if obstacle_map is None:
            obstacle_map = self.obstacle_map

        if obstacle_map is None:
            return None

        # Raycast from drone to target
        direction = target_pos - drone_pos
        distance = np.linalg.norm(direction)

        if distance < 1e-6:
            return None

        direction_normalized = direction / distance

        # Sample along ray to find first obstacle intersection
        num_samples = int(distance / 0.1) + 1
        for i in range(num_samples):
            sample_dist = (i / num_samples) * distance
            sample_pos = drone_pos + direction_normalized * sample_dist

            if obstacle_map.is_collision(sample_pos[0], sample_pos[1]):
                # Found collision - identify which obstacle
                obstacle = self._identify_obstacle_at_point(
                    sample_pos, obstacle_map
                )
                return obstacle

        return None

    def _identify_obstacle_at_point(self, point: np.ndarray,
                                    obstacle_map) -> Optional[ObstacleInfo]:
        """
        Identify which obstacle exists at a given point

        Args:
            point: Point to check
            obstacle_map: Obstacle map

        Returns:
            ObstacleInfo or None
        """
        # This is a simplified implementation
        # In a real system, the obstacle_map would track individual obstacles
        # For now, we return a generic obstacle
        if hasattr(obstacle_map, 'get_obstacle_at'):
            return obstacle_map.get_obstacle_at(point[0], point[1])

        # Generic obstacle
        return ObstacleInfo(
            center=point[:2],
            size=np.array([1.0, 1.0]),
            bounds=(point[0] - 0.5, point[0] + 0.5,
                   point[1] - 0.5, point[1] + 0.5)
        )

    def calculate_flanking_positions(self, obstacle: ObstacleInfo,
                                     target_velocity: np.ndarray) -> List[np.ndarray]:
        """
        Calculate flanking positions around an obstacle

        When target goes behind obstacle moving left->right,
        calculate positions on both sides to reacquire quickly.

        Args:
            obstacle: Obstacle blocking view
            target_velocity: Target velocity vector

        Returns:
            List of flanking positions [left, right]
        """
        # Normalize velocity to get direction
        vel_2d = target_velocity[:2]
        speed = np.linalg.norm(vel_2d)

        if speed < 0.1:
            # No movement - use cardinal directions
            vel_2d = np.array([1.0, 0.0])
        else:
            vel_2d = vel_2d / speed

        # Perpendicular vector (left side)
        perp_left = np.array([-vel_2d[1], vel_2d[0]])
        perp_right = -perp_left

        # Get obstacle bounds
        x_min, x_max, y_min, y_max = obstacle.bounds

        # Calculate edge positions
        # Left edge
        left_pos = np.array([
            x_min if vel_2d[0] > 0 else x_max,
            (y_min + y_max) / 2,
            0.0  # z coordinate
        ])

        # Right edge
        right_pos = np.array([
            x_max if vel_2d[0] > 0 else x_min,
            (y_min + y_max) / 2,
            0.0
        ])

        # Project ahead slightly based on velocity
        projection_dist = speed * 0.5  # Half second ahead
        emergence_offset = vel_2d * projection_dist

        left_pos[:2] += emergence_offset
        right_pos[:2] += emergence_offset

        return [left_pos, right_pos]

    def predict_target_trajectory(self, last_pos: np.ndarray,
                                  last_vel: np.ndarray,
                                  dt: float) -> np.ndarray:
        """
        Predict target position after time dt using constant velocity model

        Args:
            last_pos: Last known position
            last_vel: Last known velocity
            dt: Time delta

        Returns:
            Predicted position
        """
        # Simple constant velocity prediction
        predicted_pos = last_pos + last_vel * dt
        return predicted_pos

    def should_coast(self, target_confidence: float,
                    time_since_seen: float,
                    obstacle_detected: bool) -> bool:
        """
        Decide if drone should coast (predict) vs search

        Args:
            target_confidence: Target detection confidence
            time_since_seen: Time since last valid detection
            obstacle_detected: Whether an obstacle is blocking view

        Returns:
            True if should coast to predicted position
        """
        # Coast if:
        # 1. Recently lost (< 2 seconds)
        # 2. Known to be behind obstacle
        # 3. Previously had good confidence

        if time_since_seen > 2.0:
            return False

        if not obstacle_detected:
            # Lost in open space - search instead
            return False

        return True


class ShadowHunter:
    """
    High-level shadow hunting coordinator
    Implements the "Ghost Mode" logic for tracking occluded targets
    """

    def __init__(self, occlusion_strategy: OcclusionStrategy):
        """
        Initialize shadow hunter

        Args:
            occlusion_strategy: Occlusion handling strategy
        """
        self.strategy = occlusion_strategy
        self.ghost_target_pos = None
        self.ghost_target_vel = None
        self.time_in_shadow = 0.0

    def update(self, drone_pos: np.ndarray,
              target_visible: bool,
              target_pos: Optional[np.ndarray],
              target_vel: Optional[np.ndarray],
              dt: float,
              obstacle_map=None) -> Tuple[Optional[np.ndarray], str]:
        """
        Update shadow hunting logic

        Args:
            drone_pos: Current drone position
            target_visible: Whether target is currently visible
            target_pos: Current/last target position
            target_vel: Current/last target velocity
            dt: Time delta
            obstacle_map: Obstacle map

        Returns:
            Tuple of (waypoint_to_fly_to, status_string)
        """
        if target_visible and target_pos is not None:
            # Target visible - reset ghost tracking
            self.ghost_target_pos = target_pos
            self.ghost_target_vel = target_vel
            self.time_in_shadow = 0.0
            return target_pos, "TRACKING_VISIBLE"

        # Target not visible - enter shadow mode
        self.time_in_shadow += dt

        if self.ghost_target_pos is None:
            # No previous track - cannot hunt shadow
            return None, "NO_TRACK"

        # Predict where target should be now
        predicted_pos = self.strategy.predict_target_trajectory(
            self.ghost_target_pos, self.ghost_target_vel, self.time_in_shadow
        )

        # Calculate emergence point
        emergence_point = self.strategy.calculate_emergence_point(
            self.ghost_target_pos, self.ghost_target_vel, obstacle_map
        )

        if emergence_point is not None:
            # Fly to emergence point to intercept
            return emergence_point, "INTERCEPTING_EMERGENCE"
        else:
            # Fly to predicted position
            return predicted_pos, "COASTING_PREDICTED"

    def reset(self):
        """Reset shadow hunter state"""
        self.ghost_target_pos = None
        self.ghost_target_vel = None
        self.time_in_shadow = 0.0
