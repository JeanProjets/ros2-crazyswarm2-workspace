"""
Obstacle Pursuit Behavior for Scenario 4
Implements path-following pursuit using Pure Pursuit algorithm
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Waypoint:
    """Represents a waypoint in 3D space"""
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        """Convert waypoint to numpy array"""
        return np.array([self.x, self.y, self.z])

    @staticmethod
    def from_array(arr: np.ndarray) -> 'Waypoint':
        """Create waypoint from numpy array"""
        return Waypoint(float(arr[0]), float(arr[1]), float(arr[2]))


class PathFollowerBehavior:
    """
    Implements path-following pursuit behavior for Scenario 4.

    Uses Pure Pursuit algorithm to follow a dynamic path while
    avoiding obstacles. Combines path following with feedforward
    velocity matching when line-of-sight to target exists.
    """

    def __init__(self, lookahead_dist: float = 0.5, max_speed: float = 1.0):
        """
        Initialize PathFollowerBehavior.

        Args:
            lookahead_dist: Lookahead distance for pure pursuit (meters)
            max_speed: Maximum velocity magnitude (m/s)
        """
        self.lookahead_dist = lookahead_dist
        self.max_speed = max_speed
        self.path: List[Waypoint] = []
        self.has_line_of_sight = False
        self.target_velocity = np.zeros(3)

    def update_path(self, new_path: List[Waypoint]) -> None:
        """
        Update the path to follow.

        Args:
            new_path: List of waypoints from Agent 1's path planner
        """
        self.path = new_path

    def set_line_of_sight(self, has_los: bool, target_vel: Optional[np.ndarray] = None) -> None:
        """
        Update line-of-sight status and target velocity.

        Args:
            has_los: Whether drone has line-of-sight to target
            target_vel: Target's current velocity (if available)
        """
        self.has_line_of_sight = has_los
        if target_vel is not None:
            self.target_velocity = target_vel
        else:
            self.target_velocity = np.zeros(3)

    def find_closest_point_on_path(self, current_pos: np.ndarray) -> Tuple[int, float]:
        """
        Find the closest point on the path to current position.

        Args:
            current_pos: Current position [x, y, z]

        Returns:
            Tuple of (segment_index, distance_along_segment)
        """
        if len(self.path) == 0:
            return 0, 0.0

        min_dist = float('inf')
        closest_idx = 0
        closest_t = 0.0

        for i in range(len(self.path) - 1):
            p1 = self.path[i].to_array()
            p2 = self.path[i + 1].to_array()

            # Project current position onto line segment
            segment = p2 - p1
            segment_length = np.linalg.norm(segment)

            if segment_length < 1e-6:
                continue

            segment_unit = segment / segment_length
            to_point = current_pos - p1

            # Parameter t along the segment (0 to 1)
            t = np.dot(to_point, segment_unit) / segment_length
            t = np.clip(t, 0.0, 1.0)

            # Closest point on this segment
            closest_on_segment = p1 + t * segment
            dist = np.linalg.norm(current_pos - closest_on_segment)

            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                closest_t = t

        return closest_idx, closest_t

    def find_lookahead_point(self, current_pos: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the lookahead point on the path using Pure Pursuit.

        Args:
            current_pos: Current position [x, y, z]

        Returns:
            Lookahead point as numpy array, or None if path is empty
        """
        if len(self.path) == 0:
            return None

        if len(self.path) == 1:
            return self.path[0].to_array()

        # Find closest point on path
        closest_idx, closest_t = self.find_closest_point_on_path(current_pos)

        # Start from closest point and look ahead
        accumulated_dist = 0.0
        start_point = (self.path[closest_idx].to_array() +
                      closest_t * (self.path[closest_idx + 1].to_array() -
                                   self.path[closest_idx].to_array()))

        # Search forward along path for lookahead distance
        for i in range(closest_idx, len(self.path) - 1):
            p1 = self.path[i].to_array() if i > closest_idx else start_point
            p2 = self.path[i + 1].to_array()

            segment = p2 - p1
            segment_length = np.linalg.norm(segment)

            if accumulated_dist + segment_length >= self.lookahead_dist:
                # Lookahead point is on this segment
                remaining = self.lookahead_dist - accumulated_dist
                t = remaining / segment_length if segment_length > 1e-6 else 0.0
                return p1 + t * segment

            accumulated_dist += segment_length

        # If we reach here, return the last point in the path
        return self.path[-1].to_array()

    def calculate_pursuit_velocity(self, drone_pos: np.ndarray) -> np.ndarray:
        """
        Calculate velocity command using Pure Pursuit algorithm.

        Implements the core algorithm from Scenario 4 requirements:
        1. Find the lookahead point on the path
        2. Calculate velocity vector towards that point
        3. Add feedforward if line-of-sight exists

        Args:
            drone_pos: Current drone position [x, y, z]

        Returns:
            Commanded velocity vector [vx, vy, vz]
        """
        # Find lookahead point
        carrot = self.find_lookahead_point(drone_pos)

        if carrot is None:
            return np.zeros(3)

        # Calculate error vector to lookahead point
        error_vec = carrot - drone_pos
        error_dist = np.linalg.norm(error_vec)

        if error_dist < 1e-6:
            # Already at lookahead point
            if len(self.path) > 0:
                # Move towards last waypoint
                error_vec = self.path[-1].to_array() - drone_pos
                error_dist = np.linalg.norm(error_vec)
                if error_dist < 1e-6:
                    return np.zeros(3)

        # Normalize and scale to max speed
        cmd_vel = (error_vec / error_dist) * self.max_speed

        # Add feedforward velocity if we have line-of-sight
        # This helps match target velocity when visible
        if self.has_line_of_sight:
            # Blend path-following with target velocity matching
            feedforward_gain = 0.3
            cmd_vel = cmd_vel + feedforward_gain * self.target_velocity

            # Re-limit to max speed
            cmd_mag = np.linalg.norm(cmd_vel)
            if cmd_mag > self.max_speed:
                cmd_vel = (cmd_vel / cmd_mag) * self.max_speed

        return cmd_vel

    def execute_pure_pursuit(self, current_pos: np.ndarray,
                            current_vel: np.ndarray) -> np.ndarray:
        """
        Execute pure pursuit behavior.

        Main interface method that combines path following with
        velocity feedforward when line-of-sight exists.

        Args:
            current_pos: Current position [x, y, z]
            current_vel: Current velocity [vx, vy, vz] (for future smoothing)

        Returns:
            Commanded velocity [vx, vy, vz]
        """
        return self.calculate_pursuit_velocity(current_pos)

    def is_path_complete(self, current_pos: np.ndarray, threshold: float = 0.2) -> bool:
        """
        Check if drone has reached the end of the path.

        Args:
            current_pos: Current position [x, y, z]
            threshold: Distance threshold for completion (meters)

        Returns:
            True if at end of path
        """
        if len(self.path) == 0:
            return True

        final_pos = self.path[-1].to_array()
        dist = np.linalg.norm(current_pos - final_pos)
        return dist < threshold


def calculate_pursuit_velocity(drone_pos: np.ndarray, path: List[Waypoint],
                               lookahead_dist: float, max_speed: float) -> np.ndarray:
    """
    Standalone function for Pure Pursuit velocity calculation.

    This is the algorithm from the Agent 2 specification.

    Args:
        drone_pos: Current drone position [x, y, z]
        path: List of waypoints
        lookahead_dist: Lookahead distance (meters)
        max_speed: Maximum velocity (m/s)

    Returns:
        Command velocity vector [vx, vy, vz]
    """
    follower = PathFollowerBehavior(lookahead_dist=lookahead_dist, max_speed=max_speed)
    follower.update_path(path)
    return follower.calculate_pursuit_velocity(drone_pos)
