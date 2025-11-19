"""
Elastic Formation Behavior for Scenario 4
Implements loose formation that deforms around obstacles
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class FormationOffset:
    """Represents a formation offset in 3D space"""
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([self.x, self.y, self.z])


class GridMap:
    """
    Mock GridMap class for collision checking.
    In production, this would interface with Agent 1's actual GridMap.
    """

    def __init__(self, resolution: float = 0.1, inflation_radius: float = 0.3):
        """
        Initialize GridMap.

        Args:
            resolution: Grid cell size (meters)
            inflation_radius: Safety margin around obstacles (meters)
        """
        self.resolution = resolution
        self.inflation_radius = inflation_radius
        self.obstacles = []  # List of obstacle positions

    def is_collision(self, point: np.ndarray) -> bool:
        """
        Check if a point is in collision with obstacles.

        Args:
            point: Position to check [x, y, z]

        Returns:
            True if point is inside an obstacle or its inflation zone
        """
        for obs_pos in self.obstacles:
            dist = np.linalg.norm(point - obs_pos)
            if dist < self.inflation_radius:
                return True
        return False

    def add_obstacle(self, position: np.ndarray) -> None:
        """Add an obstacle to the map"""
        self.obstacles.append(position)

    def find_nearest_free(self, point: np.ndarray, max_search_radius: float = 2.0,
                         num_samples: int = 100) -> np.ndarray:
        """
        Find nearest free space to a given point.

        Uses a spiral search pattern radiating outward from the point.

        Args:
            point: Starting point [x, y, z]
            max_search_radius: Maximum search distance (meters)
            num_samples: Number of sample points to check

        Returns:
            Nearest free point
        """
        if not self.is_collision(point):
            return point

        # Sample points in expanding circles around the original point
        best_free_point = point.copy()
        min_dist = float('inf')

        for i in range(num_samples):
            # Generate sample in spherical coordinates
            radius = max_search_radius * (i / num_samples) ** 0.5
            theta = 2 * np.pi * i / num_samples
            phi = np.arccos(1 - 2 * (i / num_samples))

            # Convert to Cartesian
            dx = radius * np.sin(phi) * np.cos(theta)
            dy = radius * np.sin(phi) * np.sin(theta)
            dz = radius * np.cos(phi)

            sample = point + np.array([dx, dy, dz])

            if not self.is_collision(sample):
                dist = np.linalg.norm(sample - point)
                if dist < min_dist:
                    min_dist = dist
                    best_free_point = sample

        return best_free_point


class ElasticFormation:
    """
    Implements elastic "rubber band" formation behavior.

    Allows follower drones to deform formation around obstacles
    while maintaining approximate formation when possible.
    """

    def __init__(self, formation_offset: FormationOffset,
                 min_separation: float = 0.5,
                 max_stretch: float = 3.0):
        """
        Initialize ElasticFormation.

        Args:
            formation_offset: Desired offset from leader (default formation)
            min_separation: Minimum safe distance between drones (meters)
            max_stretch: Maximum allowed formation stretch (meters)
        """
        self.formation_offset = formation_offset
        self.min_separation = min_separation
        self.max_stretch = max_stretch
        self.grid_map: Optional[GridMap] = None

    def set_grid_map(self, grid_map: GridMap) -> None:
        """
        Set the occupancy grid map from Agent 1.

        Args:
            grid_map: GridMap instance with obstacle information
        """
        self.grid_map = grid_map

    def calculate_ideal_follower_position(self, leader_pos: np.ndarray,
                                         leader_vel: np.ndarray) -> np.ndarray:
        """
        Calculate the ideal formation position for the follower.

        Args:
            leader_pos: Leader position [x, y, z]
            leader_vel: Leader velocity [vx, vy, vz]

        Returns:
            Ideal follower position [x, y, z]
        """
        # Standard formation: offset behind and to the side
        ideal_pos = leader_pos + self.formation_offset.to_array()

        return ideal_pos

    def calculate_loose_follower_goal(self, leader_pos: np.ndarray,
                                     leader_vel: np.ndarray,
                                     follower_pos: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Calculate adaptive follower goal that deforms around obstacles.

        This is the core "elastic" behavior:
        1. Compute ideal formation position
        2. If ideal position is in obstacle, project to nearest free space
        3. If stretch exceeds maximum, signal for path planning

        Args:
            leader_pos: Leader position [x, y, z]
            leader_vel: Leader velocity [vx, vy, vz]
            follower_pos: Current follower position [x, y, z]

        Returns:
            Tuple of (goal_position, needs_path_planning)
        """
        # Calculate ideal formation position
        ideal_pos = self.calculate_ideal_follower_position(leader_pos, leader_vel)

        needs_planning = False

        # Check if ideal position is safe (if map available)
        if self.grid_map is not None:
            if self.grid_map.is_collision(ideal_pos):
                # Ideal position is inside obstacle - find nearest free space
                safe_pos = self.grid_map.find_nearest_free(ideal_pos)

                # Check if formation is stretched too far
                stretch_distance = np.linalg.norm(safe_pos - ideal_pos)

                if stretch_distance > self.max_stretch:
                    # Formation is stretched beyond limits
                    # Follower should use path planner to reach ideal position
                    needs_planning = True

                ideal_pos = safe_pos

        # Ensure minimum separation from leader (always check, regardless of map)
        to_follower = ideal_pos - leader_pos
        dist = np.linalg.norm(to_follower)

        if dist < self.min_separation and dist > 1e-6:
            # Too close - push away
            ideal_pos = leader_pos + (to_follower / dist) * self.min_separation

        return ideal_pos, needs_planning

    def calculate_formation_velocity(self, leader_pos: np.ndarray,
                                   leader_vel: np.ndarray,
                                   follower_pos: np.ndarray,
                                   follower_vel: np.ndarray,
                                   max_speed: float = 1.0) -> np.ndarray:
        """
        Calculate velocity command for follower to maintain elastic formation.

        Args:
            leader_pos: Leader position [x, y, z]
            leader_vel: Leader velocity [vx, vy, vz]
            follower_pos: Current follower position [x, y, z]
            follower_vel: Current follower velocity [vx, vy, vz]
            max_speed: Maximum velocity (m/s)

        Returns:
            Command velocity for follower [vx, vy, vz]
        """
        # Get adaptive goal position
        goal_pos, needs_planning = self.calculate_loose_follower_goal(
            leader_pos, leader_vel, follower_pos
        )

        # Calculate error
        error = goal_pos - follower_pos
        error_mag = np.linalg.norm(error)

        if error_mag < 1e-6:
            # Already at goal - match leader velocity
            return leader_vel

        # Proportional controller with feedforward
        kp = 2.0  # Proportional gain
        feedforward_gain = 0.5

        # Position error term
        vel_command = kp * error

        # Add leader velocity feedforward for smooth following
        vel_command = vel_command + feedforward_gain * leader_vel

        # Limit to max speed
        vel_mag = np.linalg.norm(vel_command)
        if vel_mag > max_speed:
            vel_command = (vel_command / vel_mag) * max_speed

        return vel_command


def get_valid_formation_point(ideal_point: np.ndarray, grid_map: GridMap) -> np.ndarray:
    """
    Standalone function to validate and adjust formation point.

    If ideal point is inside an obstacle, move it to the surface.
    This is the algorithm from the Agent 2 specification.

    Args:
        ideal_point: Desired formation position [x, y, z]
        grid_map: GridMap for collision checking

    Returns:
        Valid formation point [x, y, z]
    """
    if not grid_map.is_collision(ideal_point):
        return ideal_point

    # Search for nearest free cell
    safe_point = grid_map.find_nearest_free(ideal_point)
    return safe_point
