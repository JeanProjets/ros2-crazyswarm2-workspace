"""
Patrol Patterns Module for Crazyflie Drone Swarm

Implements search patterns for different drone roles in Scenario 1.
Provides efficient coverage algorithms for safety zone verification and area patrol.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import logging


@dataclass
class Waypoint:
    """Represents a waypoint in 3D space with associated flight parameters."""
    position: Tuple[float, float, float]
    yaw: float = 0.0
    speed: float = 0.5
    wait_time: float = 0.0

    def __post_init__(self):
        """Validate waypoint parameters."""
        x, y, z = self.position
        if z < 0:
            raise ValueError(f"Invalid altitude: {z}. Must be >= 0")
        if self.speed <= 0:
            raise ValueError(f"Invalid speed: {self.speed}. Must be > 0")


class PatternType(Enum):
    """Types of patrol patterns available."""
    RECTANGULAR_SWEEP = "rectangular_sweep"
    SPIRAL_SEARCH = "spiral_search"
    LAWN_MOWER = "lawn_mower"
    PERIMETER_SCAN = "perimeter_scan"


class SafetyZonePatrol:
    """
    Implements patrol patterns for safety zone verification by neutral drones.

    Safety zone requirements:
    - 3m x 3m area
    - Complete coverage verification
    - Two drones working in parallel
    - Maximum time: 30 seconds
    """

    def __init__(self, drone_id: str, zone_id: int = 0):
        """
        Initialize safety zone patrol.

        Args:
            drone_id: Unique identifier for the drone
            zone_id: Which half of the zone this drone covers (0 or 1)
        """
        self.drone_id = drone_id
        self.zone_id = zone_id
        self.logger = logging.getLogger(f"SafetyZonePatrol_{drone_id}")
        self.coverage_percentage = 0.0

    def rectangular_sweep(
        self,
        bounds: Dict[str, Tuple[float, float]],
        height: float,
        speed: float = 0.5
    ) -> List[Waypoint]:
        """
        Generate rectangular sweep pattern for safety zone.

        Args:
            bounds: Dictionary with 'x_range' and 'y_range' tuples
            height: Flight altitude in meters
            speed: Flight speed in m/s

        Returns:
            List of waypoints covering the rectangular area
        """
        waypoints = []
        x_min, x_max = bounds['x_range']
        y_min, y_max = bounds['y_range']

        # Calculate sweep spacing based on camera FOV (60° horizontal, 2m detection)
        # At 1.5m altitude, FOV covers ~1.7m width
        spacing = 1.5  # meters between parallel sweeps

        # Generate parallel sweeps
        y_positions = np.arange(y_min + 0.2, y_max, spacing)

        for i, y in enumerate(y_positions):
            if i % 2 == 0:
                # Left to right
                waypoints.append(Waypoint((x_min, y, height), speed=speed))
                waypoints.append(Waypoint((x_max, y, height), speed=speed))
            else:
                # Right to left
                waypoints.append(Waypoint((x_max, y, height), speed=speed))
                waypoints.append(Waypoint((x_min, y, height), speed=speed))

        self.logger.info(f"Generated {len(waypoints)} waypoints for rectangular sweep")
        return waypoints

    def spiral_search(
        self,
        center: Tuple[float, float, float],
        radius: float,
        height: float,
        num_loops: int = 3
    ) -> List[Waypoint]:
        """
        Generate spiral search pattern from center outward.

        Args:
            center: (x, y, z) center point of spiral
            radius: Maximum radius of spiral in meters
            height: Flight altitude
            num_loops: Number of spiral loops

        Returns:
            List of waypoints forming a spiral pattern
        """
        waypoints = []
        cx, cy, _ = center

        # Generate spiral with increasing radius
        num_points = num_loops * 8  # 8 points per loop
        for i in range(num_points):
            angle = (i / num_points) * 2 * np.pi * num_loops
            r = (i / num_points) * radius

            x = cx + r * np.cos(angle)
            y = cy + r * np.sin(angle)

            waypoints.append(Waypoint((x, y, height), yaw=angle, speed=0.4))

        self.logger.info(f"Generated {len(waypoints)} waypoints for spiral search")
        return waypoints

    def coverage_percentage(self) -> float:
        """
        Calculate the percentage of area covered by the patrol.

        Returns:
            Coverage percentage (0-100)
        """
        return self.coverage_percentage


class AreaPatrol:
    """
    Implements search patterns for main area patrol.

    Requirements:
    - Cover x=[3, 10], y=[0, 6], z=4
    - 70% minimum coverage
    - Maximum time: 2 minutes
    - Camera FOV: 60° horizontal, 2m detection range
    """

    def __init__(self, drone_id: str, area_bounds: Dict[str, Tuple[float, float]]):
        """
        Initialize area patrol.

        Args:
            drone_id: Unique identifier for the drone
            area_bounds: Dictionary with 'x_range' and 'y_range'
        """
        self.drone_id = drone_id
        self.area_bounds = area_bounds
        self.waypoints: List[Waypoint] = []
        self.current_waypoint_idx = 0
        self.searched_areas: List[Tuple[float, float, float, float]] = []
        self.logger = logging.getLogger(f"AreaPatrol_{drone_id}")

        # Initialize coverage grid for tracking
        x_range = area_bounds['x_range']
        y_range = area_bounds['y_range']
        grid_size = 0.5  # 0.5m resolution
        self.grid_x = int((x_range[1] - x_range[0]) / grid_size) + 1
        self.grid_y = int((y_range[1] - y_range[0]) / grid_size) + 1
        self.coverage_map = np.zeros((self.grid_x, self.grid_y))

    def lawn_mower_pattern(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        height: float,
        spacing: float = 2.0
    ) -> List[Waypoint]:
        """
        Generate optimized lawn-mower pattern for area coverage.

        Args:
            x_range: (x_min, x_max) bounds
            y_range: (y_min, y_max) bounds
            height: Flight altitude
            spacing: Distance between parallel lines in meters

        Returns:
            List of waypoints forming lawn-mower pattern
        """
        waypoints = []
        x_min, x_max = x_range
        y_min, y_max = y_range

        # Start position as specified in requirements
        start_pos = (3.0, 5.0, 4.0)
        waypoints.append(Waypoint(start_pos, speed=0.3))  # Slow start

        # Generate lawn-mower lines with 2m spacing
        y_lines = np.arange(y_min + 0.5, y_max, spacing)

        for i, y in enumerate(y_lines):
            # Determine speed: slower when scanning, faster in transit
            scan_speed = 0.2  # Slow speed for scanning
            cruise_speed = 0.5  # Faster speed for transit

            if i % 2 == 0:
                # Left to right sweep
                waypoints.append(Waypoint((x_min, y, height), speed=scan_speed))
                waypoints.append(Waypoint((x_max, y, height), speed=scan_speed))
            else:
                # Right to left sweep
                waypoints.append(Waypoint((x_max, y, height), speed=scan_speed))
                waypoints.append(Waypoint((x_min, y, height), speed=scan_speed))

            # Add brief wait at end of each line for camera to stabilize
            waypoints[-1] = Waypoint(
                waypoints[-1].position,
                yaw=waypoints[-1].yaw,
                speed=waypoints[-1].speed,
                wait_time=0.5
            )

        self.waypoints = waypoints
        self.logger.info(f"Generated {len(waypoints)} waypoints for lawn-mower pattern")
        return waypoints

    def perimeter_scan(
        self,
        bounds: Dict[str, Tuple[float, float]],
        height: float
    ) -> List[Waypoint]:
        """
        Generate perimeter scanning pattern around area boundary.

        Args:
            bounds: Dictionary with 'x_range' and 'y_range'
            height: Flight altitude

        Returns:
            List of waypoints forming perimeter path
        """
        waypoints = []
        x_min, x_max = bounds['x_range']
        y_min, y_max = bounds['y_range']

        # Perimeter corners (clockwise)
        corners = [
            (x_min, y_min, height),
            (x_max, y_min, height),
            (x_max, y_max, height),
            (x_min, y_max, height),
            (x_min, y_min, height)  # Return to start
        ]

        for corner in corners:
            waypoints.append(Waypoint(corner, speed=0.4))

        self.logger.info(f"Generated {len(waypoints)} waypoints for perimeter scan")
        return waypoints

    def adaptive_search(
        self,
        searched_areas: List[Tuple[float, float, float, float]],
        remaining_areas: List[Tuple[float, float, float, float]],
        height: float = 4.0
    ) -> List[Waypoint]:
        """
        Generate adaptive search pattern focusing on unsearched areas.

        Args:
            searched_areas: List of (x_min, x_max, y_min, y_max) tuples already searched
            remaining_areas: List of (x_min, x_max, y_min, y_max) tuples to search
            height: Flight altitude

        Returns:
            List of waypoints prioritizing unsearched areas
        """
        waypoints = []

        # Sort remaining areas by size (search larger areas first)
        sorted_areas = sorted(
            remaining_areas,
            key=lambda a: (a[1] - a[0]) * (a[3] - a[2]),
            reverse=True
        )

        for area in sorted_areas:
            x_min, x_max, y_min, y_max = area

            # Generate mini lawn-mower for this area
            area_waypoints = self.lawn_mower_pattern(
                (x_min, x_max),
                (y_min, y_max),
                height,
                spacing=1.5
            )
            waypoints.extend(area_waypoints)

        self.logger.info(f"Generated {len(waypoints)} waypoints for adaptive search")
        return waypoints

    def get_coverage_percentage(self) -> float:
        """
        Calculate current coverage percentage of the patrol area.

        Returns:
            Percentage of area covered (0-100)
        """
        if self.coverage_map.size == 0:
            return 0.0

        covered_cells = np.sum(self.coverage_map > 0)
        total_cells = self.coverage_map.size

        return (covered_cells / total_cells) * 100.0

    def mark_area_searched(self, position: Tuple[float, float], fov_radius: float = 1.0):
        """
        Mark an area as searched in the coverage map.

        Args:
            position: (x, y) position of drone
            fov_radius: Radius of camera field of view in meters
        """
        x, y = position
        x_min, x_max = self.area_bounds['x_range']
        y_min, y_max = self.area_bounds['y_range']

        # Convert world coordinates to grid indices
        grid_size = 0.5
        grid_x = int((x - x_min) / grid_size)
        grid_y = int((y - y_min) / grid_size)

        # Mark surrounding cells as covered based on FOV
        fov_cells = int(fov_radius / grid_size)
        for dx in range(-fov_cells, fov_cells + 1):
            for dy in range(-fov_cells, fov_cells + 1):
                gx = grid_x + dx
                gy = grid_y + dy
                if 0 <= gx < self.grid_x and 0 <= gy < self.grid_y:
                    self.coverage_map[gx, gy] = 1


def generate_coverage_path(
    area_bounds: Tuple[float, float, float, float],
    drone_fov: float,
    overlap: float = 0.1,
    height: float = 4.0
) -> List[Waypoint]:
    """
    Generate waypoints for complete area coverage.

    Args:
        area_bounds: (x_min, x_max, y_min, y_max) area boundaries
        drone_fov: Camera field of view in meters
        overlap: Percentage overlap between sweeps (0.0-1.0)
        height: Flight altitude

    Returns:
        List of (x, y, z) waypoints
    """
    x_min, x_max, y_min, y_max = area_bounds

    # Calculate spacing with overlap
    spacing = drone_fov * (1.0 - overlap)

    waypoints = []
    y_positions = np.arange(y_min, y_max + spacing, spacing)

    for i, y in enumerate(y_positions):
        if i % 2 == 0:
            waypoints.append(Waypoint((x_min, y, height)))
            waypoints.append(Waypoint((x_max, y, height)))
        else:
            waypoints.append(Waypoint((x_max, y, height)))
            waypoints.append(Waypoint((x_min, y, height)))

    return waypoints


def smooth_trajectory(waypoints: List[Waypoint], smoothing_factor: float = 0.3) -> List[Waypoint]:
    """
    Apply trajectory smoothing to waypoint list for smoother flight.

    Args:
        waypoints: Original waypoint list
        smoothing_factor: Amount of smoothing (0.0 = none, 1.0 = maximum)

    Returns:
        Smoothed waypoint list
    """
    if len(waypoints) < 3:
        return waypoints

    smoothed = [waypoints[0]]  # Keep first waypoint

    for i in range(1, len(waypoints) - 1):
        prev_pos = waypoints[i - 1].position
        curr_pos = waypoints[i].position
        next_pos = waypoints[i + 1].position

        # Apply smoothing: blend current position with average of neighbors
        smoothed_x = curr_pos[0] * (1 - smoothing_factor) + \
                     (prev_pos[0] + next_pos[0]) / 2 * smoothing_factor
        smoothed_y = curr_pos[1] * (1 - smoothing_factor) + \
                     (prev_pos[1] + next_pos[1]) / 2 * smoothing_factor
        smoothed_z = curr_pos[2]  # Don't smooth altitude

        smoothed.append(Waypoint(
            (smoothed_x, smoothed_y, smoothed_z),
            yaw=waypoints[i].yaw,
            speed=waypoints[i].speed,
            wait_time=waypoints[i].wait_time
        ))

    smoothed.append(waypoints[-1])  # Keep last waypoint

    return smoothed
