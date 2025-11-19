"""
Corner-Aware Search Pattern for Scenario 2

This module implements search patterns optimized for finding targets in corners
and along perimeters, unlike the center-focused patterns used in Scenario 1.
"""

from typing import List, Tuple
from dataclasses import dataclass
import math


@dataclass
class Waypoint:
    """Represents a 3D waypoint with velocity constraint"""
    x: float
    y: float
    z: float
    velocity: float = 1.0  # m/s


@dataclass
class CageDimensions:
    """Cage boundary dimensions"""
    x_min: float = 0.0
    x_max: float = 10.0
    y_min: float = 0.0
    y_max: float = 6.0
    z_min: float = 0.0
    z_max: float = 10.0


class CornerBiasPatrol:
    """
    Search pattern optimized for corner targets.

    Unlike Scenario 1 (center focus), Scenario 2's target at (9.5, 0.5, 5)
    requires prioritizing extremities. Standard lawn-mower patterns are
    inefficient for corner searches.

    Strategy: "Perimeter then Fill"
    1. Fly X-axis edge (Y=0.5) from X=3 to X=9.5 - high probability zone
    2. If not found, sweep Y-axis edges
    3. Finally, fill center if needed
    """

    def __init__(self, cage_dims: CageDimensions = None):
        """
        Initialize corner-biased patrol.

        Args:
            cage_dims: Cage dimensions (defaults to standard 10x6x10)
        """
        self.cage_dims = cage_dims or CageDimensions()

        # Flight parameters
        self.cruise_speed = 1.0  # m/s - battery efficient
        self.slow_speed = 0.5    # m/s - for corners
        self.crawl_speed = 0.2   # m/s - near walls

        # Safety margins
        self.wall_safety_margin = 0.5  # meters from walls
        self.corner_deceleration_distance = 1.5  # start slowing down

        # Search altitude
        self.search_altitude = 5.0  # meters

    def prioritize_corners(self) -> List[Waypoint]:
        """
        Generate waypoints that prioritize corner and edge searches.

        Returns:
            List of waypoints forming the corner-priority search pattern
        """
        waypoints = []

        # Phase 1: Right edge sweep (Y near 0, high X values)
        # This is the PRIMARY search zone for Scenario 2
        waypoints.extend(self._generate_right_edge_sweep())

        # Phase 2: Other perimeter sections
        waypoints.extend(self._generate_remaining_perimeter())

        # Phase 3: Interior fill (if target still not found)
        waypoints.extend(self._generate_interior_fill())

        return waypoints

    def scan_perimeter_first(self) -> List[Waypoint]:
        """
        Generate a complete perimeter scan pattern.

        This focuses entirely on edges before checking interior.

        Returns:
            List of waypoints following the perimeter
        """
        waypoints = []

        # Start at safe entry point
        entry_point = Waypoint(
            x=3.0,
            y=self.wall_safety_margin,
            z=self.search_altitude,
            velocity=self.cruise_speed
        )
        waypoints.append(entry_point)

        # Right edge (Y ~ 0): Most critical for Scenario 2
        waypoints.extend(self._sweep_right_edge())

        # Top edge (X ~ 10)
        waypoints.extend(self._sweep_top_edge())

        # Left edge (Y ~ 6)
        waypoints.extend(self._sweep_left_edge())

        # Bottom edge (X ~ 0)
        waypoints.extend(self._sweep_bottom_edge())

        return waypoints

    def _generate_right_edge_sweep(self) -> List[Waypoint]:
        """
        Generate waypoints for right edge sweep (Y near 0).

        Target at (9.5, 0.5, 5) means this is the highest priority zone.
        """
        waypoints = []

        y_pos = self.cage_dims.y_min + self.wall_safety_margin
        z_pos = self.search_altitude

        # Sweep from X=3.0 to X=9.5
        x_start = 3.0
        x_end = self.cage_dims.x_max - self.wall_safety_margin

        # Create waypoints with progressive deceleration
        num_points = 8
        for i in range(num_points):
            x_pos = x_start + (x_end - x_start) * (i / (num_points - 1))

            # Calculate velocity based on distance to far wall
            velocity = self._calculate_corner_velocity_for_x(x_pos)

            waypoints.append(Waypoint(
                x=x_pos,
                y=y_pos,
                z=z_pos,
                velocity=velocity
            ))

        return waypoints

    def _generate_remaining_perimeter(self) -> List[Waypoint]:
        """Generate waypoints for remaining perimeter sections."""
        waypoints = []

        # Top edge (high X values, varying Y)
        for y in [1.5, 3.0, 4.5]:
            waypoints.append(Waypoint(
                x=self.cage_dims.x_max - self.wall_safety_margin,
                y=y,
                z=self.search_altitude,
                velocity=self.crawl_speed
            ))

        # Left edge (low X values, high Y)
        y_pos = self.cage_dims.y_max - self.wall_safety_margin
        for x in [8.0, 6.0, 4.0, 2.0]:
            waypoints.append(Waypoint(
                x=x,
                y=y_pos,
                z=self.search_altitude,
                velocity=self.cruise_speed
            ))

        return waypoints

    def _generate_interior_fill(self) -> List[Waypoint]:
        """Generate lawn-mower pattern for interior (fallback search)."""
        waypoints = []

        x_step = 2.0
        y_step = 1.5

        x_values = list(range(
            int(self.cage_dims.x_min + 1),
            int(self.cage_dims.x_max - 1),
            int(x_step)
        ))

        y_values = list(range(
            int(self.cage_dims.y_min + 1),
            int(self.cage_dims.y_max - 1),
            int(y_step)
        ))

        # Alternate Y direction for efficiency
        for i, x in enumerate(x_values):
            y_list = y_values if i % 2 == 0 else reversed(y_values)
            for y in y_list:
                waypoints.append(Waypoint(
                    x=float(x),
                    y=float(y),
                    z=self.search_altitude,
                    velocity=self.cruise_speed
                ))

        return waypoints

    def _sweep_right_edge(self) -> List[Waypoint]:
        """Detailed sweep of right edge (Y ~ 0)."""
        waypoints = []
        y_pos = self.cage_dims.y_min + self.wall_safety_margin

        # Move along X axis with deceleration near far wall
        x_positions = [3.0, 5.0, 7.0, 8.5, 9.0, 9.5]

        for x_pos in x_positions:
            velocity = self._calculate_corner_velocity_for_x(x_pos)
            waypoints.append(Waypoint(
                x=x_pos,
                y=y_pos,
                z=self.search_altitude,
                velocity=velocity
            ))

        return waypoints

    def _sweep_top_edge(self) -> List[Waypoint]:
        """Sweep along top edge (X ~ 10)."""
        waypoints = []
        x_pos = self.cage_dims.x_max - self.wall_safety_margin

        # Move along Y axis
        for y_pos in [1.0, 2.0, 3.0, 4.0, 5.0, 5.5]:
            waypoints.append(Waypoint(
                x=x_pos,
                y=y_pos,
                z=self.search_altitude,
                velocity=self.crawl_speed
            ))

        return waypoints

    def _sweep_left_edge(self) -> List[Waypoint]:
        """Sweep along left edge (Y ~ 6)."""
        waypoints = []
        y_pos = self.cage_dims.y_max - self.wall_safety_margin

        # Move back along X axis
        for x_pos in [9.5, 8.0, 6.0, 4.0, 2.0]:
            waypoints.append(Waypoint(
                x=x_pos,
                y=y_pos,
                z=self.search_altitude,
                velocity=self.cruise_speed
            ))

        return waypoints

    def _sweep_bottom_edge(self) -> List[Waypoint]:
        """Sweep along bottom edge (X ~ 0)."""
        waypoints = []
        x_pos = self.cage_dims.x_min + self.wall_safety_margin

        # Move along Y axis back to start
        for y_pos in [5.0, 4.0, 3.0, 2.0, 1.0, 0.5]:
            waypoints.append(Waypoint(
                x=x_pos,
                y=y_pos,
                z=self.search_altitude,
                velocity=self.cruise_speed
            ))

        return waypoints

    def _calculate_corner_velocity_for_x(self, x_pos: float) -> float:
        """
        Calculate velocity based on distance to X wall.

        Implements aggressive braking curve when approaching cage X limit.

        Args:
            x_pos: Current X position

        Returns:
            Safe velocity in m/s
        """
        dist_to_wall = self.cage_dims.x_max - x_pos

        if dist_to_wall < 1.5:
            return self.crawl_speed  # 0.2 m/s
        elif dist_to_wall < 3.0:
            return self.slow_speed   # 0.5 m/s
        else:
            return self.cruise_speed # 1.0 m/s


def calculate_corner_velocity(current_pos: Tuple[float, float, float],
                              corner_pos: Tuple[float, float, float],
                              max_x: float = 10.0) -> float:
    """
    Standalone function for calculating velocity with aggressive braking curve.

    Args:
        current_pos: Current (x, y, z) position
        corner_pos: Target corner (x, y, z) position
        max_x: Maximum X boundary

    Returns:
        Safe velocity in m/s
    """
    x_current = current_pos[0]
    dist_to_wall = max_x - x_current

    if dist_to_wall < 1.5:
        return 0.2  # m/s (Crawl speed)
    elif dist_to_wall < 3.0:
        return 0.5  # m/s
    else:
        return 1.0  # m/s (Cruise)
