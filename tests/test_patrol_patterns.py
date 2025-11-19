"""
Unit tests for patrol_patterns module.

Tests patrol pattern generation, coverage calculation, and waypoint smoothing.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from behaviors.patrol_patterns import (
    Waypoint,
    SafetyZonePatrol,
    AreaPatrol,
    PatternType,
    generate_coverage_path,
    smooth_trajectory
)


class TestWaypoint:
    """Test Waypoint dataclass."""

    def test_waypoint_creation(self):
        """Test creating a basic waypoint."""
        wp = Waypoint(position=(1.0, 2.0, 3.0))
        assert wp.position == (1.0, 2.0, 3.0)
        assert wp.yaw == 0.0
        assert wp.speed == 0.5
        assert wp.wait_time == 0.0

    def test_waypoint_invalid_altitude(self):
        """Test that negative altitude raises error."""
        with pytest.raises(ValueError, match="Invalid altitude"):
            Waypoint(position=(1.0, 2.0, -1.0))

    def test_waypoint_invalid_speed(self):
        """Test that zero or negative speed raises error."""
        with pytest.raises(ValueError, match="Invalid speed"):
            Waypoint(position=(1.0, 2.0, 3.0), speed=0.0)


class TestSafetyZonePatrol:
    """Test SafetyZonePatrol class."""

    def test_initialization(self):
        """Test SafetyZonePatrol initialization."""
        patrol = SafetyZonePatrol(drone_id="neutral_1", zone_id=0)
        assert patrol.drone_id == "neutral_1"
        assert patrol.zone_id == 0

    def test_rectangular_sweep(self):
        """Test rectangular sweep pattern generation."""
        patrol = SafetyZonePatrol(drone_id="neutral_1")
        bounds = {
            'x_range': (0.0, 3.0),
            'y_range': (0.0, 3.0)
        }

        waypoints = patrol.rectangular_sweep(bounds, height=1.5, speed=0.5)

        # Should generate multiple waypoints
        assert len(waypoints) > 0

        # All waypoints should be within bounds
        for wp in waypoints:
            x, y, z = wp.position
            assert bounds['x_range'][0] <= x <= bounds['x_range'][1]
            assert bounds['y_range'][0] <= y <= bounds['y_range'][1]
            assert z == 1.5

    def test_spiral_search(self):
        """Test spiral search pattern generation."""
        patrol = SafetyZonePatrol(drone_id="neutral_1")
        center = (1.5, 1.5, 1.5)
        radius = 1.0

        waypoints = patrol.spiral_search(center, radius, height=1.5, num_loops=3)

        # Should generate waypoints
        assert len(waypoints) > 0

        # First waypoint should be near center
        first_x, first_y, first_z = waypoints[0].position
        assert abs(first_x - center[0]) < 0.1
        assert abs(first_y - center[1]) < 0.1

        # All waypoints should be at correct height
        for wp in waypoints:
            assert wp.position[2] == 1.5


class TestAreaPatrol:
    """Test AreaPatrol class."""

    def test_initialization(self):
        """Test AreaPatrol initialization."""
        bounds = {
            'x_range': (3.0, 10.0),
            'y_range': (0.0, 6.0)
        }
        patrol = AreaPatrol(drone_id="patrol_1", area_bounds=bounds)

        assert patrol.drone_id == "patrol_1"
        assert patrol.area_bounds == bounds
        assert patrol.coverage_map.shape[0] > 0
        assert patrol.coverage_map.shape[1] > 0

    def test_lawn_mower_pattern(self):
        """Test lawn-mower pattern generation."""
        bounds = {
            'x_range': (3.0, 10.0),
            'y_range': (0.0, 6.0)
        }
        patrol = AreaPatrol(drone_id="patrol_1", area_bounds=bounds)

        waypoints = patrol.lawn_mower_pattern(
            x_range=(3.0, 10.0),
            y_range=(0.0, 6.0),
            height=4.0,
            spacing=2.0
        )

        # Should generate waypoints
        assert len(waypoints) > 0

        # First waypoint should be at start position (approximately)
        first_pos = waypoints[0].position
        assert abs(first_pos[0] - 3.0) < 0.5
        assert abs(first_pos[1] - 5.0) < 1.0
        assert first_pos[2] == 4.0

    def test_perimeter_scan(self):
        """Test perimeter scanning pattern."""
        bounds = {
            'x_range': (3.0, 10.0),
            'y_range': (0.0, 6.0)
        }
        patrol = AreaPatrol(drone_id="patrol_1", area_bounds=bounds)

        waypoints = patrol.perimeter_scan(bounds, height=4.0)

        # Should generate corner waypoints
        assert len(waypoints) >= 4

        # All waypoints should be at correct height
        for wp in waypoints:
            assert wp.position[2] == 4.0

    def test_coverage_percentage(self):
        """Test coverage percentage calculation."""
        bounds = {
            'x_range': (0.0, 5.0),
            'y_range': (0.0, 5.0)
        }
        patrol = AreaPatrol(drone_id="patrol_1", area_bounds=bounds)

        # Initially 0% coverage
        assert patrol.get_coverage_percentage() == 0.0

        # Mark some areas as searched
        patrol.mark_area_searched((2.5, 2.5), fov_radius=1.0)

        # Coverage should increase
        assert patrol.get_coverage_percentage() > 0.0
        assert patrol.get_coverage_percentage() <= 100.0

    def test_adaptive_search(self):
        """Test adaptive search pattern."""
        bounds = {
            'x_range': (0.0, 10.0),
            'y_range': (0.0, 6.0)
        }
        patrol = AreaPatrol(drone_id="patrol_1", area_bounds=bounds)

        searched = [(0.0, 3.0, 0.0, 3.0)]
        remaining = [(3.0, 10.0, 0.0, 6.0)]

        waypoints = patrol.adaptive_search(
            searched_areas=searched,
            remaining_areas=remaining,
            height=4.0
        )

        # Should generate waypoints for remaining areas
        assert len(waypoints) > 0


class TestUtilityFunctions:
    """Test utility functions."""

    def test_generate_coverage_path(self):
        """Test coverage path generation."""
        area_bounds = (0.0, 10.0, 0.0, 6.0)
        drone_fov = 2.0

        waypoints = generate_coverage_path(
            area_bounds,
            drone_fov,
            overlap=0.1,
            height=4.0
        )

        # Should generate waypoints
        assert len(waypoints) > 0

        # All waypoints should be at correct height
        for wp in waypoints:
            assert wp.position[2] == 4.0

    def test_smooth_trajectory(self):
        """Test trajectory smoothing."""
        # Create simple zigzag path
        waypoints = [
            Waypoint((0.0, 0.0, 1.0)),
            Waypoint((1.0, 1.0, 1.0)),
            Waypoint((2.0, 0.0, 1.0)),
            Waypoint((3.0, 1.0, 1.0))
        ]

        smoothed = smooth_trajectory(waypoints, smoothing_factor=0.3)

        # Should have same number of waypoints
        assert len(smoothed) == len(waypoints)

        # First and last should be unchanged
        assert smoothed[0].position == waypoints[0].position
        assert smoothed[-1].position == waypoints[-1].position

        # Middle waypoints should be smoothed (different positions)
        assert smoothed[1].position != waypoints[1].position

    def test_smooth_trajectory_short_path(self):
        """Test smoothing with path too short to smooth."""
        waypoints = [
            Waypoint((0.0, 0.0, 1.0)),
            Waypoint((1.0, 1.0, 1.0))
        ]

        smoothed = smooth_trajectory(waypoints)

        # Should return original path
        assert len(smoothed) == len(waypoints)
        assert smoothed == waypoints


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
