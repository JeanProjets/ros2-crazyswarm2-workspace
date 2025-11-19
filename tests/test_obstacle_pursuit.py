"""
Tests for obstacle_pursuit module
"""

import pytest
import numpy as np
from src.behaviors.obstacle_pursuit import (
    PathFollowerBehavior,
    Waypoint,
    calculate_pursuit_velocity
)


class TestWaypoint:
    """Tests for Waypoint class"""

    def test_waypoint_creation(self):
        """Test creating a waypoint"""
        wp = Waypoint(1.0, 2.0, 3.0)
        assert wp.x == 1.0
        assert wp.y == 2.0
        assert wp.z == 3.0

    def test_waypoint_to_array(self):
        """Test converting waypoint to numpy array"""
        wp = Waypoint(1.0, 2.0, 3.0)
        arr = wp.to_array()
        np.testing.assert_array_equal(arr, np.array([1.0, 2.0, 3.0]))

    def test_waypoint_from_array(self):
        """Test creating waypoint from array"""
        arr = np.array([1.5, 2.5, 3.5])
        wp = Waypoint.from_array(arr)
        assert wp.x == 1.5
        assert wp.y == 2.5
        assert wp.z == 3.5


class TestPathFollowerBehavior:
    """Tests for PathFollowerBehavior class"""

    def test_initialization(self):
        """Test PathFollowerBehavior initialization"""
        follower = PathFollowerBehavior(lookahead_dist=0.5, max_speed=1.0)
        assert follower.lookahead_dist == 0.5
        assert follower.max_speed == 1.0
        assert len(follower.path) == 0

    def test_update_path(self):
        """Test updating path"""
        follower = PathFollowerBehavior()
        path = [
            Waypoint(0.0, 0.0, 0.0),
            Waypoint(1.0, 0.0, 0.0),
            Waypoint(1.0, 1.0, 0.0)
        ]
        follower.update_path(path)
        assert len(follower.path) == 3

    def test_set_line_of_sight(self):
        """Test setting line of sight"""
        follower = PathFollowerBehavior()
        target_vel = np.array([1.0, 0.0, 0.0])

        follower.set_line_of_sight(True, target_vel)
        assert follower.has_line_of_sight is True
        np.testing.assert_array_equal(follower.target_velocity, target_vel)

    def test_find_closest_point_on_path(self):
        """Test finding closest point on path"""
        follower = PathFollowerBehavior()
        path = [
            Waypoint(0.0, 0.0, 0.0),
            Waypoint(10.0, 0.0, 0.0)
        ]
        follower.update_path(path)

        current_pos = np.array([5.0, 1.0, 0.0])
        idx, t = follower.find_closest_point_on_path(current_pos)

        assert idx == 0
        assert 0.4 < t < 0.6  # Should be roughly in middle

    def test_find_lookahead_point_simple_path(self):
        """Test finding lookahead point on simple path"""
        follower = PathFollowerBehavior(lookahead_dist=1.0)
        path = [
            Waypoint(0.0, 0.0, 0.0),
            Waypoint(5.0, 0.0, 0.0)
        ]
        follower.update_path(path)

        current_pos = np.array([0.0, 0.0, 0.0])
        lookahead = follower.find_lookahead_point(current_pos)

        assert lookahead is not None
        # Should be 1.0m ahead along x-axis
        assert abs(lookahead[0] - 1.0) < 0.1
        assert abs(lookahead[1]) < 0.1

    def test_calculate_pursuit_velocity_straight_line(self):
        """Test pursuit velocity on straight path"""
        follower = PathFollowerBehavior(lookahead_dist=1.0, max_speed=2.0)
        path = [
            Waypoint(0.0, 0.0, 1.0),
            Waypoint(10.0, 0.0, 1.0)
        ]
        follower.update_path(path)

        current_pos = np.array([0.0, 0.0, 1.0])
        cmd_vel = follower.calculate_pursuit_velocity(current_pos)

        # Should command velocity in +x direction
        assert cmd_vel[0] > 0
        assert abs(cmd_vel[1]) < 0.1
        # Speed should not exceed max
        assert np.linalg.norm(cmd_vel) <= follower.max_speed * 1.01

    def test_execute_pure_pursuit(self):
        """Test execute_pure_pursuit method"""
        follower = PathFollowerBehavior(lookahead_dist=0.5, max_speed=1.0)
        path = [
            Waypoint(0.0, 0.0, 0.0),
            Waypoint(2.0, 0.0, 0.0)
        ]
        follower.update_path(path)

        current_pos = np.array([0.0, 0.0, 0.0])
        current_vel = np.array([0.0, 0.0, 0.0])

        cmd_vel = follower.execute_pure_pursuit(current_pos, current_vel)

        assert cmd_vel is not None
        assert len(cmd_vel) == 3
        assert np.linalg.norm(cmd_vel) <= follower.max_speed * 1.01

    def test_is_path_complete(self):
        """Test checking if path is complete"""
        follower = PathFollowerBehavior()
        path = [
            Waypoint(0.0, 0.0, 0.0),
            Waypoint(1.0, 0.0, 0.0)
        ]
        follower.update_path(path)

        # Far from end
        assert not follower.is_path_complete(np.array([0.0, 0.0, 0.0]))

        # Near end
        assert follower.is_path_complete(np.array([1.0, 0.05, 0.0]), threshold=0.2)

    def test_empty_path_handling(self):
        """Test behavior with empty path"""
        follower = PathFollowerBehavior()

        current_pos = np.array([1.0, 1.0, 1.0])
        cmd_vel = follower.calculate_pursuit_velocity(current_pos)

        np.testing.assert_array_equal(cmd_vel, np.zeros(3))

    def test_feedforward_with_line_of_sight(self):
        """Test that feedforward is applied with line of sight"""
        follower = PathFollowerBehavior(lookahead_dist=1.0, max_speed=2.0)
        path = [
            Waypoint(0.0, 0.0, 0.0),
            Waypoint(5.0, 0.0, 0.0)
        ]
        follower.update_path(path)

        target_vel = np.array([1.0, 0.5, 0.0])
        follower.set_line_of_sight(True, target_vel)

        current_pos = np.array([0.0, 0.0, 0.0])
        cmd_vel = follower.calculate_pursuit_velocity(current_pos)

        # With LOS and feedforward, should have some y component
        # from target velocity
        assert abs(cmd_vel[1]) > 0.01


def test_calculate_pursuit_velocity_function():
    """Test standalone calculate_pursuit_velocity function"""
    path = [
        Waypoint(0.0, 0.0, 0.0),
        Waypoint(5.0, 0.0, 0.0)
    ]

    drone_pos = np.array([0.0, 0.0, 0.0])
    cmd_vel = calculate_pursuit_velocity(drone_pos, path, lookahead_dist=1.0, max_speed=1.5)

    assert cmd_vel is not None
    assert len(cmd_vel) == 3
    assert np.linalg.norm(cmd_vel) <= 1.5 * 1.01
