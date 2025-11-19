"""
Tests for elastic_formation module
"""

import pytest
import numpy as np
from src.behaviors.elastic_formation import (
    ElasticFormation,
    FormationOffset,
    GridMap,
    get_valid_formation_point
)


class TestFormationOffset:
    """Tests for FormationOffset class"""

    def test_formation_offset_creation(self):
        """Test creating a formation offset"""
        offset = FormationOffset(-1.0, 0.5, 0.0)
        assert offset.x == -1.0
        assert offset.y == 0.5
        assert offset.z == 0.0

    def test_formation_offset_to_array(self):
        """Test converting offset to array"""
        offset = FormationOffset(-1.0, 0.5, 0.0)
        arr = offset.to_array()
        np.testing.assert_array_equal(arr, np.array([-1.0, 0.5, 0.0]))


class TestGridMap:
    """Tests for GridMap class"""

    def test_gridmap_initialization(self):
        """Test GridMap initialization"""
        grid = GridMap(resolution=0.1, inflation_radius=0.3)
        assert grid.resolution == 0.1
        assert grid.inflation_radius == 0.3

    def test_add_obstacle(self):
        """Test adding obstacle to map"""
        grid = GridMap()
        obstacle_pos = np.array([1.0, 1.0, 1.0])
        grid.add_obstacle(obstacle_pos)
        assert len(grid.obstacles) == 1

    def test_is_collision_free_space(self):
        """Test collision check in free space"""
        grid = GridMap(inflation_radius=0.3)
        grid.add_obstacle(np.array([5.0, 5.0, 1.0]))

        # Point far from obstacle
        assert not grid.is_collision(np.array([0.0, 0.0, 1.0]))

    def test_is_collision_in_obstacle(self):
        """Test collision check inside obstacle"""
        grid = GridMap(inflation_radius=0.3)
        obstacle_pos = np.array([1.0, 1.0, 1.0])
        grid.add_obstacle(obstacle_pos)

        # Point at obstacle
        assert grid.is_collision(obstacle_pos)

    def test_is_collision_in_inflation_zone(self):
        """Test collision check in inflation zone"""
        grid = GridMap(inflation_radius=0.5)
        grid.add_obstacle(np.array([0.0, 0.0, 0.0]))

        # Point within inflation radius
        assert grid.is_collision(np.array([0.3, 0.0, 0.0]))

    def test_find_nearest_free(self):
        """Test finding nearest free space"""
        grid = GridMap(inflation_radius=0.4)
        grid.add_obstacle(np.array([0.0, 0.0, 0.0]))

        # Point inside obstacle
        point = np.array([0.0, 0.0, 0.0])
        free_point = grid.find_nearest_free(point)

        # Should find a free point
        assert not grid.is_collision(free_point)
        # Should be at or outside inflation radius
        assert np.linalg.norm(free_point) >= 0.4 - 0.01  # Small tolerance

    def test_find_nearest_free_already_free(self):
        """Test find_nearest_free when point is already free"""
        grid = GridMap(inflation_radius=0.3)
        grid.add_obstacle(np.array([5.0, 5.0, 1.0]))

        point = np.array([0.0, 0.0, 0.0])
        free_point = grid.find_nearest_free(point)

        # Should return same point
        np.testing.assert_array_almost_equal(free_point, point)


class TestElasticFormation:
    """Tests for ElasticFormation class"""

    def test_initialization(self):
        """Test ElasticFormation initialization"""
        offset = FormationOffset(-1.0, 0.5, 0.0)
        formation = ElasticFormation(
            formation_offset=offset,
            min_separation=0.5,
            max_stretch=3.0
        )
        assert formation.min_separation == 0.5
        assert formation.max_stretch == 3.0

    def test_set_grid_map(self):
        """Test setting grid map"""
        formation = ElasticFormation(FormationOffset(-1.0, 0.0, 0.0))
        grid = GridMap()

        formation.set_grid_map(grid)
        assert formation.grid_map is grid

    def test_calculate_ideal_follower_position(self):
        """Test calculating ideal follower position"""
        offset = FormationOffset(-1.0, 0.5, 0.0)
        formation = ElasticFormation(offset)

        leader_pos = np.array([2.0, 2.0, 1.0])
        leader_vel = np.array([1.0, 0.0, 0.0])

        ideal_pos = formation.calculate_ideal_follower_position(leader_pos, leader_vel)

        # Should be leader + offset
        expected = np.array([1.0, 2.5, 1.0])
        np.testing.assert_array_almost_equal(ideal_pos, expected)

    def test_calculate_loose_follower_goal_no_obstacles(self):
        """Test follower goal with no obstacles"""
        offset = FormationOffset(-1.0, 0.0, 0.0)
        formation = ElasticFormation(offset)
        grid = GridMap()
        formation.set_grid_map(grid)

        leader_pos = np.array([5.0, 5.0, 1.0])
        leader_vel = np.array([1.0, 0.0, 0.0])
        follower_pos = np.array([4.0, 5.0, 1.0])

        goal, needs_planning = formation.calculate_loose_follower_goal(
            leader_pos, leader_vel, follower_pos
        )

        # No obstacles, should return ideal position
        expected = np.array([4.0, 5.0, 1.0])
        np.testing.assert_array_almost_equal(goal, expected)
        assert not needs_planning

    def test_calculate_loose_follower_goal_with_obstacle(self):
        """Test follower goal when ideal position has obstacle"""
        offset = FormationOffset(-1.0, 0.0, 0.0)
        formation = ElasticFormation(offset, max_stretch=2.0)
        grid = GridMap(inflation_radius=0.5)

        # Put obstacle at ideal follower position
        grid.add_obstacle(np.array([4.0, 5.0, 1.0]))
        formation.set_grid_map(grid)

        leader_pos = np.array([5.0, 5.0, 1.0])
        leader_vel = np.array([1.0, 0.0, 0.0])
        follower_pos = np.array([3.0, 5.0, 1.0])

        goal, needs_planning = formation.calculate_loose_follower_goal(
            leader_pos, leader_vel, follower_pos
        )

        # Goal should be adjusted to avoid obstacle
        assert not grid.is_collision(goal)

    def test_minimum_separation_enforcement(self):
        """Test that minimum separation is enforced"""
        offset = FormationOffset(-0.2, 0.0, 0.0)  # Very close offset
        formation = ElasticFormation(offset, min_separation=0.5)

        leader_pos = np.array([5.0, 5.0, 1.0])
        leader_vel = np.array([0.0, 0.0, 0.0])
        follower_pos = np.array([4.9, 5.0, 1.0])

        goal, _ = formation.calculate_loose_follower_goal(
            leader_pos, leader_vel, follower_pos
        )

        # Distance should be at least min_separation
        dist = np.linalg.norm(goal - leader_pos)
        assert dist >= 0.5 - 0.01  # Small tolerance

    def test_calculate_formation_velocity(self):
        """Test formation velocity calculation"""
        offset = FormationOffset(-1.0, 0.0, 0.0)
        formation = ElasticFormation(offset)

        leader_pos = np.array([5.0, 0.0, 1.0])
        leader_vel = np.array([1.0, 0.0, 0.0])
        follower_pos = np.array([3.0, 0.0, 1.0])
        follower_vel = np.array([0.5, 0.0, 0.0])

        cmd_vel = formation.calculate_formation_velocity(
            leader_pos, leader_vel, follower_pos, follower_vel, max_speed=2.0
        )

        # Should command velocity towards goal
        assert len(cmd_vel) == 3
        assert np.linalg.norm(cmd_vel) <= 2.0 * 1.01

    def test_formation_velocity_at_goal(self):
        """Test formation velocity when already at goal"""
        offset = FormationOffset(-1.0, 0.0, 0.0)
        formation = ElasticFormation(offset)

        leader_pos = np.array([5.0, 0.0, 1.0])
        leader_vel = np.array([1.0, 0.0, 0.0])
        follower_pos = np.array([4.0, 0.0, 1.0])  # Exactly at ideal position
        follower_vel = np.array([0.8, 0.0, 0.0])

        cmd_vel = formation.calculate_formation_velocity(
            leader_pos, leader_vel, follower_pos, follower_vel, max_speed=2.0
        )

        # Should match leader velocity when at goal
        np.testing.assert_array_almost_equal(cmd_vel, leader_vel, decimal=1)


def test_get_valid_formation_point_free():
    """Test standalone function with free point"""
    grid = GridMap(inflation_radius=0.3)
    grid.add_obstacle(np.array([5.0, 5.0, 1.0]))

    ideal_point = np.array([0.0, 0.0, 0.0])
    valid_point = get_valid_formation_point(ideal_point, grid)

    np.testing.assert_array_almost_equal(valid_point, ideal_point)


def test_get_valid_formation_point_occupied():
    """Test standalone function with occupied point"""
    grid = GridMap(inflation_radius=0.4)
    grid.add_obstacle(np.array([0.0, 0.0, 0.0]))

    ideal_point = np.array([0.0, 0.0, 0.0])
    valid_point = get_valid_formation_point(ideal_point, grid)

    # Should find free point
    assert not grid.is_collision(valid_point)
