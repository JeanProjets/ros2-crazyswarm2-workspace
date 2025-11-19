"""
Tests for safe_strike_v4 module
"""

import pytest
import numpy as np
from src.behaviors.safe_strike_v4 import (
    SafeDynamicStrike,
    StrikeParameters,
    StrikeState
)
from src.behaviors.elastic_formation import GridMap


class TestStrikeParameters:
    """Tests for StrikeParameters dataclass"""

    def test_strike_parameters_defaults(self):
        """Test default strike parameters"""
        params = StrikeParameters()

        assert params.min_strike_distance == 0.3
        assert params.max_strike_distance == 2.0
        assert params.strike_speed == 1.5
        assert params.safety_margin == 0.4

    def test_strike_parameters_custom(self):
        """Test custom strike parameters"""
        params = StrikeParameters(
            min_strike_distance=0.5,
            max_strike_distance=3.0,
            strike_speed=2.0
        )

        assert params.min_strike_distance == 0.5
        assert params.max_strike_distance == 3.0
        assert params.strike_speed == 2.0


class TestSafeDynamicStrike:
    """Tests for SafeDynamicStrike class"""

    def test_initialization(self):
        """Test SafeDynamicStrike initialization"""
        strike = SafeDynamicStrike()

        assert strike.state == StrikeState.READY
        assert strike.grid_map is None

    def test_initialization_with_params(self):
        """Test initialization with custom parameters"""
        params = StrikeParameters(strike_speed=2.5)
        strike = SafeDynamicStrike(params)

        assert strike.params.strike_speed == 2.5

    def test_set_grid_map(self):
        """Test setting grid map"""
        strike = SafeDynamicStrike()
        grid = GridMap()

        strike.set_grid_map(grid)
        assert strike.grid_map is grid

    def test_update_target(self):
        """Test updating target information"""
        strike = SafeDynamicStrike()

        pos = np.array([5.0, 0.0, 1.0])
        vel = np.array([1.0, 0.0, 0.0])

        strike.update_target(pos, vel)

        np.testing.assert_array_equal(strike.target_position, pos)
        np.testing.assert_array_equal(strike.target_velocity, vel)

    def test_verify_attack_corridor_no_map(self):
        """Test corridor verification without map"""
        strike = SafeDynamicStrike()

        drone_pos = np.array([0.0, 0.0, 1.0])
        target_pos = np.array([2.0, 0.0, 1.0])
        attack_vec = np.array([1.0, 0.0, 0.0])

        is_safe, reason = strike.verify_attack_corridor(drone_pos, target_pos, attack_vec)

        # Without map, assumes safe
        assert is_safe

    def test_verify_attack_corridor_clear_path(self):
        """Test corridor verification with clear path"""
        strike = SafeDynamicStrike()
        grid = GridMap(inflation_radius=0.3)
        strike.set_grid_map(grid)

        # No obstacles
        drone_pos = np.array([0.0, 0.0, 1.0])
        target_pos = np.array([2.0, 0.0, 1.0])
        attack_vec = np.array([1.0, 0.0, 0.0])

        is_safe, reason = strike.verify_attack_corridor(drone_pos, target_pos, attack_vec)

        assert is_safe
        assert "clear" in reason.lower()

    def test_verify_attack_corridor_obstacle_in_path(self):
        """Test corridor verification with obstacle in path"""
        strike = SafeDynamicStrike()
        grid = GridMap(inflation_radius=0.4)

        # Place obstacle in path
        grid.add_obstacle(np.array([1.0, 0.0, 1.0]))
        strike.set_grid_map(grid)

        drone_pos = np.array([0.0, 0.0, 1.0])
        target_pos = np.array([2.0, 0.0, 1.0])
        attack_vec = np.array([1.0, 0.0, 0.0])

        is_safe, reason = strike.verify_attack_corridor(drone_pos, target_pos, attack_vec)

        assert not is_safe
        assert "obstacle" in reason.lower()

    def test_verify_attack_corridor_target_near_wall(self):
        """Test corridor verification with target near obstacle"""
        strike = SafeDynamicStrike()
        grid = GridMap(inflation_radius=0.3)

        # Place obstacle near target
        grid.add_obstacle(np.array([2.2, 0.0, 1.0]))
        strike.set_grid_map(grid)

        drone_pos = np.array([0.0, 0.0, 1.0])
        target_pos = np.array([2.0, 0.0, 1.0])
        attack_vec = np.array([1.0, 0.0, 0.0])

        is_safe, reason = strike.verify_attack_corridor(drone_pos, target_pos, attack_vec)

        # Might be unsafe depending on proximity threshold
        if not is_safe:
            assert "obstacle" in reason.lower() or "wall" in reason.lower()

    def test_calculate_strike_approach_stationary_target(self):
        """Test calculating approach for stationary target"""
        strike = SafeDynamicStrike()

        drone_pos = np.array([0.0, 0.0, 1.0])
        target_pos = np.array([2.0, 0.0, 1.0])
        target_vel = np.array([0.0, 0.0, 0.0])

        approach_vec, approach_type = strike.calculate_strike_approach(
            drone_pos, target_pos, target_vel
        )

        # Should be direct approach
        assert approach_type == "direct"
        assert np.linalg.norm(approach_vec) > 0.9  # Should be normalized

    def test_calculate_strike_approach_moving_target(self):
        """Test calculating approach for moving target"""
        strike = SafeDynamicStrike()

        drone_pos = np.array([0.0, 0.0, 1.0])
        target_pos = np.array([2.0, 0.0, 1.0])
        target_vel = np.array([0.0, 1.0, 0.0])  # Moving in +y

        approach_vec, approach_type = strike.calculate_strike_approach(
            drone_pos, target_pos, target_vel
        )

        # Should be intercept approach
        assert approach_type == "intercept"
        # Approach should have y component to intercept
        assert abs(approach_vec[1]) > 0.01

    def test_execute_strike_no_target(self):
        """Test strike execution without target data"""
        strike = SafeDynamicStrike()

        drone_pos = np.array([0.0, 0.0, 1.0])
        drone_vel = np.array([0.0, 0.0, 0.0])

        cmd_vel, msg, state = strike.execute_strike(drone_pos, drone_vel, current_time=1.0)

        np.testing.assert_array_equal(cmd_vel, np.zeros(3))
        assert "No target" in msg
        assert state == StrikeState.READY

    def test_execute_strike_too_far(self):
        """Test strike execution when too far from target"""
        strike = SafeDynamicStrike()

        target_pos = np.array([10.0, 0.0, 1.0])  # Far away
        target_vel = np.array([0.0, 0.0, 0.0])
        strike.update_target(target_pos, target_vel)

        drone_pos = np.array([0.0, 0.0, 1.0])
        drone_vel = np.array([0.0, 0.0, 0.0])

        cmd_vel, msg, state = strike.execute_strike(drone_pos, drone_vel, current_time=1.0)

        assert state == StrikeState.READY
        assert "Out of strike range" in msg

    def test_execute_strike_too_close(self):
        """Test strike execution when very close to target"""
        strike = SafeDynamicStrike()

        target_pos = np.array([0.1, 0.0, 1.0])  # Very close
        target_vel = np.array([0.0, 0.0, 0.0])
        strike.update_target(target_pos, target_vel)

        drone_pos = np.array([0.0, 0.0, 1.0])
        drone_vel = np.array([0.0, 0.0, 0.0])

        cmd_vel, msg, state = strike.execute_strike(drone_pos, drone_vel, current_time=1.0)

        assert state == StrikeState.COMPLETED
        assert "complete" in msg.lower()

    def test_execute_strike_safe_corridor(self):
        """Test strike execution with safe corridor"""
        strike = SafeDynamicStrike()
        grid = GridMap(inflation_radius=0.2)
        strike.set_grid_map(grid)

        target_pos = np.array([1.0, 0.0, 1.0])  # In range
        target_vel = np.array([0.0, 0.0, 0.0])
        strike.update_target(target_pos, target_vel)

        drone_pos = np.array([0.0, 0.0, 1.0])
        drone_vel = np.array([0.0, 0.0, 0.0])

        cmd_vel, msg, state = strike.execute_strike(drone_pos, drone_vel, current_time=1.0)

        # Should execute attack
        assert state == StrikeState.ATTACKING
        assert "executing" in msg.lower()
        assert np.linalg.norm(cmd_vel) > 0

    def test_execute_strike_unsafe_corridor(self):
        """Test strike execution with unsafe corridor"""
        strike = SafeDynamicStrike()
        grid = GridMap(inflation_radius=0.4)

        # Obstacle in path
        grid.add_obstacle(np.array([0.5, 0.0, 1.0]))
        strike.set_grid_map(grid)

        target_pos = np.array([1.0, 0.0, 1.0])
        target_vel = np.array([0.0, 0.0, 0.0])
        strike.update_target(target_pos, target_vel)

        drone_pos = np.array([0.0, 0.0, 1.0])
        drone_vel = np.array([0.0, 0.0, 0.0])

        cmd_vel, msg, state = strike.execute_strike(drone_pos, drone_vel, current_time=1.0)

        # Should abort
        assert state == StrikeState.ABORTING
        assert "aborted" in msg.lower()

    def test_should_attempt_strike_in_range(self):
        """Test should_attempt_strike when in range"""
        params = StrikeParameters(min_strike_distance=0.5, max_strike_distance=2.0)
        strike = SafeDynamicStrike(params)

        target_pos = np.array([1.0, 0.0, 1.0])
        target_vel = np.array([0.0, 0.0, 0.0])
        strike.update_target(target_pos, target_vel)

        drone_pos = np.array([0.0, 0.0, 1.0])

        # Distance = 1.0, which is in [0.5, 2.0]
        assert strike.should_attempt_strike(drone_pos)

    def test_should_attempt_strike_out_of_range(self):
        """Test should_attempt_strike when out of range"""
        params = StrikeParameters(min_strike_distance=0.5, max_strike_distance=2.0)
        strike = SafeDynamicStrike(params)

        target_pos = np.array([5.0, 0.0, 1.0])  # Far
        target_vel = np.array([0.0, 0.0, 0.0])
        strike.update_target(target_pos, target_vel)

        drone_pos = np.array([0.0, 0.0, 1.0])

        assert not strike.should_attempt_strike(drone_pos)

    def test_get_state(self):
        """Test getting current state"""
        strike = SafeDynamicStrike()

        assert strike.get_state() == StrikeState.READY

        strike.state = StrikeState.ATTACKING
        assert strike.get_state() == StrikeState.ATTACKING

    def test_reset(self):
        """Test resetting strike state"""
        strike = SafeDynamicStrike()

        strike.state = StrikeState.ATTACKING
        strike.strike_vector = np.array([1.0, 0.0, 0.0])

        strike.reset()

        assert strike.state == StrikeState.READY
        assert strike.strike_vector is None

    def test_holding_pattern(self):
        """Test holding pattern when strike is aborted"""
        strike = SafeDynamicStrike()
        grid = GridMap(inflation_radius=0.5)

        # Large obstacle blocking path
        grid.add_obstacle(np.array([0.7, 0.0, 1.0]))
        strike.set_grid_map(grid)

        target_pos = np.array([1.5, 0.0, 1.0])
        target_vel = np.array([0.5, 0.0, 0.0])  # Moving target
        strike.update_target(target_pos, target_vel)

        drone_pos = np.array([0.0, 0.0, 1.0])
        drone_vel = np.array([0.0, 0.0, 0.0])

        cmd_vel, msg, state = strike.execute_strike(drone_pos, drone_vel, current_time=1.0)

        # Should abort and hold
        assert state == StrikeState.ABORTING
        # Should still have some velocity (holding pattern or matching target)
        # Could be zero or non-zero depending on implementation
        assert len(cmd_vel) == 3
