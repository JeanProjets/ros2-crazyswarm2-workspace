"""
Unit tests for formation_controller module.

Tests formation assignment, maintenance, collision avoidance, and leader-follower behavior.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from behaviors.formation_controller import (
    FormationType,
    FormationController,
    FormationOffset,
    LeaderFollowerBehavior,
    PIDController,
    DronePosition,
    calculate_follower_position,
    avoid_collision
)


class TestFormationType:
    """Test FormationType enum."""

    def test_formation_types_exist(self):
        """Test that all formation types are defined."""
        assert FormationType.LEADER_FOLLOWER
        assert FormationType.LINE_ABREAST
        assert FormationType.TRIANGLE
        assert FormationType.DEFENSIVE_SCREEN


class TestFormationOffset:
    """Test FormationOffset dataclass."""

    def test_offset_creation(self):
        """Test creating a formation offset."""
        offset = FormationOffset(x=-0.5, y=-0.5, z=-0.5)
        assert offset.x == -0.5
        assert offset.y == -0.5
        assert offset.z == -0.5

    def test_offset_as_tuple(self):
        """Test converting offset to tuple."""
        offset = FormationOffset(x=1.0, y=2.0, z=3.0)
        assert offset.as_tuple() == (1.0, 2.0, 3.0)


class TestPIDController:
    """Test PID controller."""

    def test_pid_initialization(self):
        """Test PID controller initialization."""
        pid = PIDController(kp=1.0, ki=0.1, kd=0.5)
        assert pid.kp == 1.0
        assert pid.ki == 0.1
        assert pid.kd == 0.5

    def test_pid_compute(self):
        """Test PID computation."""
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0)
        error = np.array([1.0, 0.0, 0.0])

        output = pid.compute(error)

        # With only P term, output should equal kp * error
        assert output[0] == pytest.approx(1.0, abs=0.1)

    def test_pid_reset(self):
        """Test PID reset."""
        pid = PIDController(kp=1.0, ki=0.1, kd=0.5)

        # Compute once to set state
        pid.compute(np.array([1.0, 1.0, 1.0]))

        # Reset
        pid.reset()

        # State should be cleared
        assert np.allclose(pid.integral, 0.0)
        assert np.allclose(pid.prev_error, 0.0)


class TestFormationController:
    """Test FormationController class."""

    def test_initialization(self):
        """Test formation controller initialization."""
        controller = FormationController(min_separation=0.5)
        assert controller.min_separation == 0.5

    def test_assign_leader_follower_formation(self):
        """Test leader-follower formation assignment."""
        controller = FormationController()
        leader_pos = (5.0, 3.0, 4.0)

        positions = controller.assign_formation_positions(
            leader_pos,
            FormationType.LEADER_FOLLOWER,
            num_drones=2
        )

        # Should have leader and follower positions
        assert "leader" in positions
        assert "follower_1" in positions

        # Leader should be at specified position
        assert positions["leader"] == leader_pos

        # Follower should be behind and below leader
        follower_x, follower_y, follower_z = positions["follower_1"]
        assert follower_x < leader_pos[0]
        assert follower_y < leader_pos[1]
        assert follower_z < leader_pos[2]

    def test_assign_line_abreast_formation(self):
        """Test line abreast formation assignment."""
        controller = FormationController()
        leader_pos = (5.0, 3.0, 4.0)

        positions = controller.assign_formation_positions(
            leader_pos,
            FormationType.LINE_ABREAST,
            num_drones=3
        )

        # Should have 3 positions
        assert len(positions) == 3

        # All should be at same x and z
        x_coords = [pos[0] for pos in positions.values()]
        z_coords = [pos[2] for pos in positions.values()]
        assert len(set(x_coords)) == 1
        assert len(set(z_coords)) == 1

    def test_assign_triangle_formation(self):
        """Test triangle formation assignment."""
        controller = FormationController()
        leader_pos = (5.0, 3.0, 4.0)

        positions = controller.assign_formation_positions(
            leader_pos,
            FormationType.TRIANGLE,
            num_drones=3
        )

        # Should have leader and 2 followers
        assert "leader" in positions
        assert "follower_1" in positions
        assert "follower_2" in positions

    def test_calculate_follower_offset(self):
        """Test follower offset calculation."""
        controller = FormationController()
        leader_pos = (5.0, 3.0, 4.0)

        # Set offset for follower
        controller.set_formation_offset(
            "follower_1",
            FormationOffset(-0.5, -0.5, -0.5)
        )

        target_pos = controller.calculate_follower_offset(leader_pos, "follower_1")

        # Target should be offset from leader
        assert target_pos == (4.5, 2.5, 3.5)

    def test_avoid_collisions(self):
        """Test collision avoidance."""
        controller = FormationController(min_separation=0.5)

        # Two drones very close together
        drone_positions = {
            "drone_1": (0.0, 0.0, 1.0),
            "drone_2": (0.2, 0.0, 1.0)  # Only 0.2m apart
        }

        adjusted = controller.avoid_collisions(drone_positions)

        # Positions should be adjusted to increase separation
        pos1 = np.array(adjusted["drone_1"])
        pos2 = np.array(adjusted["drone_2"])
        distance = np.linalg.norm(pos1 - pos2)

        # Distance should be increased (though may not reach min_separation yet)
        original_distance = 0.2
        assert distance > original_distance


class TestLeaderFollowerBehavior:
    """Test LeaderFollowerBehavior class."""

    def test_initialization(self):
        """Test leader-follower behavior initialization."""
        behavior = LeaderFollowerBehavior(
            leader_id="leader",
            follower_id="follower",
            offset=FormationOffset(-0.5, -0.5, -0.5)
        )

        assert behavior.leader_id == "leader"
        assert behavior.follower_id == "follower"
        assert behavior.offset.x == -0.5

    def test_compute_follower_target(self):
        """Test follower target computation."""
        behavior = LeaderFollowerBehavior(
            leader_id="leader",
            follower_id="follower"
        )

        leader_pos = (5.0, 3.0, 4.0)
        target = behavior.compute_follower_target(leader_pos)

        # Target should be offset from leader
        assert target[0] == leader_pos[0] + behavior.offset.x
        assert target[1] == leader_pos[1] + behavior.offset.y
        assert target[2] == leader_pos[2] + behavior.offset.z

    def test_check_emergency_separation(self):
        """Test emergency separation check."""
        behavior = LeaderFollowerBehavior(
            leader_id="leader",
            follower_id="follower"
        )

        leader_pos = (5.0, 3.0, 4.0)

        # Follower very close - should trigger emergency
        follower_pos = (5.1, 3.0, 4.0)
        assert behavior.check_emergency_separation(leader_pos, follower_pos)

        # Follower at safe distance - should not trigger
        follower_pos = (4.0, 2.0, 3.0)
        assert not behavior.check_emergency_separation(leader_pos, follower_pos)

    def test_compute_approach_to_target(self):
        """Test formation approach to target."""
        behavior = LeaderFollowerBehavior(
            leader_id="leader",
            follower_id="follower",
            offset=FormationOffset(-0.5, -0.5, -0.5)
        )

        target_pos = (7.5, 3.0, 5.0)
        leader_pos = (5.0, 3.0, 4.0)

        leader_target, follower_target = behavior.compute_approach_to_target(
            target_pos,
            leader_pos,
            approach_distance=1.0
        )

        # Leader should be 1m in front of target
        assert leader_target[0] == target_pos[0] - 1.0
        assert leader_target[1] == target_pos[1]
        assert leader_target[2] == target_pos[2]

        # Follower should maintain offset from leader target
        assert follower_target[0] == leader_target[0] + behavior.offset.x
        assert follower_target[1] == leader_target[1] + behavior.offset.y
        assert follower_target[2] == leader_target[2] + behavior.offset.z

    def test_adjust_offset_for_obstacle(self):
        """Test dynamic offset adjustment for obstacles."""
        behavior = LeaderFollowerBehavior(
            leader_id="leader",
            follower_id="follower"
        )

        leader_pos = (5.0, 3.0, 4.0)
        obstacle_pos = (5.5, 3.0, 4.0)  # Close to leader

        adjusted_offset = behavior.adjust_offset_for_obstacle(
            obstacle_pos,
            leader_pos
        )

        # Offset should be adjusted (different from original)
        # At minimum, altitude should be increased
        assert adjusted_offset.z > behavior.offset.z


class TestUtilityFunctions:
    """Test utility functions."""

    def test_calculate_follower_position(self):
        """Test follower position calculation."""
        leader_pos = (5.0, 3.0, 4.0)
        leader_vel = (0.5, 0.0, 0.0)
        offset = (-0.5, -0.5, -0.5)

        target_pos = calculate_follower_position(
            leader_pos,
            leader_vel,
            offset
        )

        # Target should be leader position plus offset
        assert target_pos == (4.5, 2.5, 3.5)

    def test_avoid_collision_function(self):
        """Test collision avoidance function."""
        # Two drones close together
        positions = [
            (0.0, 0.0, 1.0),
            (0.3, 0.0, 1.0)  # Only 0.3m apart
        ]

        adjusted = avoid_collision(positions, min_separation=0.5)

        # Should return same number of positions
        assert len(adjusted) == len(positions)

        # Distance should be increased
        pos1 = np.array(adjusted[0])
        pos2 = np.array(adjusted[1])
        new_distance = np.linalg.norm(pos1 - pos2)
        original_distance = 0.3

        assert new_distance > original_distance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
