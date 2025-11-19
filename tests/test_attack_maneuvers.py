"""
Unit tests for attack_maneuvers module.

Tests jamming behavior, neutralization maneuvers, and attack coordination.
"""

import pytest
import numpy as np
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from behaviors.attack_maneuvers import (
    AttackRole,
    AttackPhase,
    AttackWaypoint,
    JammingBehavior,
    NeutralizationManeuver,
    AttackCoordinator,
    calculate_approach_vector
)


class TestAttackRole:
    """Test AttackRole enum."""

    def test_attack_roles_exist(self):
        """Test that all attack roles are defined."""
        assert AttackRole.JAMMER_LEADER
        assert AttackRole.JAMMER_FOLLOWER
        assert AttackRole.ATTACK_DRONE
        assert AttackRole.NEUTRAL


class TestAttackWaypoint:
    """Test AttackWaypoint dataclass."""

    def test_waypoint_creation(self):
        """Test creating an attack waypoint."""
        wp = AttackWaypoint(
            position=(7.5, 3.0, 5.0),
            hover_time=2.0,
            speed=0.3
        )

        assert wp.position == (7.5, 3.0, 5.0)
        assert wp.hover_time == 2.0
        assert wp.speed == 0.3


class TestJammingBehavior:
    """Test JammingBehavior class."""

    def test_initialization(self):
        """Test jamming behavior initialization."""
        jammer = JammingBehavior(
            drone_id="jammer_1",
            role=AttackRole.JAMMER_LEADER
        )

        assert jammer.drone_id == "jammer_1"
        assert jammer.role == AttackRole.JAMMER_LEADER
        assert jammer.jamming_duration == 20.0
        assert not jammer.is_jamming

    def test_position_for_jamming_leader(self):
        """Test jamming position calculation for leader."""
        jammer = JammingBehavior(
            drone_id="jammer_leader",
            role=AttackRole.JAMMER_LEADER
        )

        target_pos = (7.5, 3.0, 5.0)
        jamming_pos = jammer.position_for_jamming(
            target_pos,
            AttackRole.JAMMER_LEADER
        )

        # Leader should be 1m in front of target
        assert jamming_pos[0] == target_pos[0] - 1.0
        assert jamming_pos[1] == target_pos[1]
        assert jamming_pos[2] == target_pos[2]

    def test_position_for_jamming_follower(self):
        """Test jamming position calculation for follower."""
        jammer = JammingBehavior(
            drone_id="jammer_follower",
            role=AttackRole.JAMMER_FOLLOWER
        )

        target_pos = (7.5, 3.0, 5.0)
        jamming_pos = jammer.position_for_jamming(
            target_pos,
            AttackRole.JAMMER_FOLLOWER
        )

        # Follower should be behind leader
        assert jamming_pos[0] < target_pos[0] - 1.0

    def test_maintain_jamming_formation(self):
        """Test jamming formation maintenance."""
        jammer = JammingBehavior(
            drone_id="jammer_1",
            role=AttackRole.JAMMER_LEADER
        )

        # Start jamming
        current_time = time.time()
        should_continue = jammer.maintain_jamming_formation(
            duration=0.1,  # Short duration for testing
            current_time=current_time
        )

        assert should_continue
        assert jammer.is_jamming

        # After duration expires
        time.sleep(0.15)
        should_continue = jammer.maintain_jamming_formation(
            duration=0.1,
            current_time=time.time()
        )

        assert not should_continue
        assert not jammer.is_jamming

    def test_simulate_rf_interference(self):
        """Test RF interference simulation."""
        jammer = JammingBehavior(
            drone_id="jammer_1",
            role=AttackRole.JAMMER_LEADER
        )

        # Not jamming - should return zero values
        metrics = jammer.simulate_rf_interference()
        assert metrics['signal_strength'] == 0.0
        assert metrics['interference_level'] == 0.0

        # Start jamming
        jammer.is_jamming = True
        metrics = jammer.simulate_rf_interference()

        # Should return non-zero values
        assert metrics['signal_strength'] > 0
        assert metrics['interference_level'] > 0

    def test_get_jamming_status(self):
        """Test jamming status retrieval."""
        jammer = JammingBehavior(
            drone_id="jammer_1",
            role=AttackRole.JAMMER_LEADER
        )

        status = jammer.get_jamming_status()

        assert status['drone_id'] == "jammer_1"
        assert status['role'] == AttackRole.JAMMER_LEADER.value
        assert not status['is_jamming']


class TestNeutralizationManeuver:
    """Test NeutralizationManeuver class."""

    def test_initialization(self):
        """Test neutralization maneuver initialization."""
        maneuver = NeutralizationManeuver(
            drone_id="attack_1",
            safe_mode=True
        )

        assert maneuver.drone_id == "attack_1"
        assert maneuver.safe_mode
        assert maneuver.stop_distance == 0.3
        assert maneuver.current_phase == AttackPhase.APPROACH

    def test_kamikaze_approach(self):
        """Test approach waypoint generation."""
        maneuver = NeutralizationManeuver(
            drone_id="attack_1",
            safe_mode=True
        )

        attacker_pos = (7.5, 3.0, 7.0)
        target_pos = (7.5, 3.0, 5.0)

        waypoints = maneuver.kamikaze_approach(attacker_pos, target_pos)

        # Should generate waypoints
        assert len(waypoints) > 0

        # First waypoint should be above target
        first_wp = waypoints[0]
        assert first_wp.position[2] > target_pos[2]

        # In safe mode, final position should be above target
        final_wp = waypoints[-1]
        assert final_wp.position[2] == target_pos[2] + maneuver.stop_distance

    def test_safe_demonstration_stop(self):
        """Test safe demonstration stop check."""
        maneuver = NeutralizationManeuver(
            drone_id="attack_1",
            safe_mode=True
        )

        target_pos = (7.5, 3.0, 5.0)

        # At stop position (0.3m above target)
        current_pos = (7.5, 3.0, 5.3)
        at_stop = maneuver.safe_demonstration_stop(
            current_pos,
            target_pos,
            stop_distance=0.3
        )

        assert at_stop
        assert maneuver.current_phase == AttackPhase.VICTORY

    def test_check_safety_constraints_battery(self):
        """Test safety constraint checking - battery."""
        maneuver = NeutralizationManeuver(drone_id="attack_1")

        # Low battery
        should_abort, reason = maneuver.check_safety_constraints(
            battery_level=0.20,  # Below 25% threshold
            collision_imminent=False
        )

        assert should_abort
        assert "Battery critical" in reason
        assert maneuver.current_phase == AttackPhase.ABORT

    def test_check_safety_constraints_collision(self):
        """Test safety constraint checking - collision."""
        maneuver = NeutralizationManeuver(drone_id="attack_1")

        # Collision imminent
        should_abort, reason = maneuver.check_safety_constraints(
            battery_level=0.80,
            collision_imminent=True
        )

        assert should_abort
        assert "Collision" in reason
        assert maneuver.current_phase == AttackPhase.ABORT

    def test_check_safety_constraints_ok(self):
        """Test safety constraints when all OK."""
        maneuver = NeutralizationManeuver(drone_id="attack_1")

        should_abort, reason = maneuver.check_safety_constraints(
            battery_level=0.80,
            collision_imminent=False
        )

        assert not should_abort
        assert reason is None

    def test_victory_hover(self):
        """Test victory hover position calculation."""
        maneuver = NeutralizationManeuver(drone_id="attack_1")

        current_pos = (7.5, 3.0, 5.3)
        victory_pos = maneuver.victory_hover(
            current_pos,
            hover_altitude_offset=0.5
        )

        # Should be 0.5m above current position
        assert victory_pos[0] == current_pos[0]
        assert victory_pos[1] == current_pos[1]
        assert victory_pos[2] == current_pos[2] + 0.5

    def test_return_to_home(self):
        """Test return to home waypoint generation."""
        maneuver = NeutralizationManeuver(drone_id="attack_1")

        current_pos = (7.5, 3.0, 5.3)
        home_pos = (0.0, 0.0, 0.5)

        waypoints = maneuver.return_to_home(current_pos, home_pos)

        # Should generate waypoints
        assert len(waypoints) > 0

        # Final waypoint should be home position
        assert waypoints[-1].position == home_pos


class TestAttackCoordinator:
    """Test AttackCoordinator class."""

    def test_initialization(self):
        """Test attack coordinator initialization."""
        coordinator = AttackCoordinator(target_pos=(7.5, 3.0, 5.0))

        assert coordinator.target_pos == (7.5, 3.0, 5.0)
        assert coordinator.current_phase == AttackPhase.APPROACH

    def test_assign_roles(self):
        """Test role assignment."""
        coordinator = AttackCoordinator()

        coordinator.assign_roles(
            leader_id="leader",
            follower_id="follower",
            attacker_id="attacker"
        )

        assert coordinator.jammer_leader is not None
        assert coordinator.jammer_follower is not None
        assert coordinator.attack_drone is not None

    def test_get_attack_positions(self):
        """Test getting attack positions."""
        coordinator = AttackCoordinator(target_pos=(7.5, 3.0, 5.0))

        coordinator.assign_roles(
            leader_id="leader",
            follower_id="follower",
            attacker_id="attacker"
        )

        positions = coordinator.get_attack_positions()

        # Should have jammer positions
        assert 'jammer_leader' in positions
        assert 'jammer_follower' in positions

    def test_execute_attack_sequence(self):
        """Test attack sequence execution."""
        coordinator = AttackCoordinator()

        status = coordinator.execute_attack_sequence()

        assert 'current_phase' in status
        assert 'elapsed_time' in status
        assert 'target_position' in status


class TestUtilityFunctions:
    """Test utility functions."""

    def test_calculate_approach_vector_frontal(self):
        """Test frontal approach vector calculation."""
        start_pos = (5.0, 3.0, 4.0)
        target_pos = (7.5, 3.0, 5.0)

        vector = calculate_approach_vector(
            start_pos,
            target_pos,
            approach_type="frontal"
        )

        # Should be normalized
        assert np.isclose(np.linalg.norm(vector), 1.0)

        # Should have no vertical component
        assert vector[2] == 0.0

    def test_calculate_approach_vector_vertical(self):
        """Test vertical approach vector calculation."""
        start_pos = (7.5, 3.0, 7.0)
        target_pos = (7.5, 3.0, 5.0)

        vector = calculate_approach_vector(
            start_pos,
            target_pos,
            approach_type="vertical"
        )

        # Should be straight down
        assert np.allclose(vector, [0, 0, -1])

    def test_calculate_approach_vector_lateral(self):
        """Test lateral approach vector calculation."""
        start_pos = (5.0, 3.0, 5.0)
        target_pos = (7.5, 3.0, 5.0)

        vector = calculate_approach_vector(
            start_pos,
            target_pos,
            approach_type="lateral"
        )

        # Should be normalized
        assert np.isclose(np.linalg.norm(vector), 1.0)

        # Should be perpendicular to direct approach
        direct = np.array(target_pos) - np.array(start_pos)
        assert np.isclose(np.dot(vector, direct), 0.0, atol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
