"""
Unit tests for behavior_sequencer module.

Tests state machine, behavior execution, and swarm coordination.
"""

import pytest
import numpy as np
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from behaviors.behavior_sequencer import (
    BehaviorState,
    BehaviorType,
    BehaviorPriority,
    BehaviorStatus,
    StateTransition,
    BehaviorSequencer,
    SwarmBehaviorCoordinator
)


class TestBehaviorState:
    """Test BehaviorState enum."""

    def test_states_exist(self):
        """Test that all states are defined."""
        assert BehaviorState.IDLE
        assert BehaviorState.SEARCH
        assert BehaviorState.TRACK
        assert BehaviorState.FORMATION
        assert BehaviorState.ATTACK
        assert BehaviorState.RTH
        assert BehaviorState.EMERGENCY
        assert BehaviorState.COMPLETED


class TestBehaviorType:
    """Test BehaviorType enum."""

    def test_behavior_types_exist(self):
        """Test that all behavior types are defined."""
        assert BehaviorType.PATROL
        assert BehaviorType.SAFETY_CHECK
        assert BehaviorType.FORMATION_FLY
        assert BehaviorType.TRACK_TARGET
        assert BehaviorType.JAMMING
        assert BehaviorType.NEUTRALIZATION
        assert BehaviorType.VICTORY_HOVER
        assert BehaviorType.RETURN_HOME


class TestBehaviorStatus:
    """Test BehaviorStatus dataclass."""

    def test_status_creation(self):
        """Test creating a behavior status."""
        status = BehaviorStatus(
            drone_id="drone_1",
            state=BehaviorState.SEARCH,
            behavior_type=BehaviorType.PATROL,
            priority=BehaviorPriority.MEDIUM,
            start_time=time.time(),
            progress=0.5
        )

        assert status.drone_id == "drone_1"
        assert status.state == BehaviorState.SEARCH
        assert status.behavior_type == BehaviorType.PATROL
        assert status.priority == BehaviorPriority.MEDIUM
        assert status.progress == 0.5


class TestStateTransition:
    """Test StateTransition dataclass."""

    def test_transition_creation(self):
        """Test creating a state transition."""
        transition = StateTransition(
            from_state=BehaviorState.IDLE,
            to_state=BehaviorState.SEARCH,
            condition="Mission start"
        )

        assert transition.from_state == BehaviorState.IDLE
        assert transition.to_state == BehaviorState.SEARCH
        assert transition.condition == "Mission start"


class TestBehaviorSequencer:
    """Test BehaviorSequencer class."""

    def test_initialization(self):
        """Test behavior sequencer initialization."""
        sequencer = BehaviorSequencer(drone_id="drone_1")

        assert sequencer.drone_id == "drone_1"
        assert sequencer.current_state == BehaviorState.IDLE
        assert not sequencer.mission_started
        assert not sequencer.target_detected

    def test_state_transitions_defined(self):
        """Test that state transitions are defined."""
        sequencer = BehaviorSequencer(drone_id="drone_1")

        assert len(sequencer.transitions) > 0

        # Check that key transitions exist
        transition_pairs = [
            (t.from_state, t.to_state) for t in sequencer.transitions
        ]

        assert (BehaviorState.IDLE, BehaviorState.SEARCH) in transition_pairs
        assert (BehaviorState.SEARCH, BehaviorState.TRACK) in transition_pairs
        assert (BehaviorState.FORMATION, BehaviorState.ATTACK) in transition_pairs

    def test_execute_behavior_patrol(self):
        """Test executing patrol behavior."""
        sequencer = BehaviorSequencer(drone_id="drone_1")

        status = sequencer.execute_behavior(
            drone_id="drone_1",
            behavior_type=BehaviorType.PATROL,
            params={}
        )

        assert status.drone_id == "drone_1"
        assert status.behavior_type == BehaviorType.PATROL
        assert status.progress >= 0.0

    def test_execute_behavior_formation(self):
        """Test executing formation behavior."""
        sequencer = BehaviorSequencer(drone_id="drone_1")

        status = sequencer.execute_behavior(
            drone_id="drone_1",
            behavior_type=BehaviorType.FORMATION_FLY,
            params={}
        )

        assert status.behavior_type == BehaviorType.FORMATION_FLY

    def test_transition_idle_to_search(self):
        """Test transition from IDLE to SEARCH."""
        sequencer = BehaviorSequencer(drone_id="drone_1")

        assert sequencer.current_state == BehaviorState.IDLE

        # Set mission started
        sequencer.set_mission_parameter("mission_started", True)

        # Check transition
        next_state = sequencer.transition_check()
        assert next_state == BehaviorState.SEARCH

    def test_transition_search_to_track(self):
        """Test transition from SEARCH to TRACK."""
        sequencer = BehaviorSequencer(drone_id="drone_1")
        sequencer.current_state = BehaviorState.SEARCH

        # Set target detected
        sequencer.set_mission_parameter("target_detected", True)

        # Check transition
        next_state = sequencer.transition_check()
        assert next_state == BehaviorState.TRACK

    def test_transition_track_to_formation(self):
        """Test transition from TRACK to FORMATION."""
        sequencer = BehaviorSequencer(drone_id="drone_1")
        sequencer.current_state = BehaviorState.TRACK

        # Set role assigned
        sequencer.set_mission_parameter("role_assigned", True)

        # Check transition
        next_state = sequencer.transition_check()
        assert next_state == BehaviorState.FORMATION

    def test_transition_formation_to_attack(self):
        """Test transition from FORMATION to ATTACK."""
        sequencer = BehaviorSequencer(drone_id="drone_1")
        sequencer.current_state = BehaviorState.FORMATION

        # Set in position
        sequencer.set_mission_parameter("in_position", True)

        # Check transition
        next_state = sequencer.transition_check()
        assert next_state == BehaviorState.ATTACK

    def test_transition_attack_to_rth(self):
        """Test transition from ATTACK to RTH."""
        sequencer = BehaviorSequencer(drone_id="drone_1")
        sequencer.current_state = BehaviorState.ATTACK

        # Set mission complete
        sequencer.set_mission_parameter("mission_complete", True)

        # Check transition
        next_state = sequencer.transition_check()
        assert next_state == BehaviorState.RTH

    def test_emergency_transition_low_battery(self):
        """Test emergency transition on low battery."""
        sequencer = BehaviorSequencer(drone_id="drone_1")
        sequencer.current_state = BehaviorState.SEARCH

        # Set battery critical
        sequencer.set_mission_parameter("battery_level", 0.20)

        # Should transition to emergency
        next_state = sequencer.transition_check()
        assert next_state == BehaviorState.EMERGENCY

    def test_emergency_transition_collision(self):
        """Test emergency transition on collision detection."""
        sequencer = BehaviorSequencer(drone_id="drone_1")
        sequencer.current_state = BehaviorState.FORMATION

        # Set collision detected
        sequencer.set_mission_parameter("collision_detected", True)

        # Should transition to emergency
        next_state = sequencer.transition_check()
        assert next_state == BehaviorState.EMERGENCY

    def test_transition_to_new_state(self):
        """Test transitioning to a new state."""
        sequencer = BehaviorSequencer(drone_id="drone_1")

        initial_state = sequencer.current_state
        sequencer.transition_to(BehaviorState.SEARCH, "Test transition")

        assert sequencer.current_state == BehaviorState.SEARCH
        assert sequencer.current_state != initial_state

    def test_abort_behavior(self):
        """Test aborting current behavior."""
        sequencer = BehaviorSequencer(drone_id="drone_1")
        sequencer.current_state = BehaviorState.FORMATION

        # Execute a behavior
        sequencer.execute_behavior(
            drone_id="drone_1",
            behavior_type=BehaviorType.FORMATION_FLY,
            params={}
        )

        # Abort
        sequencer.abort_behavior("Test abort")

        # Should transition to RTH or EMERGENCY
        assert sequencer.current_state in [BehaviorState.RTH, BehaviorState.EMERGENCY]

    def test_get_behavior_status(self):
        """Test getting behavior status."""
        sequencer = BehaviorSequencer(drone_id="drone_1")

        # Execute a behavior
        sequencer.execute_behavior(
            drone_id="drone_1",
            behavior_type=BehaviorType.PATROL,
            params={}
        )

        # Get status
        status = sequencer.get_behavior_status("drone_1")

        assert status is not None
        assert status.drone_id == "drone_1"

    def test_update(self):
        """Test sequencer update."""
        sequencer = BehaviorSequencer(drone_id="drone_1")

        # Set mission parameters for transition
        sequencer.set_mission_parameter("mission_started", True)

        # Update should trigger state transition
        sequencer.update(dt=0.01)

        # Should have transitioned to SEARCH
        assert sequencer.current_state == BehaviorState.SEARCH

    def test_get_mission_status(self):
        """Test getting mission status."""
        sequencer = BehaviorSequencer(drone_id="drone_1")

        status = sequencer.get_mission_status()

        assert status['drone_id'] == "drone_1"
        assert 'current_state' in status
        assert 'battery_level' in status
        assert 'mission_started' in status


class TestSwarmBehaviorCoordinator:
    """Test SwarmBehaviorCoordinator class."""

    def test_initialization(self):
        """Test swarm coordinator initialization."""
        coordinator = SwarmBehaviorCoordinator(num_drones=4)

        assert coordinator.num_drones == 4
        assert len(coordinator.sequencers) == 4

    def test_start_mission(self):
        """Test starting mission for all drones."""
        coordinator = SwarmBehaviorCoordinator(num_drones=3)

        coordinator.start_mission()

        # All drones should have mission_started = True
        for sequencer in coordinator.sequencers.values():
            assert sequencer.mission_started

    def test_broadcast_target_detected(self):
        """Test broadcasting target detection."""
        coordinator = SwarmBehaviorCoordinator(num_drones=3)

        target_pos = (7.5, 3.0, 5.0)
        coordinator.broadcast_target_detected(target_pos)

        # All drones should have target_detected = True
        for sequencer in coordinator.sequencers.values():
            assert sequencer.target_detected

    def test_assign_roles(self):
        """Test assigning roles to drones."""
        coordinator = SwarmBehaviorCoordinator(num_drones=4)

        role_assignments = {
            "drone_0": "leader",
            "drone_1": "follower",
            "drone_2": "attacker"
        }

        coordinator.assign_roles(role_assignments)

        # Assigned drones should have role_assigned = True
        assert coordinator.sequencers["drone_0"].role_assigned
        assert coordinator.sequencers["drone_1"].role_assigned
        assert coordinator.sequencers["drone_2"].role_assigned

        # Unassigned drone should not
        assert not coordinator.sequencers["drone_3"].role_assigned

    def test_update_all(self):
        """Test updating all sequencers."""
        coordinator = SwarmBehaviorCoordinator(num_drones=3)

        # Start mission
        coordinator.start_mission()

        # Update all
        coordinator.update_all(dt=0.01)

        # All drones should have transitioned to SEARCH
        for sequencer in coordinator.sequencers.values():
            assert sequencer.current_state == BehaviorState.SEARCH

    def test_get_swarm_status(self):
        """Test getting swarm status."""
        coordinator = SwarmBehaviorCoordinator(num_drones=3)

        status = coordinator.get_swarm_status()

        assert status['num_drones'] == 3
        assert 'drone_statuses' in status
        assert len(status['drone_statuses']) == 3
        assert 'state_distribution' in status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
