"""
Unit tests for the Mission State Machine
"""

import pytest
import time
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "scenarios"))

from mission_state_machine import MissionState, MissionStateMachine


class TestMissionState:
    """Test the MissionState enum"""

    def test_mission_states_exist(self):
        """Test that all required mission states exist"""
        assert MissionState.INITIALIZATION.value == "init"
        assert MissionState.SAFETY_CHECK.value == "safety_check"
        assert MissionState.PATROL_SEARCH.value == "patrol_search"
        assert MissionState.TARGET_DETECTED.value == "target_detected"
        assert MissionState.ROLE_ASSIGNMENT.value == "role_assignment"
        assert MissionState.APPROACH_TARGET.value == "approach_target"
        assert MissionState.JAMMING.value == "jamming"
        assert MissionState.NEUTRALIZATION.value == "neutralization"
        assert MissionState.MISSION_COMPLETE.value == "complete"
        assert MissionState.MISSION_ABORT.value == "abort"


class TestMissionStateMachine:
    """Test the MissionStateMachine class"""

    def test_initialization(self):
        """Test state machine initialization"""
        sm = MissionStateMachine()
        assert sm.current_state == MissionState.INITIALIZATION
        assert sm.start_time is None
        assert len(sm.drone_roles) == 0

    def test_start_mission(self):
        """Test starting the mission"""
        sm = MissionStateMachine()
        sm.start_mission()
        assert sm.start_time is not None
        assert sm.get_mission_elapsed_time() >= 0

    def test_valid_transition(self):
        """Test valid state transition"""
        sm = MissionStateMachine()
        result = sm.transition_to(MissionState.SAFETY_CHECK, "test")
        assert result is True
        assert sm.current_state == MissionState.SAFETY_CHECK

    def test_invalid_transition(self):
        """Test invalid state transition"""
        sm = MissionStateMachine()
        # Can't go directly from INIT to PATROL_SEARCH
        result = sm.transition_to(MissionState.PATROL_SEARCH, "test")
        assert result is False
        assert sm.current_state == MissionState.INITIALIZATION

    def test_abort_from_any_state(self):
        """Test that abort is possible from any state"""
        sm = MissionStateMachine()
        sm.transition_to(MissionState.SAFETY_CHECK)
        sm.transition_to(MissionState.PATROL_SEARCH)

        result = sm.abort_mission("test abort")
        assert result is None  # abort_mission doesn't return a value
        assert sm.current_state == MissionState.MISSION_ABORT
        assert sm.abort_reason == "test abort"

    def test_transition_sequence(self):
        """Test a complete valid transition sequence"""
        sm = MissionStateMachine()

        states_sequence = [
            MissionState.SAFETY_CHECK,
            MissionState.PATROL_SEARCH,
            MissionState.TARGET_DETECTED,
            MissionState.ROLE_ASSIGNMENT,
            MissionState.APPROACH_TARGET,
            MissionState.JAMMING,
            MissionState.NEUTRALIZATION,
            MissionState.MISSION_COMPLETE,
        ]

        for state in states_sequence:
            result = sm.transition_to(state, "test")
            assert result is True
            assert sm.current_state == state

    def test_mission_timeout(self):
        """Test mission timeout detection"""
        sm = MissionStateMachine()
        sm.start_mission()

        # Should not be timeout initially
        assert sm.is_mission_timeout() is False

        # Simulate timeout by setting start time in the past
        sm.start_time = time.time() - (sm.MISSION_TIMEOUT + 10)
        assert sm.is_mission_timeout() is True

    def test_phase_timeout(self):
        """Test phase timeout detection"""
        sm = MissionStateMachine()
        sm.start_mission()

        # Should not timeout immediately
        assert sm.is_phase_timeout() is False

        # Simulate timeout by setting phase timer in the past
        sm.phase_timers[sm.current_state] = time.time() - (sm.STATE_TIMEOUTS[sm.current_state] + 5)
        assert sm.is_phase_timeout() is True

    def test_mission_complete_check(self):
        """Test mission complete detection"""
        sm = MissionStateMachine()
        assert sm.is_mission_complete() is False

        # Transition through to complete
        sm.transition_to(MissionState.SAFETY_CHECK)
        sm.transition_to(MissionState.PATROL_SEARCH)
        sm.transition_to(MissionState.TARGET_DETECTED)
        sm.transition_to(MissionState.ROLE_ASSIGNMENT)
        sm.transition_to(MissionState.APPROACH_TARGET)
        sm.transition_to(MissionState.JAMMING)
        sm.transition_to(MissionState.NEUTRALIZATION)
        sm.transition_to(MissionState.MISSION_COMPLETE)

        assert sm.is_mission_complete() is True

    def test_mission_abort_is_complete(self):
        """Test that aborted missions are considered complete"""
        sm = MissionStateMachine()
        sm.abort_mission("test")
        assert sm.is_mission_complete() is True

    def test_set_drone_roles(self):
        """Test setting drone roles"""
        sm = MissionStateMachine()
        roles = {
            "cf1": "leader",
            "cf2": "follower",
            "cf3": "attacker"
        }
        sm.set_drone_roles(roles)
        assert sm.drone_roles == roles

    def test_get_mission_status(self):
        """Test getting mission status"""
        sm = MissionStateMachine()
        sm.start_mission()
        sm.transition_to(MissionState.SAFETY_CHECK)

        status = sm.get_mission_status()
        assert status["current_state"] == "safety_check"
        assert status["elapsed_time"] >= 0
        assert status["is_complete"] is False
        assert status["abort_reason"] is None

    def test_state_callbacks(self):
        """Test state callback registration and execution"""
        sm = MissionStateMachine()
        callback_executed = {"value": False}

        def test_callback():
            callback_executed["value"] = True

        sm.register_state_callback(MissionState.SAFETY_CHECK, test_callback)
        sm.transition_to(MissionState.SAFETY_CHECK)

        assert callback_executed["value"] is True

    def test_transition_history(self):
        """Test that transition history is recorded"""
        sm = MissionStateMachine()
        sm.transition_to(MissionState.SAFETY_CHECK, "first")
        sm.transition_to(MissionState.PATROL_SEARCH, "second")

        assert len(sm.transition_history) == 2
        assert sm.transition_history[0].from_state == MissionState.INITIALIZATION
        assert sm.transition_history[0].to_state == MissionState.SAFETY_CHECK
        assert sm.transition_history[0].reason == "first"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
