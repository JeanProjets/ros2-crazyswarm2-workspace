"""
Tests for Scenario 2 State Machine
"""

import pytest
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scenarios.scenario_2_fsm import (
    Scenario2StateMachine,
    MissionState,
    SwarmTelemetry
)


class TestSwarmTelemetry:
    """Test suite for SwarmTelemetry"""

    def test_initialization(self):
        """Test telemetry initialization"""
        telemetry = SwarmTelemetry()

        assert len(telemetry.drones) == 0
        assert telemetry.timestamp >= 0

    def test_get_drone_position(self):
        """Test getting drone position"""
        telemetry = SwarmTelemetry()
        telemetry.drones['drone1'] = {
            'x': 3.0,
            'y': 4.0,
            'z': 2.0,
            'voltage': 3.9
        }

        position = telemetry.get_drone_position('drone1')

        assert position is not None
        assert position['x'] == 3.0
        assert position['y'] == 4.0
        assert position['z'] == 2.0

    def test_get_drone_position_unknown(self):
        """Test getting position of unknown drone"""
        telemetry = SwarmTelemetry()

        position = telemetry.get_drone_position('unknown')

        assert position is None

    def test_get_drone_voltage(self):
        """Test getting drone voltage"""
        telemetry = SwarmTelemetry()
        telemetry.drones['drone1'] = {'voltage': 3.9}

        voltage = telemetry.get_drone_voltage('drone1')

        assert voltage == 3.9


class TestScenario2StateMachine:
    """Test suite for Scenario2StateMachine"""

    def test_initialization(self):
        """Test FSM initialization"""
        fsm = Scenario2StateMachine()

        assert fsm.current_state == MissionState.IDLE
        assert fsm.mission_start_time is None
        assert fsm.leader_id is None

    def test_start_mission(self):
        """Test mission start"""
        fsm = Scenario2StateMachine()

        fsm.start_mission('leader1', ['follower1', 'follower2'])

        assert fsm.current_state == MissionState.SAFETY_CHK_AND_TRANSIT
        assert fsm.leader_id == 'leader1'
        assert fsm.follower_ids == ['follower1', 'follower2']
        assert fsm.mission_start_time is not None

    def test_check_corner_safety_safe(self):
        """Test corner safety check with safe positions"""
        fsm = Scenario2StateMachine()

        telemetry = SwarmTelemetry()
        telemetry.drones['drone1'] = {'x': 5.0, 'y': 5.0, 'z': 3.0}
        telemetry.drones['drone2'] = {'x': 6.0, 'y': 4.0, 'z': 3.0}

        result = fsm.check_corner_safety(telemetry)

        assert result is True

    def test_check_corner_safety_x_violation(self):
        """Test corner safety check with X boundary violation"""
        fsm = Scenario2StateMachine()

        telemetry = SwarmTelemetry()
        telemetry.drones['drone1'] = {'x': 9.9, 'y': 5.0, 'z': 3.0}  # Exceeds x_max

        result = fsm.check_corner_safety(telemetry)

        assert result is False

    def test_check_corner_safety_y_violation(self):
        """Test corner safety check with Y boundary violation"""
        fsm = Scenario2StateMachine()

        telemetry = SwarmTelemetry()
        telemetry.drones['drone1'] = {'x': 5.0, 'y': 0.1, 'z': 3.0}  # Below y_min

        result = fsm.check_corner_safety(telemetry)

        assert result is False

    def test_monitor_transit_battery_ok(self):
        """Test battery monitoring with safe voltage"""
        fsm = Scenario2StateMachine()
        fsm.leader_id = 'leader1'
        fsm.current_state = MissionState.SAFETY_CHK_AND_TRANSIT

        telemetry = SwarmTelemetry()
        telemetry.drones['leader1'] = {'voltage': 3.9}

        result = fsm.monitor_transit_battery(telemetry)

        assert result is True

    def test_monitor_transit_battery_critical(self):
        """Test battery monitoring with critical voltage"""
        fsm = Scenario2StateMachine()
        fsm.leader_id = 'leader1'
        fsm.current_state = MissionState.SAFETY_CHK_AND_TRANSIT

        telemetry = SwarmTelemetry()
        telemetry.drones['leader1'] = {'voltage': 3.4}  # Below critical

        result = fsm.monitor_transit_battery(telemetry)

        assert result is False

    def test_monitor_transit_battery_low_for_transit(self):
        """Test battery monitoring with voltage too low for transit"""
        fsm = Scenario2StateMachine()
        fsm.leader_id = 'leader1'
        fsm.current_state = MissionState.SAFETY_CHK_AND_TRANSIT

        telemetry = SwarmTelemetry()
        telemetry.drones['leader1'] = {'voltage': 3.65}  # Below transit minimum

        result = fsm.monitor_transit_battery(telemetry)

        assert result is False

    def test_state_transition(self):
        """Test state transitions"""
        fsm = Scenario2StateMachine()

        initial_state = fsm.current_state
        fsm.start_mission('leader1', ['follower1'])

        assert fsm.current_state != initial_state
        assert fsm.previous_state == initial_state
        assert len(fsm.state_history) > 0

    def test_get_mission_elapsed_time(self):
        """Test mission elapsed time calculation"""
        fsm = Scenario2StateMachine()

        # Before mission start
        elapsed = fsm.get_mission_elapsed_time()
        assert elapsed == 0.0

        # After mission start
        fsm.start_mission('leader1', ['follower1'])
        time.sleep(0.1)
        elapsed = fsm.get_mission_elapsed_time()
        assert elapsed > 0.0

    def test_get_state_summary(self):
        """Test state summary generation"""
        fsm = Scenario2StateMachine()
        fsm.start_mission('leader1', ['follower1', 'follower2'])

        summary = fsm.get_state_summary()

        assert 'current_state' in summary
        assert 'mission_elapsed' in summary
        assert 'leader_id' in summary
        assert summary['leader_id'] == 'leader1'

    def test_update_progresses_through_states(self):
        """Test that update can progress through states"""
        fsm = Scenario2StateMachine()
        fsm.start_mission('leader1', ['follower1'])

        # Create telemetry
        telemetry = SwarmTelemetry()
        telemetry.drones['leader1'] = {
            'x': 5.0,
            'y': 5.0,
            'z': 3.0,
            'voltage': 3.9,
            'vx': 0.0,
            'vy': 0.0,
            'vz': 0.0
        }

        initial_state = fsm.current_state

        # Update FSM
        new_state = fsm.update(telemetry)

        # State should be valid
        assert new_state in MissionState

    def test_abort_on_safety_failure(self):
        """Test that mission aborts on safety failure"""
        fsm = Scenario2StateMachine()
        fsm.start_mission('leader1', ['follower1'])

        # Create unsafe telemetry
        telemetry = SwarmTelemetry()
        telemetry.drones['leader1'] = {
            'x': 9.9,  # Exceeds boundary
            'y': 5.0,
            'z': 3.0,
            'voltage': 3.9
        }

        # Update should detect violation
        fsm.update(telemetry)

        # Should be in abort state
        assert fsm.current_state == MissionState.EMERGENCY_ABORT
        assert fsm.abort_reason is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
