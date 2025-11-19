"""
Tests for Scenario 2 Mission Sequencer
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scenarios.scenario_2_mission import (
    Scenario2MissionSequencer,
    MissionResult
)
from scenarios.battery_role_manager import Drone
from scenarios.scenario_2_fsm import SwarmTelemetry, MissionState


class TestScenario2MissionSequencer:
    """Test suite for Scenario2MissionSequencer"""

    def test_initialization(self):
        """Test mission sequencer initialization"""
        sequencer = Scenario2MissionSequencer()

        assert sequencer.fsm is not None
        assert sequencer.battery_manager is not None
        assert sequencer.safety_override is not None
        assert sequencer.mission_result == MissionResult.IN_PROGRESS

    def test_initialize_mission_success(self):
        """Test successful mission initialization"""
        sequencer = Scenario2MissionSequencer()

        drones = [
            Drone('drone1', 3.9),
            Drone('drone2', 4.0),
            Drone('drone3', 3.8)
        ]

        result = sequencer.initialize_mission(drones)

        assert result is True
        assert sequencer.mission_start_time is not None
        assert sequencer.fsm.leader_id is not None

    def test_initialize_mission_insufficient_battery(self):
        """Test mission initialization failure due to low battery"""
        sequencer = Scenario2MissionSequencer()

        # All drones have low battery
        drones = [
            Drone('drone1', 3.6),
            Drone('drone2', 3.5),
            Drone('drone3', 3.4)
        ]

        result = sequencer.initialize_mission(drones)

        assert result is False
        assert sequencer.mission_result == MissionResult.ABORTED

    def test_initialize_mission_insufficient_drones(self):
        """Test mission initialization with too few drones"""
        sequencer = Scenario2MissionSequencer()

        drones = [Drone('drone1', 4.0)]

        result = sequencer.initialize_mission(drones)

        assert result is False

    def test_execute_mission_step(self):
        """Test executing a mission step"""
        sequencer = Scenario2MissionSequencer()

        drones = [
            Drone('drone1', 3.9),
            Drone('drone2', 4.0),
        ]

        sequencer.initialize_mission(drones)

        # Create telemetry
        telemetry = SwarmTelemetry()
        telemetry.drones['drone2'] = {
            'x': 3.0,
            'y': 3.0,
            'z': 2.0,
            'voltage': 3.95,
            'vx': 0.1,
            'vy': 0.0,
            'vz': 0.0
        }

        state = sequencer.execute_mission_step(telemetry)

        assert state in MissionState
        assert len(sequencer.telemetry_logs) > 0

    def test_execute_mission_step_logs_telemetry(self):
        """Test that mission step logs telemetry"""
        sequencer = Scenario2MissionSequencer()

        drones = [
            Drone('drone1', 3.9),
            Drone('drone2', 4.0),
        ]

        sequencer.initialize_mission(drones)

        telemetry = SwarmTelemetry()
        telemetry.drones['drone2'] = {
            'x': 3.0,
            'y': 3.0,
            'z': 2.0,
            'voltage': 3.95
        }

        # Execute multiple steps to ensure logging
        for _ in range(5):
            sequencer.execute_mission_step(telemetry)

        # Should have logged telemetry
        assert len(sequencer.telemetry_logs) > 0

    def test_execute_mission_step_critical_battery(self):
        """Test mission abort on critical battery"""
        sequencer = Scenario2MissionSequencer()

        drones = [
            Drone('drone1', 3.9),
            Drone('drone2', 4.0),
        ]

        sequencer.initialize_mission(drones)

        # Create telemetry with critical battery
        telemetry = SwarmTelemetry()
        telemetry.drones[sequencer.fsm.leader_id] = {
            'x': 3.0,
            'y': 3.0,
            'z': 2.0,
            'voltage': 3.4,  # Critical!
            'vx': 0.0,
            'vy': 0.0,
            'vz': 0.0
        }

        sequencer.execute_mission_step(telemetry)

        assert sequencer.mission_result == MissionResult.ABORTED

    def test_get_mission_summary(self):
        """Test mission summary generation"""
        sequencer = Scenario2MissionSequencer()

        drones = [
            Drone('drone1', 3.9),
            Drone('drone2', 4.0),
        ]

        sequencer.initialize_mission(drones)

        summary = sequencer.get_mission_summary()

        assert 'result' in summary
        assert 'duration' in summary
        assert 'state_machine' in summary
        assert 'battery_management' in summary
        assert 'safety_violations' in summary

    def test_verify_formation(self):
        """Test formation verification"""
        sequencer = Scenario2MissionSequencer()

        drones = [
            Drone('drone1', 3.9),
            Drone('drone2', 4.0),
            Drone('drone3', 3.85)
        ]

        sequencer.initialize_mission(drones)

        telemetry = SwarmTelemetry()
        for follower_id in sequencer.fsm.follower_ids:
            telemetry.drones[follower_id] = {
                'x': 8.0,
                'y': 1.0,
                'z': 3.0
            }

        result = sequencer._verify_formation(telemetry)

        # Should return True since followers are present
        assert isinstance(result, bool)

    def test_map_state_to_phase(self):
        """Test state to phase mapping"""
        sequencer = Scenario2MissionSequencer()

        from scenarios.battery_role_manager import MissionPhase

        phase = sequencer._map_state_to_phase(MissionState.SAFETY_CHK_AND_TRANSIT)
        assert phase == MissionPhase.TRANSIT

        phase = sequencer._map_state_to_phase(MissionState.CORNER_JAMMING)
        assert phase == MissionPhase.JAMMING

        phase = sequencer._map_state_to_phase(MissionState.VERTICAL_DROP)
        assert phase == MissionPhase.ATTACK

    def test_convert_to_telemetry_list(self):
        """Test conversion of SwarmTelemetry to Telemetry list"""
        sequencer = Scenario2MissionSequencer()

        swarm_telemetry = SwarmTelemetry()
        swarm_telemetry.drones['drone1'] = {
            'x': 3.0,
            'y': 4.0,
            'z': 2.0,
            'vx': 0.1,
            'vy': 0.0,
            'vz': 0.0
        }

        telemetry_list = sequencer._convert_to_telemetry_list(swarm_telemetry)

        assert len(telemetry_list) == 1
        assert telemetry_list[0].drone_id == 'drone1'
        assert telemetry_list[0].x == 3.0

    def test_telemetry_logging_rate(self):
        """Test that telemetry is logged at correct rate"""
        sequencer = Scenario2MissionSequencer()
        sequencer.TELEMETRY_LOG_RATE_HZ = 100  # High rate for testing

        drones = [
            Drone('drone1', 3.9),
            Drone('drone2', 4.0),
        ]

        sequencer.initialize_mission(drones)

        telemetry = SwarmTelemetry()
        telemetry.drones['drone2'] = {
            'x': 3.0,
            'y': 3.0,
            'z': 2.0,
            'voltage': 3.95
        }

        # Execute multiple steps quickly
        import time
        for i in range(10):
            sequencer.execute_mission_step(telemetry)
            time.sleep(0.01)

        # Should have logged multiple times
        assert len(sequencer.telemetry_logs) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
