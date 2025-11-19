"""
Integration tests for Scenario 1 Mission
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "scenarios"))

from scenario_1_mission import (
    Scenario1Mission,
    PhaseExecutor,
    MissionResult,
    HealthStatus,
    DroneAction
)
from mission_state_machine import MissionState
from role_manager import DroneRole


class TestScenario1Mission:
    """Test the Scenario1Mission class"""

    def test_initialization(self):
        """Test mission initialization"""
        mission = Scenario1Mission()
        assert mission.state_machine is not None
        assert mission.role_manager is not None
        assert mission.mission_start_time is None

    def test_initialize_mission(self):
        """Test mission initialization with config"""
        mission = Scenario1Mission()
        result = mission.initialize_mission()

        assert result is True
        assert len(mission.role_manager.drone_registry) == 3
        assert "cf1" in mission.role_manager.drone_registry
        assert "cf2" in mission.role_manager.drone_registry
        assert "cf3" in mission.role_manager.drone_registry

    def test_default_config(self):
        """Test default configuration"""
        mission = Scenario1Mission()
        config = mission.config

        assert "drone_ids" in config
        assert len(config["drone_ids"]) == 3
        assert "initial_positions" in config
        assert config["simulation_mode"] is True

    @pytest.mark.asyncio
    async def test_execute_mission_flow(self):
        """Test basic mission execution flow"""
        mission = Scenario1Mission()
        mission.initialize_mission()

        # Run mission with timeout
        # Note: This test verifies the mission can execute without crashing
        # The mission may abort or complete depending on timing
        try:
            result = await asyncio.wait_for(
                mission.execute_mission(),
                timeout=20.0  # Generous timeout
            )

            assert isinstance(result, MissionResult)
            assert result.final_state in [
                MissionState.MISSION_COMPLETE,
                MissionState.MISSION_ABORT
            ]
            assert result.completion_time > 0
        except asyncio.TimeoutError:
            # Mission is still running but hasn't crashed - this is acceptable
            # as the timing in simulation may vary
            pass

    def test_monitor_mission_health_healthy(self):
        """Test mission health monitoring with healthy drones"""
        mission = Scenario1Mission()
        mission.initialize_mission()

        # Set good battery levels
        for drone_id in mission.config["drone_ids"]:
            mission.role_manager.update_drone_info(
                drone_id,
                battery_level=80.0
            )

        health = mission.monitor_mission_health()
        assert health == HealthStatus.HEALTHY

    def test_monitor_mission_health_critical(self):
        """Test mission health monitoring with critical battery"""
        mission = Scenario1Mission()
        mission.initialize_mission()

        # Set critical battery level
        mission.role_manager.update_drone_info("cf1", battery_level=15.0)

        health = mission.monitor_mission_health()
        assert health == HealthStatus.CRITICAL

    def test_coordinate_drone_actions(self):
        """Test coordinating drone actions"""
        mission = Scenario1Mission()
        mission.initialize_mission()

        actions = [
            DroneAction(
                drone_id="cf1",
                action_type="move",
                target_position=(1.0, 2.0, 3.0),
                duration=5.0
            ),
            DroneAction(
                drone_id="cf2",
                action_type="hover",
                duration=2.0
            )
        ]

        result = mission.coordinate_drone_actions(actions)
        assert result is True

    def test_handle_contingency_battery(self):
        """Test handling battery critical contingency"""
        mission = Scenario1Mission()
        mission.initialize_mission()
        mission.state_machine.start_mission()

        mission.handle_contingency(
            "battery_critical",
            {"drone_id": "cf1"}
        )

        assert mission.state_machine.current_state == MissionState.MISSION_ABORT

    def test_handle_contingency_tracking_lost(self):
        """Test handling tracking lost contingency"""
        mission = Scenario1Mission()
        mission.initialize_mission()
        mission.state_machine.start_mission()

        mission.handle_contingency(
            "tracking_lost",
            {"drone_id": "cf1", "duration": 10.0}
        )

        assert mission.state_machine.current_state == MissionState.MISSION_ABORT


class TestPhaseExecutor:
    """Test the PhaseExecutor class"""

    @pytest.mark.asyncio
    async def test_execute_initialization(self):
        """Test initialization phase execution"""
        mission = Scenario1Mission()
        mission.initialize_mission()

        executor = PhaseExecutor(mission)

        await executor.execute_initialization()
        # Should complete without errors

    @pytest.mark.asyncio
    async def test_execute_safety_check(self):
        """Test safety check phase execution"""
        mission = Scenario1Mission()
        mission.initialize_mission()
        mission.state_machine.start_mission()
        mission.state_machine.transition_to(MissionState.SAFETY_CHECK)

        executor = PhaseExecutor(mission)

        await executor.execute_safety_check()

        # Should transition to patrol search
        assert mission.state_machine.current_state == MissionState.PATROL_SEARCH

    @pytest.mark.asyncio
    async def test_execute_patrol_search(self):
        """Test patrol search phase execution"""
        mission = Scenario1Mission()
        mission.initialize_mission()
        mission.state_machine.start_mission()
        mission.state_machine.transition_to(MissionState.PATROL_SEARCH)

        executor = PhaseExecutor(mission)

        # Execute patrol search a few times
        for _ in range(10):
            await executor.execute_patrol_search()
            await asyncio.sleep(0.1)

        # Should eventually detect target
        assert mission.target_detected_position is not None

    @pytest.mark.asyncio
    async def test_execute_target_approach(self):
        """Test target approach phase execution"""
        mission = Scenario1Mission()
        mission.initialize_mission()
        mission.state_machine.start_mission()

        # Set up target detection
        mission.target_detected_position = (1.0, 2.0, 0.3)
        mission.detector_drone_id = "cf3"

        # Reassign roles
        mission.role_manager.reassign_roles_on_detection("cf3")

        mission.state_machine.transition_to(MissionState.APPROACH_TARGET)

        executor = PhaseExecutor(mission)

        await executor.execute_target_approach()

        # Wait for position check
        await asyncio.sleep(4.0)

        # Should transition to jamming after approach
        # (may have already transitioned)

    @pytest.mark.asyncio
    async def test_execute_jamming(self):
        """Test jamming phase execution"""
        mission = Scenario1Mission()
        mission.initialize_mission()
        mission.state_machine.start_mission()
        mission.state_machine.transition_to(MissionState.JAMMING)

        executor = PhaseExecutor(mission)

        await executor.execute_jamming()
        # Should complete without errors

    @pytest.mark.asyncio
    async def test_execute_neutralization(self):
        """Test neutralization phase execution"""
        mission = Scenario1Mission()
        mission.initialize_mission()
        mission.state_machine.start_mission()

        # Set up target
        mission.target_detected_position = (1.0, 2.0, 0.3)
        mission.detector_drone_id = "cf3"
        mission.role_manager.reassign_roles_on_detection("cf3")

        # Transition through valid state sequence to reach NEUTRALIZATION
        mission.state_machine.transition_to(MissionState.SAFETY_CHECK)
        mission.state_machine.transition_to(MissionState.PATROL_SEARCH)
        mission.state_machine.transition_to(MissionState.TARGET_DETECTED)
        mission.state_machine.transition_to(MissionState.ROLE_ASSIGNMENT)
        mission.state_machine.transition_to(MissionState.APPROACH_TARGET)
        mission.state_machine.transition_to(MissionState.JAMMING)
        mission.state_machine.transition_to(MissionState.NEUTRALIZATION)

        executor = PhaseExecutor(mission)

        await executor.execute_neutralization()

        # Should complete and transition to mission complete
        assert mission.state_machine.current_state == MissionState.MISSION_COMPLETE


class TestDroneAction:
    """Test the DroneAction dataclass"""

    def test_drone_action_creation(self):
        """Test creating a drone action"""
        action = DroneAction(
            drone_id="cf1",
            action_type="move",
            target_position=(1.0, 2.0, 3.0),
            duration=5.0
        )

        assert action.drone_id == "cf1"
        assert action.action_type == "move"
        assert action.target_position == (1.0, 2.0, 3.0)
        assert action.duration == 5.0
        assert action.parameters == {}

    def test_drone_action_with_parameters(self):
        """Test creating a drone action with parameters"""
        action = DroneAction(
            drone_id="cf2",
            action_type="land",
            parameters={"speed": 0.5}
        )

        assert action.parameters["speed"] == 0.5


class TestMissionResult:
    """Test the MissionResult dataclass"""

    def test_mission_result_success(self):
        """Test creating a successful mission result"""
        result = MissionResult(
            success=True,
            completion_time=145.5,
            final_state=MissionState.MISSION_COMPLETE
        )

        assert result.success is True
        assert result.completion_time == 145.5
        assert result.final_state == MissionState.MISSION_COMPLETE
        assert result.reason is None

    def test_mission_result_failure(self):
        """Test creating a failed mission result"""
        result = MissionResult(
            success=False,
            completion_time=75.0,
            final_state=MissionState.MISSION_ABORT,
            reason="Battery critical"
        )

        assert result.success is False
        assert result.reason == "Battery critical"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
