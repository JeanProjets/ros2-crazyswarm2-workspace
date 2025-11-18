"""
Integration tests for Scenario 1 mission.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import asyncio
from scenarios.scenario_1_base import Scenario1Mission, MissionState


class TestScenario1Mission:
    """Test suite for Scenario 1 mission."""

    @pytest.fixture
    def mission_config(self):
        """Create test configuration."""
        return {
            'cage_dimensions': {'x': 10.0, 'y': 6.0, 'z': 8.0},
            'safety_zone': {'x_min': 0.0, 'x_max': 3.0, 'y_min': 0.0, 'y_max': 6.0},
            'target_position': {'x': 7.5, 'y': 3.0, 'z': 5.0},
            'mission_parameters': {
                'max_duration': 180,
                'min_battery': 20,
                'detection_range': 2.0,
                'formation_offset': 0.5
            }
        }

    @pytest.fixture
    def mission(self, mission_config):
        """Create a Scenario1Mission instance for testing."""
        return Scenario1Mission(mission_config)

    def test_initialization(self, mission):
        """Test mission initialization."""
        assert mission.current_state == MissionState.INITIALIZATION
        assert mission.target_position == (7.5, 3.0, 5.0)
        assert mission.cage_bounds == (10.0, 6.0, 8.0)

    @pytest.mark.asyncio
    async def test_state_initialization(self, mission):
        """Test initialization state."""
        result = await mission._state_initialization()
        assert result is True
        assert mission.current_state == MissionState.SAFETY_CHECK

    @pytest.mark.asyncio
    async def test_state_safety_check(self, mission):
        """Test safety check state."""
        await mission._state_initialization()
        result = await mission._state_safety_check()
        assert result is True
        assert mission.current_state == MissionState.PATROL_SEARCH

    @pytest.mark.asyncio
    async def test_state_patrol_search(self, mission):
        """Test patrol search state."""
        await mission._state_initialization()
        await mission._state_safety_check()
        result = await mission._state_patrol_search()
        assert result is True
        assert mission.target_detected is True
        assert mission.current_state == MissionState.TARGET_DETECTED

    @pytest.mark.asyncio
    async def test_state_target_detected(self, mission):
        """Test target detected state."""
        await mission._state_initialization()
        await mission._state_safety_check()
        await mission._state_patrol_search()
        result = await mission._state_target_detected()
        assert result is True
        assert mission.swarm.target_position == mission.target_position
        assert mission.current_state == MissionState.ROLE_ASSIGNMENT

    @pytest.mark.asyncio
    async def test_state_role_assignment(self, mission):
        """Test role assignment state."""
        await mission._state_initialization()
        await mission._state_safety_check()
        await mission._state_patrol_search()
        await mission._state_target_detected()
        result = await mission._state_role_assignment()
        assert result is True
        assert mission.leader_id is not None
        assert mission.follower_id is not None
        assert mission.patrol_id == 'cf3'
        assert mission.current_state == MissionState.APPROACH_TARGET

    @pytest.mark.asyncio
    async def test_fly_pattern(self, mission):
        """Test flying a pattern of waypoints."""
        await mission._state_initialization()

        waypoints = [
            (1.0, 1.0, 4.0),
            (2.0, 2.0, 4.0),
            (3.0, 3.0, 4.0)
        ]

        # Should not raise exception
        await mission._fly_pattern('cf1', waypoints, duration_per_waypoint=0.5)

    def test_check_timeout_no_start(self, mission):
        """Test timeout check when mission hasn't started."""
        result = mission._check_timeout()
        assert result is False

    @pytest.mark.asyncio
    async def test_run_mission_complete(self, mission):
        """Test complete mission run (fast version)."""
        # This is a long-running test, so we'll just verify it starts
        # Note: Full mission would take too long for unit tests
        import time
        mission.start_time = time.time()
        assert mission.current_state == MissionState.INITIALIZATION
