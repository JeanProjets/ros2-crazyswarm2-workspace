"""
Unit tests for DroneController class.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from core.pycrazyswarm_mock import Crazyswarm
from core.drone_controller import DroneController, DroneState


class TestDroneController:
    """Test suite for DroneController."""

    @pytest.fixture
    def controller(self):
        """Create a DroneController instance for testing."""
        crazyswarm = Crazyswarm()
        return DroneController(
            drone_id='cf1',
            crazyswarm=crazyswarm,
            cage_bounds=(10.0, 6.0, 8.0),
            min_battery=20.0
        )

    def test_initialization(self, controller):
        """Test controller initialization."""
        assert controller.drone_id == 'cf1'
        assert controller.state == DroneState.IDLE
        assert controller.cage_bounds == (10.0, 6.0, 8.0)
        assert controller.min_battery == 20.0

    def test_takeoff_success(self, controller):
        """Test successful takeoff."""
        result = controller.takeoff(height=4.0, duration=2.0)
        assert result is True
        assert controller.state == DroneState.FLYING

    def test_takeoff_height_exceeds_bounds(self, controller):
        """Test takeoff fails when height exceeds cage bounds."""
        result = controller.takeoff(height=10.0, duration=2.0)
        assert result is False

    def test_land_success(self, controller):
        """Test successful landing."""
        controller.takeoff(height=4.0, duration=2.0)
        result = controller.land(duration=2.0)
        assert result is True
        assert controller.state == DroneState.IDLE

    def test_go_to_success(self, controller):
        """Test successful movement to target position."""
        controller.takeoff(height=4.0, duration=2.0)
        result = controller.go_to(5.0, 3.0, 4.0, yaw=0.0, duration=3.0)
        assert result is True

    def test_go_to_out_of_bounds(self, controller):
        """Test go_to fails when target is out of bounds."""
        controller.takeoff(height=4.0, duration=2.0)
        result = controller.go_to(15.0, 3.0, 4.0, yaw=0.0, duration=3.0)
        assert result is False

    def test_get_position(self, controller):
        """Test getting drone position."""
        position = controller.get_position()
        assert isinstance(position, tuple)
        assert len(position) == 3

    def test_get_battery_percentage(self, controller):
        """Test getting battery level."""
        battery = controller.get_battery_percentage()
        assert isinstance(battery, float)
        assert 0 <= battery <= 100

    def test_is_ready(self, controller):
        """Test ready state check."""
        # Initially ready
        assert controller.is_ready() is True

        # After emergency, not ready
        controller.state = DroneState.EMERGENCY
        assert controller.is_ready() is False

    def test_emergency_stop(self, controller):
        """Test emergency stop."""
        controller.takeoff(height=4.0, duration=2.0)
        controller.emergency_stop()
        assert controller.state == DroneState.EMERGENCY
