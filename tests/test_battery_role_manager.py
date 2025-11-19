"""
Tests for the Battery Role Manager
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scenarios.battery_role_manager import (
    BatteryRoleManager,
    Drone,
    DroneRole,
    MissionPhase,
    BatteryThresholds
)


class TestBatteryThresholds:
    """Test suite for BatteryThresholds"""

    def test_threshold_values(self):
        """Test that thresholds are set correctly"""
        thresholds = BatteryThresholds()

        assert thresholds.MIN_LEADER_VOLTAGE_INIT == 3.8
        assert thresholds.CRITICAL_VOLTAGE_FLIGHT == 3.5
        assert thresholds.WARNING_VOLTAGE == 3.6

    def test_phase_energy_costs(self):
        """Test that all phases have energy costs"""
        thresholds = BatteryThresholds()

        assert MissionPhase.TRANSIT in thresholds.PHASE_ENERGY_COST
        assert MissionPhase.JAMMING in thresholds.PHASE_ENERGY_COST
        assert thresholds.PHASE_ENERGY_COST[MissionPhase.TRANSIT] > 0


class TestBatteryRoleManager:
    """Test suite for BatteryRoleManager"""

    def test_initialization(self):
        """Test manager initialization"""
        manager = BatteryRoleManager()

        assert len(manager.assigned_drones) == 0
        assert manager.mission_aborted is False
        assert manager.abort_reason is None

    def test_select_highest_voltage_drone(self):
        """Test selection of highest voltage drone"""
        manager = BatteryRoleManager()

        drones = [
            Drone('drone1', 3.9),
            Drone('drone2', 4.1),  # Highest
            Drone('drone3', 3.7)
        ]

        selected = manager.select_highest_voltage_drone(drones)

        assert selected is not None
        assert selected.drone_id == 'drone2'
        assert selected.voltage == 4.1

    def test_select_highest_voltage_empty_list(self):
        """Test selection with empty candidate list"""
        manager = BatteryRoleManager()

        selected = manager.select_highest_voltage_drone([])

        assert selected is None

    def test_assign_roles_success(self):
        """Test successful role assignment"""
        manager = BatteryRoleManager()

        drones = [
            Drone('drone1', 3.9),
            Drone('drone2', 4.0),  # Will be Leader
            Drone('drone3', 3.8)
        ]

        assignments = manager.assign_roles(drones)

        assert DroneRole.LEADER in assignments
        assert assignments[DroneRole.LEADER].drone_id == 'drone2'
        assert assignments[DroneRole.LEADER].voltage == 4.0
        assert manager.mission_aborted is False

    def test_assign_roles_insufficient_voltage(self):
        """Test role assignment failure due to low battery"""
        manager = BatteryRoleManager()

        # All drones below minimum threshold
        drones = [
            Drone('drone1', 3.7),
            Drone('drone2', 3.6),
            Drone('drone3', 3.5)
        ]

        assignments = manager.assign_roles(drones)

        assert len(assignments) == 0
        assert manager.mission_aborted is True
        assert manager.abort_reason is not None

    def test_assign_roles_insufficient_drones(self):
        """Test role assignment with too few drones"""
        manager = BatteryRoleManager()

        drones = [Drone('drone1', 4.0)]

        assignments = manager.assign_roles(drones)

        assert len(assignments) == 0

    def test_assign_roles_multiple_followers(self):
        """Test assignment with multiple follower drones"""
        manager = BatteryRoleManager()

        drones = [
            Drone('drone1', 3.9),
            Drone('drone2', 4.0),
            Drone('drone3', 3.8),
            Drone('drone4', 3.85)
        ]

        assignments = manager.assign_roles(drones)

        # Should have leader and at least 2 followers
        assert DroneRole.LEADER in assignments
        assert DroneRole.FOLLOWER_LEFT in assignments
        assert DroneRole.FOLLOWER_RIGHT in assignments

    def test_validate_energy_budget_sufficient(self):
        """Test energy budget validation with sufficient voltage"""
        manager = BatteryRoleManager()

        # High voltage should pass
        result = manager.validate_energy_budget(MissionPhase.TRANSIT, 4.0)

        assert result is True

    def test_validate_energy_budget_insufficient(self):
        """Test energy budget validation with insufficient voltage"""
        manager = BatteryRoleManager()

        # Low voltage should fail
        result = manager.validate_energy_budget(MissionPhase.TRANSIT, 3.6)

        assert result is False

    def test_check_leader_voltage_ok(self):
        """Test leader voltage check with safe voltage"""
        manager = BatteryRoleManager()

        result = manager.check_leader_voltage(3.9, MissionPhase.TRANSIT)

        assert result['status'] == 'OK'
        assert result['action'] == 'CONTINUE'

    def test_check_leader_voltage_critical(self):
        """Test leader voltage check with critical voltage"""
        manager = BatteryRoleManager()

        result = manager.check_leader_voltage(3.4, MissionPhase.TRANSIT)

        assert result['status'] == 'CRITICAL'
        assert result['action'] == 'IMMEDIATE_RTH'
        assert manager.mission_aborted is True

    def test_check_leader_voltage_warning(self):
        """Test leader voltage check with warning level"""
        manager = BatteryRoleManager()

        result = manager.check_leader_voltage(3.55, MissionPhase.TRANSIT)

        assert result['status'] == 'WARNING'
        assert 'MONITOR' in result['action'] or 'RTH' in result['action']

    def test_get_leader(self):
        """Test getting the assigned leader"""
        manager = BatteryRoleManager()

        drones = [
            Drone('drone1', 3.9),
            Drone('drone2', 4.0),
        ]

        manager.assign_roles(drones)
        leader = manager.get_leader()

        assert leader is not None
        assert leader.drone_id == 'drone2'
        assert leader.role == DroneRole.LEADER

    def test_update_drone_voltage(self):
        """Test updating drone voltage"""
        manager = BatteryRoleManager()

        drones = [
            Drone('drone1', 3.9),
            Drone('drone2', 4.0),
        ]

        manager.assign_roles(drones)
        manager.update_drone_voltage('drone1', 3.7)

        assert manager.assigned_drones['drone1'].voltage == 3.7

    def test_get_role_summary(self):
        """Test role summary generation"""
        manager = BatteryRoleManager()

        drones = [
            Drone('drone1', 3.9),
            Drone('drone2', 4.0),
        ]

        manager.assign_roles(drones)
        summary = manager.get_role_summary()

        assert 'assignments' in summary
        assert 'voltages' in summary
        assert 'mission_aborted' in summary


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
