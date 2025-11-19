"""
Unit tests for the Role Manager
"""

import pytest
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "scenarios"))

from role_manager import DroneRole, RoleManager, DroneInfo


class TestDroneRole:
    """Test the DroneRole enum"""

    def test_drone_roles_exist(self):
        """Test that all required drone roles exist"""
        assert DroneRole.NEUTRAL_1.value == "neutral_1"
        assert DroneRole.NEUTRAL_2.value == "neutral_2"
        assert DroneRole.PATROL.value == "patrol"
        assert DroneRole.LEADER.value == "leader"
        assert DroneRole.FOLLOWER.value == "follower"
        assert DroneRole.ATTACKER.value == "attacker"


class TestDroneInfo:
    """Test the DroneInfo dataclass"""

    def test_drone_info_creation(self):
        """Test creating a DroneInfo instance"""
        info = DroneInfo(
            drone_id="cf1",
            battery_level=85.5,
            position=(1.0, 2.0, 3.0),
            status="flying"
        )
        assert info.drone_id == "cf1"
        assert info.battery_level == 85.5
        assert info.position == (1.0, 2.0, 3.0)
        assert info.status == "flying"
        assert info.has_detected_target is False


class TestRoleManager:
    """Test the RoleManager class"""

    def test_initialization(self):
        """Test role manager initialization"""
        rm = RoleManager()
        assert len(rm.drone_registry) == 0
        assert len(rm.current_assignments) == 0
        assert len(rm.role_history) == 0

    def test_register_drone(self):
        """Test registering a drone"""
        rm = RoleManager()
        result = rm.register_drone("cf1", battery_level=90.0)
        assert result is True
        assert "cf1" in rm.drone_registry
        assert rm.drone_registry["cf1"].battery_level == 90.0

    def test_register_duplicate_drone(self):
        """Test that registering a drone twice fails"""
        rm = RoleManager()
        rm.register_drone("cf1")
        result = rm.register_drone("cf1")
        assert result is False

    def test_update_drone_info(self):
        """Test updating drone information"""
        rm = RoleManager()
        rm.register_drone("cf1", battery_level=100.0)

        result = rm.update_drone_info("cf1", battery_level=75.0, status="flying")
        assert result is True
        assert rm.drone_registry["cf1"].battery_level == 75.0
        assert rm.drone_registry["cf1"].status == "flying"

    def test_update_nonexistent_drone(self):
        """Test updating a drone that doesn't exist"""
        rm = RoleManager()
        result = rm.update_drone_info("cf99", battery_level=50.0)
        assert result is False

    def test_assign_initial_roles(self):
        """Test initial role assignment"""
        rm = RoleManager()
        rm.register_drone("cf1")
        rm.register_drone("cf2")
        rm.register_drone("cf3")

        roles = rm.assign_initial_roles()

        assert roles["cf1"] == DroneRole.NEUTRAL_1
        assert roles["cf2"] == DroneRole.NEUTRAL_2
        assert roles["cf3"] == DroneRole.PATROL

    def test_reassign_roles_on_detection(self):
        """Test role reassignment after target detection"""
        rm = RoleManager()
        rm.register_drone("cf1", battery_level=80.0)
        rm.register_drone("cf2", battery_level=90.0)
        rm.register_drone("cf3", battery_level=85.0)

        # Assign initial roles
        rm.assign_initial_roles()

        # Reassign after detection
        new_roles = rm.reassign_roles_on_detection("cf3")

        # Patrol becomes attacker
        assert new_roles["cf3"] == DroneRole.ATTACKER

        # Highest battery neutral becomes leader (cf2 has 90%)
        assert new_roles["cf2"] == DroneRole.LEADER

        # Other neutral becomes follower
        assert new_roles["cf1"] == DroneRole.FOLLOWER

    def test_select_leader_highest_battery(self):
        """Test leader selection based on battery level"""
        rm = RoleManager()
        rm.register_drone("cf1", battery_level=60.0)
        rm.register_drone("cf2", battery_level=95.0)
        rm.register_drone("cf3", battery_level=75.0)

        leader = rm.select_leader(["cf1", "cf2", "cf3"])
        assert leader == "cf2"  # Highest battery

    def test_select_leader_empty_list(self):
        """Test leader selection with empty candidate list"""
        rm = RoleManager()
        leader = rm.select_leader([])
        assert leader == ""

    def test_select_leader_low_battery_warning(self):
        """Test that low battery leader generates warning"""
        rm = RoleManager()
        rm.register_drone("cf1", battery_level=35.0)  # Below 40% threshold
        rm.register_drone("cf2", battery_level=30.0)

        # Should still select cf1 (highest), but log warning
        leader = rm.select_leader(["cf1", "cf2"])
        assert leader == "cf1"

    def test_get_role_position_initial(self):
        """Test getting role position in initial phase"""
        rm = RoleManager()
        pos = rm.get_role_position(DroneRole.NEUTRAL_1, "initial")
        assert isinstance(pos, tuple)
        assert len(pos) == 3

    def test_get_role_position_attack_formation(self):
        """Test getting role position in attack formation"""
        rm = RoleManager()
        target_pos = (2.0, 3.0, 0.5)

        pos = rm.get_role_position(
            DroneRole.LEADER,
            "attack_formation",
            target_pos
        )

        # Position should be relative to target
        assert isinstance(pos, tuple)
        assert len(pos) == 3

    def test_validate_role_assignment_success(self):
        """Test successful role assignment validation"""
        rm = RoleManager()
        rm.register_drone("cf1", battery_level=90.0)
        rm.register_drone("cf2", battery_level=85.0)
        rm.register_drone("cf3", battery_level=80.0)
        rm.assign_initial_roles()

        result = rm.validate_role_assignment()
        assert result is True

    def test_validate_role_assignment_missing_role(self):
        """Test validation fails when drone has no role"""
        rm = RoleManager()
        rm.register_drone("cf1")
        # Don't assign role

        result = rm.validate_role_assignment()
        assert result is False

    def test_get_drones_by_role(self):
        """Test getting drones by role"""
        rm = RoleManager()
        rm.register_drone("cf1")
        rm.register_drone("cf2")
        rm.register_drone("cf3")
        rm.assign_initial_roles()

        neutrals = rm.get_drones_by_role([DroneRole.NEUTRAL_1, DroneRole.NEUTRAL_2])
        assert len(neutrals) == 2
        assert "cf1" in neutrals
        assert "cf2" in neutrals

    def test_get_role_for_drone(self):
        """Test getting role for a specific drone"""
        rm = RoleManager()
        rm.register_drone("cf1")
        rm.assign_initial_roles()

        role = rm.get_role_for_drone("cf1")
        assert role == DroneRole.NEUTRAL_1

    def test_get_role_for_unassigned_drone(self):
        """Test getting role for drone without assignment"""
        rm = RoleManager()
        rm.register_drone("cf1")

        role = rm.get_role_for_drone("cf1")
        assert role is None

    def test_assignment_history(self):
        """Test that role assignment history is recorded"""
        rm = RoleManager()
        rm.register_drone("cf1")
        rm.register_drone("cf2")
        rm.register_drone("cf3")

        rm.assign_initial_roles()

        history = rm.get_assignment_history()
        assert len(history) == 3  # Three initial assignments

    def test_get_swarm_status(self):
        """Test getting swarm status"""
        rm = RoleManager()
        rm.register_drone("cf1", battery_level=85.0, position=(1.0, 2.0, 3.0))
        rm.register_drone("cf2", battery_level=90.0, position=(4.0, 5.0, 6.0))
        rm.assign_initial_roles()

        status = rm.get_swarm_status()

        assert status["drone_count"] == 2
        assert "cf1" in status["battery_levels"]
        assert status["battery_levels"]["cf1"] == 85.0
        assert "cf1" in status["positions"]
        assert len(status["current_assignments"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
