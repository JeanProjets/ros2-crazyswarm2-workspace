"""
Unit tests for SwarmManagerV2.

Tests battery-based leader selection and dynamic formation adjustment.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.swarm_manager_v2 import SwarmCoordinator, DroneRole
from src.core.safe_drone_controller import SafeDroneController


class TestSwarmCoordinator:
    """Test suite for SwarmCoordinator."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return {
            'safety_bounds': {
                'x_min': 0.3,
                'x_max': 9.7,
                'y_min': 0.3,
                'y_max': 5.7,
                'z_min': 0.2,
                'z_max': 5.8
            },
            'drone_configs': {
                'cf1': {'role': 'NEUTRAL_1', 'start_pos': [2.5, 2.5, 0]},
                'cf2': {'role': 'NEUTRAL_2', 'start_pos': [2.5, 3.5, 0]},
                'cf3': {'role': 'PATROL', 'start_pos': [3.0, 5.0, 0]}
            }
        }

    @pytest.fixture
    def coordinator(self, config):
        """Create a SwarmCoordinator instance for testing."""
        coord = SwarmCoordinator(config=config)
        coord.initialize_swarm(config['drone_configs'])
        return coord

    def test_initialization(self, coordinator):
        """Test coordinator initializes correctly."""
        assert len(coordinator.drones) == 3
        assert 'cf1' in coordinator.drones
        assert 'cf2' in coordinator.drones
        assert 'cf3' in coordinator.drones

    def test_drone_roles_assignment(self, coordinator):
        """Test initial drone roles are assigned correctly."""
        assert coordinator.roles['cf1'] == DroneRole.NEUTRAL_1
        assert coordinator.roles['cf2'] == DroneRole.NEUTRAL_2
        assert coordinator.roles['cf3'] == DroneRole.PATROL

    def test_add_drone(self, config):
        """Test adding a drone to the swarm."""
        coord = SwarmCoordinator(config=config)
        controller = SafeDroneController('test_drone', config=config)

        coord.add_drone('test_drone', controller, DroneRole.NEUTRAL_1)

        assert 'test_drone' in coord.drones
        assert coord.roles['test_drone'] == DroneRole.NEUTRAL_1

    def test_select_optimal_leader_by_voltage(self, coordinator):
        """Test leader selection based on highest battery voltage."""
        # Set different voltages
        coordinator.drones['cf1'].state.battery_voltage = 4.0
        coordinator.drones['cf2'].state.battery_voltage = 4.1  # Highest

        candidates = ['cf1', 'cf2']
        leader = coordinator.select_optimal_leader(candidates)

        # cf2 should be selected due to higher voltage
        assert leader == 'cf2'

    def test_select_optimal_leader_voltage_tiebreaker(self, coordinator):
        """Test leader selection tiebreaker when voltages are within 0.1V."""
        # Set voltages within 0.1V
        coordinator.drones['cf1'].state.battery_voltage = 4.05
        coordinator.drones['cf2'].state.battery_voltage = 4.03  # Within 0.1V

        # Set positions - cf1 closer to X center
        x_center = (0.3 + 9.7) / 2.0  # 5.0
        coordinator.drones['cf1'].state.x = 5.1  # Closer to center
        coordinator.drones['cf2'].state.x = 3.0  # Further from center

        candidates = ['cf1', 'cf2']
        leader = coordinator.select_optimal_leader(candidates)

        # cf1 should be selected due to proximity to X center
        assert leader == 'cf1'

    def test_select_optimal_leader_no_candidates(self, coordinator):
        """Test that selecting leader with no candidates raises error."""
        with pytest.raises(ValueError, match="No candidates provided"):
            coordinator.select_optimal_leader([])

    def test_calculate_safe_formation_normal(self, coordinator):
        """Test formation calculation in normal (non-boundary) situation."""
        leader_pos = (5.0, 3.0, 3.0)  # Center of arena

        follower_pos = coordinator.calculate_safe_formation(leader_pos, 'cf1')

        # Should apply standard offset: (-0.5, -0.5, -0.5)
        assert follower_pos[0] == 4.5  # 5.0 - 0.5
        assert follower_pos[1] == 2.5  # 3.0 - 0.5
        assert follower_pos[2] == 2.5  # 3.0 - 0.5

    def test_calculate_safe_formation_near_y_min_boundary(self, coordinator):
        """
        Test formation adjustment when leader is near Y=0 boundary.

        This is THE critical test for Scenario 2.
        If leader is at Y=0.5 (near boundary), follower Y-offset should invert.
        """
        leader_pos = (9.5, 0.5, 5.0)  # Scenario 2 corner target

        follower_pos = coordinator.calculate_safe_formation(leader_pos, 'cf1')

        # Y offset should be POSITIVE (inverted) to avoid wall
        # Expected: (9.0, 1.0, 4.5) instead of (9.0, 0.0, 4.5)
        assert follower_pos[0] == 9.0  # 9.5 - 0.5
        assert follower_pos[1] == 1.0  # 0.5 + 0.5 (INVERTED)
        assert follower_pos[2] == 4.5  # 5.0 - 0.5

        # Verify position is safe
        assert coordinator.drones['cf1'].is_position_safe(*follower_pos) == True

    def test_calculate_safe_formation_near_y_max_boundary(self, coordinator):
        """Test formation adjustment when leader is near Y_max boundary."""
        leader_pos = (5.0, 5.5, 3.0)  # Near Y_max = 5.7

        follower_pos = coordinator.calculate_safe_formation(leader_pos, 'cf1')

        # Y offset should remain negative
        assert follower_pos[1] == 5.0  # 5.5 - 0.5

    def test_calculate_safe_formation_clamping(self, coordinator):
        """Test that unsafe formation positions are clamped."""
        # Leader at extreme position
        leader_pos = (9.7, 0.3, 5.8)  # At boundaries

        follower_pos = coordinator.calculate_safe_formation(leader_pos, 'cf1')

        # Follower position should be clamped to safe bounds
        controller = coordinator.drones['cf1']
        assert controller.is_position_safe(*follower_pos) == True

    def test_assign_roles(self, coordinator):
        """Test manual role assignment."""
        coordinator.assign_roles('cf1', 'cf2')

        assert coordinator.roles['cf1'] == DroneRole.LEADER
        assert coordinator.roles['cf2'] == DroneRole.FOLLOWER

    def test_get_drone_by_role(self, coordinator):
        """Test retrieving drone by role."""
        # Find patrol drone
        patrol_id = coordinator.get_drone_by_role(DroneRole.PATROL)
        assert patrol_id == 'cf3'

        # Find neutral drones
        n1_id = coordinator.get_drone_by_role(DroneRole.NEUTRAL_1)
        assert n1_id == 'cf1'

        # Non-existent role
        leader_id = coordinator.get_drone_by_role(DroneRole.LEADER)
        assert leader_id is None  # No leader assigned yet

    def test_get_swarm_status(self, coordinator):
        """Test swarm status reporting."""
        status = coordinator.get_swarm_status()

        assert status['total_drones'] == 3
        assert 'roles' in status
        assert 'drones' in status
        assert len(status['drones']) == 3

        # Check individual drone stats
        assert 'cf1' in status['drones']
        assert 'battery_voltage' in status['drones']['cf1']

    def test_log_battery_status(self, coordinator):
        """Test battery status logging (should not raise errors)."""
        # This should execute without errors
        coordinator.log_battery_status()

    def test_scenario_2_complete_workflow(self, coordinator):
        """
        Integration test: Complete Scenario 2 workflow.

        1. Select leader from N1/N2 based on battery
        2. Assign roles
        3. Calculate safe formation near corner target
        4. Verify no boundary violations
        """
        # Step 1: Select leader
        coordinator.drones['cf1'].state.battery_voltage = 4.1
        coordinator.drones['cf2'].state.battery_voltage = 4.0

        leader_id = coordinator.select_optimal_leader(['cf1', 'cf2'])
        assert leader_id == 'cf1'  # Higher voltage

        follower_id = 'cf2' if leader_id == 'cf1' else 'cf1'

        # Step 2: Assign roles
        coordinator.assign_roles(leader_id, follower_id)

        assert coordinator.roles[leader_id] == DroneRole.LEADER
        assert coordinator.roles[follower_id] == DroneRole.FOLLOWER

        # Step 3: Move leader to corner target
        corner_target = (9.5, 0.5, 5.0)
        leader = coordinator.drones[leader_id]
        leader.clamped_navigate(*corner_target)

        # Step 4: Calculate safe formation for follower
        follower_pos = coordinator.calculate_safe_formation(
            leader.get_position(),
            follower_id
        )

        # Step 5: Verify follower position is safe
        follower = coordinator.drones[follower_id]
        assert follower.is_position_safe(*follower_pos) == True

        # Step 6: Verify Y-offset was inverted (should be at Y=1.0, not Y=0.0)
        assert follower_pos[1] > 0.5  # Must be away from Y=0 wall

    def test_boundary_threshold_parameter(self, coordinator):
        """Test Y_BOUNDARY_THRESHOLD parameter is working correctly."""
        threshold = coordinator.Y_BOUNDARY_THRESHOLD

        # Position just below threshold should trigger inversion
        leader_pos = (5.0, threshold - 0.1, 3.0)
        follower_pos = coordinator.calculate_safe_formation(leader_pos, 'cf1')
        # Y offset should be positive (inverted)
        assert follower_pos[1] > leader_pos[1]

        # Position above threshold should use normal offset
        leader_pos = (5.0, threshold + 0.5, 3.0)
        follower_pos = coordinator.calculate_safe_formation(leader_pos, 'cf1')
        # Y offset should be negative (normal)
        assert follower_pos[1] < leader_pos[1]
