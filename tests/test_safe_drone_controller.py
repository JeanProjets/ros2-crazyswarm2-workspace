"""
Unit tests for SafeDroneController.

Tests boundary enforcement, battery management, and safety features.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.safe_drone_controller import SafeDroneController


class TestSafeDroneController:
    """Test suite for SafeDroneController."""

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
            }
        }

    @pytest.fixture
    def controller(self, config):
        """Create a SafeDroneController instance for testing."""
        return SafeDroneController('test_drone', crazyswarm=None, config=config)

    def test_initialization(self, controller):
        """Test controller initializes correctly."""
        assert controller.drone_id == 'test_drone'
        assert controller.bounds['x_min'] == 0.3
        assert controller.bounds['x_max'] == 9.7
        assert controller.clamp_count == 0

    def test_clamp_position_within_bounds(self, controller):
        """Test that positions within bounds are not clamped."""
        x, y, z = 5.0, 3.0, 3.0
        safe_x, safe_y, safe_z = controller.clamp_position(x, y, z)

        assert safe_x == x
        assert safe_y == y
        assert safe_z == z
        assert controller.clamp_count == 0

    def test_clamp_position_x_min_violation(self, controller):
        """Test clamping when X is below minimum."""
        x, y, z = 0.1, 3.0, 3.0  # X below 0.3
        safe_x, safe_y, safe_z = controller.clamp_position(x, y, z)

        assert safe_x == 0.3  # Clamped to x_min
        assert safe_y == y
        assert safe_z == z
        assert controller.clamp_count == 1

    def test_clamp_position_x_max_violation(self, controller):
        """Test clamping when X is above maximum."""
        x, y, z = 9.9, 3.0, 3.0  # X above 9.7
        safe_x, safe_y, safe_z = controller.clamp_position(x, y, z)

        assert safe_x == 9.7  # Clamped to x_max
        assert safe_y == y
        assert safe_z == z
        assert controller.clamp_count == 1

    def test_clamp_position_y_boundary(self, controller):
        """Test clamping at Y boundaries (critical for Scenario 2)."""
        # Test Y min
        x, y, z = 5.0, 0.0, 3.0  # Y below 0.3
        safe_x, safe_y, safe_z = controller.clamp_position(x, y, z)
        assert safe_y == 0.3

        # Test Y max
        x, y, z = 5.0, 6.0, 3.0  # Y above 5.7
        safe_x, safe_y, safe_z = controller.clamp_position(x, y, z)
        assert safe_y == 5.7

    def test_clamp_position_z_boundary(self, controller):
        """Test clamping at Z boundaries."""
        # Test Z min
        x, y, z = 5.0, 3.0, 0.1  # Z below 0.2
        safe_x, safe_y, safe_z = controller.clamp_position(x, y, z)
        assert safe_z == 0.2

        # Test Z max
        x, y, z = 5.0, 3.0, 6.0  # Z above 5.8
        safe_x, safe_y, safe_z = controller.clamp_position(x, y, z)
        assert safe_z == 5.8

    def test_clamp_position_multiple_violations(self, controller):
        """Test clamping when multiple coordinates violate bounds."""
        x, y, z = 10.0, -1.0, 10.0  # All outside bounds
        safe_x, safe_y, safe_z = controller.clamp_position(x, y, z)

        assert safe_x == 9.7
        assert safe_y == 0.3
        assert safe_z == 5.8
        assert controller.clamp_count == 1  # One clamp event

    def test_clamped_navigate(self, controller):
        """Test clamped_navigate method."""
        controller.clamped_navigate(5.0, 3.0, 2.0, yaw=0.0)

        # Verify drone moved to position
        assert controller.state.x == 5.0
        assert controller.state.y == 3.0
        assert controller.state.z == 2.0

    def test_clamped_navigate_with_boundary_violation(self, controller):
        """Test clamped_navigate automatically clamps unsafe positions."""
        controller.clamped_navigate(10.0, 0.0, 3.0, yaw=0.0)

        # Verify position was clamped
        assert controller.state.x == 9.7
        assert controller.state.y == 0.3
        assert controller.state.z == 3.0
        assert controller.clamp_count == 1

    def test_precision_hover(self, controller):
        """Test precision_hover method."""
        controller.state.x = 5.0
        controller.state.y = 3.0

        controller.precision_hover(height=2.5, duration=1.0)

        assert controller.state.z == 2.5

    def test_is_position_safe(self, controller):
        """Test is_position_safe method."""
        # Safe position
        assert controller.is_position_safe(5.0, 3.0, 3.0) == True

        # Unsafe positions
        assert controller.is_position_safe(0.1, 3.0, 3.0) == False  # X too small
        assert controller.is_position_safe(10.0, 3.0, 3.0) == False  # X too large
        assert controller.is_position_safe(5.0, 0.1, 3.0) == False  # Y too small
        assert controller.is_position_safe(5.0, 6.0, 3.0) == False  # Y too large
        assert controller.is_position_safe(5.0, 3.0, 6.0) == False  # Z too large

    def test_get_distance_to_boundary(self, controller):
        """Test distance to boundary calculation."""
        # Center of arena
        dist = controller.get_distance_to_boundary(5.0, 3.0)
        assert dist > 0

        # Near X min boundary
        dist = controller.get_distance_to_boundary(0.5, 3.0)
        assert abs(dist - 0.2) < 0.01  # 0.5 - 0.3 = 0.2

        # Near Y=0 boundary (critical for Scenario 2)
        dist = controller.get_distance_to_boundary(5.0, 0.5)
        assert abs(dist - 0.2) < 0.01  # 0.5 - 0.3 = 0.2

    def test_battery_voltage_tracking(self, controller):
        """Test battery voltage monitoring."""
        # Initial voltage should be full
        voltage = controller.get_battery_voltage()
        assert voltage > 3.5

        # Check battery status
        assert controller.check_battery_status() == 'OK'

    def test_battery_warning_threshold(self, controller):
        """Test battery warning threshold detection."""
        # Manually set low voltage
        controller.state.battery_voltage = 3.45

        status = controller.check_battery_status()
        assert status == 'WARNING'

    def test_battery_critical_threshold(self, controller):
        """Test battery critical threshold detection."""
        # Manually set critical voltage
        controller.state.battery_voltage = 3.3

        status = controller.check_battery_status()
        assert status == 'CRITICAL'

    def test_get_safety_stats(self, controller):
        """Test safety statistics reporting."""
        controller.clamp_position(10.0, 0.0, 3.0)  # Trigger a clamp

        stats = controller.get_safety_stats()

        assert stats['drone_id'] == 'test_drone'
        assert stats['clamp_count'] == 1
        assert 'current_position' in stats
        assert 'battery_voltage' in stats
        assert 'battery_status' in stats
        assert 'is_flying' in stats
        assert 'distance_to_boundary' in stats

    def test_scenario_2_corner_target_safety(self, controller):
        """
        Test safety for Scenario 2's corner target at (9.5, 0.5, 5.0).

        This is a critical test to ensure the controller can safely handle
        the corner position without violating boundaries.
        """
        # Target position
        target_x, target_y, target_z = 9.5, 0.5, 5.0

        # Verify target is safe
        assert controller.is_position_safe(target_x, target_y, target_z) == True

        # Navigate to target
        controller.clamped_navigate(target_x, target_y, target_z)

        # Verify position reached (should not be clamped)
        assert controller.state.x == target_x
        assert controller.state.y == target_y
        assert controller.state.z == target_z
        assert controller.clamp_count == 0  # Should not clamp valid target

    def test_scenario_2_formation_near_boundary(self, controller):
        """
        Test formation follower position when leader is near Y=0 boundary.

        If leader is at (9.5, 0.5), follower with offset (-0.5, -0.5, -0.5)
        would be at (9.0, 0.0, 4.5) which violates Y_min=0.3.
        """
        leader_pos = (9.5, 0.5, 5.0)
        follower_offset = (-0.5, -0.5, -0.5)

        follower_x = leader_pos[0] + follower_offset[0]
        follower_y = leader_pos[1] + follower_offset[1]
        follower_z = leader_pos[2] + follower_offset[2]

        # This would be (9.0, 0.0, 4.5) - Y=0.0 violates boundary
        assert controller.is_position_safe(follower_x, follower_y, follower_z) == False

        # Clamp to safe position
        safe_x, safe_y, safe_z = controller.clamp_position(follower_x, follower_y, follower_z)

        # Y should be clamped to 0.3
        assert safe_y == 0.3
        assert controller.is_position_safe(safe_x, safe_y, safe_z) == True
