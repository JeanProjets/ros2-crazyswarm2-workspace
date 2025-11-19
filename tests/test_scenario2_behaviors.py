"""
Test suite for Scenario 2 Behavior modules

Tests the corner-aware navigation and energy-efficient flight behaviors
required for Scenario 2 where the target is at (9.5, 0.5, 5).
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from behaviors.adaptive_formation import (
    AdaptiveFormationController,
    Position,
    Offset,
    calculate_adaptive_offset
)
from behaviors.corner_search import (
    CornerBiasPatrol,
    CageDimensions,
    Waypoint,
    calculate_corner_velocity
)
from behaviors.corner_approach import (
    CornerApproachBehavior,
    Position as ApproachPosition,
    ApproachPhase
)
from behaviors.vertical_strike import (
    VerticalStrikeManeuver,
    Position as StrikePosition,
    AttackPhase
)


class TestAdaptiveFormation:
    """Test adaptive formation controller"""

    def test_initialization(self):
        """Test controller initialization with default bounds"""
        controller = AdaptiveFormationController()
        assert controller.bounds_x_min == 0.0
        assert controller.bounds_x_max == 10.0
        assert controller.bounds_y_min == 0.0
        assert controller.bounds_y_max == 6.0

    def test_scenario2_corner_offset(self):
        """
        Test the critical scenario 2 case: target at (9.5, 0.5, 5)
        Leader at (9.5, 0.5, 5) should get follower at (9.0, 1.0, 4.5)
        NOT at (9.0, 0.0, 4.5) which would hit Y=0 wall
        """
        controller = AdaptiveFormationController()
        leader_pos = Position(x=9.5, y=0.5, z=5.0)

        safe_offset = controller.get_safe_offset(leader_pos)

        # Y offset should be +0.5 (left) not -0.5 (right/wall)
        assert safe_offset.dy == 0.5, f"Expected dy=0.5, got {safe_offset.dy}"

    def test_standard_offset_when_safe(self):
        """Test that standard offset is used in safe zones"""
        controller = AdaptiveFormationController()
        leader_pos = Position(x=5.0, y=3.0, z=5.0)  # Center, safe

        safe_offset = controller.get_safe_offset(leader_pos)

        # Should use standard offset
        assert safe_offset.dy == -0.5

    def test_left_wall_offset(self):
        """Test offset adaptation near left wall (Y=6)"""
        controller = AdaptiveFormationController()
        leader_pos = Position(x=5.0, y=5.5, z=5.0)

        safe_offset = controller.get_safe_offset(leader_pos)

        # Should keep standard offset (push follower right)
        assert safe_offset.dy == -0.5

    def test_follower_position_calculation(self):
        """Test complete follower position calculation"""
        controller = AdaptiveFormationController()
        leader_pos = Position(x=9.5, y=0.5, z=5.0)

        follower_pos = controller.calculate_follower_position(leader_pos)

        # Verify follower is not at Y=0 (wall)
        assert follower_pos.y > 0.3, f"Follower too close to Y=0 wall: {follower_pos.y}"
        assert follower_pos.y == 1.0, f"Expected follower Y=1.0, got {follower_pos.y}"

    def test_position_validation(self):
        """Test position safety validation"""
        controller = AdaptiveFormationController()

        # Safe position
        safe_pos = Position(x=5.0, y=3.0, z=5.0)
        is_safe, msg = controller.validate_follower_position(safe_pos)
        assert is_safe

        # Unsafe position (too close to Y=0 wall)
        unsafe_pos = Position(x=5.0, y=0.1, z=5.0)
        is_safe, msg = controller.validate_follower_position(unsafe_pos)
        assert not is_safe
        assert "Y min" in msg


class TestCornerSearch:
    """Test corner search patterns"""

    def test_initialization(self):
        """Test patrol initialization"""
        patrol = CornerBiasPatrol()
        assert patrol.cruise_speed == 1.0
        assert patrol.search_altitude == 5.0

    def test_prioritize_corners_returns_waypoints(self):
        """Test that corner prioritization returns waypoints"""
        patrol = CornerBiasPatrol()
        waypoints = patrol.prioritize_corners()

        assert len(waypoints) > 0
        assert all(isinstance(wp, Waypoint) for wp in waypoints)

    def test_right_edge_prioritized(self):
        """Test that right edge (Y~0) is searched first"""
        patrol = CornerBiasPatrol()
        waypoints = patrol.prioritize_corners()

        # First waypoints should have low Y values (right edge)
        first_waypoints = waypoints[:5]
        avg_y = sum(wp.y for wp in first_waypoints) / len(first_waypoints)
        assert avg_y < 2.0, "Right edge should be prioritized"

    def test_corner_velocity_calculation(self):
        """Test velocity reduction near walls"""
        patrol = CornerBiasPatrol()

        # Far from wall - fast
        vel_far = patrol._calculate_corner_velocity_for_x(5.0)
        assert vel_far == 1.0

        # Medium distance - medium speed
        vel_medium = patrol._calculate_corner_velocity_for_x(8.0)
        assert vel_medium == 0.5

        # Close to wall - slow
        vel_close = patrol._calculate_corner_velocity_for_x(9.0)
        assert vel_close == 0.2

    def test_perimeter_scan_coverage(self):
        """Test perimeter scan covers all edges"""
        patrol = CornerBiasPatrol()
        waypoints = patrol.scan_perimeter_first()

        # Extract X and Y coordinates
        x_coords = [wp.x for wp in waypoints]
        y_coords = [wp.y for wp in waypoints]

        # Should cover range of X values
        assert max(x_coords) > 8.0
        assert min(x_coords) < 2.0

        # Should cover range of Y values
        assert max(y_coords) > 4.0
        assert min(y_coords) < 2.0


class TestCornerApproach:
    """Test corner approach behavior"""

    def test_initialization(self):
        """Test approach behavior initialization"""
        approach = CornerApproachBehavior()
        assert approach.max_approach_speed == 0.5
        assert approach.standoff_distance == 1.0

    def test_corner_detection(self):
        """Test corner position detection"""
        approach = CornerApproachBehavior()

        # Corner position (near two walls)
        corner = ApproachPosition(x=9.5, y=0.5, z=5.0)
        assert approach._is_corner_position(corner)

        # Center position (not corner)
        center = ApproachPosition(x=5.0, y=3.0, z=5.0)
        assert not approach._is_corner_position(center)

    def test_scenario2_standoff_calculation(self):
        """Test standoff position for Scenario 2 target"""
        approach = CornerApproachBehavior()
        target = ApproachPosition(x=9.5, y=0.5, z=5.0)

        # Leader standoff
        leader_standoff = approach._calculate_standoff_position(target, "leader")
        assert leader_standoff.x == 8.5  # 1m back
        assert leader_standoff.y == 0.5  # Same Y
        assert leader_standoff.z == 5.0  # Same Z

        # Follower standoff
        follower_standoff = approach._calculate_standoff_position(target, "follower")
        assert follower_standoff.x == 8.5
        assert follower_standoff.y == 1.0  # Offset left (+0.5)
        assert follower_standoff.z == 4.5  # Offset down (-0.5)

    def test_approach_phases_generation(self):
        """Test approach phases are generated correctly"""
        approach = CornerApproachBehavior()
        target = ApproachPosition(x=9.5, y=0.5, z=5.0)

        phases = approach.approach_corner_target(target, "leader")

        assert len(phases) > 0
        assert all(isinstance(p, ApproachPhase) for p in phases)

    def test_safe_approach_vector(self):
        """Test safe approach vector calculation"""
        approach = CornerApproachBehavior()

        current = ApproachPosition(x=5.0, y=2.0, z=5.0)
        target = ApproachPosition(x=9.5, y=0.5, z=5.0)

        dx, dy, dz = approach.calculate_safe_approach_vector(current, target)

        # Should have valid direction
        magnitude = (dx**2 + dy**2 + dz**2)**0.5
        assert magnitude > 0


class TestVerticalStrike:
    """Test vertical strike maneuver"""

    def test_initialization(self):
        """Test strike maneuver initialization"""
        strike = VerticalStrikeManeuver()
        assert strike.descent_speed == 0.3
        assert strike.stop_buffer == 0.3
        assert strike.hold_duration == 2.0

    def test_alignment_calculation(self):
        """Test alignment position above target"""
        strike = VerticalStrikeManeuver()
        target = StrikePosition(x=9.5, y=0.5, z=5.0)

        phases = strike.align_above_target(target)

        # Should have alignment phases
        assert len(phases) > 0

        # Final position should be above target
        final_phase = phases[-1]
        assert final_phase.position.x == target.x
        assert final_phase.position.y == target.y
        assert final_phase.position.z > target.z

    def test_descent_phases(self):
        """Test descent phase generation"""
        strike = VerticalStrikeManeuver()
        target = StrikePosition(x=9.5, y=0.5, z=5.0)

        phases = strike.descend_controlled(target)

        # Should have descent and hold phases
        assert len(phases) >= 2

        # Strike position should be above target by buffer
        strike_phase = phases[0]
        assert strike_phase.position.z == target.z + strike.stop_buffer

    def test_full_strike_sequence(self):
        """Test complete strike sequence"""
        strike = VerticalStrikeManeuver()
        target = StrikePosition(x=9.5, y=0.5, z=5.0)

        phases = strike.execute_full_strike(target)

        # Should have multiple phases
        assert len(phases) >= 5  # Align, descend, hold, ascend

    def test_alignment_precision_check(self):
        """Test alignment precision validation"""
        strike = VerticalStrikeManeuver()
        target = StrikePosition(x=9.5, y=0.5, z=5.0)

        # Perfectly aligned
        current_aligned = StrikePosition(x=9.5, y=0.5, z=6.0)
        is_aligned, error = strike.check_alignment_precision(current_aligned, target)
        assert is_aligned
        assert error < 0.01

        # Not aligned
        current_offset = StrikePosition(x=9.3, y=0.5, z=6.0)
        is_aligned, error = strike.check_alignment_precision(current_offset, target)
        assert not is_aligned

    def test_drift_detection(self):
        """Test drift detection during descent"""
        strike = VerticalStrikeManeuver()

        start = StrikePosition(x=9.5, y=0.5, z=6.0)

        # No drift
        current_no_drift = StrikePosition(x=9.5, y=0.5, z=5.5)
        is_safe, drift = strike.check_drift_during_descent(start, current_no_drift)
        assert is_safe

        # Excessive drift
        current_drift = StrikePosition(x=9.7, y=0.5, z=5.5)
        is_safe, drift = strike.check_drift_during_descent(start, current_drift)
        assert not is_safe

    def test_strike_geometry_validation(self):
        """Test strike geometry validation"""
        strike = VerticalStrikeManeuver()

        # Valid geometry
        valid_target = StrikePosition(x=9.5, y=0.5, z=5.0)
        is_valid, msg = strike.validate_strike_geometry(valid_target)
        assert is_valid

        # Invalid - too close to ceiling
        invalid_target = StrikePosition(x=9.5, y=0.5, z=9.5)
        is_valid, msg = strike.validate_strike_geometry(invalid_target)
        assert not is_valid


class TestStandaloneUtilities:
    """Test standalone utility functions"""

    def test_calculate_adaptive_offset_function(self):
        """Test standalone adaptive offset calculation"""
        # Near right wall
        leader_near_right = Position(x=5.0, y=0.5, z=5.0)
        offset = calculate_adaptive_offset(leader_near_right)
        assert offset == 0.5  # Force left

        # Safe zone
        leader_safe = Position(x=5.0, y=3.0, z=5.0)
        offset = calculate_adaptive_offset(leader_safe)
        assert offset == -0.5  # Standard

    def test_calculate_corner_velocity_function(self):
        """Test standalone corner velocity calculation"""
        corner = (9.5, 0.5, 5.0)

        # Far from wall
        current_far = (5.0, 0.5, 5.0)
        vel = calculate_corner_velocity(current_far, corner)
        assert vel == 1.0

        # Close to wall
        current_close = (9.0, 0.5, 5.0)
        vel = calculate_corner_velocity(current_close, corner)
        assert vel == 0.2


# Run all tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
