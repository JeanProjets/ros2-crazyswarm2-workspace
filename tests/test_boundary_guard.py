"""
Tests for the Boundary Guard System
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scenarios.boundary_guard import (
    GeofenceMonitor,
    SafetyOverride,
    Telemetry,
    ViolationType
)


class TestGeofenceMonitor:
    """Test suite for GeofenceMonitor"""

    def test_initialization(self):
        """Test that monitor initializes with correct limits"""
        monitor = GeofenceMonitor()

        assert monitor.hard_limits['x_max'] == 9.8
        assert monitor.hard_limits['y_min'] == 0.2
        assert monitor.soft_limits['x_max'] == 9.5
        assert monitor.soft_limits['y_min'] == 0.5

    def test_check_bounds_within_limits(self):
        """Test that positions within bounds return NONE"""
        monitor = GeofenceMonitor()

        safe_position = {'x': 5.0, 'y': 5.0, 'z': 3.0}
        result = monitor.check_bounds(safe_position, 'hard')

        assert result == ViolationType.NONE

    def test_check_bounds_exceeds_hard_limit(self):
        """Test detection of hard limit violation"""
        monitor = GeofenceMonitor()

        # Position exceeds x_max
        unsafe_position = {'x': 9.9, 'y': 5.0, 'z': 3.0}
        result = monitor.check_bounds(unsafe_position, 'hard')

        assert result == ViolationType.HARD_LIMIT

    def test_check_bounds_below_hard_limit(self):
        """Test detection of minimum boundary violation"""
        monitor = GeofenceMonitor()

        # Position below y_min
        unsafe_position = {'x': 5.0, 'y': 0.1, 'z': 3.0}
        result = monitor.check_bounds(unsafe_position, 'hard')

        assert result == ViolationType.HARD_LIMIT

    def test_predict_violation_safe_trajectory(self):
        """Test that safe trajectory is not flagged"""
        monitor = GeofenceMonitor()

        position = {'x': 5.0, 'y': 5.0, 'z': 3.0}
        velocity = {'x': 0.5, 'y': 0.0, 'z': 0.0}  # Moving slowly right

        result = monitor.predict_violation(position, velocity)

        assert result is False

    def test_predict_violation_collision_course(self):
        """Test prediction of wall collision"""
        monitor = GeofenceMonitor()

        # Drone near wall moving fast toward it
        position = {'x': 9.5, 'y': 5.0, 'z': 3.0}
        velocity = {'x': 1.0, 'y': 0.0, 'z': 0.0}  # Moving right at 1 m/s

        result = monitor.predict_violation(position, velocity)

        assert result is True

    def test_check_swarm_bounds_all_safe(self):
        """Test swarm check with all drones safe"""
        monitor = GeofenceMonitor()

        telemetry = [
            Telemetry('drone1', 3.0, 3.0, 2.0, 0.1, 0.1, 0.0),
            Telemetry('drone2', 4.0, 4.0, 2.5, 0.0, 0.1, 0.0)
        ]

        violations = monitor.check_swarm_bounds(telemetry)

        assert len(violations) == 0

    def test_check_swarm_bounds_one_violation(self):
        """Test swarm check with one drone in violation"""
        monitor = GeofenceMonitor()

        telemetry = [
            Telemetry('drone1', 3.0, 3.0, 2.0, 0.0, 0.0, 0.0),
            Telemetry('drone2', 9.9, 0.5, 2.5, 0.0, 0.0, 0.0)  # Exceeds x_max
        ]

        violations = monitor.check_swarm_bounds(telemetry)

        assert 'drone2' in violations
        assert violations['drone2']['severity'] == 'CRITICAL'


class TestSafetyOverride:
    """Test suite for SafetyOverride"""

    def test_initialization(self):
        """Test safety override initialization"""
        safety = SafetyOverride()

        assert safety.geofence is not None
        assert safety.emergency_stop_active is False
        assert len(safety.violation_history) == 0

    def test_monitor_loop_no_violations(self):
        """Test monitoring with safe telemetry"""
        safety = SafetyOverride()

        telemetry = [
            Telemetry('drone1', 3.0, 3.0, 2.0, 0.1, 0.1, 0.0)
        ]

        violations = safety.monitor_loop(telemetry)

        assert len(violations) == 0
        assert safety.emergency_stop_active is False

    def test_monitor_loop_triggers_emergency_stop(self):
        """Test that critical violations trigger emergency stop"""
        safety = SafetyOverride()

        telemetry = [
            Telemetry('drone1', 9.9, 0.5, 2.0, 0.0, 0.0, 0.0)  # Critical violation
        ]

        violations = safety.monitor_loop(telemetry)

        assert len(violations) > 0
        assert safety.emergency_stop_active is True

    def test_get_violation_summary(self):
        """Test violation summary generation"""
        safety = SafetyOverride()

        # Trigger a violation
        telemetry = [
            Telemetry('drone1', 9.9, 0.5, 2.0, 0.0, 0.0, 0.0)
        ]
        safety.monitor_loop(telemetry)

        summary = safety.get_violation_summary()

        assert summary['total_violations'] > 0
        assert 'history' in summary

    def test_reset_emergency_stop(self):
        """Test emergency stop reset"""
        safety = SafetyOverride()

        # Trigger emergency
        safety.trigger_emergency_stop('drone1', 'TEST')
        assert safety.emergency_stop_active is True

        # Reset
        safety.reset_emergency_stop()
        assert safety.emergency_stop_active is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
