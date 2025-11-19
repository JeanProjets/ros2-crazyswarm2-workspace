"""
Tests for tracking_controller module
"""

import pytest
import numpy as np
from src.core.tracking_controller import (
    DynamicTracker,
    TrackingController,
    TargetState,
    VelocityFilter
)


class TestTargetState:
    """Test TargetState dataclass"""

    def test_initialization(self):
        """Test target state initialization"""
        state = TargetState(
            position=[1.0, 2.0, 3.0],
            velocity=[0.1, 0.2, 0.3],
            timestamp=100.0
        )

        assert isinstance(state.position, np.ndarray)
        assert isinstance(state.velocity, np.ndarray)
        assert np.allclose(state.position, [1.0, 2.0, 3.0])
        assert np.allclose(state.velocity, [0.1, 0.2, 0.3])
        assert state.timestamp == 100.0


class TestVelocityFilter:
    """Test VelocityFilter class"""

    def test_filter_initialization(self):
        """Test filter initialization"""
        vf = VelocityFilter(cutoff_freq=2.0, sample_rate=10.0)
        assert vf.cutoff_freq == 2.0
        assert vf.sample_rate == 10.0
        assert len(vf.history) == 0

    def test_filter_with_few_samples(self):
        """Test filter with insufficient samples"""
        vf = VelocityFilter()
        vel1 = np.array([1.0, 0.0, 0.0])
        result = vf.filter(vel1)
        assert np.allclose(result, vel1)

    def test_filter_with_many_samples(self):
        """Test filter with sufficient samples"""
        vf = VelocityFilter()

        # Add samples
        for i in range(5):
            vf.filter(np.array([1.0 + 0.1*i, 0.0, 0.0]))

        # Should get smoothed result
        result = vf.filter(np.array([1.5, 0.0, 0.0]))
        assert isinstance(result, np.ndarray)
        assert len(result) == 3


class TestDynamicTracker:
    """Test DynamicTracker class"""

    def test_initialization(self):
        """Test tracker initialization"""
        tracker = DynamicTracker("drone1")
        assert tracker.drone_id == "drone1"
        assert tracker.max_velocity == 2.0
        assert tracker.target_state is None

    def test_update_target_state(self):
        """Test target state update"""
        tracker = DynamicTracker("drone1")
        tracker.update_target_state(
            position=(1.0, 2.0, 3.0),
            velocity=(0.5, 0.0, 0.0),
            timestamp=100.0
        )

        assert tracker.target_state is not None
        assert np.allclose(tracker.target_state.position, [1.0, 2.0, 3.0])

    def test_compute_intercept_vector_stationary_target(self):
        """Test intercept computation for stationary target"""
        tracker = DynamicTracker("drone1")

        drone_pos = np.array([0.0, 0.0, 1.0])
        target_pos = np.array([2.0, 0.0, 1.0])
        target_vel = np.array([0.0, 0.0, 0.0])

        cmd_vel = tracker.compute_intercept_vector(drone_pos, target_pos, target_vel)

        # Should point toward target
        assert cmd_vel[0] > 0  # Positive x direction
        assert np.linalg.norm(cmd_vel) <= tracker.max_velocity

    def test_compute_intercept_vector_moving_target(self):
        """Test intercept computation for moving target"""
        tracker = DynamicTracker("drone1")

        drone_pos = np.array([0.0, 0.0, 1.0])
        target_pos = np.array([2.0, 0.0, 1.0])
        target_vel = np.array([0.0, 0.5, 0.0])  # Moving in +y

        cmd_vel = tracker.compute_intercept_vector(drone_pos, target_pos, target_vel)

        # Should have both x and y components (lead pursuit)
        assert cmd_vel[0] > 0  # Toward target
        assert cmd_vel[1] > 0  # Leading the target
        assert np.linalg.norm(cmd_vel) <= tracker.max_velocity

    def test_compute_intercept_vector_velocity_limit(self):
        """Test that intercept vector respects velocity limits"""
        tracker = DynamicTracker("drone1", max_velocity=1.0)

        drone_pos = np.array([0.0, 0.0, 1.0])
        target_pos = np.array([10.0, 10.0, 1.0])  # Far away
        target_vel = np.array([1.0, 1.0, 0.0])

        cmd_vel = tracker.compute_intercept_vector(drone_pos, target_pos, target_vel)

        # Should not exceed max velocity
        assert np.linalg.norm(cmd_vel) <= tracker.max_velocity + 0.01

    def test_match_velocity_hover(self):
        """Test velocity matching for hovering"""
        tracker = DynamicTracker("drone1")

        target_vel = np.array([0.5, 0.3, 0.0])
        cmd_vel = tracker.match_velocity_hover(target_vel)

        # Should match target velocity (within limits)
        assert np.allclose(cmd_vel, target_vel)
        assert np.linalg.norm(cmd_vel) <= tracker.max_velocity

    def test_match_velocity_hover_with_limit(self):
        """Test velocity matching with speed limiting"""
        tracker = DynamicTracker("drone1", max_velocity=0.5)

        target_vel = np.array([1.0, 1.0, 0.0])  # Faster than limit
        cmd_vel = tracker.match_velocity_hover(target_vel)

        # Should be limited
        assert np.linalg.norm(cmd_vel) <= tracker.max_velocity + 0.01

    def test_track_target_no_target(self):
        """Test tracking when no target is set"""
        tracker = DynamicTracker("drone1")
        result = tracker.track_target()
        assert result is False

    def test_track_target_with_target(self):
        """Test tracking with target set"""
        tracker = DynamicTracker("drone1")
        tracker.update_target_state(
            position=(5.0, 5.0, 2.0),
            velocity=(0.5, 0.0, 0.0)
        )

        result = tracker.track_target()
        assert result is True


class TestTrackingController:
    """Test TrackingController class"""

    def test_initialization(self):
        """Test controller initialization"""
        controller = TrackingController(max_velocity=1.5)
        assert controller.max_velocity == 1.5

    def test_compute_lead_pursuit_stationary(self):
        """Test lead pursuit for stationary target"""
        controller = TrackingController()

        drone_pos = (0.0, 0.0, 1.0)
        target_pos = (3.0, 0.0, 1.0)
        target_vel = (0.0, 0.0, 0.0)

        cmd_vel = controller.compute_lead_pursuit(
            drone_pos, target_pos, target_vel, speed_gain=1.0
        )

        # Should point toward target
        assert cmd_vel[0] > 0
        assert np.linalg.norm(cmd_vel) <= controller.max_velocity + 0.01

    def test_compute_lead_pursuit_moving(self):
        """Test lead pursuit for moving target"""
        controller = TrackingController()

        drone_pos = (0.0, 0.0, 1.0)
        target_pos = (3.0, 0.0, 1.0)
        target_vel = (0.0, 1.0, 0.0)  # Moving in +y

        cmd_vel = controller.compute_lead_pursuit(
            drone_pos, target_pos, target_vel, speed_gain=1.0
        )

        # Should lead the target
        assert cmd_vel[0] > 0  # Toward target x
        assert cmd_vel[1] > 0  # Leading in y
        assert np.linalg.norm(cmd_vel) <= controller.max_velocity + 0.01

    def test_compute_lead_pursuit_at_target(self):
        """Test lead pursuit when already at target"""
        controller = TrackingController()

        drone_pos = (1.0, 1.0, 1.0)
        target_pos = (1.0, 1.0, 1.0)  # Same position
        target_vel = (0.0, 0.0, 0.0)

        cmd_vel = controller.compute_lead_pursuit(
            drone_pos, target_pos, target_vel, speed_gain=1.0
        )

        # Should be zero or very small
        assert np.linalg.norm(cmd_vel) < 0.1

    def test_compute_lead_pursuit_velocity_limit(self):
        """Test that lead pursuit respects velocity limits"""
        controller = TrackingController(max_velocity=0.5)

        drone_pos = (0.0, 0.0, 1.0)
        target_pos = (10.0, 10.0, 1.0)
        target_vel = (1.0, 1.0, 0.0)

        cmd_vel = controller.compute_lead_pursuit(
            drone_pos, target_pos, target_vel, speed_gain=2.0
        )

        # Should not exceed max velocity
        assert np.linalg.norm(cmd_vel) <= controller.max_velocity + 0.01


class TestIntegration:
    """Integration tests for tracking system"""

    def test_complete_tracking_scenario(self):
        """Test complete tracking scenario"""
        # Create tracker
        tracker = DynamicTracker("drone1", max_velocity=2.0)

        # Simulate target moving in circle
        time_steps = 10
        radius = 3.0
        angular_vel = 0.2  # rad/s

        for i in range(time_steps):
            t = i * 0.1
            angle = angular_vel * t

            # Target position and velocity
            target_pos = (
                radius * np.cos(angle),
                radius * np.sin(angle),
                2.0
            )
            target_vel = (
                -radius * angular_vel * np.sin(angle),
                radius * angular_vel * np.cos(angle),
                0.0
            )

            # Update tracker
            tracker.update_target_state(target_pos, target_vel, t)

            # Compute intercept
            drone_pos = np.array([0.0, 0.0, 2.0])
            cmd_vel = tracker.compute_intercept_vector(
                drone_pos,
                tracker.target_state.position,
                tracker.target_state.velocity
            )

            # Verify command is reasonable
            assert np.linalg.norm(cmd_vel) <= tracker.max_velocity + 0.01
            assert not np.any(np.isnan(cmd_vel))

    def test_no_oscillation(self):
        """Test that tracker doesn't oscillate around target"""
        tracker = DynamicTracker("drone1", max_velocity=1.0)

        # Stationary target
        target_pos = np.array([2.0, 2.0, 1.5])
        target_vel = np.array([0.0, 0.0, 0.0])

        # Drone approaching target
        drone_positions = [
            np.array([0.0, 0.0, 1.5]),
            np.array([1.0, 1.0, 1.5]),
            np.array([1.5, 1.5, 1.5]),
            np.array([1.9, 1.9, 1.5]),
        ]

        for drone_pos in drone_positions:
            cmd_vel = tracker.compute_intercept_vector(
                drone_pos, target_pos, target_vel
            )

            # Velocity should decrease as we get closer
            distance = np.linalg.norm(target_pos - drone_pos)
            if distance < 0.5:
                # Close to target, velocity should be small
                assert np.linalg.norm(cmd_vel) < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
