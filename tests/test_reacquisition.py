"""
Tests for reacquisition module
"""

import pytest
import numpy as np
from src.behaviors.reacquisition import (
    OcclusionHandler,
    TargetState,
    SearchState
)
from src.behaviors.elastic_formation import GridMap


class TestTargetState:
    """Tests for TargetState dataclass"""

    def test_target_state_creation(self):
        """Test creating TargetState"""
        pos = np.array([1.0, 2.0, 3.0])
        vel = np.array([0.5, 0.0, 0.0])

        state = TargetState(
            position=pos,
            velocity=vel,
            timestamp=10.0,
            confidence=0.9
        )

        assert state.timestamp == 10.0
        assert state.confidence == 0.9
        np.testing.assert_array_equal(state.position, pos)


class TestOcclusionHandler:
    """Tests for OcclusionHandler class"""

    def test_initialization(self):
        """Test OcclusionHandler initialization"""
        handler = OcclusionHandler(
            confidence_threshold=0.4,
            prediction_time_limit=3.0
        )
        assert handler.confidence_threshold == 0.4
        assert handler.prediction_time_limit == 3.0
        assert handler.state == SearchState.TRACKING

    def test_update_target_observation_good_confidence(self):
        """Test updating with good target observation"""
        handler = OcclusionHandler()

        pos = np.array([1.0, 2.0, 1.0])
        vel = np.array([0.5, 0.0, 0.0])

        handler.update_target_observation(pos, vel, timestamp=1.0, confidence=0.8)

        assert handler.state == SearchState.TRACKING
        assert handler.last_known_target is not None
        assert handler.occlusion_start_time is None

    def test_update_target_observation_low_confidence(self):
        """Test updating with low confidence (target lost)"""
        handler = OcclusionHandler(confidence_threshold=0.5)

        # First, good observation
        pos = np.array([1.0, 2.0, 1.0])
        vel = np.array([0.5, 0.0, 0.0])
        handler.update_target_observation(pos, vel, timestamp=1.0, confidence=0.8)

        # Then, lost target
        handler.update_target_observation(pos, vel, timestamp=2.0, confidence=0.2)

        assert handler.state == SearchState.PREDICTING
        assert handler.occlusion_start_time == 2.0

    def test_predict_emergence_point_no_target(self):
        """Test prediction with no target data"""
        handler = OcclusionHandler()

        prediction = handler.predict_emergence_point(current_time=5.0)
        assert prediction is None

    def test_predict_emergence_point_stationary_target(self):
        """Test prediction with stationary target"""
        handler = OcclusionHandler(min_velocity_threshold=0.1)

        pos = np.array([1.0, 2.0, 1.0])
        vel = np.array([0.0, 0.0, 0.0])  # Stationary

        handler.update_target_observation(pos, vel, timestamp=1.0, confidence=0.8)
        handler.update_target_observation(pos, vel, timestamp=2.0, confidence=0.2)

        prediction = handler.predict_emergence_point(current_time=3.0)

        # Should return last known position for stationary target
        np.testing.assert_array_almost_equal(prediction, pos)

    def test_predict_emergence_point_moving_target(self):
        """Test prediction with moving target"""
        handler = OcclusionHandler()

        pos = np.array([0.0, 0.0, 1.0])
        vel = np.array([1.0, 0.0, 0.0])  # Moving in +x

        handler.update_target_observation(pos, vel, timestamp=0.0, confidence=0.8)
        handler.state = SearchState.PREDICTING
        handler.occlusion_start_time = 0.0

        prediction = handler.predict_emergence_point(current_time=2.0)

        # Should extrapolate: pos + vel * dt = [0,0,1] + [1,0,0] * 2 = [2,0,1]
        assert prediction is not None
        assert prediction[0] > 1.5  # Should have moved in +x direction

    def test_predict_emergence_with_map(self):
        """Test prediction using grid map"""
        handler = OcclusionHandler()
        grid = GridMap(inflation_radius=0.3)

        # Add obstacle
        grid.add_obstacle(np.array([2.0, 0.0, 1.0]))

        pos = np.array([0.0, 0.0, 1.0])
        vel = np.array([1.0, 0.0, 0.0])

        handler.update_target_observation(pos, vel, timestamp=0.0, confidence=0.8)
        handler.state = SearchState.PREDICTING
        handler.occlusion_start_time = 0.0

        prediction = handler.predict_emergence_point(current_time=1.0, grid_map=grid)

        # Should predict emergence point (behavior depends on implementation)
        assert prediction is not None

    def test_execute_search_maneuver_no_data(self):
        """Test search maneuver with no target data"""
        handler = OcclusionHandler()

        drone_pos = np.array([0.0, 0.0, 1.0])
        cmd_vel, msg = handler.execute_search_maneuver(drone_pos, current_time=5.0)

        np.testing.assert_array_equal(cmd_vel, np.zeros(3))
        assert "No target data" in msg

    def test_execute_search_maneuver_recent_loss(self):
        """Test search maneuver soon after losing target"""
        handler = OcclusionHandler(prediction_time_limit=3.0)

        pos = np.array([5.0, 0.0, 1.0])
        vel = np.array([1.0, 0.0, 0.0])

        handler.update_target_observation(pos, vel, timestamp=0.0, confidence=0.8)
        handler.state = SearchState.PREDICTING
        handler.occlusion_start_time = 0.0

        drone_pos = np.array([0.0, 0.0, 1.0])
        cmd_vel, msg = handler.execute_search_maneuver(
            drone_pos, current_time=1.0, max_speed=1.0
        )

        # Should pursue to predicted emergence point
        assert "emergence" in msg.lower()
        assert np.linalg.norm(cmd_vel) > 0

    def test_execute_search_maneuver_long_loss(self):
        """Test search maneuver after long time"""
        handler = OcclusionHandler(prediction_time_limit=2.0)

        pos = np.array([5.0, 0.0, 1.0])
        vel = np.array([1.0, 0.0, 0.0])

        handler.update_target_observation(pos, vel, timestamp=0.0, confidence=0.8)
        handler.state = SearchState.PREDICTING
        handler.occlusion_start_time = 0.0

        drone_pos = np.array([0.0, 0.0, 1.0])
        cmd_vel, msg = handler.execute_search_maneuver(
            drone_pos, current_time=5.0, max_speed=1.0
        )

        # Should execute spiral search after time limit
        assert handler.state == SearchState.SEARCHING
        assert "spiral" in msg.lower()

    def test_is_target_lost(self):
        """Test checking if target is lost"""
        handler = OcclusionHandler()

        # Initially tracking
        assert not handler.is_target_lost()

        # Change to predicting
        handler.state = SearchState.PREDICTING
        assert handler.is_target_lost()

        # Change to searching
        handler.state = SearchState.SEARCHING
        assert handler.is_target_lost()

        # Back to tracking
        handler.state = SearchState.TRACKING
        assert not handler.is_target_lost()

    def test_get_state(self):
        """Test getting current state"""
        handler = OcclusionHandler()

        assert handler.get_state() == SearchState.TRACKING

        handler.state = SearchState.PREDICTING
        assert handler.get_state() == SearchState.PREDICTING

    def test_should_continue_mission(self):
        """Test that mission always continues"""
        handler = OcclusionHandler()

        # Should always return True (never give up!)
        assert handler.should_continue_mission()

        handler.state = SearchState.SEARCHING
        assert handler.should_continue_mission()

    def test_reacquisition_state_transition(self):
        """Test state transitions during reacquisition"""
        handler = OcclusionHandler()

        pos = np.array([1.0, 0.0, 1.0])
        vel = np.array([1.0, 0.0, 0.0])

        # Start tracking
        handler.update_target_observation(pos, vel, timestamp=1.0, confidence=0.9)
        assert handler.state == SearchState.TRACKING

        # Lose target
        handler.update_target_observation(pos, vel, timestamp=2.0, confidence=0.2)
        assert handler.state == SearchState.PREDICTING

        # Reacquire target
        handler.update_target_observation(pos, vel, timestamp=3.0, confidence=0.8)
        assert handler.state == SearchState.TRACKING
