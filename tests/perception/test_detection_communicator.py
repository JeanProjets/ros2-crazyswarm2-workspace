"""
Tests for detection communication and broadcasting.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'src'))

import pytest
import numpy as np
import time
from perception.detection_communicator import (
    DetectionBroadcaster,
    TargetTracker,
    DetectionMessage
)


class TestDetectionMessage:
    """Test suite for DetectionMessage."""

    def test_creation(self):
        """Test message creation."""
        msg = DetectionMessage(
            drone_id='cf1',
            timestamp=time.time(),
            target_position=(7.5, 3.0, 5.0),
            confidence=0.85,
            target_type='hostile'
        )

        assert msg.drone_id == 'cf1'
        assert msg.target_position == (7.5, 3.0, 5.0)
        assert msg.confidence == 0.85

    def test_to_dict(self):
        """Test conversion to dictionary."""
        msg = DetectionMessage(
            drone_id='cf1',
            timestamp=12345.0,
            target_position=(7.5, 3.0, 5.0),
            confidence=0.85,
            target_type='hostile'
        )

        msg_dict = msg.to_dict()
        assert isinstance(msg_dict, dict)
        assert msg_dict['drone_id'] == 'cf1'
        assert msg_dict['confidence'] == 0.85

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            'drone_id': 'cf2',
            'timestamp': 12345.0,
            'target_position': (8.0, 3.5, 5.0),
            'confidence': 0.9,
            'target_type': 'hostile',
            'detection_image': None
        }

        msg = DetectionMessage.from_dict(data)
        assert msg.drone_id == 'cf2'
        assert msg.confidence == 0.9


class TestTargetTracker:
    """Test suite for TargetTracker."""

    @pytest.fixture
    def tracker(self):
        """Create tracker instance."""
        return TargetTracker(drone_id='cf1')

    def test_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker.drone_id == 'cf1'
        assert not tracker.is_initialized
        assert tracker.track_confidence == 0.0

    def test_initialize_track(self, tracker):
        """Test track initialization."""
        position = (7.5, 3.0, 5.0)
        tracker.initialize_track(position)

        assert tracker.is_initialized is True
        assert tracker.state[0] == 7.5
        assert tracker.state[1] == 3.0
        assert tracker.state[2] == 5.0

    def test_update_track(self, tracker):
        """Test track update."""
        tracker.initialize_track((7.5, 3.0, 5.0))
        time.sleep(0.1)
        tracker.update_track((7.6, 3.0, 5.0))

        assert tracker.missed_updates == 0
        assert tracker.track_confidence > 0.5

    def test_predict_position_not_initialized(self, tracker):
        """Test prediction without initialization."""
        pos = tracker.predict_position(1.0)
        assert pos == (0.0, 0.0, 0.0)

    def test_predict_position(self, tracker):
        """Test position prediction."""
        tracker.initialize_track((7.5, 3.0, 5.0))
        tracker.update_track((7.6, 3.0, 5.0))  # Adds velocity estimate

        predicted = tracker.predict_position(1.0)
        assert isinstance(predicted, tuple)
        assert len(predicted) == 3

    def test_get_position(self, tracker):
        """Test getting current position."""
        tracker.initialize_track((7.5, 3.0, 5.0))
        pos = tracker.get_position()
        assert pos == (7.5, 3.0, 5.0)

    def test_get_velocity(self, tracker):
        """Test getting velocity."""
        tracker.initialize_track((7.5, 3.0, 5.0))
        vel = tracker.get_velocity()
        assert isinstance(vel, tuple)
        assert len(vel) == 3

    def test_get_track_confidence(self, tracker):
        """Test getting track confidence."""
        tracker.initialize_track((7.5, 3.0, 5.0))
        confidence = tracker.get_track_confidence()
        assert 0.0 <= confidence <= 1.0

    def test_is_track_lost_no(self, tracker):
        """Test track not lost."""
        tracker.initialize_track((7.5, 3.0, 5.0))
        assert tracker.is_track_lost() is False

    def test_is_track_lost_yes(self, tracker):
        """Test track lost after missed updates."""
        tracker.initialize_track((7.5, 3.0, 5.0))

        # Mark multiple missed updates
        for _ in range(6):
            tracker.mark_missed_update()

        assert tracker.is_track_lost() is True

    def test_mark_missed_update(self, tracker):
        """Test marking missed update."""
        tracker.initialize_track((7.5, 3.0, 5.0))
        initial_confidence = tracker.track_confidence

        tracker.mark_missed_update()

        assert tracker.missed_updates == 1
        assert tracker.track_confidence < initial_confidence


class TestDetectionBroadcaster:
    """Test suite for DetectionBroadcaster."""

    @pytest.fixture
    def broadcaster(self):
        """Create broadcaster instance."""
        return DetectionBroadcaster(drone_id='cf1')

    def test_initialization(self, broadcaster):
        """Test broadcaster initialization."""
        assert broadcaster.drone_id == 'cf1'
        assert broadcaster.topic_name == '/swarm/target_detection'
        assert broadcaster.message_rate == 10

    def test_broadcast_detection(self, broadcaster):
        """Test detection broadcasting."""
        result = broadcaster.broadcast_detection(
            detection_position=(7.5, 3.0, 5.0),
            confidence=0.85,
            target_type='hostile'
        )
        assert result is True

    def test_broadcast_detection_rate_limiting(self, broadcaster):
        """Test rate limiting of broadcasts."""
        # First broadcast should succeed
        result1 = broadcaster.broadcast_detection(
            detection_position=(7.5, 3.0, 5.0),
            confidence=0.85
        )
        assert result1 is True

        # Immediate second broadcast should fail (rate limited)
        result2 = broadcaster.broadcast_detection(
            detection_position=(7.5, 3.0, 5.0),
            confidence=0.85
        )
        assert result2 is False

    def test_receive_detection(self, broadcaster):
        """Test receiving detection message."""
        msg_data = {
            'drone_id': 'cf2',
            'timestamp': time.time(),
            'target_position': (7.5, 3.0, 5.0),
            'confidence': 0.9,
            'target_type': 'hostile',
            'detection_image': None
        }

        message = broadcaster.receive_detection(msg_data)
        assert message is not None
        assert message.drone_id == 'cf2'

    def test_merge_detections_empty(self, broadcaster):
        """Test merging empty detection list."""
        result = broadcaster.merge_detections([])
        assert result is None

    def test_merge_detections(self, broadcaster):
        """Test merging multiple detections."""
        det1 = DetectionMessage(
            drone_id='cf1',
            timestamp=time.time(),
            target_position=(7.5, 3.0, 5.0),
            confidence=0.8,
            target_type='hostile'
        )

        det2 = DetectionMessage(
            drone_id='cf2',
            timestamp=time.time(),
            target_position=(7.4, 3.1, 5.0),
            confidence=0.9,
            target_type='hostile'
        )

        merged = broadcaster.merge_detections([det1, det2])
        assert merged is not None
        assert isinstance(merged, tuple)
        assert len(merged) == 3

    def test_prioritize_detections_empty(self, broadcaster):
        """Test prioritizing empty list."""
        result = broadcaster.prioritize_detections([])
        assert result is None

    def test_prioritize_detections(self, broadcaster):
        """Test prioritizing detections."""
        det1 = DetectionMessage(
            drone_id='cf1',
            timestamp=time.time(),
            target_position=(7.5, 3.0, 5.0),
            confidence=0.7,
            target_type='hostile'
        )

        det2 = DetectionMessage(
            drone_id='cf2',
            timestamp=time.time(),
            target_position=(7.4, 3.1, 5.0),
            confidence=0.9,  # Higher confidence
            target_type='hostile'
        )

        best = broadcaster.prioritize_detections([det1, det2])
        assert best is not None
        assert best.confidence == 0.9

    def test_update_tracker_with_detections_empty(self, broadcaster):
        """Test tracker update with no detections."""
        result = broadcaster.update_tracker_with_detections([])
        assert result is False

    def test_update_tracker_with_detections(self, broadcaster):
        """Test tracker update with detections."""
        det = DetectionMessage(
            drone_id='cf1',
            timestamp=time.time(),
            target_position=(7.5, 3.0, 5.0),
            confidence=0.85,
            target_type='hostile'
        )

        result = broadcaster.update_tracker_with_detections([det])
        assert result is True

    def test_get_current_target_estimate_not_initialized(self, broadcaster):
        """Test getting estimate before tracker initialized."""
        result = broadcaster.get_current_target_estimate()
        assert result is None

    def test_get_current_target_estimate(self, broadcaster):
        """Test getting current target estimate."""
        det = DetectionMessage(
            drone_id='cf1',
            timestamp=time.time(),
            target_position=(7.5, 3.0, 5.0),
            confidence=0.85,
            target_type='hostile'
        )

        broadcaster.update_tracker_with_detections([det])
        estimate = broadcaster.get_current_target_estimate()

        assert estimate is not None
        assert 'position' in estimate
        assert 'velocity' in estimate
        assert 'confidence' in estimate

    def test_get_all_detections(self, broadcaster):
        """Test getting all detections."""
        msg_data = {
            'drone_id': 'cf2',
            'timestamp': time.time(),
            'target_position': (7.5, 3.0, 5.0),
            'confidence': 0.9,
            'target_type': 'hostile',
            'detection_image': None
        }

        broadcaster.receive_detection(msg_data)
        detections = broadcaster.get_all_detections()

        assert isinstance(detections, list)
        assert len(detections) == 1

    def test_clear_old_detections(self, broadcaster):
        """Test clearing old detections."""
        # Add old detection
        msg_data = {
            'drone_id': 'cf2',
            'timestamp': time.time() - 10.0,  # 10 seconds old
            'target_position': (7.5, 3.0, 5.0),
            'confidence': 0.9,
            'target_type': 'hostile',
            'detection_image': None
        }

        broadcaster.receive_detection(msg_data)
        broadcaster.clear_old_detections(max_age=5.0)

        # Should be cleared
        assert len(broadcaster.received_detections) == 0
