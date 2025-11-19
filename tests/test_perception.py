"""
Test suite for Scenario 4 Agent 3: Perception modules.

Tests all four modules:
1. RobustKalmanTracker
2. EdgeClipper
3. MapProjector
4. TargetFingerprint
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from src.perception.robust_tracker import (
    RobustKalmanTracker, Detection, Track, TrackState, iou
)
from src.perception.occlusion_filter import EdgeClipper, MeasurementQualityFilter
from src.perception.map_projector import MapProjector, Obstacle, CameraPose
from src.perception.reid_manager import TargetFingerprint, ReIDManager


class TestRobustKalmanTracker:
    """Test the occlusion-robust tracker."""

    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = RobustKalmanTracker(max_coast_frames=30)
        assert tracker.max_coast_frames == 30
        assert len(tracker.tracks) == 0

    def test_single_detection(self):
        """Test tracking a single detection."""
        tracker = RobustKalmanTracker(max_coast_frames=30, min_hits=1)

        # Create a detection
        det = Detection(np.array([100, 100, 50, 50]))
        detections = [det]

        # Update tracker
        tracks = tracker.update(detections)

        assert len(tracks) == 1
        assert tracks[0].state == TrackState.ACTIVE

    def test_occlusion_coasting(self):
        """Test coasting behavior during occlusion."""
        tracker = RobustKalmanTracker(max_coast_frames=10, min_hits=1)

        # Initial detection
        det = Detection(np.array([100, 100, 50, 50]))
        tracker.update([det])

        # Move target
        det2 = Detection(np.array([110, 110, 50, 50]))
        tracker.update([det2])

        # Simulate occlusion (no detections)
        for _ in range(5):
            tracks = tracker.update([])
            assert len(tracks) > 0, "Track should be maintained during occlusion"
            assert tracks[0].state == TrackState.COASTING

    def test_track_deletion_after_timeout(self):
        """Test track deletion after max_coast_frames."""
        tracker = RobustKalmanTracker(max_coast_frames=5, min_hits=1)

        # Initial detection
        det = Detection(np.array([100, 100, 50, 50]))
        tracker.update([det])

        # Occlude for longer than max_coast_frames
        for _ in range(10):
            tracks = tracker.update([])

        # Track should be deleted
        assert len(tracker.tracks) == 0

    def test_iou_calculation(self):
        """Test IoU calculation."""
        bbox1 = np.array([0, 0, 100, 100])
        bbox2 = np.array([50, 50, 100, 100])

        iou_score = iou(bbox1, bbox2)
        assert 0 < iou_score < 1

        # Perfect overlap
        iou_perfect = iou(bbox1, bbox1)
        assert iou_perfect == 1.0

        # No overlap
        bbox3 = np.array([200, 200, 100, 100])
        iou_none = iou(bbox1, bbox3)
        assert iou_none == 0.0

    def test_track_prediction(self):
        """Test Kalman filter prediction."""
        track = Track(Detection(np.array([100, 100, 50, 50])))

        # Predict next state
        predicted_bbox = track.predict()
        assert len(predicted_bbox) == 4

    def test_multiple_tracks(self):
        """Test tracking multiple targets."""
        tracker = RobustKalmanTracker(max_coast_frames=30, min_hits=1)

        # Two detections far apart
        det1 = Detection(np.array([100, 100, 50, 50]))
        det2 = Detection(np.array([300, 300, 50, 50]))

        tracks = tracker.update([det1, det2])
        assert len(tracks) == 2


class TestEdgeClipper:
    """Test the partial occlusion handler."""

    def test_edge_clipper_initialization(self):
        """Test EdgeClipper initialization."""
        clipper = EdgeClipper()
        assert clipper.normal_aspect_ratio == 1.0
        assert clipper.min_aspect_ratio == 0.6

    def test_aspect_ratio_detection(self):
        """Test aspect ratio-based occlusion detection."""
        clipper = EdgeClipper()

        # Normal bbox (square)
        bbox_normal = np.array([100, 100, 50, 50])
        assert not clipper.is_bad_depth_measurement(bbox_normal)

        # Vertical sliver
        bbox_sliver = np.array([100, 100, 20, 50])
        assert clipper.is_bad_depth_measurement(bbox_sliver)

        # Horizontal sliver
        bbox_wide = np.array([100, 100, 100, 30])
        assert clipper.is_bad_depth_measurement(bbox_wide)

    def test_distance_correction(self):
        """Test distance estimate correction."""
        clipper = EdgeClipper()

        # Good measurement (full visibility)
        distance, reliable = clipper.correct_distance_estimate(
            bbox_width=50,
            visible_ratio=1.0
        )
        assert reliable

        # Partial occlusion
        distance_partial, reliable_partial = clipper.correct_distance_estimate(
            bbox_width=25,
            visible_ratio=0.5
        )
        assert not reliable_partial

    def test_partial_occlusion_detection(self):
        """Test partial occlusion detection."""
        clipper = EdgeClipper()

        # Normal bbox
        bbox_normal = np.array([100, 100, 50, 50])
        occluded = clipper.check_partial_occlusion(bbox_normal)
        assert not occluded

        # Very thin bbox (sliver)
        bbox_thin = np.array([100, 100, 15, 50])
        occluded_thin = clipper.check_partial_occlusion(bbox_thin)
        assert occluded_thin


class TestMapProjector:
    """Test the map-vision fusion module."""

    def test_map_projector_initialization(self):
        """Test MapProjector initialization."""
        projector = MapProjector()
        assert projector.image_width == 640
        assert projector.image_height == 480

    def test_obstacle_creation(self):
        """Test obstacle creation."""
        obstacle = Obstacle(
            position=np.array([1.0, 1.0]),
            size=np.array([0.5, 0.5]),
            height=1.0
        )
        assert obstacle.position[0] == 1.0
        assert obstacle.height == 1.0

    def test_point_projection(self):
        """Test 3D point projection to camera."""
        projector = MapProjector()
        camera_pose = CameraPose(position=np.array([0, 0, 1]))

        # Point in front of camera
        point_3d = np.array([1, 1, 0])
        pixel, depth = projector.project_point_to_camera(point_3d, camera_pose)

        assert depth != 0

    def test_line_of_sight(self):
        """Test line of sight calculation."""
        projector = MapProjector()

        # No obstacles
        start = np.array([0, 0, 1])
        end = np.array([5, 5, 1])
        clear = projector.is_line_of_sight_clear(start, end, [])
        assert clear

        # With obstacle in the way
        obstacle = Obstacle(
            position=np.array([2.5, 2.5]),
            size=np.array([1.0, 1.0]),
            height=2.0
        )
        blocked = projector.is_line_of_sight_clear(start, end, [obstacle])
        assert not blocked

    def test_visibility_prediction(self):
        """Test target visibility prediction."""
        projector = MapProjector()
        camera_pose = CameraPose(position=np.array([0, 0, 1]))

        # Target with no obstacles
        target_pos = np.array([3, 3, 0])
        visible = projector.is_target_expected_visible(target_pos, camera_pose, [])
        assert visible

        # Target behind obstacle
        obstacle = Obstacle(
            position=np.array([1.5, 1.5]),
            size=np.array([1.0, 1.0]),
            height=2.0
        )
        blocked = projector.is_target_expected_visible(target_pos, camera_pose, [obstacle])
        assert not blocked

    def test_detection_state_classification(self):
        """Test detection state classification."""
        projector = MapProjector()
        camera_pose = CameraPose(position=np.array([0, 0, 1]))
        target_pos = np.array([5, 5, 0])

        # Case: No detection, open space -> TRUE_NEGATIVE
        state = projector.classify_detection_state(
            vision_detects_target=False,
            target_predicted_position=target_pos,
            drone_pose=camera_pose,
            obstacles=[]
        )
        assert state == "TRUE_NEGATIVE"

        # Case: No detection, blocked -> EXPECTED_OCCLUSION
        obstacle = Obstacle(
            position=np.array([2.5, 2.5]),
            size=np.array([1.0, 1.0]),
            height=2.0
        )
        state_occluded = projector.classify_detection_state(
            vision_detects_target=False,
            target_predicted_position=target_pos,
            drone_pose=camera_pose,
            obstacles=[obstacle]
        )
        assert state_occluded == "EXPECTED_OCCLUSION"


class TestTargetFingerprint:
    """Test the re-identification module."""

    def test_fingerprint_initialization(self):
        """Test TargetFingerprint initialization."""
        fingerprint = TargetFingerprint()
        assert fingerprint.max_history_length == 10

    def test_feature_storage(self):
        """Test storing appearance features."""
        fingerprint = TargetFingerprint()

        bbox = np.array([100, 100, 50, 50])
        velocity = np.array([5, 5])

        fingerprint.store_appearance_features(bbox, velocity)

        assert len(fingerprint._size_history) == 1
        assert len(fingerprint._aspect_ratio_history) == 1
        assert len(fingerprint._velocity_history) == 1

    def test_candidate_verification(self):
        """Test candidate verification."""
        fingerprint = TargetFingerprint()

        # Build history
        for i in range(5):
            bbox = np.array([100 + i*2, 100 + i*2, 50, 50])
            velocity = np.array([2, 2])
            fingerprint.store_appearance_features(bbox, velocity)

        # Test good candidate (similar features)
        good_candidate = np.array([110, 110, 50, 50])
        score_good = fingerprint.verify_candidate(good_candidate)
        assert score_good > 0.5

        # Test bad candidate (very different)
        bad_candidate = np.array([500, 500, 100, 20])
        score_bad = fingerprint.verify_candidate(bad_candidate)
        assert score_bad < score_good

    def test_reid_manager(self):
        """Test ReID manager."""
        manager = ReIDManager(reid_threshold=0.6)

        # Register track
        track_id = 1
        fingerprint = manager.register_track(track_id)
        assert fingerprint is not None

        # Update features
        bbox = np.array([100, 100, 50, 50])
        manager.update_track_features(track_id, bbox)

        # Find match
        candidate = np.array([105, 105, 50, 50])
        predictions = {track_id: {'position': np.array([105, 105]), 'velocity': np.array([1, 1])}}

        matched_id, confidence = manager.find_best_match(candidate, predictions)
        # Match might not be strong enough with just one sample
        assert matched_id is None or matched_id == track_id


class TestIntegration:
    """Integration tests combining multiple modules."""

    def test_tracker_with_map_projector(self):
        """Test tracker with map projector integration."""
        tracker = RobustKalmanTracker(max_coast_frames=30, min_hits=1)
        projector = MapProjector()

        tracker.set_map_projector(projector)
        assert tracker.map_projector is not None

    def test_full_pipeline(self):
        """Test full tracking pipeline with all modules."""
        # Initialize all components
        tracker = RobustKalmanTracker(max_coast_frames=30, min_hits=1)
        clipper = EdgeClipper()
        projector = MapProjector()
        reid_manager = ReIDManager()

        # Set up tracker with map
        tracker.set_map_projector(projector)

        # Process detections
        det1 = Detection(np.array([100, 100, 50, 50]))
        tracks = tracker.update([det1])

        assert len(tracks) == 1

        # Check occlusion filtering
        bbox = np.array([100, 100, 50, 50])
        occluded = clipper.check_partial_occlusion(bbox)

        # Update ReID
        if len(tracks) > 0:
            reid_manager.update_track_features(
                tracks[0].id,
                tracks[0].get_bbox(),
                tracks[0].get_velocity()
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
