"""
Occlusion-Robust Tracker for Scenario 4.

This module implements a SORT-lite tracker that can handle intermittent visibility
when targets pass behind obstacles. Key features:
- Kalman filter with prediction and correction steps
- IOU-based detection-to-track association
- Coasting state for maintaining tracks during occlusion
- Map-aware track lifetime management
"""

from enum import Enum
from typing import List, Tuple, Optional
import numpy as np


class TrackState(Enum):
    """Track state for handling occlusion."""
    ACTIVE = 1      # Track is matched with a detection
    COASTING = 2    # Track is predicted (no detection, likely occluded)
    LOST = 3        # Track is lost and should be deleted


class Detection:
    """Simple detection class for bounding boxes."""

    def __init__(self, bbox: np.ndarray, confidence: float = 1.0):
        """
        Args:
            bbox: [x, y, w, h] bounding box
            confidence: detection confidence score
        """
        self.bbox = bbox  # [x, y, w, h]
        self.confidence = confidence

    def to_xyxy(self) -> np.ndarray:
        """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
        x, y, w, h = self.bbox
        return np.array([x, y, x + w, y + h])


class KalmanFilter:
    """
    Simple Kalman Filter for 2D target tracking with constant velocity model.
    State: [x, y, vx, vy, w, h, vw, vh]
    """

    def __init__(self, dt: float = 1/30.0):
        """
        Args:
            dt: Time step (default 1/30s for 30fps)
        """
        self.dt = dt
        self.state_dim = 8  # [x, y, vx, vy, w, h, vw, vh]
        self.meas_dim = 4   # [x, y, w, h]

        # State vector
        self.x = np.zeros((self.state_dim, 1))

        # State transition matrix (constant velocity model)
        self.F = np.eye(self.state_dim)
        self.F[0, 2] = dt  # x = x + vx*dt
        self.F[1, 3] = dt  # y = y + vy*dt
        self.F[4, 6] = dt  # w = w + vw*dt
        self.F[5, 7] = dt  # h = h + vh*dt

        # Measurement matrix (we measure position and size)
        self.H = np.zeros((self.meas_dim, self.state_dim))
        self.H[0, 0] = 1  # measure x
        self.H[1, 1] = 1  # measure y
        self.H[2, 4] = 1  # measure w
        self.H[3, 5] = 1  # measure h

        # Process noise covariance
        self.Q = np.eye(self.state_dim) * 0.01
        self.Q[2:4, 2:4] *= 10  # Higher uncertainty in velocity
        self.Q[6:8, 6:8] *= 10  # Higher uncertainty in size velocity

        # Measurement noise covariance
        self.R = np.eye(self.meas_dim) * 1.0

        # State covariance
        self.P = np.eye(self.state_dim) * 10.0

    def init_from_detection(self, bbox: np.ndarray):
        """Initialize Kalman filter from first detection."""
        self.x[0] = bbox[0]  # x
        self.x[1] = bbox[1]  # y
        self.x[4] = bbox[2]  # w
        self.x[5] = bbox[3]  # h
        # velocities initialized to zero

    def predict(self):
        """Prediction step: x = F*x, P = F*P*F' + Q."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, measurement: np.ndarray):
        """Correction step using measurement."""
        z = measurement.reshape(-1, 1)

        # Innovation
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P

        return self.x

    def get_bbox(self) -> np.ndarray:
        """Get current bounding box [x, y, w, h]."""
        return np.array([self.x[0, 0], self.x[1, 0], self.x[4, 0], self.x[5, 0]])

    def get_velocity(self) -> np.ndarray:
        """Get current velocity [vx, vy]."""
        return np.array([self.x[2, 0], self.x[3, 0]])


class Track:
    """A single target track with Kalman filter."""

    _next_id = 0

    def __init__(self, detection: Detection, track_id: Optional[int] = None):
        """Initialize track from detection."""
        self.id = track_id if track_id is not None else Track._next_id
        Track._next_id += 1

        self.kf = KalmanFilter()
        self.kf.init_from_detection(detection.bbox)

        self.state = TrackState.ACTIVE
        self.missed_frames = 0
        self.hit_streak = 1
        self.age = 0

        # For flagging predicted vs observed
        self.is_predicted = False

    def predict(self):
        """Predict next state."""
        self.kf.predict()
        self.age += 1
        self.is_predicted = True
        return self.kf.get_bbox()

    def update(self, detection: Detection):
        """Update track with matched detection."""
        self.kf.update(detection.bbox)
        self.state = TrackState.ACTIVE
        self.missed_frames = 0
        self.hit_streak += 1
        self.is_predicted = False

    def mark_missed(self):
        """Mark track as missed (no matching detection)."""
        self.missed_frames += 1
        self.hit_streak = 0
        self.state = TrackState.COASTING
        self.is_predicted = True

    def get_bbox(self) -> np.ndarray:
        """Get current bounding box."""
        return self.kf.get_bbox()

    def get_velocity(self) -> np.ndarray:
        """Get current velocity."""
        return self.kf.get_velocity()


def iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.

    Args:
        bbox1, bbox2: [x, y, w, h] format

    Returns:
        IoU score
    """
    # Convert to [x1, y1, x2, y2]
    box1 = np.array([bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]])
    box2 = np.array([bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]])

    # Intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def associate_detections_to_tracks(
    detections: List[Detection],
    tracks: List[Track],
    iou_threshold: float = 0.3
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Associate detections to tracks using IoU matching.

    Args:
        detections: List of detections
        tracks: List of tracks
        iou_threshold: Minimum IoU for matching

    Returns:
        matches: List of (track_idx, detection_idx) tuples
        unmatched_tracks: List of track indices
        unmatched_detections: List of detection indices
    """
    if len(tracks) == 0:
        return [], [], list(range(len(detections)))

    if len(detections) == 0:
        return [], list(range(len(tracks))), []

    # Compute IoU matrix
    iou_matrix = np.zeros((len(tracks), len(detections)))
    for t, track in enumerate(tracks):
        for d, det in enumerate(detections):
            iou_matrix[t, d] = iou(track.get_bbox(), det.bbox)

    # Simple greedy matching (Hungarian algorithm would be better but this is lighter)
    matches = []
    unmatched_tracks = []
    unmatched_detections = list(range(len(detections)))

    matched_tracks = set()
    matched_detections = set()

    # Match greedily by highest IoU
    for _ in range(min(len(tracks), len(detections))):
        max_iou = iou_threshold
        max_t, max_d = -1, -1

        for t in range(len(tracks)):
            if t in matched_tracks:
                continue
            for d in range(len(detections)):
                if d in matched_detections:
                    continue
                if iou_matrix[t, d] > max_iou:
                    max_iou = iou_matrix[t, d]
                    max_t, max_d = t, d

        if max_t >= 0:
            matches.append((max_t, max_d))
            matched_tracks.add(max_t)
            matched_detections.add(max_d)
        else:
            break

    for t in range(len(tracks)):
        if t not in matched_tracks:
            unmatched_tracks.append(t)

    unmatched_detections = [d for d in range(len(detections)) if d not in matched_detections]

    return matches, unmatched_tracks, unmatched_detections


class RobustKalmanTracker:
    """
    Occlusion-robust multi-object tracker using Kalman filtering.

    Features:
    - Track persistence through occlusion (coasting mode)
    - Map-aware track management
    - Re-identification support
    """

    def __init__(
        self,
        max_coast_frames: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        """
        Args:
            max_coast_frames: Maximum frames to coast without detection (default 30 = 1s @ 30fps)
            min_hits: Minimum consecutive hits before track is confirmed
            iou_threshold: Minimum IoU for detection-track association
        """
        self.max_coast_frames = max_coast_frames
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.tracks: List[Track] = []
        self.frame_count = 0

        # Map projector will be set externally if available
        self.map_projector = None

    def set_map_projector(self, map_projector):
        """Set map projector for occlusion-aware track management."""
        self.map_projector = map_projector

    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detections for current frame

        Returns:
            List of active tracks (ACTIVE or COASTING)
        """
        self.frame_count += 1

        # Predict all tracks
        for track in self.tracks:
            track.predict()

        # Associate detections to tracks
        matches, unmatched_tracks, unmatched_detections = associate_detections_to_tracks(
            detections, self.tracks, self.iou_threshold
        )

        # Update matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx])

        # Handle unmatched tracks (critical for occlusion handling)
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            track.mark_missed()

            # Determine coast limit based on map (if available)
            coast_limit = self.max_coast_frames

            if self.map_projector is not None:
                # If track is in occluded area, allow longer coasting
                if self.map_projector.is_area_occluded(track.get_bbox()[:2]):
                    coast_limit = 60  # 2 seconds behind obstacle
                else:
                    coast_limit = 10  # Delete quickly in open space

            # Delete track if coasted too long
            if track.missed_frames > coast_limit:
                track.state = TrackState.LOST

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            new_track = Track(detections[det_idx])
            self.tracks.append(new_track)

        # Remove lost tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.LOST]

        # Return only confirmed tracks (hit_streak >= min_hits)
        return [t for t in self.tracks if t.hit_streak >= self.min_hits or t.state == TrackState.COASTING]

    def get_target_state(self, track: Track) -> dict:
        """
        Get target state for a track with status flag.

        Returns:
            Dictionary with position, velocity, and status
        """
        bbox = track.get_bbox()
        velocity = track.get_velocity()

        status = "VISIBLE" if track.state == TrackState.ACTIVE else "OCCLUDED_PREDICTED"

        return {
            'id': track.id,
            'position': bbox[:2],  # [x, y]
            'size': bbox[2:],      # [w, h]
            'velocity': velocity,   # [vx, vy]
            'status': status,
            'is_predicted': track.is_predicted,
            'age': track.age,
            'hit_streak': track.hit_streak,
            'missed_frames': track.missed_frames
        }
