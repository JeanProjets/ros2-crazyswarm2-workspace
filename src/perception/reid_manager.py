"""
Re-Identification (ReID) Manager for Scenario 4.

This module confirms that a drone emerging from occlusion is the same
target we were tracking before. Uses lightweight heuristic features
suitable for GAP8 AI Deck.
"""

from typing import Optional, List
import numpy as np


class TargetFingerprint:
    """
    Lightweight target fingerprinting for re-identification.

    GAP8 is too weak for deep learning ReID, so we use heuristic features:
    - Size (normalized area)
    - Aspect ratio history
    - Velocity vector consistency
    - Position continuity
    """

    def __init__(
        self,
        max_history_length: int = 10,
        size_tolerance: float = 0.3,
        velocity_tolerance: float = 2.0,
        position_tolerance: float = 100.0
    ):
        """
        Args:
            max_history_length: Maximum number of frames to keep in history
            size_tolerance: Relative tolerance for size matching (0.3 = 30%)
            velocity_tolerance: Velocity difference tolerance (pixels/frame)
            position_tolerance: Position difference tolerance (pixels)
        """
        self.max_history_length = max_history_length
        self.size_tolerance = size_tolerance
        self.velocity_tolerance = velocity_tolerance
        self.position_tolerance = position_tolerance

        # Feature storage
        self._size_history: List[float] = []
        self._aspect_ratio_history: List[float] = []
        self._velocity_history: List[np.ndarray] = []
        self._position_history: List[np.ndarray] = []

        # Last known features
        self._last_known_size: Optional[float] = None
        self._last_known_aspect_ratio: Optional[float] = None
        self._last_known_velocity: Optional[np.ndarray] = None
        self._last_known_position: Optional[np.ndarray] = None

    def store_appearance_features(self, bbox: np.ndarray, velocity: Optional[np.ndarray] = None):
        """
        Store appearance features from a bounding box.

        Args:
            bbox: Bounding box [x, y, w, h]
            velocity: Optional velocity [vx, vy]
        """
        x, y, w, h = bbox

        # Calculate features
        size = w * h  # Area
        aspect_ratio = w / h if h > 0 else 1.0
        position = np.array([x + w/2, y + h/2])  # Center

        # Store in history
        self._size_history.append(size)
        self._aspect_ratio_history.append(aspect_ratio)
        self._position_history.append(position)

        if velocity is not None:
            self._velocity_history.append(velocity)

        # Trim history
        if len(self._size_history) > self.max_history_length:
            self._size_history.pop(0)
            self._aspect_ratio_history.pop(0)
            self._position_history.pop(0)

        if len(self._velocity_history) > self.max_history_length:
            self._velocity_history.pop(0)

        # Update last known features
        self._last_known_size = size
        self._last_known_aspect_ratio = aspect_ratio
        self._last_known_position = position
        if velocity is not None:
            self._last_known_velocity = velocity

    def verify_candidate(
        self,
        candidate_bbox: np.ndarray,
        predicted_position: Optional[np.ndarray] = None,
        predicted_velocity: Optional[np.ndarray] = None
    ) -> float:
        """
        Verify if a candidate detection matches the stored fingerprint.

        Args:
            candidate_bbox: Candidate bounding box [x, y, w, h]
            predicted_position: Optional predicted position [x, y]
            predicted_velocity: Optional predicted velocity [vx, vy]

        Returns:
            Confidence score (0-1, higher is better match)
        """
        if len(self._size_history) == 0:
            # No history, cannot verify
            return 0.5

        x, y, w, h = candidate_bbox
        candidate_size = w * h
        candidate_aspect_ratio = w / h if h > 0 else 1.0
        candidate_position = np.array([x + w/2, y + h/2])

        scores = []

        # 1. Size similarity
        avg_size = np.mean(self._size_history)
        if avg_size > 0:
            size_diff = abs(candidate_size - avg_size) / avg_size
            size_score = max(0, 1 - size_diff / self.size_tolerance)
            scores.append(size_score)

        # 2. Aspect ratio similarity
        avg_aspect_ratio = np.mean(self._aspect_ratio_history)
        aspect_diff = abs(candidate_aspect_ratio - avg_aspect_ratio)
        aspect_score = max(0, 1 - aspect_diff / 0.5)  # Tolerance 0.5
        scores.append(aspect_score)

        # 3. Position continuity
        if predicted_position is not None:
            position_diff = np.linalg.norm(candidate_position - predicted_position)
            position_score = max(0, 1 - position_diff / self.position_tolerance)
            scores.append(position_score * 1.5)  # Higher weight
        elif self._last_known_position is not None:
            position_diff = np.linalg.norm(candidate_position - self._last_known_position)
            position_score = max(0, 1 - position_diff / self.position_tolerance)
            scores.append(position_score)

        # 4. Velocity consistency
        if predicted_velocity is not None and len(self._velocity_history) > 0:
            avg_velocity = np.mean(self._velocity_history, axis=0)
            # Compute candidate velocity from position change
            if self._last_known_position is not None:
                candidate_velocity = candidate_position - self._last_known_position
                velocity_diff = np.linalg.norm(candidate_velocity - predicted_velocity)
                velocity_score = max(0, 1 - velocity_diff / self.velocity_tolerance)
                scores.append(velocity_score)

        # Compute overall confidence
        if len(scores) == 0:
            return 0.5

        confidence = np.mean(scores)
        return min(1.0, max(0.0, confidence))

    def get_expected_features(self) -> dict:
        """
        Get expected features for the target.

        Returns:
            Dictionary with expected size, aspect ratio, position, velocity
        """
        return {
            'size': np.mean(self._size_history) if self._size_history else None,
            'aspect_ratio': np.mean(self._aspect_ratio_history) if self._aspect_ratio_history else None,
            'position': self._last_known_position,
            'velocity': self._last_known_velocity
        }

    def reset(self):
        """Reset all stored features."""
        self._size_history = []
        self._aspect_ratio_history = []
        self._velocity_history = []
        self._position_history = []
        self._last_known_size = None
        self._last_known_aspect_ratio = None
        self._last_known_velocity = None
        self._last_known_position = None


class ReIDManager:
    """
    Manages re-identification of targets after occlusion.

    Maintains fingerprints for active tracks and provides matching scores
    for candidate detections.
    """

    def __init__(self, reid_threshold: float = 0.6):
        """
        Args:
            reid_threshold: Minimum confidence for positive re-identification
        """
        self.reid_threshold = reid_threshold
        self._fingerprints: dict = {}  # track_id -> TargetFingerprint

    def register_track(self, track_id: int) -> TargetFingerprint:
        """
        Register a new track for re-identification.

        Args:
            track_id: Track ID

        Returns:
            TargetFingerprint instance for this track
        """
        fingerprint = TargetFingerprint()
        self._fingerprints[track_id] = fingerprint
        return fingerprint

    def update_track_features(
        self,
        track_id: int,
        bbox: np.ndarray,
        velocity: Optional[np.ndarray] = None
    ):
        """
        Update features for a track.

        Args:
            track_id: Track ID
            bbox: Bounding box [x, y, w, h]
            velocity: Optional velocity [vx, vy]
        """
        if track_id not in self._fingerprints:
            self.register_track(track_id)

        self._fingerprints[track_id].store_appearance_features(bbox, velocity)

    def find_best_match(
        self,
        candidate_bbox: np.ndarray,
        track_predictions: dict
    ) -> tuple:
        """
        Find the best matching track for a candidate detection.

        Args:
            candidate_bbox: Candidate bounding box [x, y, w, h]
            track_predictions: Dict of {track_id: {'position': [...], 'velocity': [...]}}

        Returns:
            (best_track_id, confidence) or (None, 0.0) if no good match
        """
        best_track_id = None
        best_confidence = 0.0

        for track_id, fingerprint in self._fingerprints.items():
            predicted_position = track_predictions.get(track_id, {}).get('position')
            predicted_velocity = track_predictions.get(track_id, {}).get('velocity')

            confidence = fingerprint.verify_candidate(
                candidate_bbox,
                predicted_position,
                predicted_velocity
            )

            if confidence > best_confidence:
                best_confidence = confidence
                best_track_id = track_id

        # Only return match if confidence exceeds threshold
        if best_confidence >= self.reid_threshold:
            return best_track_id, best_confidence
        else:
            return None, 0.0

    def remove_track(self, track_id: int):
        """Remove a track from re-identification."""
        if track_id in self._fingerprints:
            del self._fingerprints[track_id]

    def get_fingerprint(self, track_id: int) -> Optional[TargetFingerprint]:
        """Get fingerprint for a track."""
        return self._fingerprints.get(track_id)

    def clear_all(self):
        """Clear all fingerprints."""
        self._fingerprints = {}
