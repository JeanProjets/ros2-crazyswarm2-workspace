"""
Detection communication and sharing across swarm.

Implements detection broadcasting, fusion from multiple drones,
and target tracking with Kalman filtering.
"""
import numpy as np
import logging
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, asdict
import time
import json
import base64

from .target_detector import Detection


@dataclass
class DetectionMessage:
    """
    Detection message for swarm communication.

    Attributes:
        drone_id: ID of detecting drone
        timestamp: Detection timestamp
        target_position: (x, y, z) in world frame
        confidence: Detection confidence
        target_type: Classification result
        detection_image: Optional base64 encoded image
    """
    drone_id: str
    timestamp: float
    target_position: Tuple[float, float, float]
    confidence: float
    target_type: str
    detection_image: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'DetectionMessage':
        """Create from dictionary."""
        return cls(**data)


class TargetTracker:
    """
    Track target over time using Kalman filtering.

    Maintains target state estimate with prediction and update.
    """

    def __init__(self, drone_id: str):
        """
        Initialize target tracker.

        Args:
            drone_id: Unique identifier for the drone
        """
        self.drone_id = drone_id
        self.logger = logging.getLogger(f"TargetTracker_{drone_id}")

        # Track state: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)
        self.covariance = np.eye(6) * 1.0

        # Kalman filter parameters
        self.process_noise = 0.1
        self.measurement_noise = 0.5

        # Tracking state
        self.is_initialized = False
        self.last_update_time = 0.0
        self.track_confidence = 0.0
        self.missed_updates = 0
        self.max_missed_updates = 5

        self.logger.info(f"Target tracker initialized for {drone_id}")

    def initialize_track(self, first_detection: Tuple[float, float, float]) -> None:
        """
        Initialize track with first detection.

        Args:
            first_detection: (x, y, z) position
        """
        x, y, z = first_detection

        # Initialize state [x, y, z, vx, vy, vz]
        self.state = np.array([x, y, z, 0.0, 0.0, 0.0])

        # Initialize covariance
        self.covariance = np.eye(6)
        self.covariance[0:3, 0:3] *= 0.5  # Position uncertainty
        self.covariance[3:6, 3:6] *= 1.0  # Velocity uncertainty

        self.is_initialized = True
        self.last_update_time = time.time()
        self.track_confidence = 0.5
        self.missed_updates = 0

        self.logger.info(f"Track initialized at {first_detection}")

    def update_track(self, new_detection: Tuple[float, float, float]) -> None:
        """
        Update track with new detection (Kalman update).

        Args:
            new_detection: (x, y, z) position
        """
        if not self.is_initialized:
            self.initialize_track(new_detection)
            return

        current_time = time.time()
        dt = current_time - self.last_update_time

        # Prediction step
        self._predict(dt)

        # Update step
        measurement = np.array(new_detection)
        self._update(measurement)

        self.last_update_time = current_time
        self.missed_updates = 0
        self.track_confidence = min(1.0, self.track_confidence + 0.1)

        self.logger.debug(f"Track updated: {self.get_position()}")

    def predict_position(self, time_delta: float) -> Tuple[float, float, float]:
        """
        Predict position after time delta.

        Args:
            time_delta: Time into future (seconds)

        Returns:
            Predicted (x, y, z) position
        """
        if not self.is_initialized:
            return (0.0, 0.0, 0.0)

        # Simple linear prediction
        x = self.state[0] + self.state[3] * time_delta
        y = self.state[1] + self.state[4] * time_delta
        z = self.state[2] + self.state[5] * time_delta

        return (x, y, z)

    def get_position(self) -> Tuple[float, float, float]:
        """Get current tracked position."""
        return (self.state[0], self.state[1], self.state[2])

    def get_velocity(self) -> Tuple[float, float, float]:
        """Get current tracked velocity."""
        return (self.state[3], self.state[4], self.state[5])

    def get_track_confidence(self) -> float:
        """Get track confidence score."""
        return self.track_confidence

    def is_track_lost(self) -> bool:
        """Check if track is lost."""
        return self.missed_updates >= self.max_missed_updates

    def mark_missed_update(self) -> None:
        """Mark a missed detection update."""
        self.missed_updates += 1
        self.track_confidence = max(0.0, self.track_confidence - 0.2)

        if self.is_track_lost():
            self.logger.warning("Track lost!")

    def _predict(self, dt: float) -> None:
        """
        Kalman prediction step.

        Args:
            dt: Time delta since last update
        """
        # State transition matrix (constant velocity model)
        F = np.eye(6)
        F[0:3, 3:6] = np.eye(3) * dt

        # Predict state
        self.state = F @ self.state

        # Predict covariance
        Q = np.eye(6) * self.process_noise
        Q[3:6, 3:6] *= dt  # Velocity noise scales with time
        self.covariance = F @ self.covariance @ F.T + Q

    def _update(self, measurement: np.ndarray) -> None:
        """
        Kalman update step.

        Args:
            measurement: Measurement vector [x, y, z]
        """
        # Measurement matrix (we only measure position)
        H = np.zeros((3, 6))
        H[0:3, 0:3] = np.eye(3)

        # Measurement noise
        R = np.eye(3) * self.measurement_noise

        # Innovation
        y = measurement - H @ self.state

        # Innovation covariance
        S = H @ self.covariance @ H.T + R

        # Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # Update state
        self.state = self.state + K @ y

        # Update covariance
        I = np.eye(6)
        self.covariance = (I - K @ H) @ self.covariance


class DetectionBroadcaster:
    """
    Broadcast and receive detection messages across swarm.

    Handles multi-drone detection fusion and consensus.
    """

    def __init__(self, drone_id: str):
        """
        Initialize detection broadcaster.

        Args:
            drone_id: Unique identifier for the drone
        """
        self.drone_id = drone_id
        self.logger = logging.getLogger(f"DetectionBroadcaster_{drone_id}")

        # Communication state
        self.topic_name = "/swarm/target_detection"
        self.message_rate = 10  # Hz

        # Detection storage
        self.received_detections: Dict[str, DetectionMessage] = {}
        self.last_broadcast_time = 0.0

        # Target tracker
        self.tracker = TargetTracker(drone_id)

        # Fusion parameters
        self.outlier_threshold = 2.0  # sigma

        self.logger.info(f"Detection broadcaster initialized for {drone_id}")

    def broadcast_detection(
        self,
        detection_position: Tuple[float, float, float],
        confidence: float,
        target_type: str = 'hostile',
        image_data: Optional[np.ndarray] = None
    ) -> bool:
        """
        Broadcast detection to swarm.

        Args:
            detection_position: (x, y, z) in world frame
            confidence: Detection confidence
            target_type: Target classification
            image_data: Optional detection image

        Returns:
            True if broadcast successful
        """
        try:
            # Rate limiting
            current_time = time.time()
            min_interval = 1.0 / self.message_rate

            if current_time - self.last_broadcast_time < min_interval:
                return False

            # Encode image if provided
            image_b64 = None
            if image_data is not None:
                image_b64 = self._encode_image(image_data)

            # Create message
            message = DetectionMessage(
                drone_id=self.drone_id,
                timestamp=current_time,
                target_position=detection_position,
                confidence=confidence,
                target_type=target_type,
                detection_image=image_b64
            )

            # Mock ROS2 publish
            self._mock_publish(message)

            self.last_broadcast_time = current_time
            self.logger.info(
                f"Broadcasted detection: {detection_position}, "
                f"confidence={confidence:.2f}"
            )

            return True

        except Exception as e:
            self.logger.error(f"Broadcast failed: {e}")
            return False

    def receive_detection(self, msg_data: dict) -> Optional[DetectionMessage]:
        """
        Receive and parse detection message.

        Args:
            msg_data: Raw message data

        Returns:
            Parsed DetectionMessage or None
        """
        try:
            message = DetectionMessage.from_dict(msg_data)

            # Store detection
            self.received_detections[message.drone_id] = message

            self.logger.debug(
                f"Received detection from {message.drone_id}: "
                f"{message.target_position}"
            )

            return message

        except Exception as e:
            self.logger.error(f"Message parsing failed: {e}")
            return None

    def merge_detections(
        self,
        detections: List[DetectionMessage]
    ) -> Optional[Tuple[float, float, float]]:
        """
        Merge multiple detections into consolidated estimate.

        Uses weighted average based on confidence with outlier rejection.

        Args:
            detections: List of detection messages

        Returns:
            Merged position or None
        """
        if not detections:
            return None

        try:
            # Extract positions and confidences
            positions = np.array([d.target_position for d in detections])
            confidences = np.array([d.confidence for d in detections])

            # Remove outliers (simple median-based approach)
            median_pos = np.median(positions, axis=0)
            distances = np.linalg.norm(positions - median_pos, axis=1)
            median_dist = np.median(distances)

            # Keep inliers
            inlier_mask = distances < (median_dist + self.outlier_threshold)
            inlier_positions = positions[inlier_mask]
            inlier_confidences = confidences[inlier_mask]

            if len(inlier_positions) == 0:
                self.logger.warning("All detections rejected as outliers")
                return None

            # Weighted average
            total_weight = np.sum(inlier_confidences)
            if total_weight == 0:
                weights = np.ones(len(inlier_confidences)) / len(inlier_confidences)
            else:
                weights = inlier_confidences / total_weight

            merged_pos = np.average(inlier_positions, axis=0, weights=weights)

            self.logger.info(
                f"Merged {len(inlier_positions)}/{len(detections)} detections"
            )

            return tuple(merged_pos)

        except Exception as e:
            self.logger.error(f"Detection merging failed: {e}")
            return None

    def prioritize_detections(
        self,
        detections: List[DetectionMessage]
    ) -> Optional[DetectionMessage]:
        """
        Select best detection from multiple sources.

        Prioritizes based on confidence and recency.

        Args:
            detections: List of detection messages

        Returns:
            Best detection or None
        """
        if not detections:
            return None

        current_time = time.time()

        # Score each detection
        best_score = -1.0
        best_detection = None

        for det in detections:
            # Confidence score
            conf_score = det.confidence

            # Recency score (exponential decay)
            age = current_time - det.timestamp
            recency_score = np.exp(-age / 2.0)  # 2 second half-life

            # Combined score
            score = conf_score * 0.7 + recency_score * 0.3

            if score > best_score:
                best_score = score
                best_detection = det

        return best_detection

    def update_tracker_with_detections(
        self,
        detections: List[DetectionMessage]
    ) -> bool:
        """
        Update internal tracker with detection messages.

        Args:
            detections: List of detection messages

        Returns:
            True if tracker updated
        """
        if not detections:
            self.tracker.mark_missed_update()
            return False

        # Merge detections
        merged_pos = self.merge_detections(detections)

        if merged_pos is None:
            self.tracker.mark_missed_update()
            return False

        # Update tracker
        self.tracker.update_track(merged_pos)

        return True

    def get_current_target_estimate(self) -> Optional[Dict]:
        """
        Get current best estimate of target.

        Returns:
            Dictionary with target state or None
        """
        if not self.tracker.is_initialized:
            return None

        if self.tracker.is_track_lost():
            self.logger.warning("Track is lost")
            return None

        position = self.tracker.get_position()
        velocity = self.tracker.get_velocity()
        confidence = self.tracker.get_track_confidence()

        return {
            'position': position,
            'velocity': velocity,
            'confidence': confidence,
            'timestamp': time.time()
        }

    def _encode_image(self, image: np.ndarray) -> str:
        """
        Encode image to base64 string.

        Args:
            image: Image array

        Returns:
            Base64 encoded string
        """
        import io
        from PIL import Image

        # Convert to PIL Image
        if len(image.shape) == 2:  # Grayscale
            img = Image.fromarray(image, mode='L')
        else:
            img = Image.fromarray(image)

        # Encode to bytes
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()

        # Encode to base64
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')

        return img_b64

    def _mock_publish(self, message: DetectionMessage) -> None:
        """
        Mock ROS2 publish (for testing without ROS2).

        Args:
            message: Detection message to publish
        """
        # In real implementation, this would publish to ROS2 topic
        # For now, just log
        self.logger.debug(f"Publishing to {self.topic_name}: {message.to_dict()}")

    def get_all_detections(self) -> List[DetectionMessage]:
        """Get all received detections."""
        return list(self.received_detections.values())

    def clear_old_detections(self, max_age: float = 5.0) -> None:
        """
        Clear detections older than max_age.

        Args:
            max_age: Maximum age in seconds
        """
        current_time = time.time()
        to_remove = []

        for drone_id, detection in self.received_detections.items():
            if current_time - detection.timestamp > max_age:
                to_remove.append(drone_id)

        for drone_id in to_remove:
            del self.received_detections[drone_id]
            self.logger.debug(f"Cleared old detection from {drone_id}")
