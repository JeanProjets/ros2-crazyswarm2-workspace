"""
Visual positioning and odometry for target localization.

Provides visual position estimation to augment OptiTrack data
with camera-based relative positioning and tracking.
"""
import numpy as np
import logging
from typing import Tuple, Optional, List
from dataclasses import dataclass
import json
import os

from .target_detector import Detection


@dataclass
class CameraCalibrationData:
    """Camera calibration parameters."""
    focal_length_px: float = 120.0
    principal_point: Tuple[float, float] = (80.0, 60.0)  # Image center
    fov_horizontal: float = 60.0  # degrees
    fov_vertical: float = 45.0    # degrees
    distortion_coeffs: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)


class CameraCalibration:
    """
    Camera calibration and coordinate transformation.

    Handles pixel-to-world transformations and calibration data.
    """

    def __init__(self, drone_id: str):
        """
        Initialize camera calibration.

        Args:
            drone_id: Unique identifier for the drone
        """
        self.drone_id = drone_id
        self.logger = logging.getLogger(f"CameraCalib_{drone_id}")

        # Load default calibration
        self.calib = CameraCalibrationData()

        self.logger.info(f"Camera calibration initialized for {drone_id}")

    def load_calibration(self, file_path: str) -> bool:
        """
        Load calibration from file.

        Args:
            file_path: Path to calibration JSON file

        Returns:
            True if loaded successfully
        """
        try:
            if not os.path.exists(file_path):
                self.logger.warning(f"Calibration file not found: {file_path}")
                return False

            with open(file_path, 'r') as f:
                data = json.load(f)

            self.calib.focal_length_px = data.get('focal_length', 120.0)
            self.calib.principal_point = tuple(data.get('principal_point', [80.0, 60.0]))
            self.calib.fov_horizontal = data.get('fov_horizontal', 60.0)
            self.calib.fov_vertical = data.get('fov_vertical', 45.0)

            self.logger.info("Calibration loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Calibration load failed: {e}")
            return False

    def pixel_to_world(
        self,
        pixel_coords: Tuple[int, int],
        depth: float
    ) -> Tuple[float, float, float]:
        """
        Convert pixel coordinates to world coordinates.

        Args:
            pixel_coords: (x, y) in pixel coordinates
            depth: Distance to object in meters

        Returns:
            (x, y, z) in camera frame coordinates (meters)
        """
        px, py = pixel_coords
        cx, cy = self.calib.principal_point
        f = self.calib.focal_length_px

        # Pinhole camera model
        x = (px - cx) * depth / f
        y = (py - cy) * depth / f
        z = depth

        return (x, y, z)

    def world_to_pixel(
        self,
        world_coords: Tuple[float, float, float]
    ) -> Tuple[int, int]:
        """
        Convert world coordinates to pixel coordinates.

        Args:
            world_coords: (x, y, z) in camera frame (meters)

        Returns:
            (x, y) in pixel coordinates
        """
        x, y, z = world_coords

        if z <= 0:
            return (0, 0)

        cx, cy = self.calib.principal_point
        f = self.calib.focal_length_px

        # Project to image plane
        px = int(f * x / z + cx)
        py = int(f * y / z + cy)

        return (px, py)

    def calculate_drone_bearing(
        self,
        bbox_center: Tuple[int, int]
    ) -> float:
        """
        Calculate bearing angle to drone from camera center.

        Args:
            bbox_center: (x, y) center of detection bbox

        Returns:
            Bearing angle in degrees
        """
        cx, cy = self.calib.principal_point
        px, py = bbox_center

        # Calculate angle from image center
        delta_x = px - cx
        focal_length = self.calib.focal_length_px

        # Bearing angle (horizontal)
        bearing = np.arctan2(delta_x, focal_length)
        bearing_deg = np.degrees(bearing)

        return bearing_deg

    def get_calibration_data(self) -> CameraCalibrationData:
        """Get current calibration data."""
        return self.calib


class VisualPositionEstimator:
    """
    Estimate target position using visual detection.

    Combines detection data with drone pose to estimate target
    position in world coordinates.
    """

    def __init__(self, drone_id: str):
        """
        Initialize visual position estimator.

        Args:
            drone_id: Unique identifier for the drone
        """
        self.drone_id = drone_id
        self.logger = logging.getLogger(f"VisualPosEst_{drone_id}")

        # Camera calibration
        self.calibration = CameraCalibration(drone_id)

        # Position estimation history
        self.position_history: List[Tuple[float, float, float]] = []
        self.max_history = 30

        # Validation threshold
        self.optitrack_validation_threshold = 0.5  # meters

        self.logger.info(f"Visual position estimator initialized for {drone_id}")

    def estimate_relative_position(
        self,
        target_bbox: Tuple[int, int, int, int],
        camera_params: dict,
        estimated_distance: float
    ) -> Tuple[float, float, float]:
        """
        Estimate relative position of target in camera frame.

        Args:
            target_bbox: Bounding box (x, y, width, height)
            camera_params: Camera calibration parameters
            estimated_distance: Distance estimate from detection

        Returns:
            (x, y, z) relative position in camera frame (meters)
        """
        try:
            # Calculate bbox center
            x, y, w, h = target_bbox
            center_x = x + w / 2
            center_y = y + h / 2

            # Convert to camera frame coordinates
            rel_x, rel_y, rel_z = self.calibration.pixel_to_world(
                (center_x, center_y),
                estimated_distance
            )

            return (rel_x, rel_y, rel_z)

        except Exception as e:
            self.logger.error(f"Relative position estimation failed: {e}")
            return (0.0, 0.0, 0.0)

    def calculate_approach_vector(
        self,
        current_pos: Tuple[float, float, float],
        target_pos: Tuple[float, float, float]
    ) -> np.ndarray:
        """
        Calculate approach vector from current position to target.

        Args:
            current_pos: Current drone position (x, y, z)
            target_pos: Target position (x, y, z)

        Returns:
            Normalized approach vector
        """
        current = np.array(current_pos)
        target = np.array(target_pos)

        # Calculate vector
        vector = target - current

        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def estimate_target_velocity(
        self,
        detections_history: List[Detection]
    ) -> Tuple[float, float, float]:
        """
        Estimate target velocity from detection history.

        Args:
            detections_history: List of recent detections

        Returns:
            (vx, vy, vz) velocity in m/s
        """
        if len(detections_history) < 2:
            return (0.0, 0.0, 0.0)

        try:
            # Use last two detections
            det1 = detections_history[-2]
            det2 = detections_history[-1]

            # Time delta
            dt = det2.timestamp - det1.timestamp
            if dt <= 0:
                return (0.0, 0.0, 0.0)

            # Position change (simplified - assumes camera hasn't moved much)
            # In real implementation, would transform to world frame first
            dx = (det2.bbox[0] - det1.bbox[0]) / 100.0  # Rough pixel to meter
            dy = (det2.bbox[1] - det1.bbox[1]) / 100.0
            dz = det2.estimated_distance - det1.estimated_distance

            # Velocity
            vx = dx / dt
            vy = dy / dt
            vz = dz / dt

            return (vx, vy, vz)

        except Exception as e:
            self.logger.error(f"Velocity estimation failed: {e}")
            return (0.0, 0.0, 0.0)

    def validate_with_optitrack(
        self,
        visual_pos: Tuple[float, float, float],
        optitrack_pos: Tuple[float, float, float]
    ) -> bool:
        """
        Validate visual position estimate with OptiTrack data.

        Args:
            visual_pos: Position from visual estimation
            optitrack_pos: Position from OptiTrack

        Returns:
            True if positions agree within threshold
        """
        visual = np.array(visual_pos)
        opti = np.array(optitrack_pos)

        # Calculate distance between estimates
        distance = np.linalg.norm(visual - opti)

        if distance > self.optitrack_validation_threshold:
            self.logger.warning(
                f"Vision/OptiTrack disagreement: {distance:.2f}m "
                f"(threshold: {self.optitrack_validation_threshold}m)"
            )
            return False

        return True

    def transform_to_world_frame(
        self,
        camera_frame_pos: Tuple[float, float, float],
        drone_world_pos: Tuple[float, float, float],
        drone_yaw: float
    ) -> Tuple[float, float, float]:
        """
        Transform position from camera frame to world frame.

        Args:
            camera_frame_pos: Position in camera frame
            drone_world_pos: Drone position in world frame
            drone_yaw: Drone yaw angle in degrees

        Returns:
            Position in world frame
        """
        # Camera position relative to drone body
        # Assuming camera is facing forward (0° yaw relative to body)

        cx, cy, cz = camera_frame_pos
        dx, dy, dz = drone_world_pos

        # Rotate by drone yaw
        yaw_rad = np.radians(drone_yaw)
        rotation_matrix = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad),  np.cos(yaw_rad), 0],
            [0,                0,                1]
        ])

        camera_vec = np.array([cx, cy, cz])
        rotated_vec = rotation_matrix @ camera_vec

        # Translate to world frame
        world_x = dx + rotated_vec[0]
        world_y = dy + rotated_vec[1]
        world_z = dz + rotated_vec[2]

        return (world_x, world_y, world_z)

    def estimate_world_position(
        self,
        detection: Detection,
        drone_world_pos: Tuple[float, float, float],
        drone_yaw: float = 0.0
    ) -> Tuple[float, float, float]:
        """
        Estimate target position in world coordinates.

        Combines detection data with drone pose for full position estimate.

        Args:
            detection: Target detection
            drone_world_pos: Drone position in world frame
            drone_yaw: Drone yaw angle in degrees

        Returns:
            Target position in world frame
        """
        # Get camera parameters
        camera_params = self.calibration.get_calibration_data()

        # Estimate relative position
        rel_pos = self.estimate_relative_position(
            detection.bbox,
            camera_params.__dict__,
            detection.estimated_distance
        )

        # Transform to world frame
        world_pos = self.transform_to_world_frame(
            rel_pos,
            drone_world_pos,
            drone_yaw
        )

        # Add to history
        self.position_history.append(world_pos)
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)

        return world_pos

    def get_smoothed_position(self, window_size: int = 5) -> Optional[Tuple[float, float, float]]:
        """
        Get smoothed position estimate using moving average.

        Args:
            window_size: Number of recent positions to average

        Returns:
            Smoothed position or None if insufficient data
        """
        if len(self.position_history) < window_size:
            return None

        recent = self.position_history[-window_size:]
        avg_x = np.mean([p[0] for p in recent])
        avg_y = np.mean([p[1] for p in recent])
        avg_z = np.mean([p[2] for p in recent])

        return (avg_x, avg_y, avg_z)
