"""
Map-Vision Fusion Module for Scenario 4.

This module uses known obstacle locations to predict whether a target
should be visible from the drone's current pose. This enables intelligent
distinction between "target lost" vs "expected occlusion".
"""

from typing import List, Tuple, Optional
import numpy as np


class Obstacle:
    """Simple obstacle representation."""

    def __init__(self, position: np.ndarray, size: np.ndarray, height: float = 1.0):
        """
        Args:
            position: [x, y] center position in world frame
            size: [width, depth] of obstacle
            height: Height of obstacle
        """
        self.position = position
        self.size = size
        self.height = height

    def get_corners_2d(self) -> np.ndarray:
        """Get 2D corners of obstacle footprint."""
        x, y = self.position
        w, d = self.size

        corners = np.array([
            [x - w/2, y - d/2],
            [x + w/2, y - d/2],
            [x + w/2, y + d/2],
            [x - w/2, y + d/2]
        ])
        return corners

    def contains_point_2d(self, point: np.ndarray) -> bool:
        """Check if 2D point is inside obstacle footprint."""
        x, y = point
        ox, oy = self.position
        w, d = self.size

        return (abs(x - ox) <= w/2) and (abs(y - oy) <= d/2)


class CameraPose:
    """Camera/drone pose in 3D space."""

    def __init__(
        self,
        position: np.ndarray,
        rotation: np.ndarray = None,
        fov: float = 60.0
    ):
        """
        Args:
            position: [x, y, z] position in world frame
            rotation: Rotation matrix (3x3) or None for identity
            fov: Field of view in degrees
        """
        self.position = position
        self.rotation = rotation if rotation is not None else np.eye(3)
        self.fov = fov


class MapProjector:
    """
    Projects obstacles from world frame to camera view to predict visibility.

    Uses the known map (from Agent 1) to answer:
    "Should I be seeing the target right now?"
    """

    def __init__(
        self,
        image_width: int = 640,
        image_height: int = 480,
        focal_length: float = 500.0
    ):
        """
        Args:
            image_width: Camera image width in pixels
            image_height: Camera image height in pixels
            focal_length: Camera focal length in pixels
        """
        self.image_width = image_width
        self.image_height = image_height
        self.focal_length = focal_length

        # Camera intrinsic matrix
        self.K = np.array([
            [focal_length, 0, image_width / 2],
            [0, focal_length, image_height / 2],
            [0, 0, 1]
        ])

    def project_point_to_camera(
        self,
        point_3d: np.ndarray,
        camera_pose: CameraPose
    ) -> Tuple[np.ndarray, float]:
        """
        Project a 3D point to camera image plane.

        Args:
            point_3d: [x, y, z] in world frame
            camera_pose: Camera pose

        Returns:
            (pixel_coords, depth) where pixel_coords is [u, v]
        """
        # Transform point to camera frame
        point_cam = camera_pose.rotation.T @ (point_3d - camera_pose.position)

        # Check if point is behind camera
        if point_cam[2] <= 0:
            return np.array([-1, -1]), point_cam[2]

        # Project to image plane
        point_2d_hom = self.K @ point_cam
        pixel = point_2d_hom[:2] / point_2d_hom[2]

        return pixel, point_cam[2]

    def project_obstacles_to_camera(
        self,
        drone_pose: CameraPose,
        obstacles: List[Obstacle]
    ) -> List[np.ndarray]:
        """
        Project obstacle corners to camera view and create masks.

        Args:
            drone_pose: Current drone/camera pose
            obstacles: List of obstacles

        Returns:
            List of projected polygon masks (Nx2 arrays of pixel coordinates)
        """
        masks = []

        for obstacle in obstacles:
            corners_2d = obstacle.get_corners_2d()

            # Add height dimension (project top and bottom of obstacle)
            corners_3d_bottom = np.hstack([corners_2d, np.zeros((4, 1))])
            corners_3d_top = np.hstack([corners_2d, np.ones((4, 1)) * obstacle.height])

            projected_corners = []

            # Project all corners
            for corner in np.vstack([corners_3d_bottom, corners_3d_top]):
                pixel, depth = self.project_point_to_camera(corner, drone_pose)

                # Only keep points in front of camera and in image bounds
                if depth > 0 and 0 <= pixel[0] < self.image_width and 0 <= pixel[1] < self.image_height:
                    projected_corners.append(pixel)

            if len(projected_corners) > 0:
                masks.append(np.array(projected_corners))

        return masks

    def is_line_of_sight_clear(
        self,
        start: np.ndarray,
        end: np.ndarray,
        obstacles: List[Obstacle],
        num_samples: int = 20
    ) -> bool:
        """
        Check if line of sight between two points is clear of obstacles.

        Args:
            start: Start point [x, y, z]
            end: End point [x, y, z]
            obstacles: List of obstacles
            num_samples: Number of points to sample along ray

        Returns:
            True if line of sight is clear
        """
        # Sample points along the ray
        for i in range(num_samples):
            t = i / (num_samples - 1)
            point = start + t * (end - start)

            # Check if point is inside any obstacle (2D check at z=0 plane)
            point_2d = point[:2]

            for obstacle in obstacles:
                if obstacle.contains_point_2d(point_2d):
                    # Check height
                    if point[2] <= obstacle.height:
                        return False  # Blocked

        return True

    def is_target_expected_visible(
        self,
        target_position: np.ndarray,
        drone_pose: CameraPose,
        obstacles: List[Obstacle]
    ) -> bool:
        """
        Determine if target should be visible from drone's current pose.

        Args:
            target_position: Target position [x, y, z]
            drone_pose: Current drone pose
            obstacles: List of obstacles

        Returns:
            True if target should be visible (not occluded by obstacles)
        """
        # Check if line of sight is clear
        return self.is_line_of_sight_clear(
            drone_pose.position,
            target_position,
            obstacles
        )

    def is_area_occluded(
        self,
        position_2d: np.ndarray,
        drone_pose: Optional[CameraPose] = None,
        obstacles: Optional[List[Obstacle]] = None
    ) -> bool:
        """
        Check if a 2D position is currently in an occluded region.

        This is a simplified version for use by the tracker when detailed
        geometry is not available.

        Args:
            position_2d: [x, y] position to check
            drone_pose: Optional drone pose
            obstacles: Optional list of obstacles

        Returns:
            True if position is likely occluded
        """
        if obstacles is None or drone_pose is None:
            # No map information available, assume not occluded
            return False

        # Extend 2D position to 3D (assume z=0 ground level)
        position_3d = np.array([position_2d[0], position_2d[1], 0.0])

        # Check if visible
        return not self.is_target_expected_visible(position_3d, drone_pose, obstacles)

    def classify_detection_state(
        self,
        vision_detects_target: bool,
        target_predicted_position: np.ndarray,
        drone_pose: CameraPose,
        obstacles: List[Obstacle]
    ) -> str:
        """
        Classify the detection state based on vision and map.

        Cases:
        - Vision: No Target, Map: Open Space -> TRUE_NEGATIVE (target lost)
        - Vision: No Target, Map: Blocked -> EXPECTED_OCCLUSION (keep tracking)
        - Vision: Target, Map: Open Space -> TRUE_POSITIVE (good detection)
        - Vision: Target, Map: Blocked -> FALSE_POSITIVE (unlikely, might be noise)

        Args:
            vision_detects_target: Whether vision system detected target
            target_predicted_position: Where we predict target to be
            drone_pose: Current drone pose
            obstacles: List of obstacles

        Returns:
            Classification string
        """
        expected_visible = self.is_target_expected_visible(
            target_predicted_position,
            drone_pose,
            obstacles
        )

        if not vision_detects_target and expected_visible:
            return "TRUE_NEGATIVE"  # Target is gone
        elif not vision_detects_target and not expected_visible:
            return "EXPECTED_OCCLUSION"  # Target hidden by wall
        elif vision_detects_target and expected_visible:
            return "TRUE_POSITIVE"  # Good detection
        else:
            return "FALSE_POSITIVE"  # Detected but should be blocked (noise?)

    def get_occlusion_mask(
        self,
        drone_pose: CameraPose,
        obstacles: List[Obstacle]
    ) -> np.ndarray:
        """
        Generate a binary mask of occluded regions in the camera view.

        Args:
            drone_pose: Current drone pose
            obstacles: List of obstacles

        Returns:
            Binary mask (H x W) where 1 = occluded
        """
        mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)

        # Project obstacles and fill masks
        projected_masks = self.project_obstacles_to_camera(drone_pose, obstacles)

        for polygon in projected_masks:
            if len(polygon) >= 3:
                # Fill polygon (simple rasterization)
                # For production, use cv2.fillPoly
                # Here we do a simple bounding box fill
                min_x = int(max(0, np.min(polygon[:, 0])))
                max_x = int(min(self.image_width - 1, np.max(polygon[:, 0])))
                min_y = int(max(0, np.min(polygon[:, 1])))
                max_y = int(min(self.image_height - 1, np.max(polygon[:, 1])))

                mask[min_y:max_y+1, min_x:max_x+1] = 1

        return mask
