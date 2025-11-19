"""
Fallback zone scanner for long-range target detection.

This module implements motion-based scanning for the fallback maneuver when
the target is lost. Uses background subtraction and frame differencing to
detect moving objects at range, which is cheaper than CNN inference.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


@dataclass
class MovingBlob:
    """Detected moving blob in the scene."""
    x: int
    y: int
    width: int
    height: int
    area: int
    centroid: Tuple[float, float]


class ScannerMode(Enum):
    """Scanner operating modes."""
    INACTIVE = "inactive"
    MOTION_DETECTION = "motion_detection"
    CNN_CONFIRMATION = "cnn_confirmation"


class MotionScanner:
    """
    Motion-based scanner for fallback target search.

    When the main tracker loses the target, this scanner uses frame differencing
    and background subtraction to detect moving objects. This is much cheaper
    computationally than running CNN on the full frame.

    Triggered when Mission State = FALLBACK_SCAN. The drone hovers at the 3m line
    and scans for moving pixels that could be the target.
    """

    def __init__(
        self,
        motion_threshold: int = 25,
        min_blob_area: int = 100,
        max_blob_area: int = 10000,
        blur_kernel_size: int = 5,
        morphology_kernel_size: int = 3,
        frame_diff_alpha: float = 0.2,
        background_history: int = 30
    ):
        """
        Initialize motion scanner.

        Args:
            motion_threshold: Threshold for motion detection (0-255)
            min_blob_area: Minimum blob area in pixels to consider
            max_blob_area: Maximum blob area in pixels to consider
            blur_kernel_size: Gaussian blur kernel size for noise reduction
            morphology_kernel_size: Kernel size for morphological operations
            frame_diff_alpha: Learning rate for background model (0-1)
            background_history: Number of frames to build background model
        """
        self.motion_threshold = motion_threshold
        self.min_blob_area = min_blob_area
        self.max_blob_area = max_blob_area
        self.blur_kernel_size = blur_kernel_size
        self.morphology_kernel_size = morphology_kernel_size
        self.frame_diff_alpha = frame_diff_alpha

        # Background model (running average)
        self.background: Optional[np.ndarray] = None
        self.background_initialized = False
        self.frame_count = 0
        self.background_history = background_history

        # Previous frame for frame differencing
        self.prev_frame: Optional[np.ndarray] = None

        # Morphological kernels
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morphology_kernel_size, morphology_kernel_size)
        )

        self.mode = ScannerMode.INACTIVE

    def detect_moving_objects(self, frame: np.ndarray) -> List[MovingBlob]:
        """
        Detect moving objects in the frame.

        Uses both background subtraction and frame differencing for robust
        motion detection. Returns list of potential target blobs.

        Args:
            frame: Input frame (color or grayscale)

        Returns:
            List of detected moving blobs
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(
            gray,
            (self.blur_kernel_size, self.blur_kernel_size),
            0
        )

        self.frame_count += 1

        # Method 1: Background Subtraction
        motion_mask_bg = self._background_subtraction(blurred)

        # Method 2: Frame Differencing
        motion_mask_fd = self._frame_differencing(blurred)

        # Combine both methods (logical OR)
        if motion_mask_fd is not None:
            motion_mask = cv2.bitwise_or(motion_mask_bg, motion_mask_fd)
        else:
            motion_mask = motion_mask_bg

        # Apply morphological operations to clean up mask
        motion_mask = self._apply_morphology(motion_mask)

        # Find contours (blobs) in motion mask
        blobs = self._extract_blobs(motion_mask)

        # Update previous frame for next iteration
        self.prev_frame = blurred.copy()

        return blobs

    def _background_subtraction(self, frame: np.ndarray) -> np.ndarray:
        """
        Perform background subtraction.

        Maintains a running average background model and subtracts it
        from the current frame to detect moving pixels.

        Args:
            frame: Grayscale input frame

        Returns:
            Binary motion mask
        """
        # Initialize background on first frame
        if self.background is None:
            self.background = frame.astype(np.float32)
            return np.zeros(frame.shape, dtype=np.uint8)

        # Update background model (running average)
        if self.frame_count < self.background_history:
            # Still building initial background model
            alpha = 1.0 / (self.frame_count + 1)
        else:
            # Use configured learning rate
            alpha = self.frame_diff_alpha

        cv2.accumulateWeighted(frame, self.background, alpha)

        # Compute absolute difference
        bg_uint8 = self.background.astype(np.uint8)
        diff = cv2.absdiff(frame, bg_uint8)

        # Threshold to get binary motion mask
        _, motion_mask = cv2.threshold(
            diff,
            self.motion_threshold,
            255,
            cv2.THRESH_BINARY
        )

        return motion_mask

    def _frame_differencing(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Perform frame differencing.

        Computes absolute difference between consecutive frames to detect motion.

        Args:
            frame: Grayscale input frame

        Returns:
            Binary motion mask or None if no previous frame
        """
        if self.prev_frame is None:
            return None

        # Compute absolute difference between frames
        diff = cv2.absdiff(frame, self.prev_frame)

        # Threshold to get binary motion mask
        _, motion_mask = cv2.threshold(
            diff,
            self.motion_threshold,
            255,
            cv2.THRESH_BINARY
        )

        return motion_mask

    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up motion mask.

        Removes noise and fills holes in detected regions.

        Args:
            mask: Binary motion mask

        Returns:
            Cleaned motion mask
        """
        # Opening to remove noise (erode then dilate)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)

        # Closing to fill holes (dilate then erode)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)

        return mask

    def _extract_blobs(self, mask: np.ndarray) -> List[MovingBlob]:
        """
        Extract blob information from motion mask.

        Args:
            mask: Binary motion mask

        Returns:
            List of detected moving blobs
        """
        # Find contours
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        blobs = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if area < self.min_blob_area or area > self.max_blob_area:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Get centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx = x + w / 2
                cy = y + h / 2

            blob = MovingBlob(
                x=x,
                y=y,
                width=w,
                height=h,
                area=int(area),
                centroid=(cx, cy)
            )

            blobs.append(blob)

        # Sort by area (largest first)
        blobs.sort(key=lambda b: b.area, reverse=True)

        return blobs

    def get_best_target_candidate(
        self,
        frame: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the most likely target candidate from motion detection.

        This is the main interface for the fallback scan behavior.
        Returns the bounding box of the most likely target.

        Args:
            frame: Input frame

        Returns:
            Bounding box (x, y, w, h) of best candidate or None
        """
        blobs = self.detect_moving_objects(frame)

        if len(blobs) == 0:
            return None

        # Return largest blob (most likely to be target)
        best_blob = blobs[0]
        return (best_blob.x, best_blob.y, best_blob.width, best_blob.height)

    def reset(self):
        """Reset scanner to initial state."""
        self.background = None
        self.background_initialized = False
        self.frame_count = 0
        self.prev_frame = None
        self.mode = ScannerMode.INACTIVE

    def is_ready(self) -> bool:
        """
        Check if background model is ready for reliable detection.

        Returns:
            True if background model is sufficiently trained
        """
        return self.frame_count >= self.background_history

    def visualize_motion_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Create visualization of motion detection for debugging.

        Args:
            frame: Input frame

        Returns:
            Visualization image with motion overlay
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis_frame = frame.copy()
        else:
            gray = frame.copy()
            vis_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Get motion blobs
        blobs = self.detect_moving_objects(frame)

        # Draw bounding boxes
        for blob in blobs:
            cv2.rectangle(
                vis_frame,
                (blob.x, blob.y),
                (blob.x + blob.width, blob.y + blob.height),
                (0, 255, 0),
                2
            )

            # Draw centroid
            cx, cy = blob.centroid
            cv2.circle(vis_frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)

            # Add area label
            cv2.putText(
                vis_frame,
                f"A:{blob.area}",
                (blob.x, blob.y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        return vis_frame
