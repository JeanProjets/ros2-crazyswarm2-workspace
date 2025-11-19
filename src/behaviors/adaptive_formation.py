"""
Adaptive Formation Controller for Scenario 2

This module implements formation logic that dynamically adjusts to cage boundaries,
preventing wall collisions while maintaining formation integrity.
"""

from typing import Tuple
from dataclasses import dataclass


@dataclass
class Position:
    """3D Position representation"""
    x: float
    y: float
    z: float


@dataclass
class Offset:
    """Formation offset representation"""
    dx: float
    dy: float
    dz: float


class AdaptiveFormationController:
    """
    Manages formation offsets that adapt to cage boundaries.

    Critical for Scenario 2 where the target at (9.5, 0.5, 5) is near walls.
    Standard offset (-0.5, -0.5, -0.5) would crash the follower into Y=0 wall.
    """

    def __init__(self,
                 bounds_x_min: float = 0.0,
                 bounds_x_max: float = 10.0,
                 bounds_y_min: float = 0.0,
                 bounds_y_max: float = 6.0,
                 bounds_z_min: float = 0.0,
                 bounds_z_max: float = 10.0):
        """
        Initialize the adaptive formation controller.

        Args:
            bounds_x_min: Minimum X boundary (default: 0.0)
            bounds_x_max: Maximum X boundary (default: 10.0)
            bounds_y_min: Minimum Y boundary (default: 0.0)
            bounds_y_max: Maximum Y boundary (default: 6.0)
            bounds_z_min: Minimum Z boundary (default: 0.0)
            bounds_z_max: Maximum Z boundary (default: 10.0)
        """
        self.bounds_x_min = bounds_x_min
        self.bounds_x_max = bounds_x_max
        self.bounds_y_min = bounds_y_min
        self.bounds_y_max = bounds_y_max
        self.bounds_z_min = bounds_z_min
        self.bounds_z_max = bounds_z_max

        # Standard follower offset (right, back, below)
        self.default_offset = Offset(dx=-0.5, dy=-0.5, dz=-0.5)

        # Safety margin from walls
        self.safety_margin = 0.8  # meters

    def get_safe_offset(self,
                       leader_pos: Position,
                       ideal_offset: Offset = None) -> Offset:
        """
        Calculate a safe formation offset that respects cage boundaries.

        Args:
            leader_pos: Current position of the leader drone
            ideal_offset: Desired offset (defaults to standard offset)

        Returns:
            Offset that keeps follower inside safe zone
        """
        if ideal_offset is None:
            ideal_offset = self.default_offset

        # Start with ideal offset
        safe_offset = Offset(
            dx=ideal_offset.dx,
            dy=ideal_offset.dy,
            dz=ideal_offset.dz
        )

        # Calculate follower position with ideal offset
        follower_y = leader_pos.y + ideal_offset.dy
        follower_x = leader_pos.x + ideal_offset.dx
        follower_z = leader_pos.z + ideal_offset.dz

        # Adapt Y offset to avoid walls
        safe_offset.dy = self._calculate_adaptive_y_offset(leader_pos.y)

        # Adapt X offset if needed
        if follower_x < self.bounds_x_min + self.safety_margin:
            safe_offset.dx = abs(ideal_offset.dx)  # Push to right
        elif follower_x > self.bounds_x_max - self.safety_margin:
            safe_offset.dx = -abs(ideal_offset.dx)  # Push to left

        # Adapt Z offset if needed
        if follower_z < self.bounds_z_min + self.safety_margin:
            safe_offset.dz = abs(ideal_offset.dz)  # Push upward
        elif follower_z > self.bounds_z_max - self.safety_margin:
            safe_offset.dz = -abs(ideal_offset.dz)  # Push downward

        return safe_offset

    def _calculate_adaptive_y_offset(self, leader_y: float) -> float:
        """
        Dynamically flip Y-offset to avoid walls.

        This is the critical algorithm for Scenario 2:
        - When leader is near Y=0 wall (right), place follower on left (+0.5)
        - When leader is near Y=6 wall (left), place follower on right (-0.5)
        - Otherwise use standard offset (-0.5)

        Args:
            leader_y: Leader's Y position

        Returns:
            Safe Y offset value
        """
        base_offset_y = -0.5

        # Check Right Wall (Y=0)
        if leader_y - self.safety_margin < self.bounds_y_min:
            return 0.5  # Force follower to LEFT of leader

        # Check Left Wall (Y=6)
        elif leader_y + self.safety_margin > self.bounds_y_max:
            return -0.5  # Force follower to RIGHT of leader

        # Safe zone - use standard offset
        return base_offset_y

    def invert_formation_if_needed(self, leader_pos: Position) -> Offset:
        """
        Determine if formation inversion is needed based on leader position.

        This is a convenience method that returns the fully adapted offset.

        Args:
            leader_pos: Current position of the leader drone

        Returns:
            Adapted formation offset
        """
        return self.get_safe_offset(leader_pos)

    def validate_follower_position(self, follower_pos: Position) -> Tuple[bool, str]:
        """
        Check if a follower position is safe (within boundaries).

        Args:
            follower_pos: Position to validate

        Returns:
            Tuple of (is_safe, reason_if_unsafe)
        """
        min_clearance = 0.3  # Absolute minimum distance from walls

        if follower_pos.x < self.bounds_x_min + min_clearance:
            return False, f"Too close to X min boundary: {follower_pos.x:.2f}"

        if follower_pos.x > self.bounds_x_max - min_clearance:
            return False, f"Too close to X max boundary: {follower_pos.x:.2f}"

        if follower_pos.y < self.bounds_y_min + min_clearance:
            return False, f"Too close to Y min boundary: {follower_pos.y:.2f}"

        if follower_pos.y > self.bounds_y_max - min_clearance:
            return False, f"Too close to Y max boundary: {follower_pos.y:.2f}"

        if follower_pos.z < self.bounds_z_min + min_clearance:
            return False, f"Too close to Z min boundary: {follower_pos.z:.2f}"

        if follower_pos.z > self.bounds_z_max - min_clearance:
            return False, f"Too close to Z max boundary: {follower_pos.z:.2f}"

        return True, ""

    def calculate_follower_position(self, leader_pos: Position) -> Position:
        """
        Calculate the safe follower position given a leader position.

        Args:
            leader_pos: Current position of the leader drone

        Returns:
            Safe position for the follower drone
        """
        safe_offset = self.get_safe_offset(leader_pos)

        return Position(
            x=leader_pos.x + safe_offset.dx,
            y=leader_pos.y + safe_offset.dy,
            z=leader_pos.z + safe_offset.dz
        )


def calculate_adaptive_offset(leader_pos: Position,
                              bounds_y_min: float = 0.0,
                              bounds_y_max: float = 6.0) -> float:
    """
    Standalone function for adaptive Y-offset calculation.

    Dynamically flip Y-offset to avoid walls.
    Default offset is -0.5 (Right side).

    Args:
        leader_pos: Position of the leader drone
        bounds_y_min: Minimum Y boundary
        bounds_y_max: Maximum Y boundary

    Returns:
        Safe Y offset value
    """
    safety_margin = 0.8  # meters
    base_offset_y = -0.5

    # Check Right Wall (Y=0)
    if leader_pos.y - safety_margin < bounds_y_min:
        return 0.5  # Force Left

    # Check Left Wall (Y=6)
    elif leader_pos.y + safety_margin > bounds_y_max:
        return -0.5  # Force Right (Standard)

    return base_offset_y  # Standard
