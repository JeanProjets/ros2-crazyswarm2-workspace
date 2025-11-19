"""
Precision Corner Approach Behavior for Scenario 2

This module implements high-precision approach behaviors for confined spaces
where drones must navigate to targets very close to walls.
"""

from typing import Tuple, List
from dataclasses import dataclass
import math


@dataclass
class Position:
    """3D Position representation"""
    x: float
    y: float
    z: float


@dataclass
class ApproachPhase:
    """Represents a phase in the approach maneuver"""
    name: str
    position: Position
    velocity: float
    hold_time: float = 0.0  # seconds


class CornerApproachBehavior:
    """
    High-precision approach behavior for targets in confined spaces.

    For Scenario 2, the target at (9.5, 0.5, 5) is extremely close to walls.
    Drones cannot circle around the back (X > 9.5) or right (Y < 0.5).
    Must approach from the "open" side with careful positioning.

    Jamming Geometry:
    - Leader Standoff: (8.5, 0.5, 5) - 1m in front
    - Follower Standoff: (8.5, 1.0, 4.5) - using adaptive offset
    """

    def __init__(self,
                 max_approach_speed: float = 0.5,
                 standoff_distance: float = 1.0,
                 corner_threshold: float = 1.5):
        """
        Initialize corner approach behavior.

        Args:
            max_approach_speed: Maximum speed during approach (m/s)
            standoff_distance: Distance to maintain from target (m)
            corner_threshold: Distance from wall to consider "corner mode" (m)
        """
        self.max_approach_speed = max_approach_speed  # Slower than Scenario 1
        self.standoff_distance = standoff_distance
        self.corner_threshold = corner_threshold

        # Speed profiles
        self.cruise_speed = 0.5    # m/s
        self.precision_speed = 0.3  # m/s - for final approach
        self.station_speed = 0.1   # m/s - for station keeping

        # Timing
        self.hold_time = 2.0  # seconds to hold jamming position

    def approach_corner_target(self,
                               target_pos: Position,
                               drone_role: str = "leader") -> List[ApproachPhase]:
        """
        Generate approach phases for a corner target.

        Args:
            target_pos: Position of the target (e.g., (9.5, 0.5, 5))
            drone_role: "leader" or "follower"

        Returns:
            List of approach phases with positions and velocities
        """
        phases = []

        # Determine if target is in a corner
        is_corner = self._is_corner_position(target_pos)

        if is_corner:
            phases = self._generate_corner_approach(target_pos, drone_role)
        else:
            phases = self._generate_standard_approach(target_pos, drone_role)

        return phases

    def _is_corner_position(self, pos: Position) -> bool:
        """
        Determine if a position is in a corner (near multiple walls).

        Args:
            pos: Position to check

        Returns:
            True if position is in a corner zone
        """
        # Check distance to walls (assuming 10x6x10 cage)
        x_near_wall = pos.x < self.corner_threshold or pos.x > (10.0 - self.corner_threshold)
        y_near_wall = pos.y < self.corner_threshold or pos.y > (6.0 - self.corner_threshold)

        return x_near_wall and y_near_wall

    def _generate_corner_approach(self,
                                  target_pos: Position,
                                  drone_role: str) -> List[ApproachPhase]:
        """
        Generate approach phases optimized for corner targets.

        For target at (9.5, 0.5, 5):
        - Cannot approach from behind (X > 9.5)
        - Cannot approach from right (Y < 0.5)
        - Must approach from the "open" quadrant

        Args:
            target_pos: Target position
            drone_role: "leader" or "follower"

        Returns:
            List of approach phases
        """
        phases = []

        # Phase 1: Move to staging position (safe distance, same altitude)
        staging_pos = self._calculate_staging_position(target_pos, drone_role)
        phases.append(ApproachPhase(
            name="staging",
            position=staging_pos,
            velocity=self.cruise_speed,
            hold_time=0.5
        ))

        # Phase 2: Move to approach corridor (align axis)
        corridor_pos = self._calculate_approach_corridor(target_pos, drone_role)
        phases.append(ApproachPhase(
            name="corridor",
            position=corridor_pos,
            velocity=self.precision_speed,
            hold_time=0.5
        ))

        # Phase 3: Final approach to standoff position
        standoff_pos = self._calculate_standoff_position(target_pos, drone_role)
        phases.append(ApproachPhase(
            name="standoff",
            position=standoff_pos,
            velocity=self.precision_speed,
            hold_time=self.hold_time
        ))

        return phases

    def _generate_standard_approach(self,
                                    target_pos: Position,
                                    drone_role: str) -> List[ApproachPhase]:
        """
        Generate approach phases for non-corner targets.

        Args:
            target_pos: Target position
            drone_role: "leader" or "follower"

        Returns:
            List of approach phases
        """
        phases = []

        # Direct approach to standoff
        standoff_pos = self._calculate_standoff_position(target_pos, drone_role)
        phases.append(ApproachPhase(
            name="direct_standoff",
            position=standoff_pos,
            velocity=self.max_approach_speed,
            hold_time=self.hold_time
        ))

        return phases

    def _calculate_staging_position(self,
                                    target_pos: Position,
                                    drone_role: str) -> Position:
        """
        Calculate staging position - safe distance before final approach.

        Args:
            target_pos: Target position
            drone_role: "leader" or "follower"

        Returns:
            Staging position
        """
        # Stage 2-3 meters away from target, on the "open" side
        staging_distance = 3.0

        # For target at (9.5, 0.5, 5), stage at approximately (6.5, 0.5, 5)
        staging_x = target_pos.x - staging_distance

        if drone_role == "leader":
            return Position(
                x=staging_x,
                y=target_pos.y,
                z=target_pos.z
            )
        else:  # follower
            # Follower offset to side (using adaptive formation logic)
            y_offset = 0.5 if target_pos.y < 1.5 else -0.5
            return Position(
                x=staging_x,
                y=target_pos.y + y_offset,
                z=target_pos.z - 0.5
            )

    def _calculate_approach_corridor(self,
                                     target_pos: Position,
                                     drone_role: str) -> Position:
        """
        Calculate position in the approach corridor.

        Args:
            target_pos: Target position
            drone_role: "leader" or "follower"

        Returns:
            Corridor position
        """
        # Position 1.5m from target
        corridor_distance = 1.5

        corridor_x = target_pos.x - corridor_distance

        if drone_role == "leader":
            return Position(
                x=corridor_x,
                y=target_pos.y,
                z=target_pos.z
            )
        else:  # follower
            y_offset = 0.5 if target_pos.y < 1.5 else -0.5
            return Position(
                x=corridor_x,
                y=target_pos.y + y_offset,
                z=target_pos.z - 0.5
            )

    def _calculate_standoff_position(self,
                                     target_pos: Position,
                                     drone_role: str) -> Position:
        """
        Calculate final standoff position for jamming.

        For Scenario 2 target at (9.5, 0.5, 5):
        - Leader: (8.5, 0.5, 5) - directly in front
        - Follower: (8.5, 1.0, 4.5) - offset to side and below

        Args:
            target_pos: Target position
            drone_role: "leader" or "follower"

        Returns:
            Standoff position
        """
        standoff_x = target_pos.x - self.standoff_distance

        if drone_role == "leader":
            return Position(
                x=standoff_x,
                y=target_pos.y,
                z=target_pos.z
            )
        else:  # follower
            # Use adaptive offset for corner scenarios
            # If target Y < 1.0, place follower on left (+Y)
            y_offset = 0.5 if target_pos.y < 1.0 else -0.5

            return Position(
                x=standoff_x,
                y=target_pos.y + y_offset,
                z=target_pos.z - 0.5
            )

    def hold_jamming_position_corner(self,
                                     current_pos: Position,
                                     standoff_pos: Position) -> Tuple[float, float, float]:
        """
        Calculate velocity commands to hold jamming position in corner.

        Uses proportional control to maintain precise position.

        Args:
            current_pos: Current drone position
            standoff_pos: Desired standoff position

        Returns:
            Tuple of (vx, vy, vz) velocity commands
        """
        # Proportional gain
        kp = 0.5

        # Calculate position error
        error_x = standoff_pos.x - current_pos.x
        error_y = standoff_pos.y - current_pos.y
        error_z = standoff_pos.z - current_pos.z

        # Calculate velocity commands
        vx = kp * error_x
        vy = kp * error_y
        vz = kp * error_z

        # Limit velocity for station keeping
        max_station_vel = self.station_speed

        vx = max(-max_station_vel, min(max_station_vel, vx))
        vy = max(-max_station_vel, min(max_station_vel, vy))
        vz = max(-max_station_vel, min(max_station_vel, vz))

        return (vx, vy, vz)

    def calculate_safe_approach_vector(self,
                                       current_pos: Position,
                                       target_pos: Position) -> Tuple[float, float, float]:
        """
        Calculate safe approach vector that avoids walls.

        For corner targets, this calculates a safe approach path that avoids
        approaching from behind walls. Instead of approaching directly,
        it guides the drone to approach from the "open" side.

        Args:
            current_pos: Current position
            target_pos: Target position

        Returns:
            Unit vector (dx, dy, dz) for safe approach direction
        """
        # Calculate direct vector to target
        dx = target_pos.x - current_pos.x
        dy = target_pos.y - current_pos.y
        dz = target_pos.z - current_pos.z

        # Normalize
        magnitude = math.sqrt(dx**2 + dy**2 + dz**2)

        if magnitude < 0.01:
            return (0.0, 0.0, 0.0)

        dx_norm = dx / magnitude
        dy_norm = dy / magnitude
        dz_norm = dz / magnitude

        # Check if target is in corner and adjust approach
        target_in_corner = (target_pos.x > 9.0 and target_pos.y < 1.0)

        if target_in_corner:
            # For corner targets, approach from the open quadrant
            # Instead of blocking directions, guide to safe approach corridor

            # If we're approaching from behind the wall, redirect
            if dx_norm > 0 and current_pos.x < target_pos.x - 1.0:
                # We're in front of target, can approach normally along X
                pass
            elif dx_norm > 0:
                # Too close to wall, stop X approach
                dx_norm = 0.0

            # Similar logic for Y
            if dy_norm < 0 and current_pos.y > target_pos.y + 0.5:
                # We're above target in Y, can approach normally
                pass
            elif dy_norm < 0:
                # Would push toward Y wall, stop Y approach
                dy_norm = 0.0

            # If both are blocked, prefer X approach (safer for this geometry)
            if abs(dx_norm) < 0.01 and abs(dy_norm) < 0.01:
                # Return approach along Z if available, or minimal X
                if abs(dz_norm) > 0.01:
                    return (0.0, 0.0, dz_norm)
                else:
                    # Approach from in front (negative X direction relative to approach)
                    # This means moving to a position with lower X
                    if current_pos.x > target_pos.x:
                        return (-1.0, 0.0, 0.0)
                    else:
                        return (1.0, 0.0, 0.0)

        # Re-normalize
        magnitude = math.sqrt(dx_norm**2 + dy_norm**2 + dz_norm**2)
        if magnitude > 0.01:
            dx_norm /= magnitude
            dy_norm /= magnitude
            dz_norm /= magnitude
            return (dx_norm, dy_norm, dz_norm)

        return (0.0, 0.0, 0.0)
