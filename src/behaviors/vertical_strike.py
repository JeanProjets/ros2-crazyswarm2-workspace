"""
Vertical Strike Maneuver for Scenario 2

This module implements strictly vertical attack patterns for confined spaces
where lateral movement is extremely limited due to wall proximity.
"""

from typing import Tuple, List
from dataclasses import dataclass
import time


@dataclass
class Position:
    """3D Position representation"""
    x: float
    y: float
    z: float


@dataclass
class AttackPhase:
    """Represents a phase in the attack maneuver"""
    name: str
    position: Position
    velocity: float
    hold_time: float = 0.0  # seconds


class VerticalStrikeManeuver:
    """
    Strictly vertical attack pattern for confined spaces.

    Critical for Scenario 2 where target at (9.5, 0.5, 5) is close to walls.
    No room for circling or lateral approaches.

    Attack Sequence:
    1. Attack drone (P) approaches target (C)
    2. C is at Z=5
    3. P aligns above at (9.5, 0.5, 6.0)
    4. Descend to (9.5, 0.5, 5.3) - stop buffer above target
    5. HOLD 2 seconds - jamming
    6. ASCEND immediately to 6.0

    Critical Safety:
    - No X/Y drift allowed during descent (drift > 10cm = wall hit)
    - Use Position Hold mode with high gain before descent
    """

    def __init__(self,
                 descent_speed: float = 0.3,
                 ascent_speed: float = 0.5,
                 stop_buffer: float = 0.3,
                 hold_duration: float = 2.0):
        """
        Initialize vertical strike maneuver.

        Args:
            descent_speed: Speed during descent (m/s)
            ascent_speed: Speed during ascent (m/s)
            stop_buffer: Distance to stop above target (m)
            hold_duration: Time to hold strike position (seconds)
        """
        self.descent_speed = descent_speed
        self.ascent_speed = ascent_speed
        self.stop_buffer = stop_buffer
        self.hold_duration = hold_duration

        # Alignment parameters
        self.alignment_altitude_offset = 1.0  # meters above target
        self.alignment_precision = 0.05       # meters (5cm tolerance)
        self.max_drift_allowed = 0.10         # meters (10cm max drift)

        # Safety parameters
        self.position_hold_time = 0.5  # seconds to stabilize before descent
        self.emergency_stop_buffer = 0.5  # meters - emergency stop distance

    def align_above_target(self,
                          target_pos: Position,
                          current_pos: Position = None) -> List[AttackPhase]:
        """
        Align attack drone directly above target position.

        Args:
            target_pos: Position of the target
            current_pos: Current position of attack drone (optional)

        Returns:
            List of phases to align above target
        """
        phases = []

        # Calculate alignment position (directly above target)
        alignment_pos = Position(
            x=target_pos.x,
            y=target_pos.y,
            z=target_pos.z + self.alignment_altitude_offset
        )

        # Phase 1: Move to X,Y coordinates at safe altitude
        if current_pos and current_pos.z < alignment_pos.z:
            # If below target altitude, first climb
            climb_pos = Position(
                x=current_pos.x,
                y=current_pos.y,
                z=alignment_pos.z
            )
            phases.append(AttackPhase(
                name="pre_climb",
                position=climb_pos,
                velocity=self.ascent_speed,
                hold_time=0.3
            ))

        # Phase 2: Move to alignment position (above target)
        phases.append(AttackPhase(
            name="align_above",
            position=alignment_pos,
            velocity=0.3,  # Slow approach
            hold_time=self.position_hold_time
        ))

        # Phase 3: Stabilize in position hold
        phases.append(AttackPhase(
            name="stabilize",
            position=alignment_pos,
            velocity=0.0,  # Position hold
            hold_time=self.position_hold_time
        ))

        return phases

    def descend_controlled(self,
                          target_pos: Position,
                          stop_buffer: float = None) -> List[AttackPhase]:
        """
        Execute controlled vertical descent to strike position.

        Critical: Maintains X,Y position while descending Z only.

        Args:
            target_pos: Target position
            stop_buffer: Distance to stop above target (defaults to self.stop_buffer)

        Returns:
            List of descent phases
        """
        if stop_buffer is None:
            stop_buffer = self.stop_buffer

        phases = []

        # Calculate strike position (above target by stop_buffer)
        strike_pos = Position(
            x=target_pos.x,
            y=target_pos.y,
            z=target_pos.z + stop_buffer
        )

        # Descent phase
        phases.append(AttackPhase(
            name="descend",
            position=strike_pos,
            velocity=self.descent_speed,
            hold_time=0.0
        ))

        # Hold strike position
        phases.append(AttackPhase(
            name="strike_hold",
            position=strike_pos,
            velocity=0.0,  # Position hold
            hold_time=self.hold_duration
        ))

        return phases

    def ascend_escape(self, target_pos: Position) -> List[AttackPhase]:
        """
        Execute rapid vertical ascent after strike.

        Args:
            target_pos: Target position (to calculate escape position)

        Returns:
            List of ascent phases
        """
        phases = []

        # Calculate escape position (back to alignment altitude)
        escape_pos = Position(
            x=target_pos.x,
            y=target_pos.y,
            z=target_pos.z + self.alignment_altitude_offset
        )

        # Rapid ascent
        phases.append(AttackPhase(
            name="ascend_escape",
            position=escape_pos,
            velocity=self.ascent_speed,
            hold_time=0.5
        ))

        return phases

    def execute_full_strike(self, target_pos: Position) -> List[AttackPhase]:
        """
        Execute complete vertical strike maneuver.

        Combines alignment, descent, hold, and ascent.

        Args:
            target_pos: Position of the target

        Returns:
            Complete list of strike phases
        """
        phases = []

        # Phase 1: Align above target
        phases.extend(self.align_above_target(target_pos))

        # Phase 2: Descend to strike position
        phases.extend(self.descend_controlled(target_pos))

        # Phase 3: Ascend to escape
        phases.extend(self.ascend_escape(target_pos))

        return phases

    def check_alignment_precision(self,
                                  current_pos: Position,
                                  target_pos: Position) -> Tuple[bool, float]:
        """
        Check if drone is precisely aligned above target.

        Args:
            current_pos: Current drone position
            target_pos: Target position

        Returns:
            Tuple of (is_aligned, horizontal_error)
        """
        # Calculate horizontal error
        dx = current_pos.x - target_pos.x
        dy = current_pos.y - target_pos.y
        horizontal_error = (dx**2 + dy**2)**0.5

        is_aligned = horizontal_error <= self.alignment_precision

        return is_aligned, horizontal_error

    def check_drift_during_descent(self,
                                   start_pos: Position,
                                   current_pos: Position) -> Tuple[bool, float]:
        """
        Monitor X/Y drift during vertical descent.

        Critical safety check for Scenario 2.

        Args:
            start_pos: Position at start of descent
            current_pos: Current position during descent

        Returns:
            Tuple of (is_safe, drift_distance)
        """
        # Calculate horizontal drift
        dx = current_pos.x - start_pos.x
        dy = current_pos.y - start_pos.y
        drift = (dx**2 + dy**2)**0.5

        is_safe = drift <= self.max_drift_allowed

        return is_safe, drift

    def calculate_descent_velocity_commands(self,
                                           current_pos: Position,
                                           target_pos: Position,
                                           descent_start_pos: Position) -> Tuple[float, float, float]:
        """
        Calculate velocity commands for controlled descent.

        Maintains X,Y position while controlling Z descent.

        Args:
            current_pos: Current drone position
            target_pos: Target position
            descent_start_pos: Position where descent started

        Returns:
            Tuple of (vx, vy, vz) velocity commands
        """
        # High gain for X,Y position hold
        kp_xy = 2.0
        kp_z = 0.5

        # Calculate X,Y errors (should maintain descent_start position)
        error_x = descent_start_pos.x - current_pos.x
        error_y = descent_start_pos.y - current_pos.y

        # Calculate Z error (descend toward strike altitude)
        strike_altitude = target_pos.z + self.stop_buffer
        error_z = strike_altitude - current_pos.z

        # Velocity commands
        vx = kp_xy * error_x
        vy = kp_xy * error_y

        # Z velocity: negative (down) if above strike altitude
        if error_z < 0:  # Below strike altitude - slow down
            vz = max(error_z * kp_z, -0.1)  # Gentle slowdown
        else:  # Above strike altitude - descend
            vz = -self.descent_speed

        # Limit velocities
        max_xy_correction = 0.2
        vx = max(-max_xy_correction, min(max_xy_correction, vx))
        vy = max(-max_xy_correction, min(max_xy_correction, vy))

        return (vx, vy, vz)

    def emergency_stop(self, current_pos: Position) -> Position:
        """
        Calculate emergency stop position (hold current location).

        Args:
            current_pos: Current position

        Returns:
            Emergency hold position
        """
        return Position(
            x=current_pos.x,
            y=current_pos.y,
            z=current_pos.z
        )

    def validate_strike_geometry(self,
                                 target_pos: Position,
                                 bounds_x_max: float = 10.0,
                                 bounds_y_min: float = 0.0) -> Tuple[bool, str]:
        """
        Validate that vertical strike is geometrically feasible.

        Args:
            target_pos: Target position
            bounds_x_max: Maximum X boundary
            bounds_y_min: Minimum Y boundary

        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        # Check clearance above target
        clearance_above = (10.0 - target_pos.z)  # Assuming 10m ceiling
        required_clearance = self.alignment_altitude_offset

        if clearance_above < required_clearance:
            return False, f"Insufficient clearance above target: {clearance_above:.2f}m"

        # Check wall proximity
        min_wall_clearance = 0.3
        x_clearance = bounds_x_max - target_pos.x
        y_clearance = target_pos.y - bounds_y_min

        if x_clearance < min_wall_clearance:
            return False, f"Target too close to X wall: {x_clearance:.2f}m clearance"

        if y_clearance < min_wall_clearance:
            return False, f"Target too close to Y wall: {y_clearance:.2f}m clearance"

        return True, ""
