"""
Attack Maneuvers Module for Crazyflie Drone Swarm

Implements coordinated attack behaviors including jamming positioning,
neutralization maneuvers, and victory procedures for Scenario 1.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import logging
import time


class AttackRole(Enum):
    """Roles for drones during attack sequence."""
    JAMMER_LEADER = "jammer_leader"
    JAMMER_FOLLOWER = "jammer_follower"
    ATTACK_DRONE = "attack_drone"
    NEUTRAL = "neutral"


class AttackPhase(Enum):
    """Phases of the attack sequence."""
    APPROACH = "approach"
    JAMMING = "jamming"
    NEUTRALIZATION = "neutralization"
    VICTORY = "victory"
    ABORT = "abort"


@dataclass
class AttackWaypoint:
    """Waypoint for attack maneuvers with timing information."""
    position: Tuple[float, float, float]
    hover_time: float = 0.0  # seconds to hover at this position
    speed: float = 0.3  # m/s
    yaw: float = 0.0


class JammingBehavior:
    """
    Implements jamming positioning and formation maintenance.

    Scenario 1 requirements:
    - Position jammers at target front
    - Hold jamming position for 20 seconds
    - Maintain formation during jamming
    """

    def __init__(self, drone_id: str, role: AttackRole):
        """
        Initialize jamming behavior.

        Args:
            drone_id: Unique identifier for the drone
            role: Attack role (JAMMER_LEADER or JAMMER_FOLLOWER)
        """
        self.drone_id = drone_id
        self.role = role
        self.logger = logging.getLogger(f"JammingBehavior_{drone_id}")
        self.jamming_start_time: Optional[float] = None
        self.jamming_duration = 20.0  # seconds
        self.is_jamming = False

    def position_for_jamming(
        self,
        target_pos: Tuple[float, float, float],
        drone_role: AttackRole
    ) -> Tuple[float, float, float]:
        """
        Calculate jamming position based on target location and drone role.

        Args:
            target_pos: Position of the target (7.5, 3, 5)
            drone_role: Role of this drone in attack

        Returns:
            Jamming position (x, y, z)
        """
        target_x, target_y, target_z = target_pos

        if drone_role == AttackRole.JAMMER_LEADER:
            # Leader positions 1m in front of target
            jamming_pos = (
                target_x - 1.0,  # 1m in front
                target_y,
                target_z
            )
            self.logger.info(f"Jammer leader position: {jamming_pos}")

        elif drone_role == AttackRole.JAMMER_FOLLOWER:
            # Follower maintains offset from leader position
            jamming_pos = (
                target_x - 1.5,  # 0.5m behind leader
                target_y - 0.5,
                target_z - 0.5
            )
            self.logger.info(f"Jammer follower position: {jamming_pos}")

        else:
            # Default to frontal position
            jamming_pos = (target_x - 1.0, target_y, target_z)

        return jamming_pos

    def maintain_jamming_formation(
        self,
        duration: float = 20.0,
        current_time: Optional[float] = None
    ) -> bool:
        """
        Maintain jamming formation for specified duration.

        Args:
            duration: Duration to maintain jamming in seconds
            current_time: Current timestamp (optional)

        Returns:
            True if jamming should continue, False if complete
        """
        if current_time is None:
            current_time = time.time()

        # Start jamming if not already started
        if self.jamming_start_time is None:
            self.jamming_start_time = current_time
            self.is_jamming = True
            self.logger.info(f"Starting jamming sequence for {duration}s")

        # Check if jamming duration complete
        elapsed = current_time - self.jamming_start_time
        if elapsed >= duration:
            self.is_jamming = False
            self.logger.info("Jamming sequence complete")
            return False

        # Jamming still in progress
        remaining = duration - elapsed
        if int(remaining) % 5 == 0 and remaining % 1.0 < 0.1:  # Log every 5 seconds
            self.logger.info(f"Jamming in progress: {remaining:.1f}s remaining")

        return True

    def simulate_rf_interference(self) -> Dict[str, float]:
        """
        Simulate RF interference signal strength.

        Returns:
            Dictionary with interference metrics
        """
        if not self.is_jamming:
            return {
                'signal_strength': 0.0,
                'interference_level': 0.0,
                'effective_radius': 0.0
            }

        # Simulate jamming effectiveness (placeholder values)
        return {
            'signal_strength': 85.0,  # dBm
            'interference_level': 0.95,  # 95% interference
            'effective_radius': 2.0  # meters
        }

    def get_jamming_status(self) -> Dict[str, any]:
        """
        Get current jamming status.

        Returns:
            Dictionary with jamming status information
        """
        if self.jamming_start_time is None:
            elapsed = 0.0
        else:
            elapsed = time.time() - self.jamming_start_time

        return {
            'drone_id': self.drone_id,
            'role': self.role.value,
            'is_jamming': self.is_jamming,
            'elapsed_time': elapsed,
            'remaining_time': max(0, self.jamming_duration - elapsed)
        }


class NeutralizationManeuver:
    """
    Implements neutralization approach maneuvers.

    Scenario 1 sequence:
    1. Patrol drone approaches from above
    2. Descends to 30cm above target
    3. Stops safely (demonstration mode)
    4. Executes victory hover
    """

    def __init__(self, drone_id: str, safe_mode: bool = True):
        """
        Initialize neutralization maneuver.

        Args:
            drone_id: Unique identifier for the drone
            safe_mode: If True, stops at safe distance (for demo)
        """
        self.drone_id = drone_id
        self.safe_mode = safe_mode
        self.logger = logging.getLogger(f"Neutralization_{drone_id}")
        self.min_battery_level = 0.25  # 25% minimum battery
        self.stop_distance = 0.3  # meters above target
        self.current_phase = AttackPhase.APPROACH

    def kamikaze_approach(
        self,
        attacker_pos: Tuple[float, float, float],
        target_pos: Tuple[float, float, float]
    ) -> List[AttackWaypoint]:
        """
        Generate approach waypoints from above the target.

        Args:
            attacker_pos: Current position of attack drone
            target_pos: Position of target (7.5, 3, 5)

        Returns:
            List of waypoints for the approach
        """
        waypoints = []
        target_x, target_y, target_z = target_pos

        # Phase 1: Move to position above target
        above_target = (target_x, target_y, target_z + 2.0)
        waypoints.append(AttackWaypoint(
            position=above_target,
            hover_time=2.0,
            speed=0.4
        ))

        # Phase 2: Descend to safe demonstration distance
        if self.safe_mode:
            final_pos = (target_x, target_y, target_z + self.stop_distance)
        else:
            final_pos = target_pos  # Would be actual neutralization

        waypoints.append(AttackWaypoint(
            position=final_pos,
            hover_time=3.0,
            speed=0.2  # Slow descent
        ))

        self.logger.info(f"Generated {len(waypoints)} waypoints for approach")
        return waypoints

    def safe_demonstration_stop(
        self,
        current_pos: Tuple[float, float, float],
        target_pos: Tuple[float, float, float],
        stop_distance: float = 0.3
    ) -> bool:
        """
        Check if drone has reached safe demonstration stop position.

        Args:
            current_pos: Current drone position
            target_pos: Target position
            stop_distance: Safe stop distance in meters

        Returns:
            True if at safe stop position
        """
        current_array = np.array(current_pos)
        target_array = np.array(target_pos)

        distance = np.linalg.norm(current_array - target_array)

        # Check if within tolerance of stop distance
        tolerance = 0.1  # 10cm tolerance
        at_stop_position = abs(distance - stop_distance) < tolerance

        if at_stop_position:
            self.logger.info(
                f"Reached safe demonstration stop at {stop_distance}m from target"
            )
            self.current_phase = AttackPhase.VICTORY

        return at_stop_position

    def check_safety_constraints(
        self,
        battery_level: float,
        collision_imminent: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if maneuver should abort due to safety constraints.

        Args:
            battery_level: Current battery level (0.0-1.0)
            collision_imminent: Whether collision is imminent

        Returns:
            Tuple of (should_abort, abort_reason)
        """
        # Check battery level
        if battery_level < self.min_battery_level:
            reason = f"Battery critical: {battery_level*100:.1f}%"
            self.logger.warning(reason)
            self.current_phase = AttackPhase.ABORT
            return True, reason

        # Check collision risk
        if collision_imminent:
            reason = "Collision imminent - emergency stop"
            self.logger.error(reason)
            self.current_phase = AttackPhase.ABORT
            return True, reason

        return False, None

    def victory_hover(
        self,
        current_pos: Tuple[float, float, float],
        hover_altitude_offset: float = 0.5
    ) -> Tuple[float, float, float]:
        """
        Calculate victory hover position.

        Args:
            current_pos: Current drone position
            hover_altitude_offset: Additional altitude for victory hover

        Returns:
            Victory hover position
        """
        x, y, z = current_pos
        victory_pos = (x, y, z + hover_altitude_offset)

        self.logger.info(f"Victory hover position: {victory_pos}")
        return victory_pos

    def return_to_home(
        self,
        current_pos: Tuple[float, float, float],
        home_pos: Tuple[float, float, float]
    ) -> List[AttackWaypoint]:
        """
        Generate return-to-home waypoints.

        Args:
            current_pos: Current drone position
            home_pos: Home/landing position

        Returns:
            List of waypoints for return journey
        """
        waypoints = []

        # First, climb to safe altitude
        safe_altitude = max(current_pos[2], home_pos[2]) + 1.0
        waypoints.append(AttackWaypoint(
            position=(current_pos[0], current_pos[1], safe_altitude),
            hover_time=1.0,
            speed=0.4
        ))

        # Navigate to home position at safe altitude
        waypoints.append(AttackWaypoint(
            position=(home_pos[0], home_pos[1], safe_altitude),
            hover_time=1.0,
            speed=0.5
        ))

        # Descend to home position
        waypoints.append(AttackWaypoint(
            position=home_pos,
            hover_time=2.0,
            speed=0.2
        ))

        self.logger.info(f"Generated {len(waypoints)} waypoints for return-to-home")
        return waypoints


class AttackCoordinator:
    """
    Coordinates the complete attack sequence for Scenario 1.

    Manages timing and sequencing of all attack phases.
    """

    def __init__(self, target_pos: Tuple[float, float, float] = (7.5, 3.0, 5.0)):
        """
        Initialize attack coordinator.

        Args:
            target_pos: Position of target to attack
        """
        self.target_pos = target_pos
        self.logger = logging.getLogger("AttackCoordinator")
        self.current_phase = AttackPhase.APPROACH
        self.phase_start_time: Optional[float] = None

        # Drone assignments
        self.jammer_leader: Optional[JammingBehavior] = None
        self.jammer_follower: Optional[JammingBehavior] = None
        self.attack_drone: Optional[NeutralizationManeuver] = None

    def assign_roles(
        self,
        leader_id: str,
        follower_id: str,
        attacker_id: str
    ):
        """
        Assign attack roles to drones.

        Args:
            leader_id: ID for jammer leader
            follower_id: ID for jammer follower
            attacker_id: ID for attack drone
        """
        self.jammer_leader = JammingBehavior(leader_id, AttackRole.JAMMER_LEADER)
        self.jammer_follower = JammingBehavior(follower_id, AttackRole.JAMMER_FOLLOWER)
        self.attack_drone = NeutralizationManeuver(attacker_id, safe_mode=True)

        self.logger.info(
            f"Roles assigned - Leader: {leader_id}, "
            f"Follower: {follower_id}, Attacker: {attacker_id}"
        )

    def execute_attack_sequence(self) -> Dict[str, any]:
        """
        Execute the complete attack sequence.

        Scenario 1 sequence:
        1. Jammers approach and hold position (20s)
        2. Attack drone approaches from above
        3. Attack drone descends to safe distance
        4. All drones victory hover

        Returns:
            Status dictionary with current phase information
        """
        current_time = time.time()

        if self.phase_start_time is None:
            self.phase_start_time = current_time

        elapsed = current_time - self.phase_start_time

        # Phase 1: Approach (jammers move to position)
        if self.current_phase == AttackPhase.APPROACH:
            if elapsed > 5.0:  # Assume approach takes 5 seconds
                self.current_phase = AttackPhase.JAMMING
                self.phase_start_time = current_time
                self.logger.info("Transitioning to JAMMING phase")

        # Phase 2: Jamming (hold position for 20s)
        elif self.current_phase == AttackPhase.JAMMING:
            if elapsed > 20.0:  # Jamming duration
                self.current_phase = AttackPhase.NEUTRALIZATION
                self.phase_start_time = current_time
                self.logger.info("Transitioning to NEUTRALIZATION phase")

        # Phase 3: Neutralization (attack drone descends)
        elif self.current_phase == AttackPhase.NEUTRALIZATION:
            if elapsed > 8.0:  # Assume neutralization takes 8 seconds
                self.current_phase = AttackPhase.VICTORY
                self.phase_start_time = current_time
                self.logger.info("Transitioning to VICTORY phase")

        # Phase 4: Victory hover
        elif self.current_phase == AttackPhase.VICTORY:
            self.logger.info("Mission complete - Victory!")

        return {
            'current_phase': self.current_phase.value,
            'elapsed_time': elapsed,
            'target_position': self.target_pos
        }

    def get_attack_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Get current target positions for all attack drones.

        Returns:
            Dictionary mapping drone IDs to positions
        """
        positions = {}

        if self.jammer_leader:
            positions['jammer_leader'] = self.jammer_leader.position_for_jamming(
                self.target_pos,
                AttackRole.JAMMER_LEADER
            )

        if self.jammer_follower:
            positions['jammer_follower'] = self.jammer_follower.position_for_jamming(
                self.target_pos,
                AttackRole.JAMMER_FOLLOWER
            )

        if self.attack_drone and self.current_phase == AttackPhase.NEUTRALIZATION:
            # Attack from above
            target_x, target_y, target_z = self.target_pos
            positions['attack_drone'] = (target_x, target_y, target_z + 2.0)

        return positions


def calculate_approach_vector(
    start_pos: Tuple[float, float, float],
    target_pos: Tuple[float, float, float],
    approach_type: str = "frontal"
) -> np.ndarray:
    """
    Calculate approach vector for attack maneuver.

    Args:
        start_pos: Starting position
        target_pos: Target position
        approach_type: Type of approach ("frontal", "vertical", "lateral")

    Returns:
        Unit vector in approach direction
    """
    start = np.array(start_pos)
    target = np.array(target_pos)

    if approach_type == "frontal":
        # Direct horizontal approach
        direction = target - start
        direction[2] = 0  # Zero out vertical component
    elif approach_type == "vertical":
        # Vertical descent
        direction = np.array([0, 0, -1])
    elif approach_type == "lateral":
        # Lateral approach (perpendicular to frontal)
        direction = target - start
        direction = np.array([-direction[1], direction[0], 0])
    else:
        # Default to direct approach
        direction = target - start

    # Normalize
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm

    return direction
