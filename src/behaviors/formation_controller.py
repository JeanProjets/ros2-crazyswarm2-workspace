"""
Formation Flying Controller for Crazyflie Drone Swarm

Implements formation flying behaviors including leader-follower,
line abreast, triangle, and defensive screen formations.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import time


class FormationType(Enum):
    """Types of formations available for the swarm."""
    LEADER_FOLLOWER = "leader_follower"
    LINE_ABREAST = "line_abreast"
    TRIANGLE = "triangle"
    DEFENSIVE_SCREEN = "defensive_screen"


@dataclass
class DronePosition:
    """Represents a drone's position and velocity in 3D space."""
    drone_id: str
    position: Tuple[float, float, float]
    velocity: Optional[Tuple[float, float, float]] = None
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class FormationOffset:
    """Defines the offset of a follower relative to the leader."""
    x: float
    y: float
    z: float

    def as_tuple(self) -> Tuple[float, float, float]:
        """Return offset as tuple."""
        return (self.x, self.y, self.z)


class PIDController:
    """
    PID controller for smooth position tracking.

    Used to calculate control outputs for formation maintenance.
    """

    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.5):
        """
        Initialize PID controller.

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.prev_time = time.time()

    def compute(self, error: np.ndarray) -> np.ndarray:
        """
        Compute PID control output.

        Args:
            error: Current error vector [x, y, z]

        Returns:
            Control output vector
        """
        current_time = time.time()
        dt = current_time - self.prev_time

        if dt <= 0:
            dt = 0.01  # Prevent division by zero

        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -10, 10)  # Anti-windup
        i_term = self.ki * self.integral

        # Derivative term
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative

        # Update state
        self.prev_error = error.copy()
        self.prev_time = current_time

        return p_term + i_term + d_term

    def reset(self):
        """Reset the PID controller state."""
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.prev_time = time.time()


class FormationController:
    """
    Main formation controller for managing multi-drone formations.

    Handles formation assignment, maintenance, and collision avoidance.
    """

    def __init__(self, min_separation: float = 0.5):
        """
        Initialize formation controller.

        Args:
            min_separation: Minimum distance between drones in meters
        """
        self.min_separation = min_separation
        self.formation_offsets: Dict[str, FormationOffset] = {}
        self.logger = logging.getLogger("FormationController")
        self.pid_controllers: Dict[str, PIDController] = {}

    def assign_formation_positions(
        self,
        leader_pos: Tuple[float, float, float],
        formation_type: FormationType,
        num_drones: int
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Assign formation positions for all drones in the swarm.

        Args:
            leader_pos: Position of the leader drone (x, y, z)
            formation_type: Type of formation to create
            num_drones: Total number of drones including leader

        Returns:
            Dictionary mapping drone_id to target position
        """
        positions = {}
        leader_x, leader_y, leader_z = leader_pos

        if formation_type == FormationType.LEADER_FOLLOWER:
            # Leader position
            positions["leader"] = leader_pos

            # Follower positions (behind and below leader)
            for i in range(1, num_drones):
                offset_x = -0.5 * i
                offset_y = -0.5 * i
                offset_z = -0.5 * i
                positions[f"follower_{i}"] = (
                    leader_x + offset_x,
                    leader_y + offset_y,
                    leader_z + offset_z
                )

        elif formation_type == FormationType.LINE_ABREAST:
            # Drones in a horizontal line
            spacing = self.min_separation + 0.2
            start_y = leader_y - (num_drones - 1) * spacing / 2

            for i in range(num_drones):
                drone_id = "leader" if i == 0 else f"drone_{i}"
                positions[drone_id] = (
                    leader_x,
                    start_y + i * spacing,
                    leader_z
                )

        elif formation_type == FormationType.TRIANGLE:
            # Triangle formation
            positions["leader"] = leader_pos

            if num_drones >= 2:
                positions["follower_1"] = (
                    leader_x - 0.7,
                    leader_y - 0.5,
                    leader_z
                )
            if num_drones >= 3:
                positions["follower_2"] = (
                    leader_x - 0.7,
                    leader_y + 0.5,
                    leader_z
                )

        elif formation_type == FormationType.DEFENSIVE_SCREEN:
            # Leader in center, followers in defensive positions
            positions["leader"] = leader_pos

            angles = np.linspace(0, 2 * np.pi, num_drones, endpoint=False)[1:]
            radius = 1.0

            for i, angle in enumerate(angles, start=1):
                positions[f"defender_{i}"] = (
                    leader_x + radius * np.cos(angle),
                    leader_y + radius * np.sin(angle),
                    leader_z
                )

        self.logger.info(f"Assigned {len(positions)} positions for {formation_type.value}")
        return positions

    def maintain_formation(
        self,
        leader_velocity: Tuple[float, float, float],
        follower_positions: Dict[str, DronePosition]
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Calculate target velocities to maintain formation while leader moves.

        Args:
            leader_velocity: Current velocity of leader (vx, vy, vz)
            follower_positions: Current positions of all followers

        Returns:
            Dictionary mapping follower_id to target velocity
        """
        target_velocities = {}

        for drone_id, drone_pos in follower_positions.items():
            # Base velocity matches leader
            target_vel = np.array(leader_velocity)

            # Add correction based on position error if offset is defined
            if drone_id in self.formation_offsets:
                offset = self.formation_offsets[drone_id]

                # Get or create PID controller for this drone
                if drone_id not in self.pid_controllers:
                    self.pid_controllers[drone_id] = PIDController(kp=1.0, kd=0.5)

                # Calculate position error
                # (This would need the leader position, simplified here)
                error = np.zeros(3)  # Placeholder
                correction = self.pid_controllers[drone_id].compute(error)
                target_vel += correction

            target_velocities[drone_id] = tuple(target_vel)

        return target_velocities

    def calculate_follower_offset(
        self,
        leader_pos: Tuple[float, float, float],
        follower_id: str
    ) -> Tuple[float, float, float]:
        """
        Calculate the desired position offset for a follower.

        Args:
            leader_pos: Current position of leader
            follower_id: ID of the follower drone

        Returns:
            Target position for the follower (x, y, z)
        """
        if follower_id not in self.formation_offsets:
            # Default offset if not set
            self.formation_offsets[follower_id] = FormationOffset(-0.5, -0.5, -0.5)

        offset = self.formation_offsets[follower_id]
        leader_x, leader_y, leader_z = leader_pos

        target_pos = (
            leader_x + offset.x,
            leader_y + offset.y,
            leader_z + offset.z
        )

        return target_pos

    def set_formation_offset(self, follower_id: str, offset: FormationOffset):
        """
        Set the formation offset for a specific follower.

        Args:
            follower_id: ID of the follower drone
            offset: Desired offset from leader
        """
        self.formation_offsets[follower_id] = offset
        self.logger.info(f"Set offset for {follower_id}: {offset.as_tuple()}")

    def avoid_collisions(
        self,
        drone_positions: Dict[str, Tuple[float, float, float]]
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Adjust positions to avoid collisions between drones.

        Uses potential field method for collision avoidance.

        Args:
            drone_positions: Current target positions for all drones

        Returns:
            Adjusted positions with collision avoidance applied
        """
        adjusted_positions = drone_positions.copy()
        drone_ids = list(drone_positions.keys())

        for i, drone_id_1 in enumerate(drone_ids):
            pos1 = np.array(drone_positions[drone_id_1])
            repulsive_force = np.zeros(3)

            # Check against all other drones
            for j, drone_id_2 in enumerate(drone_ids):
                if i >= j:  # Skip self and already processed pairs
                    continue

                pos2 = np.array(drone_positions[drone_id_2])
                distance_vec = pos1 - pos2
                distance = np.linalg.norm(distance_vec)

                # Apply repulsive force if too close
                if distance < self.min_separation and distance > 0:
                    # Inverse square law for repulsion
                    force_magnitude = (self.min_separation - distance) / distance
                    direction = distance_vec / distance
                    repulsive_force += direction * force_magnitude * 0.3

            # Apply repulsive force to position
            adjusted_positions[drone_id_1] = tuple(pos1 + repulsive_force)

        return adjusted_positions


class LeaderFollowerBehavior:
    """
    Specialized behavior for leader-follower formation in Scenario 1.

    Features:
    - Follower offset: (-0.5, -0.5, -0.5) relative to leader
    - Smooth following with 0.2s delay
    - Dynamic offset adjustment for obstacles
    - Emergency separation if too close (<0.3m)
    """

    def __init__(
        self,
        leader_id: str,
        follower_id: str,
        offset: FormationOffset = FormationOffset(-0.5, -0.5, -0.5)
    ):
        """
        Initialize leader-follower behavior.

        Args:
            leader_id: ID of the leader drone
            follower_id: ID of the follower drone
            offset: Desired offset from leader
        """
        self.leader_id = leader_id
        self.follower_id = follower_id
        self.offset = offset
        self.pid = PIDController(kp=1.0, ki=0.05, kd=0.5)
        self.logger = logging.getLogger(f"LeaderFollower_{follower_id}")

        # Delay buffer for smooth following
        self.leader_position_history: List[Tuple[float, Tuple[float, float, float]]] = []
        self.follow_delay = 0.2  # seconds

        # Emergency separation threshold
        self.emergency_distance = 0.3  # meters

    def compute_follower_target(
        self,
        leader_pos: Tuple[float, float, float],
        leader_vel: Optional[Tuple[float, float, float]] = None,
        current_time: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """
        Compute target position for follower with delay.

        Args:
            leader_pos: Current leader position
            leader_vel: Current leader velocity (optional)
            current_time: Current timestamp (optional)

        Returns:
            Target position for follower
        """
        if current_time is None:
            current_time = time.time()

        # Add current leader position to history
        self.leader_position_history.append((current_time, leader_pos))

        # Remove old positions (older than 1 second)
        self.leader_position_history = [
            (t, pos) for t, pos in self.leader_position_history
            if current_time - t < 1.0
        ]

        # Find delayed leader position
        delayed_time = current_time - self.follow_delay
        delayed_pos = leader_pos  # Default to current if no history

        for i, (t, pos) in enumerate(self.leader_position_history):
            if t <= delayed_time:
                delayed_pos = pos
            else:
                break

        # Apply offset to delayed position
        target_x = delayed_pos[0] + self.offset.x
        target_y = delayed_pos[1] + self.offset.y
        target_z = delayed_pos[2] + self.offset.z

        return (target_x, target_y, target_z)

    def adjust_offset_for_obstacle(
        self,
        obstacle_pos: Tuple[float, float, float],
        leader_pos: Tuple[float, float, float]
    ) -> FormationOffset:
        """
        Dynamically adjust offset to avoid obstacles.

        Args:
            obstacle_pos: Position of detected obstacle
            leader_pos: Current leader position

        Returns:
            Adjusted formation offset
        """
        # Calculate vector from leader to obstacle
        obs_vec = np.array(obstacle_pos) - np.array(leader_pos)
        obs_distance = np.linalg.norm(obs_vec)

        if obs_distance < 1.5:  # Obstacle within 1.5m
            # Move offset perpendicular to obstacle direction
            if obs_vec[1] != 0:  # Avoid divide by zero
                perp_offset_y = -obs_vec[0] / obs_vec[1] * 0.5
            else:
                perp_offset_y = 0.5

            adjusted_offset = FormationOffset(
                self.offset.x,
                perp_offset_y,
                self.offset.z + 0.3  # Increase altitude
            )

            self.logger.info(f"Adjusted offset for obstacle: {adjusted_offset.as_tuple()}")
            return adjusted_offset

        return self.offset

    def check_emergency_separation(
        self,
        leader_pos: Tuple[float, float, float],
        follower_pos: Tuple[float, float, float]
    ) -> bool:
        """
        Check if emergency separation is needed.

        Args:
            leader_pos: Current leader position
            follower_pos: Current follower position

        Returns:
            True if emergency separation is needed
        """
        distance = np.linalg.norm(
            np.array(leader_pos) - np.array(follower_pos)
        )

        if distance < self.emergency_distance:
            self.logger.warning(
                f"Emergency separation triggered! Distance: {distance:.2f}m"
            )
            return True

        return False

    def compute_approach_to_target(
        self,
        target_pos: Tuple[float, float, float],
        leader_pos: Tuple[float, float, float],
        approach_distance: float = 1.0
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Compute formation approach to target for Scenario 1.

        Leader approaches to 1m in front of target.
        Follower maintains offset during approach.

        Args:
            target_pos: Target position (7.5, 3, 5)
            leader_pos: Current leader position
            approach_distance: Distance to stop in front of target

        Returns:
            Tuple of (leader_target_pos, follower_target_pos)
        """
        target_x, target_y, target_z = target_pos

        # Leader target: 1m in front of target
        leader_target = (
            target_x - approach_distance,
            target_y,
            target_z
        )

        # Follower maintains offset
        follower_target = (
            leader_target[0] + self.offset.x,
            leader_target[1] + self.offset.y,
            leader_target[2] + self.offset.z
        )

        self.logger.info(
            f"Approach positions - Leader: {leader_target}, Follower: {follower_target}"
        )

        return leader_target, follower_target


def calculate_follower_position(
    leader_pos: Tuple[float, float, float],
    leader_vel: Tuple[float, float, float],
    offset: Tuple[float, float, float],
    kp: float = 1.0,
    kd: float = 0.5
) -> Tuple[float, float, float]:
    """
    Calculate follower position using PD control.

    Maintains fixed offset while following smoothly.

    Args:
        leader_pos: Leader position (x, y, z)
        leader_vel: Leader velocity (vx, vy, vz)
        offset: Desired offset (dx, dy, dz)
        kp: Proportional gain
        kd: Derivative gain

    Returns:
        Target position for follower
    """
    leader_array = np.array(leader_pos)
    offset_array = np.array(offset)

    # Target position with offset
    target_pos = leader_array + offset_array

    return tuple(target_pos)


def avoid_collision(
    drone_positions: List[Tuple[float, float, float]],
    min_separation: float = 0.5
) -> List[Tuple[float, float, float]]:
    """
    Adjust velocities to maintain minimum separation.

    Uses potential field method.

    Args:
        drone_positions: List of drone positions
        min_separation: Minimum allowed distance between drones

    Returns:
        Adjusted positions
    """
    adjusted = []
    n = len(drone_positions)

    for i in range(n):
        pos_i = np.array(drone_positions[i])
        repulsive_force = np.zeros(3)

        for j in range(n):
            if i == j:
                continue

            pos_j = np.array(drone_positions[j])
            diff = pos_i - pos_j
            distance = np.linalg.norm(diff)

            if distance < min_separation and distance > 0:
                # Repulsive force inversely proportional to distance
                force_mag = (min_separation - distance) / (distance ** 2)
                repulsive_force += (diff / distance) * force_mag * 0.1

        adjusted_pos = pos_i + repulsive_force
        adjusted.append(tuple(adjusted_pos))

    return adjusted
