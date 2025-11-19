"""
Motion-Compensated Swarm Manager for Scenario 3

This module implements swarm coordination for moving formations where the
reference frame (leader) is itself in motion. It handles velocity feedforward,
collision avoidance during leader braking, and smooth role transitions.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import time


class SwarmRole(Enum):
    """Roles that drones can take in the swarm"""
    LEADER = "leader"
    FOLLOWER = "follower"
    SCOUT = "scout"
    IDLE = "idle"


@dataclass
class DroneState:
    """Represents the state of a drone in the swarm"""
    drone_id: str
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    role: SwarmRole
    timestamp: float

    def __post_init__(self):
        self.position = np.array(self.position, dtype=float)
        self.velocity = np.array(self.velocity, dtype=float)


@dataclass
class FormationOffset:
    """Defines a follower's offset relative to leader"""
    offset: np.ndarray  # [dx, dy, dz]
    damping: float = 0.8  # Velocity feedforward damping factor

    def __post_init__(self):
        self.offset = np.array(self.offset, dtype=float)


class SwarmCoordinator:
    """
    Coordinates multiple drones in a formation that can move together

    Handles:
    - Formation keeping with moving leader
    - Velocity feedforward to prevent collisions
    - Role management and transitions
    """

    def __init__(
        self,
        max_velocity: float = 2.0,
        safety_distance: float = 0.5,
        formation_tolerance: float = 0.2
    ):
        """
        Initialize swarm coordinator

        Args:
            max_velocity: Maximum allowed velocity (m/s)
            safety_distance: Minimum safe distance between drones (m)
            formation_tolerance: Acceptable position error in formation (m)
        """
        self.max_velocity = max_velocity
        self.safety_distance = safety_distance
        self.formation_tolerance = formation_tolerance

        # Drone registry
        self.drones: Dict[str, DroneState] = {}

        # Formation configuration
        self.formations: Dict[str, FormationOffset] = {}

        # Role assignments
        self.leader_id: Optional[str] = None
        self.follower_ids: List[str] = []

    def register_drone(
        self,
        drone_id: str,
        initial_position: Tuple[float, float, float],
        role: SwarmRole = SwarmRole.IDLE
    ):
        """
        Register a drone in the swarm

        Args:
            drone_id: Unique identifier for the drone
            initial_position: Starting position (x, y, z)
            role: Initial role assignment
        """
        self.drones[drone_id] = DroneState(
            drone_id=drone_id,
            position=np.array(initial_position),
            velocity=np.array([0.0, 0.0, 0.0]),
            role=role,
            timestamp=time.time()
        )

    def set_formation_offset(
        self,
        follower_id: str,
        offset: Tuple[float, float, float],
        damping: float = 0.8
    ):
        """
        Define a follower's offset relative to the leader

        Args:
            follower_id: ID of the follower drone
            offset: Relative position offset (dx, dy, dz)
            damping: Velocity feedforward damping (0-1)
        """
        self.formations[follower_id] = FormationOffset(
            offset=offset,
            damping=damping
        )

    def assign_leader(self, drone_id: str):
        """
        Assign a drone as the formation leader

        Args:
            drone_id: ID of the drone to make leader
        """
        if drone_id in self.drones:
            # Update previous leader if exists
            if self.leader_id and self.leader_id in self.drones:
                self.drones[self.leader_id].role = SwarmRole.FOLLOWER

            # Set new leader
            self.leader_id = drone_id
            self.drones[drone_id].role = SwarmRole.LEADER

    def assign_follower(self, drone_id: str):
        """
        Assign a drone as a follower

        Args:
            drone_id: ID of the drone to make follower
        """
        if drone_id in self.drones:
            self.drones[drone_id].role = SwarmRole.FOLLOWER
            if drone_id not in self.follower_ids:
                self.follower_ids.append(drone_id)

    def update_drone_state(
        self,
        drone_id: str,
        position: Tuple[float, float, float],
        velocity: Optional[Tuple[float, float, float]] = None
    ):
        """
        Update a drone's state

        Args:
            drone_id: ID of the drone
            position: Current position (x, y, z)
            velocity: Current velocity (vx, vy, vz), estimated if None
        """
        if drone_id not in self.drones:
            return

        drone = self.drones[drone_id]
        new_position = np.array(position)

        # Estimate velocity if not provided
        if velocity is None:
            dt = time.time() - drone.timestamp
            if dt > 0.001:  # Avoid division by zero
                velocity = (new_position - drone.position) / dt
            else:
                velocity = drone.velocity
        else:
            velocity = np.array(velocity)

        # Update state
        drone.position = new_position
        drone.velocity = velocity
        drone.timestamp = time.time()

    def coordinate_moving_formation(
        self,
        leader_id: str,
        follower_id: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Compute follower's target position and velocity for moving formation

        This implements velocity feedforward to prevent the follower from
        colliding with the leader during braking or sudden maneuvers.

        Args:
            leader_id: ID of the leader drone
            follower_id: ID of the follower drone

        Returns:
            Tuple of (target_position, target_velocity) or None if invalid
        """
        # Validate inputs
        if leader_id not in self.drones or follower_id not in self.drones:
            return None

        if follower_id not in self.formations:
            return None

        leader = self.drones[leader_id]
        follower = self.drones[follower_id]
        formation = self.formations[follower_id]

        # 1. Calculate target position (Leader + Offset)
        target_position = leader.position + formation.offset

        # 2. Get leader velocity
        leader_vel = leader.velocity

        # 3. Apply velocity feedforward with damping
        # This helps follower anticipate leader's motion
        target_velocity = leader_vel * formation.damping

        # 4. Add position correction term (P controller)
        position_error = target_position - follower.position
        error_magnitude = np.linalg.norm(position_error)

        if error_magnitude > self.formation_tolerance:
            # Proportional correction to get back in formation
            correction_gain = 1.0
            velocity_correction = position_error * correction_gain
            target_velocity += velocity_correction

        # 5. Safety limit on velocity
        vel_magnitude = np.linalg.norm(target_velocity)
        if vel_magnitude > self.max_velocity:
            target_velocity = (target_velocity / vel_magnitude) * self.max_velocity

        return target_position, target_velocity

    def check_collision_risk(
        self,
        drone1_id: str,
        drone2_id: str
    ) -> Tuple[bool, float]:
        """
        Check if two drones are at risk of collision

        Args:
            drone1_id: ID of first drone
            drone2_id: ID of second drone

        Returns:
            Tuple of (is_at_risk, distance)
        """
        if drone1_id not in self.drones or drone2_id not in self.drones:
            return False, float('inf')

        drone1 = self.drones[drone1_id]
        drone2 = self.drones[drone2_id]

        distance = np.linalg.norm(drone1.position - drone2.position)
        at_risk = distance < self.safety_distance

        return at_risk, distance

    def smooth_role_transition(
        self,
        old_leader_id: str,
        new_leader_id: str,
        transition_time: float = 2.0
    ) -> bool:
        """
        Smoothly transition leadership without causing sudden stops

        Args:
            old_leader_id: Current leader ID
            new_leader_id: New leader ID
            transition_time: Time for smooth transition (s)

        Returns:
            True if transition initiated successfully
        """
        if old_leader_id not in self.drones or new_leader_id not in self.drones:
            return False

        old_leader = self.drones[old_leader_id]
        new_leader = self.drones[new_leader_id]

        # Store old leader's velocity for handoff
        handoff_velocity = old_leader.velocity.copy()

        # Assign new leader
        self.assign_leader(new_leader_id)

        # Make old leader a follower
        self.assign_follower(old_leader_id)

        # New leader should initially maintain the formation velocity
        new_leader.velocity = handoff_velocity

        return True

    def get_formation_status(self) -> Dict[str, any]:
        """
        Get current formation status

        Returns:
            Dictionary with formation metrics
        """
        status = {
            'leader_id': self.leader_id,
            'num_followers': len(self.follower_ids),
            'drones': {}
        }

        if self.leader_id and self.leader_id in self.drones:
            leader = self.drones[self.leader_id]
            status['leader_position'] = leader.position.tolist()
            status['leader_velocity'] = leader.velocity.tolist()

        for follower_id in self.follower_ids:
            if follower_id not in self.drones:
                continue

            follower = self.drones[follower_id]
            formation = self.formations.get(follower_id)

            follower_status = {
                'position': follower.position.tolist(),
                'velocity': follower.velocity.tolist(),
                'role': follower.role.value
            }

            if formation and self.leader_id:
                # Calculate formation error
                result = self.coordinate_moving_formation(self.leader_id, follower_id)
                if result:
                    target_pos, _ = result
                    error = np.linalg.norm(follower.position - target_pos)
                    follower_status['formation_error'] = float(error)
                    follower_status['in_formation'] = error < self.formation_tolerance

            status['drones'][follower_id] = follower_status

        return status

    def emergency_stop(self):
        """
        Command all drones to stop immediately

        This should only be used in emergency situations as it doesn't
        maintain formation or smooth transitions.
        """
        for drone in self.drones.values():
            drone.velocity = np.array([0.0, 0.0, 0.0])


class FormationPresets:
    """
    Predefined formation configurations for common scenarios
    """

    @staticmethod
    def line_formation(spacing: float = 1.5) -> Dict[str, Tuple[float, float, float]]:
        """
        Create a line formation (useful for scanning)

        Args:
            spacing: Distance between drones (m)

        Returns:
            Dictionary mapping follower names to offsets
        """
        return {
            'N1': (0.0, spacing, 0.0),
            'N2': (0.0, -spacing, 0.0)
        }

    @staticmethod
    def triangle_formation(spacing: float = 1.5) -> Dict[str, Tuple[float, float, float]]:
        """
        Create a triangle formation

        Args:
            spacing: Distance from leader (m)

        Returns:
            Dictionary mapping follower names to offsets
        """
        return {
            'N1': (-spacing, spacing * 0.866, 0.0),  # 60 degrees
            'N2': (-spacing, -spacing * 0.866, 0.0)
        }

    @staticmethod
    def jamming_formation(radius: float = 1.0) -> Dict[str, Tuple[float, float, float]]:
        """
        Create a circular formation for jamming (surrounding target)

        Args:
            radius: Radius of formation (m)

        Returns:
            Dictionary mapping follower names to offsets
        """
        return {
            'N1': (0.0, radius, 0.0),
            'N2': (radius * 0.866, -radius * 0.5, 0.0),
            'P': (-radius * 0.866, -radius * 0.5, 0.0)
        }
