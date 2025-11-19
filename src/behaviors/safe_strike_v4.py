"""
Safe Strike Behavior for Scenario 4
Implements obstacle-aware neutralization logic
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum


class StrikeState(Enum):
    """States for strike execution"""
    READY = 1  # Ready to strike
    CHECKING = 2  # Verifying strike corridor
    ATTACKING = 3  # Executing strike
    ABORTING = 4  # Strike aborted due to obstacle
    HOLDING = 5  # Waiting for safe opportunity
    COMPLETED = 6  # Strike completed


@dataclass
class StrikeParameters:
    """Parameters for strike execution"""
    min_strike_distance: float = 0.3  # Minimum distance to initiate strike
    max_strike_distance: float = 2.0  # Maximum distance to initiate strike
    strike_speed: float = 1.5  # Speed during strike (m/s)
    safety_margin: float = 0.4  # Safety margin from obstacles (m)
    corridor_width: float = 0.3  # Width of safe corridor needed (m)
    min_hover_altitude: float = 0.5  # Minimum altitude for holding pattern


class SafeDynamicStrike:
    """
    Implements obstacle-aware strike behavior for Scenario 4.

    Before executing "kamikaze" dive, verifies that the attack
    corridor is clear of obstacles. Aborts if unsafe.
    """

    def __init__(self, params: Optional[StrikeParameters] = None):
        """
        Initialize SafeDynamicStrike.

        Args:
            params: Strike parameters (uses defaults if None)
        """
        self.params = params or StrikeParameters()
        self.state = StrikeState.READY
        self.grid_map: Optional['GridMap'] = None
        self.target_position: Optional[np.ndarray] = None
        self.target_velocity: Optional[np.ndarray] = None
        self.strike_vector: Optional[np.ndarray] = None

    def set_grid_map(self, grid_map: 'GridMap') -> None:
        """
        Set the occupancy grid map from Agent 1.

        Args:
            grid_map: GridMap instance with obstacle information
        """
        self.grid_map = grid_map

    def update_target(self, position: np.ndarray, velocity: np.ndarray) -> None:
        """
        Update target state.

        Args:
            position: Target position [x, y, z]
            velocity: Target velocity [vx, vy, vz]
        """
        self.target_position = position.copy()
        self.target_velocity = velocity.copy()

    def verify_attack_corridor(self, drone_pos: np.ndarray,
                               target_pos: np.ndarray,
                               attack_vector: np.ndarray) -> Tuple[bool, str]:
        """
        Verify that attack corridor is safe.

        Checks if the path from drone to target intersects any obstacles
        or their inflation zones.

        Args:
            drone_pos: Current drone position [x, y, z]
            target_pos: Target position [x, y, z]
            attack_vector: Planned attack direction (unit vector)

        Returns:
            Tuple of (is_safe, reason)
        """
        if self.grid_map is None:
            # No map available - assume safe (risky but necessary)
            return True, "No map available"

        # Ray-cast along attack vector
        distance = np.linalg.norm(target_pos - drone_pos)

        if distance < 1e-6:
            return False, "Already at target"

        # Check points along the attack path
        num_samples = int(distance / 0.05) + 1  # Sample every 5cm
        num_samples = max(num_samples, 10)

        for i in range(num_samples):
            t = i / (num_samples - 1)
            sample_point = drone_pos + t * (target_pos - drone_pos)

            if self.grid_map.is_collision(sample_point):
                return False, f"Obstacle at {t*100:.0f}% along path"

            # Also check corridor width (lateral clearance)
            if not self._check_lateral_clearance(sample_point, attack_vector):
                return False, f"Insufficient lateral clearance at {t*100:.0f}%"

        # Check target vicinity
        if self._is_target_near_obstacle(target_pos):
            return False, "Target too close to obstacle"

        return True, "Corridor clear"

    def _check_lateral_clearance(self, point: np.ndarray,
                                 direction: np.ndarray) -> bool:
        """
        Check if there's enough lateral clearance around a point.

        Args:
            point: Point to check
            direction: Direction of motion (for perpendicular check)

        Returns:
            True if clearance is sufficient
        """
        if self.grid_map is None:
            return True

        # Find perpendicular vectors
        direction_mag = np.linalg.norm(direction)
        if direction_mag < 1e-6:
            return True

        dir_unit = direction / direction_mag

        # Create perpendicular vectors in horizontal plane
        perp1 = np.array([-dir_unit[1], dir_unit[0], 0.0])
        perp1_mag = np.linalg.norm(perp1)
        if perp1_mag > 1e-6:
            perp1 = perp1 / perp1_mag
        else:
            perp1 = np.array([1.0, 0.0, 0.0])

        perp2 = np.cross(dir_unit, perp1)

        # Check clearance in perpendicular directions
        check_distance = self.params.corridor_width / 2.0

        for perp in [perp1, perp2, -perp1, -perp2]:
            check_point = point + perp * check_distance
            if self.grid_map.is_collision(check_point):
                return False

        return True

    def _is_target_near_obstacle(self, target_pos: np.ndarray,
                                 proximity_threshold: float = 0.5) -> bool:
        """
        Check if target is dangerously close to obstacles.

        Args:
            target_pos: Target position
            proximity_threshold: Distance threshold (meters)

        Returns:
            True if target is too close to obstacles
        """
        if self.grid_map is None:
            return False

        # Check vicinity around target
        num_samples = 8
        for i in range(num_samples):
            angle = 2 * np.pi * i / num_samples
            offset = proximity_threshold * np.array([
                np.cos(angle),
                np.sin(angle),
                0.0
            ])

            check_point = target_pos + offset
            if self.grid_map.is_collision(check_point):
                return True

        return False

    def calculate_strike_approach(self, drone_pos: np.ndarray,
                                  target_pos: np.ndarray,
                                  target_vel: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Calculate approach vector for moving strike.

        Intercept angle optimized for moving target.

        Args:
            drone_pos: Current drone position [x, y, z]
            target_pos: Target position [x, y, z]
            target_vel: Target velocity [vx, vy, vz]

        Returns:
            Tuple of (approach_vector, approach_type)
        """
        # Calculate direct vector to target
        to_target = target_pos - drone_pos
        distance = np.linalg.norm(to_target)

        if distance < 1e-6:
            return np.zeros(3), "at_target"

        # Consider target velocity for intercept
        target_speed = np.linalg.norm(target_vel)

        if target_speed > 0.1:
            # Moving target - aim ahead
            # Simple proportional navigation
            time_to_intercept = distance / self.params.strike_speed

            # Predict target position
            predicted_pos = target_pos + target_vel * time_to_intercept

            # Recalculate approach
            approach_vec = predicted_pos - drone_pos
            approach_mag = np.linalg.norm(approach_vec)

            if approach_mag > 1e-6:
                approach_vec = approach_vec / approach_mag

            return approach_vec, "intercept"
        else:
            # Stationary or slow target - direct approach
            return to_target / distance, "direct"

    def execute_strike(self, drone_pos: np.ndarray,
                      drone_vel: np.ndarray,
                      current_time: float) -> Tuple[np.ndarray, str, StrikeState]:
        """
        Execute or abort strike based on safety check.

        Main decision logic:
        1. Check distance to target
        2. Calculate attack vector
        3. Verify corridor safety
        4. Execute if safe, hold if unsafe

        Args:
            drone_pos: Current drone position [x, y, z]
            drone_vel: Current drone velocity [vx, vy, vz]
            current_time: Current time (seconds)

        Returns:
            Tuple of (command_velocity, status_message, new_state)
        """
        if self.target_position is None or self.target_velocity is None:
            return np.zeros(3), "No target data", StrikeState.READY

        # Calculate distance to target
        to_target = self.target_position - drone_pos
        distance = np.linalg.norm(to_target)

        # Check if in strike range
        if distance < self.params.min_strike_distance:
            # Too close - consider strike complete or collision
            return np.zeros(3), "Strike range - complete", StrikeState.COMPLETED

        if distance > self.params.max_strike_distance:
            # Too far - not ready yet
            return np.zeros(3), "Out of strike range", StrikeState.READY

        # Calculate approach vector
        attack_vector, approach_type = self.calculate_strike_approach(
            drone_pos, self.target_position, self.target_velocity
        )

        # Verify corridor safety
        is_safe, reason = self.verify_attack_corridor(
            drone_pos, self.target_position, attack_vector
        )

        if is_safe:
            # Execute strike
            cmd_vel = attack_vector * self.params.strike_speed

            # Match target velocity component for moving strike
            if approach_type == "intercept":
                cmd_vel = cmd_vel + 0.5 * self.target_velocity

            # Limit speed
            cmd_mag = np.linalg.norm(cmd_vel)
            max_strike = self.params.strike_speed * 1.5
            if cmd_mag > max_strike:
                cmd_vel = (cmd_vel / cmd_mag) * max_strike

            return cmd_vel, f"Strike executing: {reason}", StrikeState.ATTACKING

        else:
            # Unsafe - abort and hold
            hold_vel = self._calculate_holding_pattern(drone_pos, self.target_position)
            return hold_vel, f"Strike aborted: {reason}", StrikeState.ABORTING

    def _calculate_holding_pattern(self, drone_pos: np.ndarray,
                                  target_pos: np.ndarray) -> np.ndarray:
        """
        Calculate holding pattern velocity when strike is unsafe.

        Maintains position near target while waiting for safe opportunity.

        Args:
            drone_pos: Current drone position
            target_pos: Target position

        Returns:
            Holding pattern velocity
        """
        # Maintain safe distance from target
        to_target = target_pos - drone_pos
        distance = np.linalg.norm(to_target)

        # Ideal holding distance (edge of strike range)
        ideal_distance = (self.params.max_strike_distance +
                         self.params.min_strike_distance) / 2.0

        if distance < 1e-6:
            # At target - back off
            return np.array([0.0, 0.0, 0.5])  # Move up

        # Proportional control to maintain distance
        error = ideal_distance - distance
        kp = 0.5

        # Move towards or away to maintain ideal distance
        hold_vel = -(error * kp) * (to_target / distance)

        # Match target velocity to stay with it
        if self.target_velocity is not None:
            hold_vel = hold_vel + 0.8 * self.target_velocity

        return hold_vel

    def should_attempt_strike(self, drone_pos: np.ndarray) -> bool:
        """
        Determine if conditions are right for strike attempt.

        Args:
            drone_pos: Current drone position

        Returns:
            True if should attempt strike
        """
        if self.target_position is None:
            return False

        distance = np.linalg.norm(self.target_position - drone_pos)

        return (self.params.min_strike_distance <= distance <=
                self.params.max_strike_distance)

    def get_state(self) -> StrikeState:
        """Get current strike state"""
        return self.state

    def reset(self) -> None:
        """Reset strike state"""
        self.state = StrikeState.READY
        self.strike_vector = None
