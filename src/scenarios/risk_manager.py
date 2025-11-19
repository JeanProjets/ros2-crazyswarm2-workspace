"""
Dynamic Risk Manager for Scenario 4
Decides when it is safe to attack in a cluttered environment
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum


class AttackClearance(Enum):
    """Attack clearance status"""
    SAFE = "SAFE"
    UNSAFE_NEAR_OBSTACLE = "UNSAFE_NEAR_OBSTACLE"
    UNSAFE_DESCENT_PATH = "UNSAFE_DESCENT_PATH"
    UNSAFE_NO_HEADROOM = "UNSAFE_NO_HEADROOM"
    WAITING_FOR_CLEARANCE = "WAITING_FOR_CLEARANCE"


@dataclass
class RiskAssessment:
    """Risk assessment result"""
    clearance: AttackClearance
    risk_score: float  # 0.0 (safe) to 1.0 (very unsafe)
    blocking_obstacles: List[str]
    recommended_action: str


class AttackCorridorValidator:
    """
    Validates whether attack corridor is safe in cluttered environment
    """

    def __init__(self, obstacle_map=None):
        """
        Initialize attack corridor validator

        Args:
            obstacle_map: Map containing obstacle information
        """
        self.obstacle_map = obstacle_map

        # Safety parameters
        self.min_obstacle_clearance = 0.5  # meters
        self.descent_corridor_radius = 0.3  # meters
        self.min_ceiling_height = 2.0  # meters for safe descent

    def is_attack_safe(self, target_pos: np.ndarray,
                      obstacle_map=None) -> bool:
        """
        Check if attack is safe at target position

        Args:
            target_pos: Target position [x, y, z]
            obstacle_map: Obstacle map (optional)

        Returns:
            True if attack is safe
        """
        assessment = self.assess_attack_risk(target_pos, obstacle_map)
        return assessment.clearance == AttackClearance.SAFE

    def assess_attack_risk(self, target_pos: np.ndarray,
                          obstacle_map=None) -> RiskAssessment:
        """
        Comprehensive risk assessment for attack

        Args:
            target_pos: Target position [x, y, z]
            obstacle_map: Obstacle map (optional)

        Returns:
            RiskAssessment object
        """
        if obstacle_map is None:
            obstacle_map = self.obstacle_map

        if obstacle_map is None:
            # No map - cannot assess risk properly, assume safe
            return RiskAssessment(
                clearance=AttackClearance.SAFE,
                risk_score=0.0,
                blocking_obstacles=[],
                recommended_action="PROCEED"
            )

        blocking_obstacles = []
        risk_score = 0.0

        # Check 1: Target proximity to obstacles
        dist_to_obstacle = obstacle_map.get_distance_to_nearest_obstacle(
            target_pos[0], target_pos[1]
        )

        if dist_to_obstacle < self.min_obstacle_clearance:
            blocking_obstacles.append("Target too close to obstacle")
            risk_score += 0.5

            return RiskAssessment(
                clearance=AttackClearance.UNSAFE_NEAR_OBSTACLE,
                risk_score=min(risk_score, 1.0),
                blocking_obstacles=blocking_obstacles,
                recommended_action="HOVER_HIGH - Wait for target to move to open ground"
            )

        # Check 2: Descent path clearance
        descent_safe, descent_risk = self._check_descent_corridor(
            target_pos, obstacle_map
        )

        if not descent_safe:
            blocking_obstacles.append("Descent path blocked")
            risk_score += descent_risk

            return RiskAssessment(
                clearance=AttackClearance.UNSAFE_DESCENT_PATH,
                risk_score=min(risk_score, 1.0),
                blocking_obstacles=blocking_obstacles,
                recommended_action="ABORT - Descent corridor not clear"
            )

        # Check 3: Sufficient headroom
        if target_pos[2] < self.min_ceiling_height:
            blocking_obstacles.append("Insufficient ceiling height")
            risk_score += 0.3

            return RiskAssessment(
                clearance=AttackClearance.UNSAFE_NO_HEADROOM,
                risk_score=min(risk_score, 1.0),
                blocking_obstacles=blocking_obstacles,
                recommended_action="WAIT - Need more altitude clearance"
            )

        # All checks passed
        return RiskAssessment(
            clearance=AttackClearance.SAFE,
            risk_score=risk_score,
            blocking_obstacles=[],
            recommended_action="PROCEED - Attack corridor is clear"
        )

    def _check_descent_corridor(self, target_pos: np.ndarray,
                               obstacle_map) -> Tuple[bool, float]:
        """
        Check if descent corridor is clear of obstacles

        Args:
            target_pos: Target position
            obstacle_map: Obstacle map

        Returns:
            Tuple of (is_safe, risk_score)
        """
        # Sample points in a cylinder from current height down to target
        # Cylinder radius = descent_corridor_radius

        height_samples = 10
        angle_samples = 8
        risk_score = 0.0

        for h in range(height_samples):
            # Height from target up
            z = target_pos[2] + (h / height_samples) * 2.0

            # Sample around perimeter
            for a in range(angle_samples):
                angle = (a / angle_samples) * 2 * np.pi
                x = target_pos[0] + self.descent_corridor_radius * np.cos(angle)
                y = target_pos[1] + self.descent_corridor_radius * np.sin(angle)

                if obstacle_map.is_collision(x, y):
                    # Found obstacle in descent path
                    risk_score += 1.0 / (height_samples * angle_samples)

        # Safe if less than 10% of samples hit obstacles
        is_safe = risk_score < 0.1
        return is_safe, risk_score

    def verify_attack_vector(self, drone_pos: np.ndarray,
                            target_pos: np.ndarray,
                            obstacle_map=None) -> bool:
        """
        Verify attack vector from drone to target is clear

        Args:
            drone_pos: Drone position
            target_pos: Target position
            obstacle_map: Obstacle map

        Returns:
            True if attack vector is clear
        """
        if obstacle_map is None:
            obstacle_map = self.obstacle_map

        if obstacle_map is None:
            return True

        # Check straight line from drone to target
        direction = target_pos - drone_pos
        distance = np.linalg.norm(direction)

        if distance < 1e-6:
            return True

        direction_normalized = direction / distance

        # Sample along vector
        num_samples = int(distance / 0.1) + 1
        for i in range(num_samples):
            sample_dist = (i / num_samples) * distance
            sample_pos = drone_pos + direction_normalized * sample_dist

            # Check with safety margin
            for offset_x in [-self.descent_corridor_radius, 0, self.descent_corridor_radius]:
                for offset_y in [-self.descent_corridor_radius, 0, self.descent_corridor_radius]:
                    check_x = sample_pos[0] + offset_x
                    check_y = sample_pos[1] + offset_y

                    if obstacle_map.is_collision(check_x, check_y):
                        return False

        return True

    def get_safe_hover_position(self, target_pos: np.ndarray,
                               obstacle_map=None) -> np.ndarray:
        """
        Calculate safe hover position above target when waiting for clearance

        Args:
            target_pos: Target position
            obstacle_map: Obstacle map

        Returns:
            Safe hover position [x, y, z]
        """
        # Hover directly above target at safe altitude
        hover_pos = target_pos.copy()
        hover_pos[2] = max(target_pos[2] + 3.0, 4.0)  # At least 3m above target or 4m absolute

        return hover_pos


class DynamicRiskManager:
    """
    High-level risk management for dynamic environments
    """

    def __init__(self, validator: AttackCorridorValidator):
        """
        Initialize risk manager

        Args:
            validator: Attack corridor validator
        """
        self.validator = validator
        self.clearance_wait_time = 0.0
        self.max_wait_time = 5.0  # seconds
        self.last_assessment = None

    def evaluate_strike_conditions(self, drone_pos: np.ndarray,
                                   target_pos: np.ndarray,
                                   target_vel: np.ndarray,
                                   obstacle_map=None) -> Tuple[bool, str]:
        """
        Evaluate if strike should proceed

        Args:
            drone_pos: Current drone position
            target_pos: Target position
            target_vel: Target velocity
            obstacle_map: Obstacle map

        Returns:
            Tuple of (should_strike, reason)
        """
        # Assess risk
        assessment = self.validator.assess_attack_risk(target_pos, obstacle_map)
        self.last_assessment = assessment

        if assessment.clearance == AttackClearance.SAFE:
            # Verify attack vector
            vector_clear = self.validator.verify_attack_vector(
                drone_pos, target_pos, obstacle_map
            )

            if vector_clear:
                self.clearance_wait_time = 0.0
                return True, "CLEAR_TO_STRIKE"
            else:
                return False, "ATTACK_VECTOR_BLOCKED"

        else:
            # Not safe - wait or abort
            return False, assessment.recommended_action

    def should_abort_wait(self, wait_time: float) -> bool:
        """
        Decide if should abort waiting for clearance

        Args:
            wait_time: Time spent waiting

        Returns:
            True if should abort wait and return to pursuit
        """
        return wait_time > self.max_wait_time

    def get_evasive_position(self, drone_pos: np.ndarray,
                            target_pos: np.ndarray,
                            obstacle_map=None) -> np.ndarray:
        """
        Get safe position to maintain pursuit without attacking

        Args:
            drone_pos: Current drone position
            target_pos: Target position
            obstacle_map: Obstacle map

        Returns:
            Safe tracking position
        """
        if self.last_assessment and \
           self.last_assessment.clearance == AttackClearance.UNSAFE_NEAR_OBSTACLE:
            # Target near obstacle - hover high
            return self.validator.get_safe_hover_position(target_pos, obstacle_map)
        else:
            # Maintain current altitude and distance
            offset = drone_pos - target_pos
            offset[2] = 2.0  # Maintain 2m altitude
            if np.linalg.norm(offset[:2]) < 2.0:
                # Too close - back off
                offset[:2] = offset[:2] / np.linalg.norm(offset[:2]) * 2.0

            return target_pos + offset

    def calculate_risk_score(self, target_pos: np.ndarray,
                            drone_pos: np.ndarray,
                            obstacle_map=None) -> float:
        """
        Calculate overall risk score for current situation

        Args:
            target_pos: Target position
            drone_pos: Drone position
            obstacle_map: Obstacle map

        Returns:
            Risk score 0.0 (safe) to 1.0 (very risky)
        """
        assessment = self.validator.assess_attack_risk(target_pos, obstacle_map)
        return assessment.risk_score

    def is_target_cornered(self, target_pos: np.ndarray,
                          obstacle_map=None) -> bool:
        """
        Check if target is cornered (surrounded by obstacles)

        Args:
            target_pos: Target position
            obstacle_map: Obstacle map

        Returns:
            True if target is cornered
        """
        if obstacle_map is None:
            return False

        # Check 8 directions around target
        check_distance = 1.0  # meters
        blocked_directions = 0

        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            check_x = target_pos[0] + check_distance * np.cos(angle)
            check_y = target_pos[1] + check_distance * np.sin(angle)

            if obstacle_map.is_collision(check_x, check_y):
                blocked_directions += 1

        # Cornered if more than 5 out of 8 directions blocked
        return blocked_directions >= 5
