"""
Swarm Formation Manager for Scenario 4
Manages how the swarm moves through bottlenecks and cluttered environments
"""

import numpy as np
from typing import List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass


class FormationMode(Enum):
    """Formation modes for different environments"""
    TIGHT_FORMATION = "TIGHT_FORMATION"  # Open space, compact formation
    COMBAT_SPREAD = "COMBAT_SPREAD"      # Open space, wider spread
    SINGLE_FILE = "SINGLE_FILE"          # Narrow passage
    SPLIT_FLANKING = "SPLIT_FLANKING"    # Obstacle in middle, split around
    INDEPENDENT = "INDEPENDENT"          # Each drone navigates independently


@dataclass
class FormationParameters:
    """Parameters for formation control"""
    mode: FormationMode
    leader_offset: np.ndarray  # Offset for follower relative to leader
    min_separation: float      # Minimum separation between drones
    max_separation: float      # Maximum separation before breaking formation


class FormationManagerV4:
    """
    Formation manager that adapts to cluttered environments
    """

    def __init__(self, obstacle_map=None):
        """
        Initialize formation manager

        Args:
            obstacle_map: Map containing obstacle information
        """
        self.obstacle_map = obstacle_map

        # Formation parameters
        self.tight_offset = np.array([1.0, 0.5, 0.0])  # [x, y, z] offset
        self.combat_offset = np.array([2.0, 1.0, 0.0])
        self.single_file_offset = np.array([2.0, 0.0, 0.0])  # Behind leader

        # Environment thresholds
        self.narrow_passage_width = 1.5  # meters
        self.wide_open_width = 3.0       # meters

        # Current formation state
        self.current_mode = FormationMode.TIGHT_FORMATION
        self.formation_params = self._get_formation_params(self.current_mode)

    def check_formation_integrity(self, leader_pos: np.ndarray,
                                  follower_pos: np.ndarray,
                                  obstacle_map=None) -> bool:
        """
        Check if formation can be maintained safely

        Args:
            leader_pos: Leader drone position [x, y, z]
            follower_pos: Follower drone position [x, y, z]
            obstacle_map: Obstacle map (optional)

        Returns:
            True if formation is safe and intact
        """
        if obstacle_map is None:
            obstacle_map = self.obstacle_map

        if obstacle_map is None:
            # No map - assume formation is okay
            return True

        # Check 1: Follower not in obstacle
        if obstacle_map.is_collision(follower_pos[0], follower_pos[1]):
            return False

        # Check 2: Path between leader and follower is clear
        if not self._is_path_clear(leader_pos, follower_pos, obstacle_map):
            return False

        # Check 3: Separation within acceptable bounds
        separation = np.linalg.norm(leader_pos - follower_pos)
        if separation > self.formation_params.max_separation:
            return False

        return True

    def update_formation_mode(self, leader_pos: np.ndarray,
                             leader_vel: np.ndarray,
                             obstacle_map=None) -> FormationMode:
        """
        Determine appropriate formation mode based on environment

        Args:
            leader_pos: Leader position
            leader_vel: Leader velocity
            obstacle_map: Obstacle map

        Returns:
            Recommended formation mode
        """
        if obstacle_map is None:
            obstacle_map = self.obstacle_map

        if obstacle_map is None:
            # No map - use tight formation
            return FormationMode.TIGHT_FORMATION

        # Assess local environment
        path_width = self._estimate_path_width(leader_pos, leader_vel, obstacle_map)

        if path_width < self.narrow_passage_width:
            # Narrow passage - single file
            new_mode = FormationMode.SINGLE_FILE
        elif path_width > self.wide_open_width:
            # Wide open - combat spread
            new_mode = FormationMode.COMBAT_SPREAD
        else:
            # Medium width - tight formation
            new_mode = FormationMode.TIGHT_FORMATION

        # Update current mode
        if new_mode != self.current_mode:
            print(f"[Formation] Mode change: {self.current_mode.value} -> {new_mode.value}")
            self.current_mode = new_mode
            self.formation_params = self._get_formation_params(new_mode)

        return new_mode

    def calculate_follower_goal(self, leader_pos: np.ndarray,
                               leader_vel: np.ndarray,
                               obstacle_map=None) -> np.ndarray:
        """
        Calculate goal position for follower drone

        Args:
            leader_pos: Leader position
            leader_vel: Leader velocity
            obstacle_map: Obstacle map

        Returns:
            Goal position for follower
        """
        # Base goal is leader position + offset
        offset = self.formation_params.leader_offset

        # Rotate offset based on leader velocity direction
        if np.linalg.norm(leader_vel[:2]) > 0.1:
            # Leader is moving - align formation with velocity
            vel_angle = np.arctan2(leader_vel[1], leader_vel[0])
            rotation_matrix = np.array([
                [np.cos(vel_angle), -np.sin(vel_angle), 0],
                [np.sin(vel_angle), np.cos(vel_angle), 0],
                [0, 0, 1]
            ])
            rotated_offset = rotation_matrix @ offset
        else:
            # Leader stationary - use default offset
            rotated_offset = offset

        ideal_goal = leader_pos + rotated_offset

        # Validate goal is not in obstacle
        if obstacle_map and obstacle_map.is_collision(ideal_goal[0], ideal_goal[1]):
            # Goal is inside obstacle - project to nearest free space
            ideal_goal = self._project_to_free_space(ideal_goal, obstacle_map)

        return ideal_goal

    def _estimate_path_width(self, position: np.ndarray,
                            velocity: np.ndarray,
                            obstacle_map) -> float:
        """
        Estimate width of path in direction of travel

        Args:
            position: Current position
            velocity: Current velocity
            obstacle_map: Obstacle map

        Returns:
            Estimated path width in meters
        """
        # Check perpendicular to velocity direction
        if np.linalg.norm(velocity[:2]) < 0.1:
            # Not moving - check in current facing
            perp_dir = np.array([0, 1, 0])
        else:
            # Moving - check perpendicular to velocity
            vel_2d = velocity[:2] / np.linalg.norm(velocity[:2])
            perp_dir = np.array([-vel_2d[1], vel_2d[0], 0])

        # Ray cast in both perpendicular directions
        max_check_distance = 5.0  # meters
        resolution = 0.1  # meters

        # Left side
        left_clearance = 0.0
        for d in np.arange(0, max_check_distance, resolution):
            check_pos = position + perp_dir * d
            if obstacle_map.is_collision(check_pos[0], check_pos[1]):
                break
            left_clearance = d

        # Right side
        right_clearance = 0.0
        for d in np.arange(0, max_check_distance, resolution):
            check_pos = position - perp_dir * d
            if obstacle_map.is_collision(check_pos[0], check_pos[1]):
                break
            right_clearance = d

        total_width = left_clearance + right_clearance
        return total_width

    def _is_path_clear(self, pos1: np.ndarray, pos2: np.ndarray,
                      obstacle_map) -> bool:
        """
        Check if straight path between two positions is clear

        Args:
            pos1: Start position
            pos2: End position
            obstacle_map: Obstacle map

        Returns:
            True if path is clear
        """
        direction = pos2 - pos1
        distance = np.linalg.norm(direction)

        if distance < 1e-6:
            return True

        direction_normalized = direction / distance

        # Sample along path
        num_samples = int(distance / 0.1) + 1
        for i in range(num_samples):
            sample_dist = (i / num_samples) * distance
            sample_pos = pos1 + direction_normalized * sample_dist

            if obstacle_map.is_collision(sample_pos[0], sample_pos[1]):
                return False

        return True

    def _project_to_free_space(self, position: np.ndarray,
                               obstacle_map) -> np.ndarray:
        """
        Project position to nearest free space

        Args:
            position: Position that might be in obstacle
            obstacle_map: Obstacle map

        Returns:
            Nearest free position
        """
        # Simple spiral search for free space
        search_radius = 2.0
        search_resolution = 0.2

        for r in np.arange(0, search_radius, search_resolution):
            for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
                check_x = position[0] + r * np.cos(angle)
                check_y = position[1] + r * np.sin(angle)

                if not obstacle_map.is_collision(check_x, check_y):
                    return np.array([check_x, check_y, position[2]])

        # Fallback - return original position
        return position

    def _get_formation_params(self, mode: FormationMode) -> FormationParameters:
        """
        Get formation parameters for a given mode

        Args:
            mode: Formation mode

        Returns:
            FormationParameters
        """
        if mode == FormationMode.TIGHT_FORMATION:
            return FormationParameters(
                mode=mode,
                leader_offset=self.tight_offset,
                min_separation=0.5,
                max_separation=3.0
            )
        elif mode == FormationMode.COMBAT_SPREAD:
            return FormationParameters(
                mode=mode,
                leader_offset=self.combat_offset,
                min_separation=1.0,
                max_separation=5.0
            )
        elif mode == FormationMode.SINGLE_FILE:
            return FormationParameters(
                mode=mode,
                leader_offset=self.single_file_offset,
                min_separation=1.5,
                max_separation=4.0
            )
        else:
            # Default
            return FormationParameters(
                mode=mode,
                leader_offset=self.tight_offset,
                min_separation=0.5,
                max_separation=3.0
            )

    def should_split_formation(self, leader_pos: np.ndarray,
                               leader_vel: np.ndarray,
                               obstacle_map=None) -> bool:
        """
        Determine if formation should split around obstacle

        Args:
            leader_pos: Leader position
            leader_vel: Leader velocity
            obstacle_map: Obstacle map

        Returns:
            True if formation should split
        """
        if obstacle_map is None:
            return False

        # Check if there's an obstacle directly ahead
        if np.linalg.norm(leader_vel[:2]) < 0.1:
            return False

        lookahead_dist = 2.0  # meters
        direction = leader_vel / np.linalg.norm(leader_vel)
        check_pos = leader_pos + direction * lookahead_dist

        # If obstacle ahead, consider splitting
        if obstacle_map.is_collision(check_pos[0], check_pos[1]):
            # Check if there's room on both sides
            path_width = self._estimate_path_width(leader_pos, leader_vel, obstacle_map)
            return path_width < self.narrow_passage_width

        return False

    def get_split_positions(self, obstacle_center: np.ndarray,
                           leader_pos: np.ndarray,
                           leader_vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate split positions for flanking around obstacle

        Args:
            obstacle_center: Center of obstacle
            leader_pos: Leader position
            leader_vel: Leader velocity

        Returns:
            Tuple of (left_position, right_position)
        """
        # Perpendicular to velocity
        vel_2d = leader_vel[:2]
        if np.linalg.norm(vel_2d) < 0.1:
            perp = np.array([0, 1])
        else:
            vel_2d = vel_2d / np.linalg.norm(vel_2d)
            perp = np.array([-vel_2d[1], vel_2d[0]])

        # Positions on left and right of obstacle
        flank_distance = 1.5  # meters from obstacle center

        left_pos = obstacle_center[:2] + perp * flank_distance
        right_pos = obstacle_center[:2] - perp * flank_distance

        # Add z coordinate
        left_pos = np.append(left_pos, leader_pos[2])
        right_pos = np.append(right_pos, leader_pos[2])

        return left_pos, right_pos


class SwarmCoordinator:
    """
    High-level coordinator for multi-drone swarm in cluttered environment
    """

    def __init__(self, formation_manager: FormationManagerV4):
        """
        Initialize swarm coordinator

        Args:
            formation_manager: Formation manager
        """
        self.formation_manager = formation_manager
        self.num_drones = 2  # Leader + Follower
        self.leader_id = 0
        self.follower_ids = [1]

    def update_swarm(self, drone_states: List[dict],
                    obstacle_map=None) -> dict:
        """
        Update swarm formation and return commands

        Args:
            drone_states: List of drone state dicts with pos, vel, etc.
            obstacle_map: Obstacle map

        Returns:
            Dict of drone_id -> goal_position
        """
        if len(drone_states) < 2:
            return {}

        leader_state = drone_states[self.leader_id]
        leader_pos = leader_state.get('pos', np.zeros(3))
        leader_vel = leader_state.get('vel', np.zeros(3))

        # Update formation mode
        self.formation_manager.update_formation_mode(
            leader_pos, leader_vel, obstacle_map
        )

        # Calculate follower goals
        goals = {}

        for follower_id in self.follower_ids:
            if follower_id >= len(drone_states):
                continue

            follower_state = drone_states[follower_id]
            follower_pos = follower_state.get('pos', np.zeros(3))

            # Calculate goal
            goal = self.formation_manager.calculate_follower_goal(
                leader_pos, leader_vel, obstacle_map
            )

            goals[follower_id] = goal

            # Check integrity
            integrity = self.formation_manager.check_formation_integrity(
                leader_pos, follower_pos, obstacle_map
            )

            if not integrity:
                print(f"[Swarm] Warning: Formation integrity compromised for drone {follower_id}")

        return goals
