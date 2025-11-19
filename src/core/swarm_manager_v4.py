"""
Swarm Manager V4 for Scenario 4
Implements "Rubber Band" formation navigation with independent obstacle avoidance.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from .path_planner_v4 import GridMap, DynamicAStar


class SwarmFormation:
    """
    Defines formation offsets for follower drones relative to leader.
    """

    def __init__(self, formation_type: str = "line", spacing: float = 1.0):
        """
        Initialize formation configuration.

        Args:
            formation_type: Type of formation ("line", "triangle", "square")
            spacing: Distance between drones in meters
        """
        self.formation_type = formation_type
        self.spacing = spacing
        self.offsets = self._compute_offsets()

    def _compute_offsets(self) -> Dict[int, Tuple[float, float]]:
        """
        Compute formation offsets for each drone ID.

        Returns:
            Dictionary mapping drone_id to (x_offset, y_offset)
        """
        offsets = {}

        if self.formation_type == "line":
            # Linear formation along Y-axis
            for i in range(1, 10):  # Support up to 10 drones
                offsets[i] = (0.0, -i * self.spacing)

        elif self.formation_type == "triangle":
            # Triangular formation
            offsets[1] = (-self.spacing, -self.spacing)
            offsets[2] = (self.spacing, -self.spacing)
            offsets[3] = (-2 * self.spacing, -2 * self.spacing)
            offsets[4] = (0.0, -2 * self.spacing)
            offsets[5] = (2 * self.spacing, -2 * self.spacing)

        elif self.formation_type == "square":
            # Square formation
            offsets[1] = (self.spacing, 0.0)
            offsets[2] = (0.0, self.spacing)
            offsets[3] = (self.spacing, self.spacing)

        else:
            # Default: line formation
            for i in range(1, 10):
                offsets[i] = (0.0, -i * self.spacing)

        return offsets

    def get_desired_position(self, follower_id: int, leader_pos: Tuple[float, float],
                            leader_heading: float = 0.0) -> Tuple[float, float]:
        """
        Calculate desired position for a follower relative to leader.

        Args:
            follower_id: ID of the follower drone
            leader_pos: Current leader position (x, y)
            leader_heading: Leader heading in radians (for rotation of formation)

        Returns:
            Desired position (x, y) for the follower
        """
        if follower_id not in self.offsets:
            # Default offset if not defined
            offset = (0.0, -follower_id * self.spacing)
        else:
            offset = self.offsets[follower_id]

        # Rotate offset by leader heading
        cos_h = np.cos(leader_heading)
        sin_h = np.sin(leader_heading)

        rotated_x = offset[0] * cos_h - offset[1] * sin_h
        rotated_y = offset[0] * sin_h + offset[1] * cos_h

        # Add to leader position
        desired_x = leader_pos[0] + rotated_x
        desired_y = leader_pos[1] + rotated_y

        return (desired_x, desired_y)


class ObstacleAwareFollower:
    """
    Manages a follower drone that independently navigates around obstacles
    while trying to maintain formation with the leader.
    """

    def __init__(self, drone_id: int, grid_map: GridMap):
        """
        Initialize obstacle-aware follower controller.

        Args:
            drone_id: Unique ID for this follower drone
            grid_map: Shared GridMap for obstacle checking
        """
        self.drone_id = drone_id
        self.grid_map = grid_map
        self.planner = DynamicAStar(grid_map)

        # State
        self.current_position = (0.0, 0.0)
        self.current_path = None
        self.path_index = 0

    def update_position(self, position: Tuple[float, float]):
        """Update the current position of this drone."""
        self.current_position = position

    def compute_command(self, desired_position: Tuple[float, float],
                       leader_velocity: Optional[Tuple[float, float]] = None) -> Dict:
        """
        Compute navigation command to reach desired formation position.

        Args:
            desired_position: Target position in formation
            leader_velocity: Leader's current velocity (for feedforward)

        Returns:
            Dictionary with 'target_pos' and 'target_vel'
        """
        # Check if we have line of sight to desired position
        has_los = self.grid_map.has_line_of_sight(
            self.current_position[0], self.current_position[1],
            desired_position[0], desired_position[1]
        )

        if has_los:
            # Direct path is clear - use velocity feedforward like Scenario 3
            target_pos = desired_position

            if leader_velocity is not None:
                target_vel = leader_velocity
            else:
                # Simple proportional control
                error_x = desired_position[0] - self.current_position[0]
                error_y = desired_position[1] - self.current_position[1]
                gain = 1.0
                target_vel = (error_x * gain, error_y * gain)

        else:
            # Obstacle blocking - use A* pathfinding
            if self.current_path is None or self._needs_replan(desired_position):
                self.current_path = self.planner.plan_path(
                    self.current_position,
                    desired_position
                )
                self.path_index = 0

            # Follow the path
            if self.current_path and len(self.current_path) > 0:
                # Find next waypoint ahead
                if self.path_index < len(self.current_path):
                    target_pos = self.current_path[self.path_index]

                    # Check if we're close to current waypoint
                    dist_to_wp = np.sqrt(
                        (target_pos[0] - self.current_position[0])**2 +
                        (target_pos[1] - self.current_position[1])**2
                    )

                    if dist_to_wp < 0.3:  # Within 30cm - advance to next waypoint
                        self.path_index += 1
                        if self.path_index < len(self.current_path):
                            target_pos = self.current_path[self.path_index]
                else:
                    # Reached end of path
                    target_pos = desired_position

                # Compute velocity toward waypoint
                error_x = target_pos[0] - self.current_position[0]
                error_y = target_pos[1] - self.current_position[1]
                gain = 1.5
                target_vel = (error_x * gain, error_y * gain)

            else:
                # No path found - hover in place
                target_pos = self.current_position
                target_vel = (0.0, 0.0)

        return {
            'target_pos': target_pos,
            'target_vel': target_vel,
            'has_line_of_sight': has_los
        }

    def _needs_replan(self, new_goal: Tuple[float, float]) -> bool:
        """
        Check if we need to replan the path.

        Args:
            new_goal: New desired goal position

        Returns:
            True if replanning is needed
        """
        if self.current_path is None:
            return True

        # Check if goal has moved significantly
        if len(self.current_path) > 0:
            last_goal = self.current_path[-1]
            goal_moved = np.sqrt(
                (new_goal[0] - last_goal[0])**2 +
                (new_goal[1] - last_goal[1])**2
            )

            if goal_moved > 0.5:  # Goal moved more than 50cm
                return True

        # Check if we've completed the path
        if self.path_index >= len(self.current_path):
            return True

        return False


class SwarmManagerV4:
    """
    Main swarm coordinator for Scenario 4.
    Manages leader and multiple followers with obstacle avoidance.
    """

    def __init__(self, config: Dict):
        """
        Initialize swarm manager.

        Args:
            config: Configuration dictionary with arena_map and formation settings
        """
        # Create shared grid map
        arena_config = config.get('arena_map', {})
        self.grid_map = GridMap(arena_config)

        # Apply safety inflation
        nav_params = config.get('nav_parameters', {})
        safety_margin = nav_params.get('safety_margin', 0.4)
        self.grid_map.inflate_obstacles(safety_margin)

        # Formation configuration
        formation_config = config.get('formation', {})
        formation_type = formation_config.get('type', 'line')
        spacing = formation_config.get('spacing', 1.0)
        self.formation = SwarmFormation(formation_type, spacing)

        # Follower controllers
        self.followers: Dict[int, ObstacleAwareFollower] = {}

        # Leader state
        self.leader_id = 0
        self.leader_position = (0.0, 0.0)
        self.leader_velocity = (0.0, 0.0)
        self.leader_heading = 0.0

    def add_follower(self, drone_id: int):
        """Add a follower drone to the swarm."""
        if drone_id not in self.followers:
            self.followers[drone_id] = ObstacleAwareFollower(drone_id, self.grid_map)
            print(f"Added follower drone {drone_id} to swarm")

    def update_leader_state(self, position: Tuple[float, float],
                           velocity: Tuple[float, float],
                           heading: float = 0.0):
        """
        Update leader drone state.

        Args:
            position: Leader position (x, y)
            velocity: Leader velocity (vx, vy)
            heading: Leader heading in radians
        """
        self.leader_position = position
        self.leader_velocity = velocity
        self.leader_heading = heading

    def update_follower_position(self, drone_id: int, position: Tuple[float, float]):
        """Update a follower's current position."""
        if drone_id in self.followers:
            self.followers[drone_id].update_position(position)

    def coordinate_obstacle_swarm(self, leader_id: int, follower_ids: List[int]) -> Dict[int, Dict]:
        """
        Coordinate swarm navigation through obstacles.

        Args:
            leader_id: ID of the leader drone
            follower_ids: List of follower drone IDs

        Returns:
            Dictionary mapping drone_id to command dict with 'target_pos' and 'target_vel'
        """
        commands = {}

        # Leader follows its own path (provided externally)
        # Followers compute their paths to maintain formation

        for follower_id in follower_ids:
            if follower_id not in self.followers:
                self.add_follower(follower_id)

            # Calculate desired formation position
            desired_pos = self.formation.get_desired_position(
                follower_id,
                self.leader_position,
                self.leader_heading
            )

            # Compute obstacle-aware command
            command = self.followers[follower_id].compute_command(
                desired_pos,
                self.leader_velocity
            )

            commands[follower_id] = command

        return commands

    def get_formation_status(self) -> Dict:
        """
        Get current formation status.

        Returns:
            Dictionary with formation metrics
        """
        status = {
            'leader_position': self.leader_position,
            'num_followers': len(self.followers),
            'followers': {}
        }

        for follower_id, follower in self.followers.items():
            desired_pos = self.formation.get_desired_position(
                follower_id,
                self.leader_position,
                self.leader_heading
            )

            # Calculate formation error
            error = np.sqrt(
                (follower.current_position[0] - desired_pos[0])**2 +
                (follower.current_position[1] - desired_pos[1])**2
            )

            status['followers'][follower_id] = {
                'current_position': follower.current_position,
                'desired_position': desired_pos,
                'formation_error': error,
                'has_path': follower.current_path is not None
            }

        return status


def create_swarm_manager(config: Dict) -> SwarmManagerV4:
    """
    Factory function to create a SwarmManagerV4 instance.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized SwarmManagerV4 instance
    """
    return SwarmManagerV4(config)
