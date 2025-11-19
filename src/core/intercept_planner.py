"""
Obstacle-Aware Intercept Planner for Scenario 4
Calculates optimal intercept points considering both target motion and obstacles.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from .path_planner_v4 import GridMap


class ObstacleAwareIntercept:
    """
    Computes intercept points for moving targets while accounting for obstacles.
    Ensures predicted intercept positions are in valid free space.
    """

    def __init__(self, grid_map: GridMap, lookahead_time: float = 1.5):
        """
        Initialize the intercept planner.

        Args:
            grid_map: GridMap for obstacle checking
            lookahead_time: Time horizon for prediction in seconds
        """
        self.grid_map = grid_map
        self.lookahead_time = lookahead_time
        self.max_prediction_iterations = 10

    def calculate_valid_intercept(self, drone_pos: Tuple[float, float],
                                  target_pos: Tuple[float, float],
                                  target_vel: Tuple[float, float],
                                  drone_max_speed: float = 2.0) -> Dict:
        """
        Calculate a valid intercept point considering obstacles.

        Args:
            drone_pos: Current drone position (x, y)
            target_pos: Current target position (x, y)
            target_vel: Current target velocity (vx, vy)
            drone_max_speed: Maximum drone speed in m/s

        Returns:
            Dictionary with:
                - 'intercept_point': Best intercept position (x, y)
                - 'intercept_time': Time to intercept
                - 'is_valid': Whether intercept point is in free space
                - 'strategy': 'direct', 'prediction', or 'current_target'
        """
        # Strategy 1: Try to predict target future position
        predicted_intercept = self._predict_intercept_point(
            drone_pos, target_pos, target_vel, drone_max_speed
        )

        # Check if predicted intercept is valid (in free space)
        if self.grid_map.is_valid(predicted_intercept[0], predicted_intercept[1]):
            # Check line of sight from drone to intercept point
            has_los = self.grid_map.has_line_of_sight(
                drone_pos[0], drone_pos[1],
                predicted_intercept[0], predicted_intercept[1]
            )

            if has_los:
                # Direct intercept is possible
                intercept_time = self._calculate_intercept_time(
                    drone_pos, predicted_intercept, drone_max_speed
                )

                return {
                    'intercept_point': predicted_intercept,
                    'intercept_time': intercept_time,
                    'is_valid': True,
                    'strategy': 'direct'
                }
            else:
                # No line of sight, but point is valid - will need path planning
                intercept_time = self._calculate_intercept_time(
                    drone_pos, predicted_intercept, drone_max_speed
                )

                return {
                    'intercept_point': predicted_intercept,
                    'intercept_time': intercept_time,
                    'is_valid': True,
                    'strategy': 'prediction'
                }

        # Strategy 2: Predicted point is inside obstacle - find valid alternative
        valid_intercept = self._find_valid_intercept_near_obstacle(
            target_pos, target_vel, predicted_intercept
        )

        if valid_intercept is not None:
            intercept_time = self._calculate_intercept_time(
                drone_pos, valid_intercept, drone_max_speed
            )

            return {
                'intercept_point': valid_intercept,
                'intercept_time': intercept_time,
                'is_valid': True,
                'strategy': 'prediction'
            }

        # Strategy 3: Fall back to current target position
        # Target might be behind wall - aim for current known position
        if self.grid_map.is_valid(target_pos[0], target_pos[1]):
            intercept_time = self._calculate_intercept_time(
                drone_pos, target_pos, drone_max_speed
            )

            return {
                'intercept_point': target_pos,
                'intercept_time': intercept_time,
                'is_valid': True,
                'strategy': 'current_target'
            }

        # Strategy 4: Last resort - stay at current position (hover)
        return {
            'intercept_point': drone_pos,
            'intercept_time': 0.0,
            'is_valid': False,
            'strategy': 'hover'
        }

    def _predict_intercept_point(self, drone_pos: Tuple[float, float],
                                 target_pos: Tuple[float, float],
                                 target_vel: Tuple[float, float],
                                 drone_max_speed: float) -> Tuple[float, float]:
        """
        Predict intercept point using iterative refinement.

        This solves for the point where:
        time_for_drone_to_reach_point == time_for_target_to_reach_point

        Args:
            drone_pos: Drone position
            target_pos: Target position
            target_vel: Target velocity
            drone_max_speed: Drone maximum speed

        Returns:
            Predicted intercept point (x, y)
        """
        # Initial guess: simple forward prediction
        predicted_pos = (
            target_pos[0] + target_vel[0] * self.lookahead_time,
            target_pos[1] + target_vel[1] * self.lookahead_time
        )

        # Iteratively refine
        for iteration in range(self.max_prediction_iterations):
            # Calculate time for drone to reach predicted position
            distance_drone = np.sqrt(
                (predicted_pos[0] - drone_pos[0])**2 +
                (predicted_pos[1] - drone_pos[1])**2
            )

            if drone_max_speed > 0:
                time_for_drone = distance_drone / drone_max_speed
            else:
                time_for_drone = self.lookahead_time

            # Predict where target will be at that time
            new_predicted_pos = (
                target_pos[0] + target_vel[0] * time_for_drone,
                target_pos[1] + target_vel[1] * time_for_drone
            )

            # Check convergence
            error = np.sqrt(
                (new_predicted_pos[0] - predicted_pos[0])**2 +
                (new_predicted_pos[1] - predicted_pos[1])**2
            )

            predicted_pos = new_predicted_pos

            if error < 0.05:  # Converged to 5cm accuracy
                break

        # Clamp to map bounds
        predicted_pos = (
            np.clip(predicted_pos[0], 0, self.grid_map.width_m),
            np.clip(predicted_pos[1], 0, self.grid_map.height_m)
        )

        return predicted_pos

    def _find_valid_intercept_near_obstacle(self, target_pos: Tuple[float, float],
                                           target_vel: Tuple[float, float],
                                           predicted_pos: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        Find a valid intercept point near an obstacle.

        If the predicted intercept is inside an obstacle, find the nearest
        free space point along the target's trajectory.

        Args:
            target_pos: Current target position
            target_vel: Target velocity
            predicted_pos: Predicted position (inside obstacle)

        Returns:
            Valid intercept point or None if not found
        """
        # Sample points along the target's trajectory
        target_speed = np.sqrt(target_vel[0]**2 + target_vel[1]**2)

        if target_speed < 0.01:
            # Target is stationary
            return target_pos if self.grid_map.is_valid(target_pos[0], target_pos[1]) else None

        # Normalize velocity
        vel_norm = (target_vel[0] / target_speed, target_vel[1] / target_speed)

        # Sample points at different times
        for dt in np.linspace(0.1, self.lookahead_time * 1.5, 20):
            sample_x = target_pos[0] + vel_norm[0] * target_speed * dt
            sample_y = target_pos[1] + vel_norm[1] * target_speed * dt

            # Clamp to bounds
            sample_x = np.clip(sample_x, 0, self.grid_map.width_m)
            sample_y = np.clip(sample_y, 0, self.grid_map.height_m)

            if self.grid_map.is_valid(sample_x, sample_y):
                return (sample_x, sample_y)

        # Also try going backwards (target emerging from behind obstacle)
        for dt in np.linspace(-0.5, 0, 10):
            sample_x = target_pos[0] + vel_norm[0] * target_speed * dt
            sample_y = target_pos[1] + vel_norm[1] * target_speed * dt

            sample_x = np.clip(sample_x, 0, self.grid_map.width_m)
            sample_y = np.clip(sample_y, 0, self.grid_map.height_m)

            if self.grid_map.is_valid(sample_x, sample_y):
                return (sample_x, sample_y)

        return None

    def _calculate_intercept_time(self, drone_pos: Tuple[float, float],
                                  intercept_pos: Tuple[float, float],
                                  drone_max_speed: float) -> float:
        """
        Calculate time for drone to reach intercept point.

        Args:
            drone_pos: Current drone position
            intercept_pos: Intercept point
            drone_max_speed: Maximum drone speed

        Returns:
            Estimated time to intercept in seconds
        """
        distance = np.sqrt(
            (intercept_pos[0] - drone_pos[0])**2 +
            (intercept_pos[1] - drone_pos[1])**2
        )

        if drone_max_speed > 0:
            return distance / drone_max_speed
        else:
            return float('inf')

    def calculate_intercept_velocity(self, drone_pos: Tuple[float, float],
                                    intercept_point: Tuple[float, float],
                                    intercept_time: float,
                                    target_vel: Tuple[float, float]) -> Tuple[float, float]:
        """
        Calculate required velocity to reach intercept point.

        Args:
            drone_pos: Current drone position
            intercept_point: Desired intercept point
            intercept_time: Time to intercept
            target_vel: Target velocity (for velocity matching)

        Returns:
            Required velocity (vx, vy)
        """
        if intercept_time > 0.01:
            # Velocity to reach intercept point
            required_vx = (intercept_point[0] - drone_pos[0]) / intercept_time
            required_vy = (intercept_point[1] - drone_pos[1]) / intercept_time

            # Blend with target velocity for smooth matching
            blend_factor = 0.3
            final_vx = required_vx * (1 - blend_factor) + target_vel[0] * blend_factor
            final_vy = required_vy * (1 - blend_factor) + target_vel[1] * blend_factor

            return (final_vx, final_vy)
        else:
            # Already at intercept - match target velocity
            return target_vel


class InterceptController:
    """
    High-level controller that combines intercept planning with obstacle avoidance.
    """

    def __init__(self, config: Dict):
        """
        Initialize intercept controller.

        Args:
            config: Configuration dictionary
        """
        # Create grid map
        from .path_planner_v4 import GridMap
        arena_config = config.get('arena_map', {})
        self.grid_map = GridMap(arena_config)

        # Apply safety inflation
        nav_params = config.get('nav_parameters', {})
        safety_margin = nav_params.get('safety_margin', 0.4)
        self.grid_map.inflate_obstacles(safety_margin)

        # Create intercept planner
        lookahead_time = nav_params.get('lookahead_time', 1.5)
        self.intercept_planner = ObstacleAwareIntercept(self.grid_map, lookahead_time)

    def compute_intercept_command(self, drone_pos: Tuple[float, float],
                                 target_pos: Tuple[float, float],
                                 target_vel: Tuple[float, float],
                                 drone_max_speed: float = 2.0) -> Dict:
        """
        Compute command to intercept moving target.

        Args:
            drone_pos: Current drone position
            target_pos: Current target position
            target_vel: Current target velocity
            drone_max_speed: Maximum drone speed

        Returns:
            Command dictionary with 'target_pos', 'target_vel', and metadata
        """
        # Calculate intercept
        intercept_result = self.intercept_planner.calculate_valid_intercept(
            drone_pos, target_pos, target_vel, drone_max_speed
        )

        # Calculate required velocity
        required_vel = self.intercept_planner.calculate_intercept_velocity(
            drone_pos,
            intercept_result['intercept_point'],
            intercept_result['intercept_time'],
            target_vel
        )

        return {
            'target_pos': intercept_result['intercept_point'],
            'target_vel': required_vel,
            'intercept_time': intercept_result['intercept_time'],
            'is_valid': intercept_result['is_valid'],
            'strategy': intercept_result['strategy']
        }


def create_intercept_controller(config: Dict) -> InterceptController:
    """
    Factory function to create an InterceptController.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized InterceptController instance
    """
    return InterceptController(config)
