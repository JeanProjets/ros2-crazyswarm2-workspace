"""
Scenario 3: Mobile Target Interception

State machine for dynamic target tracking, jamming, and interception.
The target moves in a circular or square pattern, requiring predictive
tracking and velocity matching.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from enum import Enum
import time
import yaml
import os


class MissionState(Enum):
    """States in the Scenario 3 mission"""
    TAKEOFF = "takeoff"
    PATROL_SEARCH = "patrol_search"
    FALLBACK_SCAN = "fallback_scan"
    DYNAMIC_APPROACH = "dynamic_approach"
    MOVING_JAM = "moving_jam"
    INTERCEPTION_STRIKE = "interception_strike"
    RETURN_HOME = "return_home"
    LANDED = "landed"


class PatrolPattern(Enum):
    """Patrol search patterns"""
    LAWNMOWER = "lawnmower"
    SPIRAL = "spiral"
    GRID = "grid"


class TargetPattern(Enum):
    """Target movement patterns"""
    CIRCLE = "circle"
    SQUARE = "square"
    FIGURE_EIGHT = "figure_eight"


class Scenario3Mission:
    """
    Mission controller for Scenario 3: Mobile Target

    Manages the complete mission flow from patrol search through
    target interception with moving target tracking.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Scenario 3 mission

        Args:
            config_path: Path to configuration YAML file
        """
        self.state = MissionState.TAKEOFF
        self.previous_state: Optional[MissionState] = None

        # Load configuration
        self.config = self._load_config(config_path)

        # Mission timing
        self.state_start_time = time.time()
        self.mission_start_time = time.time()
        self.target_last_seen: Optional[float] = None

        # Target tracking
        self.target_position: Optional[np.ndarray] = None
        self.target_velocity: Optional[np.ndarray] = None
        self.target_detected = False

        # Patrol state
        self.patrol_waypoints: List[np.ndarray] = []
        self.current_waypoint_index = 0
        self.patrol_complete = False

        # Jamming state
        self.jamming_start_time: Optional[float] = None
        self.jamming_duration = self.config.get('jamming_duration', 20.0)

        # Strike state
        self.strike_initiated = False

        # Performance metrics
        self.metrics = {
            'detection_time': None,
            'interception_time': None,
            'jamming_distance_errors': [],
            'successful_strike': False
        }

    def _load_config(self, config_path: Optional[str]) -> dict:
        """
        Load configuration from YAML file

        Args:
            config_path: Path to config file

        Returns:
            Configuration dictionary
        """
        default_config = {
            'takeoff_height': 1.0,
            'patrol_height': 1.5,
            'patrol_pattern': 'LAWNMOWER',
            'patrol_bounds': {
                'x_min': 0.0, 'x_max': 5.0,
                'y_min': 0.0, 'y_max': 5.0
            },
            'patrol_spacing': 1.0,
            'target_timeout': 60.0,
            'fallback_positions': {
                'N1': [3.0, 1.5, 4.0],
                'N2': [3.0, 3.0, 4.0],
                'P': [3.0, 4.5, 4.0]
            },
            'scan_yaw_rate': 0.5,
            'jamming_distance': 1.0,
            'jamming_duration': 20.0,
            'strike_dive_height': 0.3,
            'strike_pullup_height': 2.0,
            'max_velocity': 2.0,
            'target_dynamics': {
                'estimated_speed': 0.5,
                'pattern_type': 'CIRCLE_OR_SQUARE',
                'prediction_lookahead': 0.5
            },
            'tracking_gains': {
                'kp_pos': 1.5,
                'kd_vel': 0.5,
                'max_vel': 1.5
            }
        }

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")

        return default_config

    def update_target_state(
        self,
        position: Optional[Tuple[float, float, float]],
        velocity: Optional[Tuple[float, float, float]] = None
    ):
        """
        Update the tracked target's state

        Args:
            position: Target position (x, y, z), None if not visible
            velocity: Target velocity (vx, vy, vz)
        """
        if position is not None:
            self.target_position = np.array(position)
            self.target_detected = True
            self.target_last_seen = time.time()

            if velocity is not None:
                self.target_velocity = np.array(velocity)
            else:
                # Estimate velocity if not provided
                if self.target_velocity is None:
                    self.target_velocity = np.array([0.0, 0.0, 0.0])
        else:
            self.target_detected = False

    def _generate_lawnmower_patrol(self) -> List[np.ndarray]:
        """
        Generate lawnmower pattern waypoints for patrol

        Returns:
            List of waypoints [x, y, z]
        """
        bounds = self.config['patrol_bounds']
        spacing = self.config['patrol_spacing']
        height = self.config['patrol_height']

        waypoints = []
        x = bounds['x_min']
        y = bounds['y_min']
        direction = 1  # 1 for forward, -1 for backward

        while x <= bounds['x_max']:
            waypoints.append(np.array([x, y, height]))

            # Move to other end of y range
            if direction == 1:
                y = bounds['y_max']
            else:
                y = bounds['y_min']

            waypoints.append(np.array([x, y, height]))

            # Move to next x position and flip direction
            x += spacing
            direction *= -1

        return waypoints

    def _check_state_timeout(self) -> bool:
        """
        Check if current state has timed out

        Returns:
            True if timeout occurred
        """
        time_in_state = time.time() - self.state_start_time

        # Different timeouts for different states
        if self.state == MissionState.PATROL_SEARCH:
            # If target not seen after patrol timeout, go to fallback
            if not self.target_detected and self.patrol_complete:
                return time_in_state > self.config['target_timeout']

        return False

    def transition_to(self, new_state: MissionState):
        """
        Transition to a new mission state

        Args:
            new_state: State to transition to
        """
        print(f"[Mission] State transition: {self.state.value} -> {new_state.value}")
        self.previous_state = self.state
        self.state = new_state
        self.state_start_time = time.time()

    def update(
        self,
        drone_positions: Dict[str, np.ndarray],
        drone_velocities: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, any]:
        """
        Update mission state machine

        Args:
            drone_positions: Dictionary of drone_id -> position
            drone_velocities: Dictionary of drone_id -> velocity

        Returns:
            Command dictionary for drones
        """
        # State machine logic
        if self.state == MissionState.TAKEOFF:
            return self._handle_takeoff(drone_positions)

        elif self.state == MissionState.PATROL_SEARCH:
            return self._handle_patrol_search(drone_positions)

        elif self.state == MissionState.FALLBACK_SCAN:
            return self._handle_fallback_scan(drone_positions)

        elif self.state == MissionState.DYNAMIC_APPROACH:
            return self._handle_dynamic_approach(drone_positions)

        elif self.state == MissionState.MOVING_JAM:
            return self._handle_moving_jam(drone_positions)

        elif self.state == MissionState.INTERCEPTION_STRIKE:
            return self._handle_interception_strike(drone_positions)

        elif self.state == MissionState.RETURN_HOME:
            return self._handle_return_home(drone_positions)

        else:
            return {'action': 'idle'}

    def _handle_takeoff(self, drone_positions: Dict[str, np.ndarray]) -> dict:
        """Handle takeoff state"""
        target_height = self.config['takeoff_height']

        # Check if all drones reached target height
        all_at_height = all(
            pos[2] >= target_height - 0.1
            for pos in drone_positions.values()
        )

        if all_at_height:
            # Initialize patrol
            self.patrol_waypoints = self._generate_lawnmower_patrol()
            self.current_waypoint_index = 0
            self.transition_to(MissionState.PATROL_SEARCH)

        return {
            'action': 'takeoff',
            'target_height': target_height
        }

    def _handle_patrol_search(self, drone_positions: Dict[str, np.ndarray]) -> dict:
        """Handle patrol search state"""
        # If target detected, transition to approach
        if self.target_detected and self.target_position is not None:
            if self.metrics['detection_time'] is None:
                self.metrics['detection_time'] = time.time() - self.mission_start_time
            self.transition_to(MissionState.DYNAMIC_APPROACH)
            return {'action': 'target_detected'}

        # Continue patrol
        if self.current_waypoint_index >= len(self.patrol_waypoints):
            self.patrol_complete = True

            # Check timeout - if no target found, go to fallback
            if self._check_state_timeout():
                self.transition_to(MissionState.FALLBACK_SCAN)
                return {'action': 'fallback_triggered'}

        # Get next waypoint
        if self.current_waypoint_index < len(self.patrol_waypoints):
            waypoint = self.patrol_waypoints[self.current_waypoint_index]

            # Check if drones reached waypoint
            leader_pos = next(iter(drone_positions.values()))
            if np.linalg.norm(leader_pos - waypoint) < 0.3:
                self.current_waypoint_index += 1

            return {
                'action': 'patrol',
                'waypoint': waypoint.tolist(),
                'waypoint_index': self.current_waypoint_index,
                'total_waypoints': len(self.patrol_waypoints)
            }

        return {'action': 'patrol_complete'}

    def _handle_fallback_scan(self, drone_positions: Dict[str, np.ndarray]) -> dict:
        """Handle fallback scanning state - line up at 3m and scan"""
        # Position drones at fallback positions
        fallback_positions = self.config['fallback_positions']

        # If target detected during scan, go to approach
        if self.target_detected and self.target_position is not None:
            self.transition_to(MissionState.DYNAMIC_APPROACH)
            return {'action': 'target_detected'}

        return {
            'action': 'fallback_scan',
            'positions': fallback_positions,
            'yaw_rate': self.config['scan_yaw_rate']
        }

    def _handle_dynamic_approach(self, drone_positions: Dict[str, np.ndarray]) -> dict:
        """Handle dynamic approach to moving target"""
        if not self.target_detected or self.target_position is None:
            # Lost target, return to patrol or fallback
            self.transition_to(MissionState.PATROL_SEARCH)
            return {'action': 'target_lost'}

        # Check if close enough to start jamming
        leader_pos = next(iter(drone_positions.values()))
        distance_to_target = np.linalg.norm(leader_pos - self.target_position)

        if distance_to_target < self.config['jamming_distance'] * 1.5:
            self.transition_to(MissionState.MOVING_JAM)
            return {'action': 'jamming_range_reached'}

        return {
            'action': 'dynamic_approach',
            'target_position': self.target_position.tolist(),
            'target_velocity': self.target_velocity.tolist() if self.target_velocity is not None else [0, 0, 0],
            'distance': float(distance_to_target)
        }

    def _handle_moving_jam(self, drone_positions: Dict[str, np.ndarray]) -> dict:
        """Handle moving jamming - match velocity with target"""
        if self.jamming_start_time is None:
            self.jamming_start_time = time.time()

        if not self.target_detected or self.target_position is None:
            # Lost target
            self.transition_to(MissionState.PATROL_SEARCH)
            return {'action': 'target_lost'}

        # Calculate jamming metrics
        leader_pos = next(iter(drone_positions.values()))
        distance_to_target = np.linalg.norm(leader_pos - self.target_position)
        self.metrics['jamming_distance_errors'].append(distance_to_target)

        # Check if jamming duration complete
        jamming_time = time.time() - self.jamming_start_time
        if jamming_time >= self.jamming_duration:
            if self.metrics['interception_time'] is None:
                self.metrics['interception_time'] = time.time() - self.mission_start_time
            self.transition_to(MissionState.INTERCEPTION_STRIKE)
            return {'action': 'jamming_complete'}

        return {
            'action': 'moving_jam',
            'target_position': self.target_position.tolist(),
            'target_velocity': self.target_velocity.tolist() if self.target_velocity is not None else [0, 0, 0],
            'distance': float(distance_to_target),
            'jamming_time': float(jamming_time),
            'jamming_duration': self.jamming_duration
        }

    def _handle_interception_strike(self, drone_positions: Dict[str, np.ndarray]) -> dict:
        """Handle semi-kamikaze strike on moving target"""
        if not self.target_detected or self.target_position is None:
            # Lost target, return home
            self.transition_to(MissionState.RETURN_HOME)
            return {'action': 'target_lost'}

        if not self.strike_initiated:
            self.strike_initiated = True

        # Predict impact point based on target velocity
        if self.target_velocity is not None:
            lookahead = self.config['target_dynamics']['prediction_lookahead']
            impact_point = self.target_position + (self.target_velocity * lookahead)
        else:
            impact_point = self.target_position

        # Check if strike completed (got very close)
        leader_pos = next(iter(drone_positions.values()))
        distance = np.linalg.norm(leader_pos[:2] - self.target_position[:2])

        if distance < 0.3 or leader_pos[2] < self.config['strike_dive_height']:
            self.metrics['successful_strike'] = True
            self.transition_to(MissionState.RETURN_HOME)
            return {'action': 'strike_complete'}

        return {
            'action': 'interception_strike',
            'impact_point': impact_point.tolist(),
            'dive_height': self.config['strike_dive_height'],
            'pullup_height': self.config['strike_pullup_height']
        }

    def _handle_return_home(self, drone_positions: Dict[str, np.ndarray]) -> dict:
        """Handle return to home position"""
        # Simple return to origin
        home_position = np.array([0.0, 0.0, self.config['takeoff_height']])

        # Check if all drones are home
        all_home = all(
            np.linalg.norm(pos - home_position) < 0.3
            for pos in drone_positions.values()
        )

        if all_home:
            self.transition_to(MissionState.LANDED)

        return {
            'action': 'return_home',
            'home_position': home_position.tolist()
        }

    def get_mission_status(self) -> dict:
        """
        Get current mission status

        Returns:
            Dictionary with mission metrics and state
        """
        return {
            'state': self.state.value,
            'mission_time': time.time() - self.mission_start_time,
            'state_time': time.time() - self.state_start_time,
            'target_detected': self.target_detected,
            'target_position': self.target_position.tolist() if self.target_position is not None else None,
            'metrics': {
                'detection_time': self.metrics['detection_time'],
                'interception_time': self.metrics['interception_time'],
                'avg_jamming_error': np.mean(self.metrics['jamming_distance_errors']) if self.metrics['jamming_distance_errors'] else None,
                'successful_strike': self.metrics['successful_strike']
            }
        }
