"""
Battery-Optimized Swarm Manager for Scenario 2.

This module implements advanced swarm coordination with:
- Battery-based leader selection
- Dynamic formation offset adjustment to prevent wall collisions
- Role management for multi-drone missions
"""

import logging
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from src.core.safe_drone_controller import SafeDroneController


class DroneRole(Enum):
    """Enumeration of possible drone roles in the swarm."""
    PATROL = "P"           # Patrol drone - scouts and detects target
    NEUTRAL_1 = "N1"       # Neutral drone 1 - can become Leader or Follower
    NEUTRAL_2 = "N2"       # Neutral drone 2 - can become Leader or Follower
    CAMERA = "C"           # Camera drone - virtual/static observer
    LEADER = "L"           # Leader - coordinates attack
    FOLLOWER = "S"         # Follower (Support) - follows leader


class SwarmCoordinator:
    """
    Coordinates multiple drones with battery-optimized role assignment.

    This coordinator is specifically designed for Scenario 2 where:
    - Target is in a corner (9.5, 0.5, 5.0)
    - Boundary avoidance is critical
    - Battery management determines role assignment
    """

    # Standard formation offset (before dynamic adjustment)
    DEFAULT_FORMATION_OFFSET = (-0.5, -0.5, -0.5)  # (dx, dy, dz) relative to leader

    # Boundary threshold for Y-axis inversion (meters from wall)
    Y_BOUNDARY_THRESHOLD = 1.0  # If leader Y < this, invert follower Y offset

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the swarm coordinator.

        Args:
            config: Configuration dictionary with drone setup
        """
        self.logger = logging.getLogger("SwarmCoordinator")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] [SwarmManager] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.config = config or {}
        self.drones: Dict[str, SafeDroneController] = {}
        self.roles: Dict[str, DroneRole] = {}

        # Extract safety bounds from config
        self.bounds = self.config.get('safety_bounds', {
            'x_min': 0.3,
            'x_max': 9.7,
            'y_min': 0.3,
            'y_max': 5.7,
            'z_min': 0.2,
            'z_max': 5.8
        })

    def add_drone(self, drone_id: str, controller: SafeDroneController, role: DroneRole):
        """
        Add a drone to the swarm.

        Args:
            drone_id: Unique identifier for the drone
            controller: SafeDroneController instance
            role: Initial role assignment
        """
        self.drones[drone_id] = controller
        self.roles[drone_id] = role
        self.logger.info(f"Added drone {drone_id} with role {role.value}")

    def initialize_swarm(self, drone_configs: Dict[str, Dict[str, Any]]):
        """
        Initialize the entire swarm from configuration.

        Args:
            drone_configs: Dictionary mapping drone IDs to their configurations
        """
        for drone_id, drone_config in drone_configs.items():
            role_str = drone_config.get('role', 'NEUTRAL_1')
            role = DroneRole[role_str]

            # Create controller for this drone
            controller = SafeDroneController(
                drone_id=drone_id,
                crazyswarm=None,  # Mock for simulation
                config=self.config
            )

            # Set initial position if provided
            if 'start_pos' in drone_config:
                x, y, z = drone_config['start_pos']
                controller.state.x = x
                controller.state.y = y
                controller.state.z = z

            self.add_drone(drone_id, controller, role)

    def select_optimal_leader(self, candidates: List[str]) -> str:
        """
        Select the optimal leader from candidate drones based on battery voltage.

        Strict Rule: Highest voltage becomes LEADER.
        Tiebreaker: If voltage difference < 0.1V, prefer drone closest to X center.

        Args:
            candidates: List of drone IDs to consider for leadership

        Returns:
            Drone ID of the selected leader
        """
        if not candidates:
            raise ValueError("No candidates provided for leader selection")

        # Get battery voltages for all candidates
        voltages = {}
        for drone_id in candidates:
            if drone_id not in self.drones:
                self.logger.warning(f"Candidate {drone_id} not found in swarm, skipping")
                continue
            voltages[drone_id] = self.drones[drone_id].get_battery_voltage()

        if not voltages:
            raise ValueError("No valid candidates with voltage data")

        # Sort by voltage (descending)
        sorted_candidates = sorted(voltages.items(), key=lambda x: x[1], reverse=True)

        # Check if top two are within 0.1V
        if len(sorted_candidates) > 1:
            top_voltage = sorted_candidates[0][1]
            second_voltage = sorted_candidates[1][1]

            if abs(top_voltage - second_voltage) < 0.1:
                # Tiebreaker: prefer drone closest to X center
                x_center = (self.bounds['x_min'] + self.bounds['x_max']) / 2.0
                self.logger.info(f"Voltage tiebreaker! Top candidates within 0.1V. Using X-center proximity.")

                # Get X positions for tied candidates
                tied_candidates = [
                    (drone_id, voltage) for drone_id, voltage in sorted_candidates
                    if abs(voltage - top_voltage) < 0.1
                ]

                # Select based on distance to X center
                best_candidate = min(
                    tied_candidates,
                    key=lambda x: abs(self.drones[x[0]].state.x - x_center)
                )
                selected_leader = best_candidate[0]
            else:
                selected_leader = sorted_candidates[0][0]
        else:
            selected_leader = sorted_candidates[0][0]

        selected_voltage = voltages[selected_leader]
        self.logger.info(
            f"✓ Selected {selected_leader} as LEADER (Voltage: {selected_voltage:.2f}V)"
        )

        # Log all candidate voltages for transparency
        for drone_id, voltage in sorted_candidates:
            self.logger.info(f"  - {drone_id}: {voltage:.2f}V")

        return selected_leader

    def calculate_safe_formation(
        self,
        leader_pos: Tuple[float, float, float],
        follower_id: str
    ) -> Tuple[float, float, float]:
        """
        Calculate safe formation position for follower relative to leader.

        Scenario 2 Check: If Leader is near Y=0 boundary, dynamically invert
        Y-offset to prevent Follower from colliding with wall.

        Standard offset: (-0.5, -0.5, -0.5)
        Near Y=0 wall:   (-0.5, +0.5, -0.5)

        Args:
            leader_pos: Leader position as (x, y, z)
            follower_id: ID of the follower drone

        Returns:
            Safe follower position as (x, y, z)
        """
        leader_x, leader_y, leader_z = leader_pos

        # Start with default offset
        dx, dy, dz = self.DEFAULT_FORMATION_OFFSET

        # Dynamic Y-offset inversion near boundaries
        if leader_y < self.Y_BOUNDARY_THRESHOLD:
            # Leader is near Y=0 wall, invert Y offset to push follower away from wall
            dy = abs(dy)  # Make positive to move follower in +Y direction
            self.logger.info(
                f"⚠️  Leader near Y=0 boundary ({leader_y:.2f}m), "
                f"inverting formation Y-offset to +{dy:.2f}m"
            )
        elif leader_y > (self.bounds['y_max'] - self.Y_BOUNDARY_THRESHOLD):
            # Leader is near Y_max wall, ensure negative offset
            dy = -abs(dy)  # Make negative to move follower in -Y direction
            self.logger.info(
                f"⚠️  Leader near Y_max boundary ({leader_y:.2f}m), "
                f"using formation Y-offset of {dy:.2f}m"
            )

        # Calculate follower position
        follower_x = leader_x + dx
        follower_y = leader_y + dy
        follower_z = leader_z + dz

        # Verify position is safe using the follower's controller
        if follower_id in self.drones:
            follower_controller = self.drones[follower_id]
            if not follower_controller.is_position_safe(follower_x, follower_y, follower_z):
                # Clamp to safe position
                follower_x, follower_y, follower_z = follower_controller.clamp_position(
                    follower_x, follower_y, follower_z
                )
                self.logger.warning(
                    f"Formation position unsafe for {follower_id}, clamped to safe bounds"
                )

        self.logger.info(
            f"Formation: Leader at ({leader_x:.2f}, {leader_y:.2f}, {leader_z:.2f}) -> "
            f"Follower {follower_id} at ({follower_x:.2f}, {follower_y:.2f}, {follower_z:.2f})"
        )

        return (follower_x, follower_y, follower_z)

    def assign_roles(self, leader_id: str, follower_id: str):
        """
        Assign LEADER and FOLLOWER roles to specific drones.

        Args:
            leader_id: Drone to assign as LEADER
            follower_id: Drone to assign as FOLLOWER
        """
        if leader_id in self.roles:
            self.roles[leader_id] = DroneRole.LEADER
            self.logger.info(f"Assigned {leader_id} -> LEADER")

        if follower_id in self.roles:
            self.roles[follower_id] = DroneRole.FOLLOWER
            self.logger.info(f"Assigned {follower_id} -> FOLLOWER")

    def get_drone_by_role(self, role: DroneRole) -> Optional[str]:
        """
        Get drone ID for a specific role.

        Args:
            role: Role to search for

        Returns:
            Drone ID if found, None otherwise
        """
        for drone_id, drone_role in self.roles.items():
            if drone_role == role:
                return drone_id
        return None

    def get_swarm_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the entire swarm.

        Returns:
            Dictionary containing swarm metrics
        """
        status = {
            'total_drones': len(self.drones),
            'roles': {drone_id: role.value for drone_id, role in self.roles.items()},
            'drones': {}
        }

        for drone_id, controller in self.drones.items():
            status['drones'][drone_id] = controller.get_safety_stats()

        return status

    def log_battery_status(self):
        """
        Log battery status for all drones in the swarm.
        Should be called every 5 seconds during mission.
        """
        self.logger.info("=== Battery Status ===")
        for drone_id, controller in self.drones.items():
            voltage = controller.get_battery_voltage()
            status = controller.check_battery_status()
            role = self.roles.get(drone_id, DroneRole.NEUTRAL_1).value
            self.logger.info(f"  [{role}] {drone_id}: {voltage:.2f}V ({status})")
