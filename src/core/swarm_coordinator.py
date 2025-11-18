"""
Swarm Coordinator for managing multiple Crazyflie drones.

This module provides coordination logic for multi-drone operations,
including role assignment, communication, and formation control.
"""
import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

try:
    from pycrazyswarm import Crazyswarm
except ImportError:
    from .pycrazyswarm_mock import Crazyswarm

from .drone_controller import DroneController


class DroneRole(Enum):
    """Roles that drones can be assigned during mission."""
    NEUTRAL_1 = "neutral_1"
    NEUTRAL_2 = "neutral_2"
    PATROL = "patrol"
    LEADER = "leader"
    FOLLOWER = "follower"


@dataclass
class DroneStatus:
    """Status information for a single drone."""
    drone_id: str
    position: Tuple[float, float, float]
    battery: float
    role: DroneRole
    target_found: bool = False


class SwarmCoordinator:
    """
    Coordinator for managing a swarm of Crazyflie drones.

    Handles initialization, role assignment, and inter-drone communication
    for coordinated swarm operations.
    """

    def __init__(
        self,
        drone_ids: List[str] = None,
        cage_bounds: Tuple[float, float, float] = (10.0, 6.0, 8.0)
    ):
        """
        Initialize swarm coordinator.

        Args:
            drone_ids: List of drone identifiers (default: ['cf1', 'cf2', 'cf3'])
            cage_bounds: Maximum bounds (x, y, z) for safety
        """
        self.drone_ids = drone_ids or ['cf1', 'cf2', 'cf3']
        self.cage_bounds = cage_bounds
        self.logger = logging.getLogger("SwarmCoordinator")

        # Initialize Crazyswarm
        self.crazyswarm = Crazyswarm()

        # Create controllers for each drone
        self.controllers: Dict[str, DroneController] = {}
        for drone_id in self.drone_ids:
            self.controllers[drone_id] = DroneController(
                drone_id=drone_id,
                crazyswarm=self.crazyswarm,
                cage_bounds=cage_bounds
            )

        # Role assignments
        self.roles: Dict[str, DroneRole] = {}

        # Shared state
        self.drone_statuses: Dict[str, DroneStatus] = {}
        self.target_position: Optional[Tuple[float, float, float]] = None

        self.logger.info(f"Initialized swarm with drones: {self.drone_ids}")

    def initialize_swarm(
        self,
        initial_roles: Optional[Dict[str, DroneRole]] = None
    ) -> bool:
        """
        Initialize swarm with default or specified roles.

        Args:
            initial_roles: Optional role assignments (default: cf1=NEUTRAL_1,
                          cf2=NEUTRAL_2, cf3=PATROL)

        Returns:
            True if initialization successful
        """
        try:
            # Set default roles for Scenario 1
            if initial_roles is None:
                initial_roles = {
                    'cf1': DroneRole.NEUTRAL_1,
                    'cf2': DroneRole.NEUTRAL_2,
                    'cf3': DroneRole.PATROL
                }

            self.roles = initial_roles

            # Initialize status for each drone
            for drone_id in self.drone_ids:
                position = self.controllers[drone_id].get_position()
                battery = self.controllers[drone_id].get_battery_percentage()
                role = self.roles.get(drone_id, DroneRole.NEUTRAL_1)

                self.drone_statuses[drone_id] = DroneStatus(
                    drone_id=drone_id,
                    position=position,
                    battery=battery,
                    role=role,
                    target_found=False
                )

            self.logger.info("Swarm initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Swarm initialization failed: {e}")
            return False

    def assign_initial_positions(
        self,
        positions: Dict[str, Tuple[float, float, float, float]]
    ) -> bool:
        """
        Assign and move drones to initial positions.

        Args:
            positions: Dict mapping drone_id to (x, y, z, height) tuples

        Returns:
            True if all drones reached positions successfully
        """
        try:
            success = True

            for drone_id, (start_x, start_y, start_z, flight_height) in positions.items():
                if drone_id not in self.controllers:
                    self.logger.warning(f"Unknown drone {drone_id}")
                    continue

                controller = self.controllers[drone_id]

                # Takeoff
                self.logger.info(f"{drone_id} taking off to {flight_height}m")
                if not controller.takeoff(height=flight_height, duration=2.0):
                    self.logger.error(f"{drone_id} takeoff failed")
                    success = False
                    continue

                # Update status
                self.drone_statuses[drone_id].position = (
                    start_x, start_y, flight_height
                )

            return success

        except Exception as e:
            self.logger.error(f"Position assignment failed: {e}")
            return False

    def broadcast_drone_status(
        self,
        drone_id: str,
        position: Tuple[float, float, float],
        battery: float,
        target_found: bool = False
    ) -> None:
        """
        Update and broadcast drone status to swarm.

        Args:
            drone_id: Drone identifier
            position: Current (x, y, z) position
            battery: Battery percentage
            target_found: Whether target has been detected
        """
        if drone_id not in self.drone_statuses:
            self.logger.warning(f"Unknown drone {drone_id}")
            return

        # Update status
        self.drone_statuses[drone_id].position = position
        self.drone_statuses[drone_id].battery = battery
        self.drone_statuses[drone_id].target_found = target_found

        self.logger.debug(
            f"{drone_id} status: pos={position}, "
            f"battery={battery:.1f}%, target={target_found}"
        )

        # If target found, update shared target position
        if target_found and self.target_position is None:
            self.logger.info(f"{drone_id} detected target at {position}")
            self.target_position = position

    def select_leader(self) -> Optional[str]:
        """
        Select leader drone based on battery levels.

        Returns:
            Drone ID of selected leader, or None if selection fails
        """
        try:
            # Filter drones that are neutral or patrol (not already assigned)
            candidates = [
                drone_id for drone_id, status in self.drone_statuses.items()
                if status.role in [DroneRole.NEUTRAL_1, DroneRole.NEUTRAL_2, DroneRole.PATROL]
            ]

            if not candidates:
                self.logger.error("No candidates available for leader selection")
                return None

            # Select drone with highest battery
            leader_id = max(
                candidates,
                key=lambda did: self.drone_statuses[did].battery
            )

            # Update role
            self.roles[leader_id] = DroneRole.LEADER
            self.drone_statuses[leader_id].role = DroneRole.LEADER

            self.logger.info(
                f"Selected {leader_id} as LEADER "
                f"(battery: {self.drone_statuses[leader_id].battery:.1f}%)"
            )

            return leader_id

        except Exception as e:
            self.logger.error(f"Leader selection failed: {e}")
            return None

    def select_follower(self, exclude_ids: List[str] = None) -> Optional[str]:
        """
        Select follower drone.

        Args:
            exclude_ids: Drone IDs to exclude from selection

        Returns:
            Drone ID of selected follower, or None if selection fails
        """
        try:
            exclude_ids = exclude_ids or []

            # Filter available drones
            candidates = [
                drone_id for drone_id, status in self.drone_statuses.items()
                if (drone_id not in exclude_ids and
                    status.role in [DroneRole.NEUTRAL_1, DroneRole.NEUTRAL_2, DroneRole.PATROL])
            ]

            if not candidates:
                self.logger.error("No candidates available for follower selection")
                return None

            # Select drone with next highest battery
            follower_id = max(
                candidates,
                key=lambda did: self.drone_statuses[did].battery
            )

            # Update role
            self.roles[follower_id] = DroneRole.FOLLOWER
            self.drone_statuses[follower_id].role = DroneRole.FOLLOWER

            self.logger.info(
                f"Selected {follower_id} as FOLLOWER "
                f"(battery: {self.drone_statuses[follower_id].battery:.1f}%)"
            )

            return follower_id

        except Exception as e:
            self.logger.error(f"Follower selection failed: {e}")
            return None

    def coordinate_formation(
        self,
        leader_id: str,
        follower_id: str,
        formation_offset: float = 0.5
    ) -> bool:
        """
        Coordinate leader-follower formation movement.

        Args:
            leader_id: ID of leader drone
            follower_id: ID of follower drone
            formation_offset: Distance offset for follower (meters)

        Returns:
            True if formation coordinated successfully
        """
        try:
            if self.target_position is None:
                self.logger.error("No target position set")
                return False

            leader_pos = self.drone_statuses[leader_id].position
            target_x, target_y, target_z = self.target_position

            # Leader goes directly to target
            self.logger.info(f"{leader_id} (LEADER) moving to target")
            self.controllers[leader_id].go_to(
                target_x, target_y, target_z, duration=3.0
            )

            # Follower stays offset behind leader
            follower_x = target_x - formation_offset
            follower_y = target_y
            follower_z = target_z

            self.logger.info(f"{follower_id} (FOLLOWER) moving to offset position")
            self.controllers[follower_id].go_to(
                follower_x, follower_y, follower_z, duration=3.0
            )

            return True

        except Exception as e:
            self.logger.error(f"Formation coordination failed: {e}")
            return False

    def get_drone_status(self, drone_id: str) -> Optional[DroneStatus]:
        """
        Get status of specific drone.

        Args:
            drone_id: Drone identifier

        Returns:
            DroneStatus object or None if drone not found
        """
        return self.drone_statuses.get(drone_id)

    def get_all_statuses(self) -> Dict[str, DroneStatus]:
        """
        Get status of all drones.

        Returns:
            Dictionary mapping drone IDs to DroneStatus objects
        """
        return self.drone_statuses.copy()

    def emergency_land_all(self) -> None:
        """
        Emergency land all drones in the swarm.
        """
        self.logger.warning("EMERGENCY LANDING ALL DRONES")
        for drone_id, controller in self.controllers.items():
            try:
                controller.emergency_stop()
            except Exception as e:
                self.logger.error(f"Emergency landing {drone_id} failed: {e}")
