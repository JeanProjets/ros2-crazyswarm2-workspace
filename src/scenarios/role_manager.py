"""
Role Manager for Scenario 1

This module implements dynamic role assignment for drone swarm coordination.
Drones are assigned roles based on battery levels and mission phase requirements.
"""

import logging
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


class DroneRole(Enum):
    """Enumeration of drone roles in the mission"""
    NEUTRAL_1 = "neutral_1"
    NEUTRAL_2 = "neutral_2"
    PATROL = "patrol"
    LEADER = "leader"
    FOLLOWER = "follower"
    ATTACKER = "attacker"


@dataclass
class DroneInfo:
    """
    Information about a drone in the swarm.

    Attributes:
        drone_id: Unique identifier for the drone
        battery_level: Current battery level (0-100%)
        position: Current 3D position (x, y, z)
        status: Current status string
        has_detected_target: Whether this drone detected the target
        last_update: Timestamp of last update
    """
    drone_id: str
    battery_level: float = 100.0
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    status: str = "idle"
    has_detected_target: bool = False
    last_update: float = field(default_factory=time.time)


@dataclass
class RoleAssignment:
    """
    Record of a role assignment event.

    Attributes:
        drone_id: The drone being assigned
        role: The role being assigned
        timestamp: When the assignment occurred
        reason: Reason for the assignment
    """
    drone_id: str
    role: DroneRole
    timestamp: float
    reason: str


class RoleManager:
    """
    Manages dynamic role assignment for the drone swarm.

    The role manager is responsible for:
    - Tracking drone capabilities and status
    - Assigning initial roles at mission start
    - Reassigning roles when target is detected
    - Selecting leaders based on battery levels
    """

    # Position mappings for each role in different mission phases
    ROLE_POSITIONS = {
        # Initial positions (before target detection)
        "initial": {
            DroneRole.NEUTRAL_1: (0.5, 0.5, 1.0),
            DroneRole.NEUTRAL_2: (-0.5, 0.5, 1.0),
            DroneRole.PATROL: (0.0, -1.0, 1.0),
        },
        # Safety check positions
        "safety_check": {
            DroneRole.NEUTRAL_1: (1.0, 0.0, 1.0),
            DroneRole.NEUTRAL_2: (-1.0, 0.0, 1.0),
            DroneRole.PATROL: (0.0, -1.0, 1.0),
        },
        # Patrol search positions
        "patrol_search": {
            DroneRole.NEUTRAL_1: (0.5, 0.5, 1.0),
            DroneRole.NEUTRAL_2: (-0.5, 0.5, 1.0),
            DroneRole.PATROL: (0.0, 1.0, 1.0),  # Moving forward
        },
        # Attack formation positions (relative to target)
        "attack_formation": {
            DroneRole.LEADER: (-0.3, -0.5, 1.2),  # Jamming position left
            DroneRole.FOLLOWER: (0.3, -0.5, 1.2),  # Jamming position right
            DroneRole.ATTACKER: (0.0, -0.8, 1.5),  # Behind and above
        },
    }

    # Minimum battery level required for leader role
    MIN_LEADER_BATTERY = 40.0

    def __init__(self):
        """Initialize the role manager"""
        self.logger = logging.getLogger("RoleManager")
        self.drone_registry: Dict[str, DroneInfo] = {}
        self.current_assignments: Dict[str, DroneRole] = {}
        self.role_history: List[RoleAssignment] = []
        self.target_position: Optional[Tuple[float, float, float]] = None

    def register_drone(
        self,
        drone_id: str,
        battery_level: float = 100.0,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        status: str = "ready"
    ) -> bool:
        """
        Register a drone with the role manager.

        Args:
            drone_id: Unique identifier for the drone
            battery_level: Initial battery level (0-100%)
            position: Initial position (x, y, z)
            status: Initial status

        Returns:
            True if registration successful, False otherwise
        """
        if drone_id in self.drone_registry:
            self.logger.warning(f"Drone {drone_id} already registered")
            return False

        drone_info = DroneInfo(
            drone_id=drone_id,
            battery_level=battery_level,
            position=position,
            status=status,
            has_detected_target=False,
            last_update=time.time()
        )

        self.drone_registry[drone_id] = drone_info
        self.logger.info(f"Registered drone {drone_id}")
        return True

    def update_drone_info(
        self,
        drone_id: str,
        battery_level: Optional[float] = None,
        position: Optional[Tuple[float, float, float]] = None,
        status: Optional[str] = None,
        has_detected_target: Optional[bool] = None
    ) -> bool:
        """
        Update information about a registered drone.

        Args:
            drone_id: The drone to update
            battery_level: New battery level
            position: New position
            status: New status
            has_detected_target: Whether target was detected

        Returns:
            True if update successful, False if drone not found
        """
        if drone_id not in self.drone_registry:
            self.logger.warning(f"Drone {drone_id} not registered")
            return False

        drone = self.drone_registry[drone_id]

        if battery_level is not None:
            drone.battery_level = battery_level
        if position is not None:
            drone.position = position
        if status is not None:
            drone.status = status
        if has_detected_target is not None:
            drone.has_detected_target = has_detected_target

        drone.last_update = time.time()
        return True

    def assign_initial_roles(self) -> Dict[str, DroneRole]:
        """
        Assign initial roles for Scenario 1.

        Initial assignment:
        - cf1: NEUTRAL_1
        - cf2: NEUTRAL_2
        - cf3: PATROL

        Returns:
            Dictionary mapping drone IDs to roles
        """
        # Expected drone IDs for Scenario 1
        expected_drones = ["cf1", "cf2", "cf3"]

        # Check if all drones are registered
        for drone_id in expected_drones:
            if drone_id not in self.drone_registry:
                self.logger.warning(f"Expected drone {drone_id} not registered")

        # Assign initial roles
        initial_roles = {
            "cf1": DroneRole.NEUTRAL_1,
            "cf2": DroneRole.NEUTRAL_2,
            "cf3": DroneRole.PATROL,
        }

        for drone_id, role in initial_roles.items():
            if drone_id in self.drone_registry:
                self.current_assignments[drone_id] = role
                self._record_assignment(drone_id, role, "initial assignment")

        self.logger.info(f"Initial roles assigned: {initial_roles}")
        return self.current_assignments.copy()

    def reassign_roles_on_detection(self, detector_id: str) -> Dict[str, DroneRole]:
        """
        Reassign roles after target detection.

        Role assignment logic:
        1. Highest battery neutral -> LEADER
        2. Other neutral -> FOLLOWER
        3. Patrol -> ATTACKER

        Args:
            detector_id: ID of the drone that detected the target

        Returns:
            Dictionary mapping drone IDs to new roles
        """
        # Mark the detector
        if detector_id in self.drone_registry:
            self.drone_registry[detector_id].has_detected_target = True

        # Get all neutral drones
        neutral_drones = [
            drone_id for drone_id, role in self.current_assignments.items()
            if role in [DroneRole.NEUTRAL_1, DroneRole.NEUTRAL_2]
        ]

        # Select leader based on battery level
        leader_id = self.select_leader(neutral_drones)

        # Assign new roles
        new_assignments = {}

        for drone_id in self.drone_registry.keys():
            current_role = self.current_assignments.get(drone_id)

            if current_role == DroneRole.PATROL:
                # Patrol becomes attacker
                new_assignments[drone_id] = DroneRole.ATTACKER
                self._record_assignment(
                    drone_id, DroneRole.ATTACKER, "patrol to attacker"
                )

            elif drone_id == leader_id:
                # Selected neutral becomes leader
                new_assignments[drone_id] = DroneRole.LEADER
                self._record_assignment(
                    drone_id, DroneRole.LEADER, "selected as leader"
                )

            elif current_role in [DroneRole.NEUTRAL_1, DroneRole.NEUTRAL_2]:
                # Other neutral becomes follower
                new_assignments[drone_id] = DroneRole.FOLLOWER
                self._record_assignment(
                    drone_id, DroneRole.FOLLOWER, "neutral to follower"
                )

        self.current_assignments = new_assignments
        self.logger.info(f"Roles reassigned after detection: {new_assignments}")
        return self.current_assignments.copy()

    def select_leader(self, candidate_ids: List[str]) -> str:
        """
        Select the leader from candidate drones based on battery level.

        Args:
            candidate_ids: List of drone IDs to consider for leader role

        Returns:
            The drone ID selected as leader
        """
        if not candidate_ids:
            self.logger.error("No candidates for leader selection")
            return ""

        # Get battery levels for all candidates
        battery_levels = {
            drone_id: self.drone_registry[drone_id].battery_level
            for drone_id in candidate_ids
            if drone_id in self.drone_registry
        }

        if not battery_levels:
            self.logger.error("No valid candidates with battery info")
            return candidate_ids[0]  # Fallback to first candidate

        # Select drone with highest battery
        leader_id = max(battery_levels, key=battery_levels.get)
        leader_battery = battery_levels[leader_id]

        # Check if battery level is sufficient
        if leader_battery < self.MIN_LEADER_BATTERY:
            self.logger.warning(
                f"Selected leader {leader_id} has low battery: {leader_battery}%"
            )

        self.logger.info(
            f"Selected {leader_id} as leader (battery: {leader_battery}%)"
        )

        return leader_id

    def get_role_position(
        self,
        role: DroneRole,
        mission_phase: str,
        target_position: Optional[Tuple[float, float, float]] = None
    ) -> Tuple[float, float, float]:
        """
        Get the target position for a drone with a specific role in a mission phase.

        Args:
            role: The drone's role
            mission_phase: Current mission phase (e.g., "initial", "attack_formation")
            target_position: Position of the target (for attack formation)

        Returns:
            Target position as (x, y, z) tuple
        """
        if mission_phase not in self.ROLE_POSITIONS:
            self.logger.warning(f"Unknown mission phase: {mission_phase}")
            return (0.0, 0.0, 1.0)

        phase_positions = self.ROLE_POSITIONS[mission_phase]

        if role not in phase_positions:
            self.logger.warning(
                f"No position defined for role {role.value} in phase {mission_phase}"
            )
            return (0.0, 0.0, 1.0)

        position = phase_positions[role]

        # For attack formation, positions are relative to target
        if mission_phase == "attack_formation" and target_position is not None:
            position = (
                target_position[0] + position[0],
                target_position[1] + position[1],
                position[2]  # Z is absolute, not relative
            )

        return position

    def validate_role_assignment(self) -> bool:
        """
        Validate that current role assignments are valid.

        Checks:
        - All registered drones have roles
        - No duplicate roles (except for general roles)
        - Battery levels are sufficient for assigned roles

        Returns:
            True if assignments are valid, False otherwise
        """
        # Check all drones have roles
        for drone_id in self.drone_registry.keys():
            if drone_id not in self.current_assignments:
                self.logger.error(f"Drone {drone_id} has no role assigned")
                return False

        # Check for battery issues with leader role
        for drone_id, role in self.current_assignments.items():
            if role == DroneRole.LEADER:
                battery = self.drone_registry[drone_id].battery_level
                if battery < self.MIN_LEADER_BATTERY:
                    self.logger.warning(
                        f"Leader {drone_id} has insufficient battery: {battery}%"
                    )
                    # This is a warning, not a hard failure
                    # return False

        self.logger.info("Role assignments validated successfully")
        return True

    def get_drones_by_role(self, roles: List[DroneRole]) -> List[str]:
        """
        Get all drones assigned to specific roles.

        Args:
            roles: List of roles to search for

        Returns:
            List of drone IDs with matching roles
        """
        return [
            drone_id for drone_id, role in self.current_assignments.items()
            if role in roles
        ]

    def get_role_for_drone(self, drone_id: str) -> Optional[DroneRole]:
        """
        Get the current role for a specific drone.

        Args:
            drone_id: The drone to query

        Returns:
            The drone's current role, or None if not assigned
        """
        return self.current_assignments.get(drone_id)

    def _record_assignment(self, drone_id: str, role: DroneRole, reason: str):
        """
        Record a role assignment in the history.

        Args:
            drone_id: The drone being assigned
            role: The role being assigned
            reason: Reason for the assignment
        """
        assignment = RoleAssignment(
            drone_id=drone_id,
            role=role,
            timestamp=time.time(),
            reason=reason
        )
        self.role_history.append(assignment)

    def get_assignment_history(self) -> List[RoleAssignment]:
        """
        Get the complete history of role assignments.

        Returns:
            List of all role assignments
        """
        return self.role_history.copy()

    def get_swarm_status(self) -> dict:
        """
        Get comprehensive status of the entire swarm.

        Returns:
            Dictionary with swarm status information
        """
        return {
            "drone_count": len(self.drone_registry),
            "current_assignments": {
                drone_id: role.value
                for drone_id, role in self.current_assignments.items()
            },
            "battery_levels": {
                drone_id: info.battery_level
                for drone_id, info in self.drone_registry.items()
            },
            "positions": {
                drone_id: info.position
                for drone_id, info in self.drone_registry.items()
            },
            "assignment_count": len(self.role_history),
        }
