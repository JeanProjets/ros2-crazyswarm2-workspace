"""
Battery-Critical Role Manager for Scenario 2

This module implements strict battery-based role assignment for long-range
corner missions where energy management is critical.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DroneRole(Enum):
    """Drone roles in the mission"""
    UNASSIGNED = "unassigned"
    LEADER = "leader"  # Primary/Pursuer - flies furthest and hovers longest
    FOLLOWER_LEFT = "follower_left"  # Neutral drone supporting from left
    FOLLOWER_RIGHT = "follower_right"  # Neutral drone supporting from right
    STANDBY = "standby"  # Backup drone


class MissionPhase(Enum):
    """Mission phases with different energy requirements"""
    INIT = "init"
    TRANSIT = "transit"
    PATROL = "patrol"
    APPROACH = "approach"
    JAMMING = "jamming"
    ATTACK = "attack"
    RTH = "return_to_home"


@dataclass
class Drone:
    """Drone information"""
    drone_id: str
    voltage: float
    role: DroneRole = DroneRole.UNASSIGNED
    battery_percentage: Optional[float] = None

    def __repr__(self):
        return f"Drone({self.drone_id}, {self.voltage}V, {self.role.value})"


class BatteryThresholds:
    """Battery voltage thresholds for mission phases"""
    # Minimum voltage to assign as Leader at mission start
    MIN_LEADER_VOLTAGE_INIT = 3.8

    # Critical voltage during mission - triggers abort
    CRITICAL_VOLTAGE_FLIGHT = 3.5

    # Warning voltage - consider role swap or extra monitoring
    WARNING_VOLTAGE = 3.6

    # Minimum safe voltage for any drone to participate
    MIN_PARTICIPATION_VOLTAGE = 3.7

    # Energy budget per phase (estimated voltage drop)
    PHASE_ENERGY_COST = {
        MissionPhase.INIT: 0.05,
        MissionPhase.TRANSIT: 0.15,  # Long distance
        MissionPhase.PATROL: 0.10,
        MissionPhase.APPROACH: 0.08,
        MissionPhase.JAMMING: 0.07,
        MissionPhase.ATTACK: 0.05,
        MissionPhase.RTH: 0.15
    }


class BatteryRoleManager:
    """
    Manages role assignment based on battery levels

    In Scenario 2, the Leader drone flies the furthest distance and hovers
    longest, so it must have the highest battery. This class enforces strict
    battery-based role assignment and monitors energy throughout the mission.
    """

    def __init__(self):
        """Initialize the battery role manager"""
        self.thresholds = BatteryThresholds()
        self.assigned_drones: Dict[str, Drone] = {}
        self.role_history: List[Dict] = []
        self.mission_aborted = False
        self.abort_reason: Optional[str] = None

    def select_highest_voltage_drone(self, candidates: List[Drone]) -> Optional[Drone]:
        """
        Select the drone with the highest voltage from candidates

        Args:
            candidates: List of Drone objects

        Returns:
            Drone with highest voltage, or None if list is empty
        """
        if not candidates:
            logger.warning("No candidate drones provided")
            return None

        # Sort by voltage (descending) and return the first
        sorted_drones = sorted(candidates, key=lambda d: d.voltage, reverse=True)
        highest_voltage_drone = sorted_drones[0]

        logger.info(f"Selected {highest_voltage_drone.drone_id} with {highest_voltage_drone.voltage}V "
                   f"from {len(candidates)} candidates")

        return highest_voltage_drone

    def assign_roles(self, available_drones: List[Drone]) -> Dict[DroneRole, Optional[Drone]]:
        """
        Assign roles to drones based on battery levels

        The drone with the most battery becomes Leader. Requirement from cite:53.

        Args:
            available_drones: List of available drones with current voltages

        Returns:
            Dict mapping DroneRole to assigned Drone, or None if assignment fails
        """
        if len(available_drones) < 2:
            logger.error(f"Insufficient drones: need at least 2, got {len(available_drones)}")
            return {}

        # Check if highest voltage meets minimum requirement
        highest = self.select_highest_voltage_drone(available_drones)
        if not highest or highest.voltage < self.thresholds.MIN_LEADER_VOLTAGE_INIT:
            self.mission_aborted = True
            self.abort_reason = f"ABORT: Highest battery ({highest.voltage if highest else 'N/A'}V) " \
                               f"< {self.thresholds.MIN_LEADER_VOLTAGE_INIT}V minimum for Leader"
            logger.critical(self.abort_reason)
            return {}

        # Assign Leader
        leader = highest
        leader.role = DroneRole.LEADER

        # Assign Followers from remaining drones
        remaining = [d for d in available_drones if d.drone_id != leader.drone_id]
        remaining_sorted = sorted(remaining, key=lambda d: d.voltage, reverse=True)

        assignments = {DroneRole.LEADER: leader}

        # Assign up to 2 followers
        if len(remaining_sorted) >= 1:
            follower_left = remaining_sorted[0]
            follower_left.role = DroneRole.FOLLOWER_LEFT
            assignments[DroneRole.FOLLOWER_LEFT] = follower_left

        if len(remaining_sorted) >= 2:
            follower_right = remaining_sorted[1]
            follower_right.role = DroneRole.FOLLOWER_RIGHT
            assignments[DroneRole.FOLLOWER_RIGHT] = follower_right

        # Any remaining drones are standby
        for i in range(2, len(remaining_sorted)):
            remaining_sorted[i].role = DroneRole.STANDBY
            assignments[DroneRole.STANDBY] = remaining_sorted[i]

        # Store assignments
        for drone in available_drones:
            self.assigned_drones[drone.drone_id] = drone

        # Log the assignment
        self._log_role_assignment(assignments)

        logger.info(f"Role assignment complete: Leader={leader.drone_id}({leader.voltage}V)")

        return assignments

    def validate_energy_budget(self, phase: MissionPhase, current_voltage: float) -> bool:
        """
        Validate if current voltage is sufficient for the given phase

        Args:
            phase: Current mission phase
            current_voltage: Current battery voltage

        Returns:
            True if voltage is sufficient, False otherwise
        """
        # Get energy cost for this phase
        energy_cost = self.thresholds.PHASE_ENERGY_COST.get(phase, 0.1)

        # Calculate minimum required voltage
        # Need enough for current phase + return to home
        rth_cost = self.thresholds.PHASE_ENERGY_COST[MissionPhase.RTH]
        min_required = self.thresholds.CRITICAL_VOLTAGE_FLIGHT + energy_cost + rth_cost

        if current_voltage < min_required:
            logger.warning(f"Insufficient energy for {phase.value}: "
                         f"{current_voltage}V < {min_required}V required")
            return False

        logger.debug(f"Energy budget OK for {phase.value}: {current_voltage}V >= {min_required}V")
        return True

    def check_leader_voltage(self, current_voltage: float, phase: MissionPhase) -> Dict[str, any]:
        """
        Check Leader's voltage and determine if action is needed

        Args:
            current_voltage: Current Leader voltage
            phase: Current mission phase

        Returns:
            Dict with 'status', 'action', and 'reason' keys
        """
        result = {
            'status': 'OK',
            'action': 'CONTINUE',
            'reason': None,
            'voltage': current_voltage
        }

        # Critical check - immediate abort
        if current_voltage < self.thresholds.CRITICAL_VOLTAGE_FLIGHT:
            result['status'] = 'CRITICAL'
            result['action'] = 'IMMEDIATE_RTH'
            result['reason'] = f"Leader voltage {current_voltage}V < critical {self.thresholds.CRITICAL_VOLTAGE_FLIGHT}V"
            logger.critical(result['reason'])
            self.mission_aborted = True
            self.abort_reason = result['reason']
            return result

        # Warning check
        if current_voltage < self.thresholds.WARNING_VOLTAGE:
            result['status'] = 'WARNING'
            result['action'] = 'MONITOR_CLOSELY'
            result['reason'] = f"Leader voltage {current_voltage}V approaching critical"
            logger.warning(result['reason'])
            return result

        # Check energy budget for current phase
        if not self.validate_energy_budget(phase, current_voltage):
            result['status'] = 'WARNING'
            result['action'] = 'CONSIDER_RTH'
            result['reason'] = f"Insufficient energy budget for {phase.value}"
            logger.warning(result['reason'])
            return result

        return result

    def _log_role_assignment(self, assignments: Dict[DroneRole, Optional[Drone]]):
        """Log role assignment to history"""
        log_entry = {
            'timestamp': 'MISSION_START',
            'assignments': {
                role.value: drone.drone_id if drone else None
                for role, drone in assignments.items()
            },
            'voltages': {
                role.value: drone.voltage if drone else None
                for role, drone in assignments.items()
            }
        }
        self.role_history.append(log_entry)

    def get_leader(self) -> Optional[Drone]:
        """Get the currently assigned Leader drone"""
        for drone in self.assigned_drones.values():
            if drone.role == DroneRole.LEADER:
                return drone
        return None

    def get_role_summary(self) -> Dict:
        """Get summary of current role assignments"""
        return {
            'assignments': {
                drone_id: drone.role.value
                for drone_id, drone in self.assigned_drones.items()
            },
            'voltages': {
                drone_id: drone.voltage
                for drone_id, drone in self.assigned_drones.items()
            },
            'mission_aborted': self.mission_aborted,
            'abort_reason': self.abort_reason
        }

    def update_drone_voltage(self, drone_id: str, new_voltage: float):
        """
        Update a drone's voltage reading

        Args:
            drone_id: ID of the drone
            new_voltage: New voltage reading
        """
        if drone_id in self.assigned_drones:
            old_voltage = self.assigned_drones[drone_id].voltage
            self.assigned_drones[drone_id].voltage = new_voltage
            logger.debug(f"Updated {drone_id} voltage: {old_voltage}V -> {new_voltage}V")
        else:
            logger.warning(f"Attempted to update voltage for unknown drone: {drone_id}")
