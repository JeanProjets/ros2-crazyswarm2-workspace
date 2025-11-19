"""
Behavior Sequencer for Crazyflie Drone Swarm

Orchestrates behavior transitions and manages state machine for mission execution.
Coordinates patrol, tracking, formation, and attack behaviors.
"""

import numpy as np
from typing import Dict, Optional, Callable, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import time


class BehaviorState(Enum):
    """States in the behavior state machine."""
    IDLE = "idle"
    SEARCH = "search"
    TRACK = "track"
    FORMATION = "formation"
    ATTACK = "attack"
    RTH = "return_to_home"
    EMERGENCY = "emergency"
    COMPLETED = "completed"


class BehaviorType(Enum):
    """Types of behaviors that can be executed."""
    PATROL = "patrol"
    SAFETY_CHECK = "safety_check"
    FORMATION_FLY = "formation_fly"
    TRACK_TARGET = "track_target"
    JAMMING = "jamming"
    NEUTRALIZATION = "neutralization"
    VICTORY_HOVER = "victory_hover"
    RETURN_HOME = "return_home"


class BehaviorPriority(Enum):
    """Priority levels for behavior execution."""
    CRITICAL = 1  # Safety, collision avoidance
    HIGH = 2      # Battery preservation, emergency procedures
    MEDIUM = 3    # Mission objectives
    LOW = 4       # Efficiency optimization


@dataclass
class BehaviorStatus:
    """Status information for a behavior."""
    drone_id: str
    state: BehaviorState
    behavior_type: Optional[BehaviorType]
    priority: BehaviorPriority
    start_time: float
    progress: float = 0.0  # 0.0 to 1.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StateTransition:
    """Defines a state transition rule."""
    from_state: BehaviorState
    to_state: BehaviorState
    condition: str  # Description of the condition
    action: Optional[Callable] = None


class BehaviorSequencer:
    """
    Orchestrates behavior transitions and manages mission state machine.

    State transitions:
    IDLE -> SEARCH: Mission start
    SEARCH -> TRACK: Target detected
    TRACK -> FORMATION: Roles assigned
    FORMATION -> ATTACK: In position
    ATTACK -> RTH: Mission complete
    * -> EMERGENCY: Critical failure
    """

    def __init__(self, drone_id: str):
        """
        Initialize behavior sequencer.

        Args:
            drone_id: Unique identifier for this drone
        """
        self.drone_id = drone_id
        self.current_state = BehaviorState.IDLE
        self.current_behavior: Optional[BehaviorType] = None
        self.state_start_time: float = time.time()
        self.logger = logging.getLogger(f"BehaviorSequencer_{drone_id}")

        # Behavior status tracking
        self.behavior_history: List[BehaviorStatus] = []
        self.active_behaviors: Dict[str, BehaviorStatus] = {}

        # State transition definitions
        self.transitions = self._define_transitions()

        # Mission parameters
        self.mission_started = False
        self.target_detected = False
        self.role_assigned = False
        self.in_position = False
        self.mission_complete = False

        # Safety parameters
        self.battery_level = 1.0
        self.min_battery_threshold = 0.25
        self.collision_detected = False

    def _define_transitions(self) -> List[StateTransition]:
        """
        Define all valid state transitions.

        Returns:
            List of StateTransition objects
        """
        return [
            StateTransition(
                BehaviorState.IDLE,
                BehaviorState.SEARCH,
                "Mission start command received"
            ),
            StateTransition(
                BehaviorState.SEARCH,
                BehaviorState.TRACK,
                "Target detected by vision system"
            ),
            StateTransition(
                BehaviorState.TRACK,
                BehaviorState.FORMATION,
                "Role assigned by coordinator"
            ),
            StateTransition(
                BehaviorState.FORMATION,
                BehaviorState.ATTACK,
                "Formation in position for attack"
            ),
            StateTransition(
                BehaviorState.ATTACK,
                BehaviorState.RTH,
                "Attack sequence completed"
            ),
            StateTransition(
                BehaviorState.RTH,
                BehaviorState.COMPLETED,
                "Returned to home position"
            ),
        ]

    def execute_behavior(
        self,
        drone_id: str,
        behavior_type: BehaviorType,
        params: Dict[str, Any]
    ) -> BehaviorStatus:
        """
        Execute a specific behavior with given parameters.

        Args:
            drone_id: ID of the drone executing the behavior
            behavior_type: Type of behavior to execute
            params: Parameters for the behavior

        Returns:
            BehaviorStatus object with execution status
        """
        self.logger.info(f"Executing {behavior_type.value} for {drone_id}")

        # Determine priority based on behavior type
        priority = self._get_behavior_priority(behavior_type)

        # Create behavior status
        status = BehaviorStatus(
            drone_id=drone_id,
            state=self.current_state,
            behavior_type=behavior_type,
            priority=priority,
            start_time=time.time(),
            metadata=params
        )

        # Execute behavior based on type
        try:
            if behavior_type == BehaviorType.PATROL:
                status.progress = self._execute_patrol(params)

            elif behavior_type == BehaviorType.SAFETY_CHECK:
                status.progress = self._execute_safety_check(params)

            elif behavior_type == BehaviorType.FORMATION_FLY:
                status.progress = self._execute_formation_fly(params)

            elif behavior_type == BehaviorType.TRACK_TARGET:
                status.progress = self._execute_track_target(params)

            elif behavior_type == BehaviorType.JAMMING:
                status.progress = self._execute_jamming(params)

            elif behavior_type == BehaviorType.NEUTRALIZATION:
                status.progress = self._execute_neutralization(params)

            elif behavior_type == BehaviorType.VICTORY_HOVER:
                status.progress = self._execute_victory_hover(params)

            elif behavior_type == BehaviorType.RETURN_HOME:
                status.progress = self._execute_return_home(params)

            # Store status
            self.active_behaviors[drone_id] = status
            self.behavior_history.append(status)

        except Exception as e:
            status.error_message = str(e)
            self.logger.error(f"Error executing {behavior_type.value}: {e}")

        return status

    def _get_behavior_priority(self, behavior_type: BehaviorType) -> BehaviorPriority:
        """Determine priority for a behavior type."""
        if behavior_type in [BehaviorType.RETURN_HOME]:
            return BehaviorPriority.HIGH
        elif behavior_type in [BehaviorType.SAFETY_CHECK]:
            return BehaviorPriority.CRITICAL
        elif behavior_type in [BehaviorType.NEUTRALIZATION, BehaviorType.JAMMING]:
            return BehaviorPriority.MEDIUM
        else:
            return BehaviorPriority.LOW

    def _execute_patrol(self, params: Dict[str, Any]) -> float:
        """Execute patrol behavior."""
        self.logger.info("Executing patrol pattern")
        # Placeholder: would interface with patrol_patterns module
        return 0.5  # 50% progress

    def _execute_safety_check(self, params: Dict[str, Any]) -> float:
        """Execute safety zone check behavior."""
        self.logger.info("Executing safety zone verification")
        return 0.8

    def _execute_formation_fly(self, params: Dict[str, Any]) -> float:
        """Execute formation flying behavior."""
        self.logger.info("Maintaining formation")
        return 0.6

    def _execute_track_target(self, params: Dict[str, Any]) -> float:
        """Execute target tracking behavior."""
        self.logger.info("Tracking target")
        return 0.7

    def _execute_jamming(self, params: Dict[str, Any]) -> float:
        """Execute jamming behavior."""
        self.logger.info("Executing jamming maneuver")
        return 0.9

    def _execute_neutralization(self, params: Dict[str, Any]) -> float:
        """Execute neutralization behavior."""
        self.logger.info("Executing neutralization approach")
        return 0.85

    def _execute_victory_hover(self, params: Dict[str, Any]) -> float:
        """Execute victory hover behavior."""
        self.logger.info("Victory hover")
        return 1.0

    def _execute_return_home(self, params: Dict[str, Any]) -> float:
        """Execute return to home behavior."""
        self.logger.info("Returning to home")
        return 0.75

    def transition_check(self) -> Optional[BehaviorState]:
        """
        Check if conditions are met for state transition.

        Returns:
            Next state if transition should occur, None otherwise
        """
        # Check for emergency conditions first (highest priority)
        if self._check_emergency_conditions():
            return BehaviorState.EMERGENCY

        # Check normal state transitions
        if self.current_state == BehaviorState.IDLE:
            if self.mission_started:
                return BehaviorState.SEARCH

        elif self.current_state == BehaviorState.SEARCH:
            if self.target_detected:
                return BehaviorState.TRACK

        elif self.current_state == BehaviorState.TRACK:
            if self.role_assigned:
                return BehaviorState.FORMATION

        elif self.current_state == BehaviorState.FORMATION:
            if self.in_position:
                return BehaviorState.ATTACK

        elif self.current_state == BehaviorState.ATTACK:
            if self.mission_complete:
                return BehaviorState.RTH

        elif self.current_state == BehaviorState.RTH:
            # Check if back at home position
            if self._at_home_position():
                return BehaviorState.COMPLETED

        return None

    def _check_emergency_conditions(self) -> bool:
        """
        Check if emergency conditions exist.

        Returns:
            True if emergency condition detected
        """
        # Battery critical
        if self.battery_level < self.min_battery_threshold:
            self.logger.error(
                f"EMERGENCY: Battery critical ({self.battery_level*100:.1f}%)"
            )
            return True

        # Collision detected
        if self.collision_detected:
            self.logger.error("EMERGENCY: Collision detected")
            return True

        return False

    def _at_home_position(self) -> bool:
        """Check if drone is at home position."""
        # Placeholder - would check actual position
        return False

    def transition_to(self, new_state: BehaviorState, reason: str = ""):
        """
        Transition to a new state.

        Args:
            new_state: State to transition to
            reason: Reason for transition (for logging)
        """
        old_state = self.current_state
        self.current_state = new_state
        self.state_start_time = time.time()

        self.logger.info(
            f"State transition: {old_state.value} -> {new_state.value}"
            + (f" ({reason})" if reason else "")
        )

        # Trigger state entry actions
        self._on_state_entry(new_state)

    def _on_state_entry(self, state: BehaviorState):
        """
        Execute actions when entering a new state.

        Args:
            state: State being entered
        """
        if state == BehaviorState.SEARCH:
            self.logger.info("Beginning search pattern")
            self.current_behavior = BehaviorType.PATROL

        elif state == BehaviorState.TRACK:
            self.logger.info("Tracking target")
            self.current_behavior = BehaviorType.TRACK_TARGET

        elif state == BehaviorState.FORMATION:
            self.logger.info("Forming up")
            self.current_behavior = BehaviorType.FORMATION_FLY

        elif state == BehaviorState.ATTACK:
            self.logger.info("Executing attack sequence")
            self.current_behavior = BehaviorType.JAMMING

        elif state == BehaviorState.RTH:
            self.logger.info("Returning to home")
            self.current_behavior = BehaviorType.RETURN_HOME

        elif state == BehaviorState.EMERGENCY:
            self.logger.error("EMERGENCY STATE - executing emergency procedures")
            self.current_behavior = BehaviorType.RETURN_HOME

    def abort_behavior(self, reason: str):
        """
        Abort current behavior and transition to safe state.

        Args:
            reason: Reason for aborting
        """
        self.logger.warning(f"Aborting behavior: {reason}")

        # Clear active behaviors
        for drone_id, status in self.active_behaviors.items():
            status.error_message = f"Aborted: {reason}"

        # Transition to RTH or EMERGENCY based on severity
        if "emergency" in reason.lower() or "critical" in reason.lower():
            self.transition_to(BehaviorState.EMERGENCY, reason)
        else:
            self.transition_to(BehaviorState.RTH, reason)

    def get_behavior_status(self, drone_id: str) -> Optional[BehaviorStatus]:
        """
        Get current behavior status for a drone.

        Args:
            drone_id: ID of the drone

        Returns:
            BehaviorStatus object or None if not found
        """
        return self.active_behaviors.get(drone_id)

    def update(self, dt: float = 0.01):
        """
        Update the behavior sequencer (called each control loop iteration).

        Args:
            dt: Time delta since last update
        """
        # Check for state transitions
        next_state = self.transition_check()
        if next_state is not None:
            self.transition_to(next_state)

        # Update current behavior
        if self.current_behavior is not None:
            # Behavior-specific updates would go here
            pass

    def set_mission_parameter(self, param: str, value: Any):
        """
        Set a mission parameter that affects state transitions.

        Args:
            param: Parameter name
            value: Parameter value
        """
        if param == "mission_started":
            self.mission_started = value
        elif param == "target_detected":
            self.target_detected = value
        elif param == "role_assigned":
            self.role_assigned = value
        elif param == "in_position":
            self.in_position = value
        elif param == "mission_complete":
            self.mission_complete = value
        elif param == "battery_level":
            self.battery_level = value
        elif param == "collision_detected":
            self.collision_detected = value
        else:
            self.logger.warning(f"Unknown parameter: {param}")

    def get_mission_status(self) -> Dict[str, Any]:
        """
        Get overall mission status.

        Returns:
            Dictionary with mission status information
        """
        time_in_state = time.time() - self.state_start_time

        return {
            'drone_id': self.drone_id,
            'current_state': self.current_state.value,
            'current_behavior': self.current_behavior.value if self.current_behavior else None,
            'time_in_state': time_in_state,
            'mission_started': self.mission_started,
            'target_detected': self.target_detected,
            'role_assigned': self.role_assigned,
            'in_position': self.in_position,
            'mission_complete': self.mission_complete,
            'battery_level': self.battery_level,
            'active_behaviors': len(self.active_behaviors)
        }


class SwarmBehaviorCoordinator:
    """
    Coordinates behavior sequencers for multiple drones in a swarm.

    Ensures synchronized state transitions and coordinated behaviors.
    """

    def __init__(self, num_drones: int):
        """
        Initialize swarm behavior coordinator.

        Args:
            num_drones: Number of drones in the swarm
        """
        self.num_drones = num_drones
        self.sequencers: Dict[str, BehaviorSequencer] = {}
        self.logger = logging.getLogger("SwarmBehaviorCoordinator")

        # Create sequencers for each drone
        for i in range(num_drones):
            drone_id = f"drone_{i}"
            self.sequencers[drone_id] = BehaviorSequencer(drone_id)

    def start_mission(self):
        """Start the mission for all drones."""
        self.logger.info("Starting swarm mission")
        for sequencer in self.sequencers.values():
            sequencer.set_mission_parameter("mission_started", True)

    def broadcast_target_detected(self, target_pos: Tuple[float, float, float]):
        """
        Broadcast target detection to all drones.

        Args:
            target_pos: Position of detected target
        """
        self.logger.info(f"Target detected at {target_pos}")
        for sequencer in self.sequencers.values():
            sequencer.set_mission_parameter("target_detected", True)

    def assign_roles(self, role_assignments: Dict[str, str]):
        """
        Assign roles to drones.

        Args:
            role_assignments: Dictionary mapping drone_id to role
        """
        self.logger.info(f"Assigning roles: {role_assignments}")
        for drone_id, role in role_assignments.items():
            if drone_id in self.sequencers:
                self.sequencers[drone_id].set_mission_parameter("role_assigned", True)

    def update_all(self, dt: float = 0.01):
        """
        Update all drone sequencers.

        Args:
            dt: Time delta since last update
        """
        for sequencer in self.sequencers.values():
            sequencer.update(dt)

    def get_swarm_status(self) -> Dict[str, Any]:
        """
        Get status of entire swarm.

        Returns:
            Dictionary with swarm status information
        """
        statuses = {
            drone_id: sequencer.get_mission_status()
            for drone_id, sequencer in self.sequencers.items()
        }

        # Count drones in each state
        state_counts = {}
        for status in statuses.values():
            state = status['current_state']
            state_counts[state] = state_counts.get(state, 0) + 1

        return {
            'num_drones': self.num_drones,
            'drone_statuses': statuses,
            'state_distribution': state_counts
        }
