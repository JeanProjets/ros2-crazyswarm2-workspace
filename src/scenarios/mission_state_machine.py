"""
Mission State Machine for Scenario 1

This module implements the core mission state machine that coordinates
the 9 mission phases for a Crazyflie drone swarm.
"""

import time
import logging
from enum import Enum
from typing import Dict, Optional, Callable
from dataclasses import dataclass


class MissionState(Enum):
    """Enumeration of all mission states"""
    INITIALIZATION = "init"
    SAFETY_CHECK = "safety_check"
    PATROL_SEARCH = "patrol_search"
    TARGET_DETECTED = "target_detected"
    ROLE_ASSIGNMENT = "role_assignment"
    APPROACH_TARGET = "approach_target"
    JAMMING = "jamming"
    NEUTRALIZATION = "neutralization"
    MISSION_COMPLETE = "complete"
    MISSION_ABORT = "abort"


@dataclass
class StateTransition:
    """Represents a state transition with timing information"""
    from_state: MissionState
    to_state: MissionState
    timestamp: float
    reason: str


class MissionStateMachine:
    """
    Core mission state machine that manages transitions between mission phases.

    Attributes:
        current_state: Current mission state
        start_time: Mission start timestamp
        phase_timers: Timing information for each phase
        drone_roles: Current role assignments for drones
    """

    # Timeout values for each state (in seconds)
    STATE_TIMEOUTS = {
        MissionState.INITIALIZATION: 10,
        MissionState.SAFETY_CHECK: 30,
        MissionState.PATROL_SEARCH: 90,
        MissionState.TARGET_DETECTED: 1,
        MissionState.ROLE_ASSIGNMENT: 1,
        MissionState.APPROACH_TARGET: 30,
        MissionState.JAMMING: 20,
        MissionState.NEUTRALIZATION: 5,
    }

    # Total mission timeout (3 minutes)
    MISSION_TIMEOUT = 180

    def __init__(self):
        """Initialize the mission state machine"""
        self.logger = logging.getLogger("MissionStateMachine")
        self.current_state = MissionState.INITIALIZATION
        self.start_time: Optional[float] = None
        self.phase_timers: Dict[MissionState, float] = {}
        self.drone_roles: Dict[str, str] = {}
        self.transition_history: list[StateTransition] = []
        self.state_callbacks: Dict[MissionState, list[Callable]] = {}
        self.transition_conditions: Dict[MissionState, Callable] = {}
        self.abort_reason: Optional[str] = None

        # Initialize phase timers
        self._reset_phase_timer()

    def start_mission(self):
        """Start the mission timer"""
        self.start_time = time.time()
        self.logger.info("Mission started")
        self._reset_phase_timer()

    def _reset_phase_timer(self):
        """Reset the phase timer for the current state"""
        self.phase_timers[self.current_state] = time.time()

    def transition_to(self, new_state: MissionState, reason: str = "automatic") -> bool:
        """
        Transition to a new mission state.

        Args:
            new_state: The state to transition to
            reason: Reason for the transition

        Returns:
            True if transition was successful, False otherwise
        """
        if not self._is_valid_transition(self.current_state, new_state):
            self.logger.warning(
                f"Invalid transition from {self.current_state.value} to {new_state.value}"
            )
            return False

        old_state = self.current_state
        self.current_state = new_state
        self._reset_phase_timer()

        # Record transition
        transition = StateTransition(
            from_state=old_state,
            to_state=new_state,
            timestamp=time.time(),
            reason=reason
        )
        self.transition_history.append(transition)

        self.logger.info(
            f"State transition: {old_state.value} -> {new_state.value} ({reason})"
        )

        # Execute state entry callbacks
        self._execute_state_callbacks(new_state)

        return True

    def _is_valid_transition(self, from_state: MissionState, to_state: MissionState) -> bool:
        """
        Check if a state transition is valid.

        Valid transitions:
        - INIT -> SAFETY_CHECK
        - SAFETY_CHECK -> PATROL_SEARCH
        - PATROL_SEARCH -> TARGET_DETECTED
        - TARGET_DETECTED -> ROLE_ASSIGNMENT
        - ROLE_ASSIGNMENT -> APPROACH_TARGET
        - APPROACH_TARGET -> JAMMING
        - JAMMING -> NEUTRALIZATION
        - NEUTRALIZATION -> MISSION_COMPLETE
        - Any state -> MISSION_ABORT
        """
        # Can always abort from any state
        if to_state == MissionState.MISSION_ABORT:
            return True

        # Define valid transitions
        valid_transitions = {
            MissionState.INITIALIZATION: [MissionState.SAFETY_CHECK],
            MissionState.SAFETY_CHECK: [MissionState.PATROL_SEARCH],
            MissionState.PATROL_SEARCH: [MissionState.TARGET_DETECTED],
            MissionState.TARGET_DETECTED: [MissionState.ROLE_ASSIGNMENT],
            MissionState.ROLE_ASSIGNMENT: [MissionState.APPROACH_TARGET],
            MissionState.APPROACH_TARGET: [MissionState.JAMMING],
            MissionState.JAMMING: [MissionState.NEUTRALIZATION],
            MissionState.NEUTRALIZATION: [MissionState.MISSION_COMPLETE],
            MissionState.MISSION_COMPLETE: [],
            MissionState.MISSION_ABORT: [],
        }

        return to_state in valid_transitions.get(from_state, [])

    def check_transition_conditions(self) -> Optional[MissionState]:
        """
        Check if conditions are met for transitioning to the next state.

        Returns:
            The next state if transition conditions are met, None otherwise
        """
        # Check for mission timeout
        if self.is_mission_timeout():
            self.logger.warning("Mission timeout exceeded")
            return MissionState.MISSION_ABORT

        # Check for phase timeout
        if self.is_phase_timeout():
            self.logger.warning(
                f"Phase timeout for {self.current_state.value}"
            )
            # Handle timeout based on current state
            return self._handle_phase_timeout()

        # Check custom transition conditions
        if self.current_state in self.transition_conditions:
            next_state = self.transition_conditions[self.current_state]()
            if next_state:
                return next_state

        return None

    def _handle_phase_timeout(self) -> Optional[MissionState]:
        """
        Handle timeout for the current phase.

        Returns:
            Next state to transition to, or MISSION_ABORT if timeout is critical
        """
        # For some states, timeout should trigger automatic transition
        if self.current_state == MissionState.TARGET_DETECTED:
            # Auto-transition after 1 second
            return MissionState.ROLE_ASSIGNMENT
        elif self.current_state == MissionState.ROLE_ASSIGNMENT:
            # Auto-transition after role assignment
            return MissionState.APPROACH_TARGET
        elif self.current_state == MissionState.JAMMING:
            # After jamming timeout, move to neutralization
            return MissionState.NEUTRALIZATION
        elif self.current_state == MissionState.PATROL_SEARCH:
            # If target not found in time, abort
            self.logger.error("Target not detected within patrol time limit")
            return MissionState.MISSION_ABORT
        else:
            # For other states, timeout is critical
            return MissionState.MISSION_ABORT

    def execute_state_actions(self, state: MissionState):
        """
        Execute actions associated with a specific state.

        Args:
            state: The state to execute actions for
        """
        self._execute_state_callbacks(state)

    def _execute_state_callbacks(self, state: MissionState):
        """Execute all callbacks registered for a state"""
        if state in self.state_callbacks:
            for callback in self.state_callbacks[state]:
                try:
                    callback()
                except Exception as e:
                    self.logger.error(
                        f"Error executing callback for {state.value}: {e}"
                    )

    def register_state_callback(self, state: MissionState, callback: Callable):
        """
        Register a callback to be executed when entering a state.

        Args:
            state: The state to register the callback for
            callback: The callback function to execute
        """
        if state not in self.state_callbacks:
            self.state_callbacks[state] = []
        self.state_callbacks[state].append(callback)

    def register_transition_condition(self, state: MissionState, condition: Callable):
        """
        Register a condition function for transitioning from a state.

        Args:
            state: The state to register the condition for
            condition: Function that returns the next state or None
        """
        self.transition_conditions[state] = condition

    def get_mission_elapsed_time(self) -> float:
        """
        Get the total elapsed time since mission start.

        Returns:
            Elapsed time in seconds, or 0 if mission not started
        """
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def get_phase_elapsed_time(self) -> float:
        """
        Get the elapsed time for the current phase.

        Returns:
            Elapsed time in seconds for current phase
        """
        if self.current_state not in self.phase_timers:
            return 0.0
        return time.time() - self.phase_timers[self.current_state]

    def is_mission_timeout(self) -> bool:
        """
        Check if the mission has exceeded its time limit.

        Returns:
            True if mission timeout exceeded, False otherwise
        """
        return self.get_mission_elapsed_time() > self.MISSION_TIMEOUT

    def is_phase_timeout(self) -> bool:
        """
        Check if the current phase has exceeded its time limit.

        Returns:
            True if phase timeout exceeded, False otherwise
        """
        timeout = self.STATE_TIMEOUTS.get(self.current_state, float('inf'))
        return self.get_phase_elapsed_time() > timeout

    def abort_mission(self, reason: str):
        """
        Abort the mission with a specific reason.

        Args:
            reason: The reason for aborting the mission
        """
        self.abort_reason = reason
        self.logger.error(f"Mission aborted: {reason}")
        self.transition_to(MissionState.MISSION_ABORT, reason=reason)

    def is_mission_complete(self) -> bool:
        """
        Check if the mission has completed (successfully or aborted).

        Returns:
            True if mission is in a terminal state, False otherwise
        """
        return self.current_state in [
            MissionState.MISSION_COMPLETE,
            MissionState.MISSION_ABORT
        ]

    def get_mission_status(self) -> dict:
        """
        Get comprehensive status information about the mission.

        Returns:
            Dictionary containing mission status information
        """
        return {
            "current_state": self.current_state.value,
            "elapsed_time": self.get_mission_elapsed_time(),
            "phase_elapsed_time": self.get_phase_elapsed_time(),
            "is_complete": self.is_mission_complete(),
            "abort_reason": self.abort_reason,
            "drone_roles": self.drone_roles.copy(),
            "transition_count": len(self.transition_history),
        }

    def set_drone_roles(self, roles: Dict[str, str]):
        """
        Update the drone role assignments.

        Args:
            roles: Dictionary mapping drone IDs to role names
        """
        self.drone_roles = roles.copy()
        self.logger.info(f"Drone roles updated: {roles}")
