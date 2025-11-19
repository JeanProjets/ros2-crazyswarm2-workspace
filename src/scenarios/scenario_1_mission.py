"""
Scenario 1 Mission Sequencer

This module implements the complete mission sequence for Scenario 1,
coordinating all drones through the 9 mission phases.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from mission_state_machine import MissionState, MissionStateMachine
from role_manager import DroneRole, RoleManager, DroneInfo


@dataclass
class MissionResult:
    """Result of a mission execution"""
    success: bool
    completion_time: float
    final_state: MissionState
    reason: Optional[str] = None
    telemetry: Optional[dict] = None


@dataclass
class DroneAction:
    """Action to be performed by a drone"""
    drone_id: str
    action_type: str  # "move", "hover", "land", etc.
    target_position: Optional[Tuple[float, float, float]] = None
    duration: float = 0.0
    parameters: dict = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class HealthStatus(Enum):
    """Health status of the mission"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


class Scenario1Mission:
    """
    Main mission orchestrator for Scenario 1.

    This class coordinates all aspects of the mission including:
    - State machine management
    - Role assignment
    - Drone control
    - Phase execution
    - Health monitoring
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the mission.

        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger("Scenario1Mission")
        self.config = config or self._get_default_config()

        # Initialize core components
        self.state_machine = MissionStateMachine()
        self.role_manager = RoleManager()

        # Mission tracking
        self.mission_start_time: Optional[float] = None
        self.telemetry_log: List[dict] = []
        self.target_detected_position: Optional[Tuple[float, float, float]] = None
        self.detector_drone_id: Optional[str] = None

        # Mock component references (would be real controllers in actual system)
        self.drone_controllers: Dict[str, object] = {}
        self.behavior_executors: Dict[str, object] = {}
        self.vision_systems: Dict[str, object] = {}

        # Phase executor
        self.phase_executor = PhaseExecutor(self)

        # Setup transition conditions
        self._setup_transition_conditions()

    def _get_default_config(self) -> dict:
        """Get default configuration for the mission"""
        return {
            "drone_ids": ["cf1", "cf2", "cf3"],
            "initial_positions": {
                "cf1": (0.5, 0.5, 0.0),
                "cf2": (-0.5, 0.5, 0.0),
                "cf3": (0.0, -1.0, 0.0),
            },
            "hover_height": 1.0,
            "target_search_area": {
                "min_x": -2.0,
                "max_x": 2.0,
                "min_y": 0.0,
                "max_y": 4.0,
            },
            "simulation_mode": True,  # Mock hardware in simulation
        }

    def initialize_mission(self, config_file: Optional[str] = None) -> bool:
        """
        Initialize the mission with configuration.

        Args:
            config_file: Optional path to configuration file

        Returns:
            True if initialization successful
        """
        self.logger.info("Initializing Scenario 1 mission")

        # Register all drones
        for drone_id in self.config["drone_ids"]:
            self.role_manager.register_drone(
                drone_id=drone_id,
                battery_level=100.0,
                position=self.config["initial_positions"][drone_id],
                status="ready"
            )

        # Assign initial roles
        self.role_manager.assign_initial_roles()

        # Initialize mock controllers if in simulation mode
        if self.config.get("simulation_mode", True):
            self._initialize_mock_controllers()

        self.logger.info("Mission initialization complete")
        return True

    def _initialize_mock_controllers(self):
        """Initialize mock controllers for simulation"""
        for drone_id in self.config["drone_ids"]:
            self.drone_controllers[drone_id] = MockDroneController(drone_id)
            self.behavior_executors[drone_id] = MockBehaviorExecutor(drone_id)
            self.vision_systems[drone_id] = MockVisionSystem(drone_id)

    def _setup_transition_conditions(self):
        """Setup automatic transition conditions for each state"""

        def check_init_complete():
            # Transition when all drones are airborne
            # In simulation, we'll auto-transition after a delay
            return None  # Manual transition

        def check_safety_complete():
            # Transition when safety check is complete or timeout
            return None  # Handled by timeout or manual

        def check_patrol_complete():
            # Transition when target is detected
            if self.target_detected_position is not None:
                return MissionState.TARGET_DETECTED
            return None

        def check_target_detected():
            # Auto-transition after 1 second
            if self.state_machine.get_phase_elapsed_time() > 1.0:
                return MissionState.ROLE_ASSIGNMENT
            return None

        def check_role_assignment():
            # Auto-transition after roles assigned
            if self.state_machine.get_phase_elapsed_time() > 0.5:
                return MissionState.APPROACH_TARGET
            return None

        # Register conditions
        self.state_machine.register_transition_condition(
            MissionState.INITIALIZATION, check_init_complete
        )
        self.state_machine.register_transition_condition(
            MissionState.SAFETY_CHECK, check_safety_complete
        )
        self.state_machine.register_transition_condition(
            MissionState.PATROL_SEARCH, check_patrol_complete
        )
        self.state_machine.register_transition_condition(
            MissionState.TARGET_DETECTED, check_target_detected
        )
        self.state_machine.register_transition_condition(
            MissionState.ROLE_ASSIGNMENT, check_role_assignment
        )

    async def execute_mission(self) -> MissionResult:
        """
        Main mission execution loop.

        Returns:
            MissionResult with outcome and telemetry
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Scenario 1 Mission Execution")
        self.logger.info("=" * 60)

        self.mission_start_time = time.time()
        self.state_machine.start_mission()

        try:
            # Execute initialization phase
            await self.phase_executor.execute_initialization()
            self.state_machine.transition_to(
                MissionState.SAFETY_CHECK, "initialization complete"
            )

            # Main mission loop
            while not self.state_machine.is_mission_complete():
                current_state = self.state_machine.current_state

                # Check for automatic transitions
                next_state = self.state_machine.check_transition_conditions()
                if next_state:
                    self.state_machine.transition_to(
                        next_state, "condition met"
                    )
                    current_state = next_state

                # Execute current phase
                await self.execute_phase(current_state)

                # Monitor mission health
                health = self.monitor_mission_health()
                if health == HealthStatus.CRITICAL:
                    self.logger.error("Critical health status detected")
                    self.state_machine.abort_mission("Critical health failure")
                    break

                # Log telemetry
                self._log_telemetry()

                # Small delay to prevent CPU overload
                await asyncio.sleep(0.1)

            # Generate final result
            return self._generate_mission_result()

        except Exception as e:
            self.logger.error(f"Mission failed with exception: {e}", exc_info=True)
            self.state_machine.abort_mission(str(e))
            return MissionResult(
                success=False,
                completion_time=time.time() - self.mission_start_time,
                final_state=self.state_machine.current_state,
                reason=str(e)
            )

    async def execute_phase(self, phase: MissionState):
        """
        Execute actions for the current phase.

        Args:
            phase: The mission phase to execute
        """
        if phase == MissionState.INITIALIZATION:
            pass  # Already executed before loop

        elif phase == MissionState.SAFETY_CHECK:
            await self.phase_executor.execute_safety_check()

        elif phase == MissionState.PATROL_SEARCH:
            await self.phase_executor.execute_patrol_search()

        elif phase == MissionState.TARGET_DETECTED:
            # Just waiting for auto-transition
            pass

        elif phase == MissionState.ROLE_ASSIGNMENT:
            # Reassign roles based on detection
            if self.detector_drone_id:
                self.role_manager.reassign_roles_on_detection(
                    self.detector_drone_id
                )
                self.state_machine.set_drone_roles(
                    {k: v.value for k, v in self.role_manager.current_assignments.items()}
                )

        elif phase == MissionState.APPROACH_TARGET:
            await self.phase_executor.execute_target_approach()

        elif phase == MissionState.JAMMING:
            await self.phase_executor.execute_jamming()

        elif phase == MissionState.NEUTRALIZATION:
            await self.phase_executor.execute_neutralization()

        elif phase in [MissionState.MISSION_COMPLETE, MissionState.MISSION_ABORT]:
            # Terminal states - do nothing
            pass

    def coordinate_drone_actions(self, actions: List[DroneAction]) -> bool:
        """
        Coordinate execution of multiple drone actions.

        Args:
            actions: List of actions to execute

        Returns:
            True if all actions completed successfully
        """
        self.logger.info(f"Coordinating {len(actions)} drone actions")

        for action in actions:
            drone_id = action.drone_id
            if drone_id in self.drone_controllers:
                controller = self.drone_controllers[drone_id]
                # Mock execution
                if action.action_type == "move" and action.target_position:
                    self.logger.info(
                        f"{drone_id}: Moving to {action.target_position}"
                    )
                elif action.action_type == "hover":
                    self.logger.info(f"{drone_id}: Hovering")
                elif action.action_type == "land":
                    self.logger.info(f"{drone_id}: Landing")

        return True

    def monitor_mission_health(self) -> HealthStatus:
        """
        Monitor the health of the mission.

        Returns:
            Current health status
        """
        # Check battery levels
        swarm_status = self.role_manager.get_swarm_status()

        for drone_id, battery in swarm_status["battery_levels"].items():
            if battery < 20.0:
                self.logger.error(
                    f"Critical battery level for {drone_id}: {battery}%"
                )
                return HealthStatus.CRITICAL
            elif battery < 40.0:
                self.logger.warning(
                    f"Low battery level for {drone_id}: {battery}%"
                )
                # Don't return yet, check other conditions

        # Check mission timeout
        if self.state_machine.is_mission_timeout():
            self.logger.error("Mission timeout exceeded")
            return HealthStatus.CRITICAL

        return HealthStatus.HEALTHY

    def handle_contingency(self, event_type: str, details: dict = None):
        """
        Handle contingency events.

        Args:
            event_type: Type of contingency event
            details: Additional details about the event
        """
        self.logger.warning(f"Contingency event: {event_type}")

        if event_type == "battery_critical":
            drone_id = details.get("drone_id")
            self.logger.error(f"Battery critical for {drone_id}")
            self.state_machine.abort_mission(f"Battery critical: {drone_id}")

        elif event_type == "tracking_lost":
            drone_id = details.get("drone_id")
            duration = details.get("duration", 0)
            if duration > 5.0:
                self.state_machine.abort_mission(f"Tracking lost: {drone_id}")

        elif event_type == "collision_imminent":
            self.state_machine.abort_mission("Collision imminent")

    def _log_telemetry(self):
        """Log current telemetry data"""
        telemetry = {
            "timestamp": time.time(),
            "mission_elapsed": self.state_machine.get_mission_elapsed_time(),
            "current_state": self.state_machine.current_state.value,
            "phase_elapsed": self.state_machine.get_phase_elapsed_time(),
            "swarm_status": self.role_manager.get_swarm_status(),
        }
        self.telemetry_log.append(telemetry)

    def _generate_mission_result(self) -> MissionResult:
        """Generate final mission result"""
        final_state = self.state_machine.current_state
        success = final_state == MissionState.MISSION_COMPLETE
        completion_time = time.time() - self.mission_start_time

        self.logger.info("=" * 60)
        self.logger.info(f"Mission Complete: {success}")
        self.logger.info(f"Final State: {final_state.value}")
        self.logger.info(f"Completion Time: {completion_time:.2f}s")
        self.logger.info("=" * 60)

        return MissionResult(
            success=success,
            completion_time=completion_time,
            final_state=final_state,
            reason=self.state_machine.abort_reason,
            telemetry={"log": self.telemetry_log}
        )


class PhaseExecutor:
    """Executes individual mission phases"""

    def __init__(self, mission: Scenario1Mission):
        """
        Initialize phase executor.

        Args:
            mission: Reference to the main mission object
        """
        self.mission = mission
        self.logger = logging.getLogger("PhaseExecutor")

    async def execute_initialization(self):
        """Execute initialization phase (10s)"""
        self.logger.info("PHASE: Initialization")

        # Launch all drones to hover positions
        actions = []
        for drone_id in self.mission.config["drone_ids"]:
            hover_height = self.mission.config["hover_height"]
            initial_pos = self.mission.config["initial_positions"][drone_id]
            target_pos = (initial_pos[0], initial_pos[1], hover_height)

            action = DroneAction(
                drone_id=drone_id,
                action_type="move",
                target_position=target_pos,
                duration=5.0
            )
            actions.append(action)

        self.mission.coordinate_drone_actions(actions)

        # Simulate takeoff time
        await asyncio.sleep(2.0)

        self.logger.info("Initialization complete - all drones airborne")

    async def execute_safety_check(self):
        """Execute safety check phase (30s)"""
        self.logger.info("PHASE: Safety Check")

        # Get neutral drones
        neutral_drones = self.mission.role_manager.get_drones_by_role([
            DroneRole.NEUTRAL_1,
            DroneRole.NEUTRAL_2
        ])

        # Execute parallel safety sweeps
        self.logger.info(f"Neutral drones {neutral_drones} performing safety sweep")

        # Simulate sweep
        await asyncio.sleep(2.0)

        # Transition to patrol
        self.mission.state_machine.transition_to(
            MissionState.PATROL_SEARCH, "safety zone clear"
        )

    async def execute_patrol_search(self):
        """Execute patrol search phase (60-90s)"""
        self.logger.info("PHASE: Patrol Search")

        # Get patrol drone
        patrol_drones = self.mission.role_manager.get_drones_by_role([
            DroneRole.PATROL
        ])

        if patrol_drones:
            patrol_id = patrol_drones[0]
            self.logger.info(f"{patrol_id} executing search pattern")

        # Simulate search (in real system, would monitor vision system)
        await asyncio.sleep(1.0)

        # Simulate target detection after some time
        if self.mission.state_machine.get_phase_elapsed_time() > 5.0:
            if self.mission.target_detected_position is None:
                # Simulate detection
                self.mission.target_detected_position = (0.0, 2.0, 0.3)
                self.mission.detector_drone_id = patrol_id if patrol_drones else "cf3"
                self.logger.info(
                    f"Target detected by {self.mission.detector_drone_id} "
                    f"at {self.mission.target_detected_position}"
                )

    async def execute_target_approach(self):
        """Execute target approach phase (30s)"""
        self.logger.info("PHASE: Target Approach")

        # Get role-based positions
        actions = []

        for drone_id, role in self.mission.role_manager.current_assignments.items():
            target_pos = self.mission.role_manager.get_role_position(
                role,
                "attack_formation",
                self.mission.target_detected_position
            )

            action = DroneAction(
                drone_id=drone_id,
                action_type="move",
                target_position=target_pos,
                duration=10.0
            )
            actions.append(action)

        self.mission.coordinate_drone_actions(actions)

        # Simulate approach time
        await asyncio.sleep(3.0)

        # Check if in position
        if self.mission.state_machine.get_phase_elapsed_time() > 3.0:
            self.logger.info("Formation in position")
            self.mission.state_machine.transition_to(
                MissionState.JAMMING, "formation ready"
            )

    async def execute_jamming(self):
        """Execute jamming phase (20s)"""
        self.logger.info("PHASE: Jamming")

        # Hold positions in front of target
        self.logger.info("Leader and Follower holding jamming positions")
        self.logger.info("Simulating RF interference")

        # Wait for jamming duration
        await asyncio.sleep(1.0)

        # Transition after timeout (handled by state machine)

    async def execute_neutralization(self):
        """Execute neutralization phase (5s)"""
        self.logger.info("PHASE: Neutralization")

        # Get attacker drone
        attacker_drones = self.mission.role_manager.get_drones_by_role([
            DroneRole.ATTACKER
        ])

        if attacker_drones:
            attacker_id = attacker_drones[0]
            self.logger.info(f"{attacker_id} descending on target")

            # Move to 30cm above target
            if self.mission.target_detected_position:
                attack_pos = (
                    self.mission.target_detected_position[0],
                    self.mission.target_detected_position[1],
                    0.3  # 30cm above ground
                )

                action = DroneAction(
                    drone_id=attacker_id,
                    action_type="move",
                    target_position=attack_pos,
                    duration=3.0
                )
                self.mission.coordinate_drone_actions([action])

        await asyncio.sleep(2.0)

        # Mission complete
        self.logger.info("Target neutralized - Mission success!")
        self.mission.state_machine.transition_to(
            MissionState.MISSION_COMPLETE, "neutralization complete"
        )


# Mock classes for simulation (would be replaced with real implementations)

class MockDroneController:
    """Mock drone controller for simulation"""
    def __init__(self, drone_id: str):
        self.drone_id = drone_id
        self.position = (0.0, 0.0, 0.0)


class MockBehaviorExecutor:
    """Mock behavior executor for simulation"""
    def __init__(self, drone_id: str):
        self.drone_id = drone_id


class MockVisionSystem:
    """Mock vision system for simulation"""
    def __init__(self, drone_id: str):
        self.drone_id = drone_id
