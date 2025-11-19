"""
Scenario 2 Finite State Machine

This module implements the state machine specific to the corner mission
in Scenario 2, where drones must fly long distances to a confined corner
(9.5, 0.5, 5) without hitting cage walls or running out of battery.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
import time

logger = logging.getLogger(__name__)


class MissionState(Enum):
    """States in the Scenario 2 mission state machine"""
    IDLE = "idle"
    SAFETY_CHK_AND_TRANSIT = "safety_chk_and_transit"
    PERIMETER_SWEEP = "perimeter_sweep"
    CORNER_IDENTIFICATION = "corner_identification"
    FORMATION_ASSEMBLE_LONG = "formation_assemble_long"
    PRECISION_CRAWL = "precision_crawl"
    CORNER_JAMMING = "corner_jamming"
    VERTICAL_DROP = "vertical_drop"
    MISSION_COMPLETE = "mission_complete"
    EMERGENCY_ABORT = "emergency_abort"


class StateTransitionResult(Enum):
    """Results of state transition checks"""
    SUCCESS = "success"
    IN_PROGRESS = "in_progress"
    FAILED = "failed"
    ABORT = "abort"


@dataclass
class SwarmTelemetry:
    """Aggregated swarm telemetry data"""
    drones: Dict[str, Dict[str, float]] = field(default_factory=dict)
    timestamp: float = 0.0

    def get_drone_position(self, drone_id: str) -> Optional[Dict[str, float]]:
        """Get position of a specific drone"""
        if drone_id in self.drones:
            drone_data = self.drones[drone_id]
            return {
                'x': drone_data.get('x', 0.0),
                'y': drone_data.get('y', 0.0),
                'z': drone_data.get('z', 0.0)
            }
        return None

    def get_drone_voltage(self, drone_id: str) -> Optional[float]:
        """Get battery voltage of a specific drone"""
        if drone_id in self.drones:
            return self.drones[drone_id].get('voltage', 0.0)
        return None


class Scenario2StateMachine:
    """
    State machine for Scenario 2 corner mission

    This FSM orchestrates the high-risk operation where drones fly long
    distances to a confined corner, managing strict boundary and energy constraints.
    """

    # Mission parameters
    CORNER_TARGET = {'x': 9.5, 'y': 0.5, 'z': 5.0}
    STANDOFF_POSITION = {'x': 8.0, 'y': 0.5, 'z': 3.0}
    PRECISION_APPROACH_POSITION = {'x': 8.5, 'y': 0.5, 'z': 4.0}

    # Timing constraints (seconds)
    MAX_TRANSIT_TIME = 120  # 2 minutes
    MAX_APPROACH_TIME = 45  # 45 seconds
    JAMMING_DURATION = 20  # 20 seconds
    TOTAL_MISSION_TIME = 210  # 3 min 30s

    # Safety thresholds
    MIN_VOLTAGE_TRANSIT = 3.7
    MIN_VOLTAGE_APPROACH = 3.6
    CRITICAL_VOLTAGE = 3.5

    # Position tolerances
    POSITION_TOLERANCE = 0.15  # meters
    VELOCITY_TOLERANCE = 0.1  # m/s (for "stationary" check)
    CORNER_BOUNDS = {'x_max': 9.8, 'y_min': 0.2}  # Hard limits

    def __init__(self):
        """Initialize the state machine"""
        self.current_state = MissionState.IDLE
        self.previous_state = MissionState.IDLE
        self.state_start_time = time.time()
        self.mission_start_time: Optional[float] = None

        # State history
        self.state_history: List[Dict] = []

        # Mission data
        self.leader_id: Optional[str] = None
        self.follower_ids: List[str] = []
        self.target_confirmed = False
        self.corner_identified_position: Optional[Dict] = None

        # Abort tracking
        self.abort_reason: Optional[str] = None

    def start_mission(self, leader_id: str, follower_ids: List[str]):
        """
        Start the mission

        Args:
            leader_id: ID of the leader/pursuer drone
            follower_ids: IDs of follower/neutral drones
        """
        self.mission_start_time = time.time()
        self.leader_id = leader_id
        self.follower_ids = follower_ids

        logger.info(f"Mission started - Leader: {leader_id}, Followers: {follower_ids}")
        self._transition_to(MissionState.SAFETY_CHK_AND_TRANSIT)

    def update(self, telemetry: SwarmTelemetry) -> MissionState:
        """
        Update the state machine based on current telemetry

        Args:
            telemetry: Current swarm telemetry

        Returns:
            Current mission state after update
        """
        if self.current_state == MissionState.IDLE:
            return self.current_state

        # Check safety conditions
        if not self._check_mission_safety(telemetry):
            return self.current_state

        # Process current state
        if self.current_state == MissionState.SAFETY_CHK_AND_TRANSIT:
            self._process_safety_and_transit(telemetry)
        elif self.current_state == MissionState.PERIMETER_SWEEP:
            self._process_perimeter_sweep(telemetry)
        elif self.current_state == MissionState.CORNER_IDENTIFICATION:
            self._process_corner_identification(telemetry)
        elif self.current_state == MissionState.FORMATION_ASSEMBLE_LONG:
            self._process_formation_assemble(telemetry)
        elif self.current_state == MissionState.PRECISION_CRAWL:
            self._process_precision_crawl(telemetry)
        elif self.current_state == MissionState.CORNER_JAMMING:
            self._process_corner_jamming(telemetry)
        elif self.current_state == MissionState.VERTICAL_DROP:
            self._process_vertical_drop(telemetry)

        return self.current_state

    def _check_mission_safety(self, telemetry: SwarmTelemetry) -> bool:
        """
        Check overall mission safety conditions

        Returns:
            True if safe to continue, False if abort needed
        """
        # Check total mission time
        if self.mission_start_time:
            elapsed = time.time() - self.mission_start_time
            if elapsed > self.TOTAL_MISSION_TIME:
                self._abort_mission("Total mission time exceeded")
                return False

        # Check corner safety
        if not self.check_corner_safety(telemetry):
            self._abort_mission("Corner safety check failed")
            return False

        # Check transit battery
        if not self.monitor_transit_battery(telemetry):
            self._abort_mission("Battery critical during transit")
            return False

        return True

    def check_corner_safety(self, telemetry: SwarmTelemetry) -> bool:
        """
        Check if all drones are inside safe bounds

        Returns:
            True if all drones are safe, False otherwise
        """
        for drone_id, drone_data in telemetry.drones.items():
            x = drone_data.get('x', 0.0)
            y = drone_data.get('y', 0.0)

            # Check hard limits
            if x > self.CORNER_BOUNDS['x_max']:
                logger.error(f"Drone {drone_id} exceeded X limit: {x} > {self.CORNER_BOUNDS['x_max']}")
                return False
            if y < self.CORNER_BOUNDS['y_min']:
                logger.error(f"Drone {drone_id} below Y limit: {y} < {self.CORNER_BOUNDS['y_min']}")
                return False

        return True

    def monitor_transit_battery(self, telemetry: SwarmTelemetry) -> bool:
        """
        Monitor battery voltage during transit

        Returns:
            True if battery levels are safe, False if critical
        """
        if not self.leader_id:
            return True

        leader_voltage = telemetry.get_drone_voltage(self.leader_id)
        if leader_voltage is None:
            logger.warning(f"No voltage data for leader {self.leader_id}")
            return True

        # Check against critical threshold
        if leader_voltage < self.CRITICAL_VOLTAGE:
            logger.error(f"Leader voltage critical: {leader_voltage}V < {self.CRITICAL_VOLTAGE}V")
            return False

        # Check against phase-specific thresholds
        if self.current_state in [MissionState.SAFETY_CHK_AND_TRANSIT, MissionState.PERIMETER_SWEEP]:
            if leader_voltage < self.MIN_VOLTAGE_TRANSIT:
                logger.warning(f"Leader voltage low for transit: {leader_voltage}V")
                return False

        if self.current_state in [MissionState.FORMATION_ASSEMBLE_LONG, MissionState.PRECISION_CRAWL]:
            if leader_voltage < self.MIN_VOLTAGE_APPROACH:
                logger.warning(f"Leader voltage low for approach: {leader_voltage}V")
                return False

        return True

    def _process_safety_and_transit(self, telemetry: SwarmTelemetry):
        """Process SAFETY_CHK_AND_TRANSIT state"""
        # Check if leader has reached X=5 (mid-point)
        if self.leader_id:
            leader_pos = telemetry.get_drone_position(self.leader_id)
            if leader_pos and leader_pos['x'] >= 5.0:
                logger.info("Leader reached transit waypoint X=5")
                self._transition_to(MissionState.PERIMETER_SWEEP)

        # Check timeout
        if self._get_state_duration() > self.MAX_TRANSIT_TIME / 2:
            logger.warning("Transit taking too long, proceeding to perimeter sweep")
            self._transition_to(MissionState.PERIMETER_SWEEP)

    def _process_perimeter_sweep(self, telemetry: SwarmTelemetry):
        """Process PERIMETER_SWEEP state"""
        # In real implementation, this would check vision system
        # For now, simulate scan completion after time
        if self._get_state_duration() > 30:
            logger.info("Perimeter sweep complete")
            self._transition_to(MissionState.CORNER_IDENTIFICATION)

    def _process_corner_identification(self, telemetry: SwarmTelemetry):
        """Process CORNER_IDENTIFICATION state"""
        # Simulate vision confirmation
        if self._get_state_duration() > 10:
            self.corner_identified_position = self.CORNER_TARGET.copy()
            self.target_confirmed = True
            logger.info(f"Corner identified at {self.corner_identified_position}")
            self._transition_to(MissionState.FORMATION_ASSEMBLE_LONG)

    def _process_formation_assemble(self, telemetry: SwarmTelemetry):
        """Process FORMATION_ASSEMBLE_LONG state"""
        # Check if all drones reached standoff position
        all_in_position = True
        if self.leader_id:
            leader_pos = telemetry.get_drone_position(self.leader_id)
            if leader_pos:
                dist = abs(leader_pos['x'] - self.STANDOFF_POSITION['x'])
                if dist > self.POSITION_TOLERANCE:
                    all_in_position = False

        if all_in_position or self._get_state_duration() > self.MAX_APPROACH_TIME / 2:
            logger.info("Formation assembled at standoff")
            self._transition_to(MissionState.PRECISION_CRAWL)

    def _process_precision_crawl(self, telemetry: SwarmTelemetry):
        """Process PRECISION_CRAWL state - slow approach to corner"""
        if self.leader_id:
            leader_pos = telemetry.get_drone_position(self.leader_id)
            if leader_pos:
                # Check if reached final corner position
                dist_x = abs(leader_pos['x'] - self.CORNER_TARGET['x'])
                dist_y = abs(leader_pos['y'] - self.CORNER_TARGET['y'])

                if dist_x < self.POSITION_TOLERANCE and dist_y < self.POSITION_TOLERANCE:
                    logger.info("Reached corner position")
                    self._transition_to(MissionState.CORNER_JAMMING)

        # Timeout check
        if self._get_state_duration() > self.MAX_APPROACH_TIME:
            logger.warning("Precision crawl timeout")
            self._transition_to(MissionState.CORNER_JAMMING)

    def _process_corner_jamming(self, telemetry: SwarmTelemetry):
        """Process CORNER_JAMMING state - hold position for 20 seconds"""
        duration = self._get_state_duration()

        # Check if leader is stationary (velocity < threshold)
        leader_stationary = False
        if self.leader_id and self.leader_id in telemetry.drones:
            leader_data = telemetry.drones[self.leader_id]
            vx = leader_data.get('vx', 0.0)
            vy = leader_data.get('vy', 0.0)
            velocity = (vx**2 + vy**2)**0.5
            leader_stationary = velocity < self.VELOCITY_TOLERANCE

        # Only count time when leader is stationary
        if leader_stationary and duration >= self.JAMMING_DURATION:
            logger.info(f"Jamming complete after {duration:.1f}s")
            self._transition_to(MissionState.VERTICAL_DROP)

    def _process_vertical_drop(self, telemetry: SwarmTelemetry):
        """Process VERTICAL_DROP state - Z-axis attack"""
        # Check if leader has completed vertical movement
        if self.leader_id:
            leader_pos = telemetry.get_drone_position(self.leader_id)
            if leader_pos and leader_pos['z'] > 5.0:
                logger.info("Vertical drop complete")
                self._transition_to(MissionState.MISSION_COMPLETE)

        # Timeout
        if self._get_state_duration() > 10:
            logger.info("Vertical drop timeout - completing mission")
            self._transition_to(MissionState.MISSION_COMPLETE)

    def _transition_to(self, new_state: MissionState):
        """Transition to a new state"""
        self.previous_state = self.current_state
        self.current_state = new_state
        self.state_start_time = time.time()

        # Log transition
        log_entry = {
            'from': self.previous_state.value,
            'to': new_state.value,
            'timestamp': self.state_start_time
        }
        self.state_history.append(log_entry)

        logger.info(f"State transition: {self.previous_state.value} -> {new_state.value}")

    def _abort_mission(self, reason: str):
        """Abort the mission"""
        self.abort_reason = reason
        logger.critical(f"MISSION ABORT: {reason}")
        self._transition_to(MissionState.EMERGENCY_ABORT)

    def _get_state_duration(self) -> float:
        """Get duration in current state (seconds)"""
        return time.time() - self.state_start_time

    def get_mission_elapsed_time(self) -> float:
        """Get total mission elapsed time (seconds)"""
        if self.mission_start_time:
            return time.time() - self.mission_start_time
        return 0.0

    def get_state_summary(self) -> Dict:
        """Get summary of current state"""
        return {
            'current_state': self.current_state.value,
            'previous_state': self.previous_state.value,
            'state_duration': self._get_state_duration(),
            'mission_elapsed': self.get_mission_elapsed_time(),
            'target_confirmed': self.target_confirmed,
            'abort_reason': self.abort_reason,
            'leader_id': self.leader_id,
            'follower_ids': self.follower_ids
        }
