"""
Mission Coordinator for Scenario 1

This module implements the swarm coordination layer, including
decision-making, telemetry aggregation, and communication protocols.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from mission_state_machine import MissionState


@dataclass
class DroneTelemetry:
    """Telemetry data from a single drone"""
    drone_id: str
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    battery_level: float
    status: str
    timestamp: float
    tracking_quality: float = 1.0  # 0-1, quality of OptiTrack tracking


@dataclass
class SensorFusion:
    """Aggregated sensor data from multiple drones"""
    timestamp: float
    all_positions: Dict[str, Tuple[float, float, float]]
    swarm_center: Tuple[float, float, float]
    average_battery: float
    detection_confidence: float
    target_position: Optional[Tuple[float, float, float]] = None


@dataclass
class Decision:
    """A tactical decision made by the decision engine"""
    decision_type: str
    priority: int  # 1-10, 10 being highest
    target_drones: List[str]
    action: str
    parameters: dict
    timestamp: float


class RiskLevel(Enum):
    """Risk level assessment"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MissionCoordinator:
    """
    Coordinates communication and synchronization between drones.

    Responsibilities:
    - Broadcast mission state to all drones
    - Collect and aggregate telemetry
    - Synchronize drones at key points
    - Make tactical decisions based on swarm state
    """

    # Communication frequencies (Hz)
    STATE_BROADCAST_FREQ = 5  # 5 Hz
    TELEMETRY_COLLECTION_FREQ = 10  # 10 Hz

    def __init__(self):
        """Initialize the mission coordinator"""
        self.logger = logging.getLogger("MissionCoordinator")
        self.current_mission_state: Optional[MissionState] = None
        self.last_state_broadcast = 0.0
        self.last_telemetry_collection = 0.0
        self.drone_telemetry: Dict[str, DroneTelemetry] = {}
        self.sync_points_reached: Dict[str, List[str]] = {}

        # Decision engine and telemetry aggregator
        self.decision_engine = DecisionEngine()
        self.telemetry_aggregator = TelemetryAggregator()

    def synchronize_drones(self, sync_point: str, drone_ids: List[str]) -> bool:
        """
        Synchronize drones at a specific sync point.

        Args:
            sync_point: Name of the synchronization point
            drone_ids: List of drone IDs that should synchronize

        Returns:
            True when all drones have reached the sync point
        """
        if sync_point not in self.sync_points_reached:
            self.sync_points_reached[sync_point] = []

        # In a real system, drones would report when they reach sync point
        # For simulation, we'll assume synchronization happens
        self.logger.info(f"Synchronizing {len(drone_ids)} drones at: {sync_point}")

        # Mark all drones as synchronized
        self.sync_points_reached[sync_point] = drone_ids.copy()

        self.logger.info(f"Synchronization complete at {sync_point}")
        return True

    def broadcast_mission_state(self, state: MissionState) -> bool:
        """
        Broadcast the current mission state to all drones.

        Args:
            state: The mission state to broadcast

        Returns:
            True if broadcast successful
        """
        current_time = time.time()

        # Check broadcast frequency
        time_since_last = current_time - self.last_state_broadcast
        min_interval = 1.0 / self.STATE_BROADCAST_FREQ

        if time_since_last < min_interval:
            return False  # Too soon, skip this broadcast

        self.current_mission_state = state
        self.last_state_broadcast = current_time

        self.logger.debug(f"Broadcasting mission state: {state.value}")

        # In a real system, this would publish to ROS2 topic /mission/state
        # For simulation, we just log it

        return True

    def collect_drone_telemetry(
        self,
        drone_states: Dict[str, dict]
    ) -> Dict[str, DroneTelemetry]:
        """
        Collect telemetry from all drones.

        Args:
            drone_states: Dictionary of drone states

        Returns:
            Dictionary of drone telemetry
        """
        current_time = time.time()

        # Check collection frequency
        time_since_last = current_time - self.last_telemetry_collection
        min_interval = 1.0 / self.TELEMETRY_COLLECTION_FREQ

        if time_since_last < min_interval:
            return self.drone_telemetry  # Return cached data

        self.last_telemetry_collection = current_time

        # Collect telemetry from each drone
        telemetry = {}
        for drone_id, state in drone_states.items():
            telemetry[drone_id] = DroneTelemetry(
                drone_id=drone_id,
                position=state.get("position", (0.0, 0.0, 0.0)),
                velocity=state.get("velocity", (0.0, 0.0, 0.0)),
                battery_level=state.get("battery", 100.0),
                status=state.get("status", "unknown"),
                timestamp=current_time,
                tracking_quality=state.get("tracking_quality", 1.0)
            )

        self.drone_telemetry = telemetry
        return telemetry

    def aggregate_sensor_data(self) -> SensorFusion:
        """
        Aggregate sensor data from all drones.

        Returns:
            Fused sensor data
        """
        return self.telemetry_aggregator.merge_telemetry(
            list(self.drone_telemetry.values())
        )

    def make_tactical_decision(self, situation: dict) -> Optional[Decision]:
        """
        Make a tactical decision based on current situation.

        Args:
            situation: Dictionary describing current situation

        Returns:
            Decision object or None if no action needed
        """
        return self.decision_engine.determine_next_action(situation)

    def broadcast_role_assignments(self, assignments: Dict[str, str]) -> bool:
        """
        Broadcast role assignments to all drones.

        Args:
            assignments: Dictionary mapping drone IDs to roles

        Returns:
            True if broadcast successful
        """
        self.logger.info(f"Broadcasting role assignments: {assignments}")

        # In a real system, publish to ROS2 topic /mission/role_assignments
        # For simulation, just log

        return True

    def signal_abort(self, reason: str) -> bool:
        """
        Send abort signal to all drones.

        Args:
            reason: Reason for abort

        Returns:
            True if signal sent successfully
        """
        self.logger.error(f"ABORT SIGNAL: {reason}")

        # In a real system, publish to ROS2 topic /mission/abort
        # This would trigger emergency landing procedures

        return True


class DecisionEngine:
    """
    Makes tactical decisions based on mission state and drone status.
    """

    # Risk thresholds
    CRITICAL_BATTERY = 20.0
    LOW_BATTERY = 40.0
    TRACKING_LOSS_THRESHOLD = 5.0  # seconds

    def __init__(self):
        """Initialize the decision engine"""
        self.logger = logging.getLogger("DecisionEngine")
        self.decision_history: List[Decision] = []
        self.tracking_loss_times: Dict[str, float] = {}

    def evaluate_mission_progress(
        self,
        current_state: MissionState,
        elapsed_time: float,
        telemetry: Dict[str, DroneTelemetry]
    ) -> float:
        """
        Evaluate mission progress as a percentage.

        Args:
            current_state: Current mission state
            elapsed_time: Time elapsed since mission start
            telemetry: Current drone telemetry

        Returns:
            Progress percentage (0-100)
        """
        # Map states to progress percentages
        state_progress = {
            MissionState.INITIALIZATION: 10,
            MissionState.SAFETY_CHECK: 20,
            MissionState.PATROL_SEARCH: 40,
            MissionState.TARGET_DETECTED: 50,
            MissionState.ROLE_ASSIGNMENT: 60,
            MissionState.APPROACH_TARGET: 75,
            MissionState.JAMMING: 90,
            MissionState.NEUTRALIZATION: 95,
            MissionState.MISSION_COMPLETE: 100,
            MissionState.MISSION_ABORT: 0,
        }

        return state_progress.get(current_state, 0)

    def determine_next_action(self, situation: dict) -> Optional[Decision]:
        """
        Determine the next action based on current situation.

        Args:
            situation: Dictionary with situation information

        Returns:
            Decision object or None
        """
        current_state = situation.get("current_state")
        telemetry = situation.get("telemetry", {})

        # Check for critical conditions first
        critical_decision = self._check_critical_conditions(telemetry)
        if critical_decision:
            return critical_decision

        # Normal decision-making based on state
        # In most cases, the state machine handles transitions
        # This is for exceptional situations

        return None

    def _check_critical_conditions(
        self,
        telemetry: Dict[str, DroneTelemetry]
    ) -> Optional[Decision]:
        """
        Check for critical conditions that require immediate action.

        Args:
            telemetry: Current drone telemetry

        Returns:
            Critical decision or None
        """
        current_time = time.time()

        # Check battery levels
        for drone_id, data in telemetry.items():
            if data.battery_level < self.CRITICAL_BATTERY:
                decision = Decision(
                    decision_type="emergency_land",
                    priority=10,
                    target_drones=[drone_id],
                    action="land_immediately",
                    parameters={"reason": "critical_battery"},
                    timestamp=current_time
                )
                self.decision_history.append(decision)
                return decision

        # Check tracking quality
        for drone_id, data in telemetry.items():
            if data.tracking_quality < 0.5:
                # Tracking is poor
                if drone_id not in self.tracking_loss_times:
                    self.tracking_loss_times[drone_id] = current_time
                else:
                    loss_duration = current_time - self.tracking_loss_times[drone_id]
                    if loss_duration > self.TRACKING_LOSS_THRESHOLD:
                        decision = Decision(
                            decision_type="abort_mission",
                            priority=9,
                            target_drones=list(telemetry.keys()),
                            action="abort",
                            parameters={"reason": f"tracking_lost_{drone_id}"},
                            timestamp=current_time
                        )
                        self.decision_history.append(decision)
                        return decision
            else:
                # Tracking restored
                if drone_id in self.tracking_loss_times:
                    del self.tracking_loss_times[drone_id]

        return None

    def assess_risk_level(
        self,
        telemetry: Dict[str, DroneTelemetry]
    ) -> RiskLevel:
        """
        Assess the current risk level.

        Args:
            telemetry: Current drone telemetry

        Returns:
            Current risk level
        """
        # Check various risk factors
        min_battery = min(
            (t.battery_level for t in telemetry.values()),
            default=100.0
        )

        min_tracking = min(
            (t.tracking_quality for t in telemetry.values()),
            default=1.0
        )

        # Determine risk level
        if min_battery < self.CRITICAL_BATTERY or min_tracking < 0.3:
            return RiskLevel.CRITICAL

        if min_battery < self.LOW_BATTERY or min_tracking < 0.7:
            return RiskLevel.HIGH

        if min_battery < 60.0:
            return RiskLevel.MEDIUM

        return RiskLevel.LOW

    def recommend_abort(
        self,
        telemetry: Dict[str, DroneTelemetry],
        mission_time: float
    ) -> Tuple[bool, str]:
        """
        Recommend whether to abort the mission.

        Args:
            telemetry: Current drone telemetry
            mission_time: Time elapsed in mission

        Returns:
            Tuple of (should_abort, reason)
        """
        # Check if any drone has critical battery
        for drone_id, data in telemetry.items():
            if data.battery_level < self.CRITICAL_BATTERY:
                return True, f"Critical battery on {drone_id}"

        # Check tracking loss
        for drone_id, loss_time in self.tracking_loss_times.items():
            if time.time() - loss_time > self.TRACKING_LOSS_THRESHOLD:
                return True, f"Tracking lost on {drone_id}"

        # Check if mission is taking too long
        if mission_time > 200.0:  # 20s buffer beyond 180s limit
            return True, "Mission timeout exceeded"

        # Check if less than 2/3 drones operational
        operational = sum(
            1 for t in telemetry.values()
            if t.status != "failed" and t.battery_level > 20.0
        )

        if operational < 2:
            return True, "Insufficient operational drones"

        return False, ""


class TelemetryAggregator:
    """
    Aggregates telemetry data from multiple drones.
    """

    def __init__(self):
        """Initialize the telemetry aggregator"""
        self.logger = logging.getLogger("TelemetryAggregator")

    def merge_telemetry(
        self,
        telemetry_list: List[DroneTelemetry]
    ) -> SensorFusion:
        """
        Merge telemetry from multiple drones.

        Args:
            telemetry_list: List of drone telemetry data

        Returns:
            Fused sensor data
        """
        if not telemetry_list:
            return SensorFusion(
                timestamp=time.time(),
                all_positions={},
                swarm_center=(0.0, 0.0, 0.0),
                average_battery=0.0,
                detection_confidence=0.0
            )

        # Merge position data
        all_positions = {
            t.drone_id: t.position
            for t in telemetry_list
        }

        # Calculate swarm center
        swarm_center = self.calculate_swarm_center(all_positions)

        # Calculate average battery
        average_battery = sum(t.battery_level for t in telemetry_list) / len(telemetry_list)

        # Detection confidence (placeholder - would use actual vision data)
        detection_confidence = 0.0

        return SensorFusion(
            timestamp=time.time(),
            all_positions=all_positions,
            swarm_center=swarm_center,
            average_battery=average_battery,
            detection_confidence=detection_confidence
        )

    def combine_detection_confidence(
        self,
        detections: List[dict]
    ) -> float:
        """
        Combine detection confidence from multiple drones.

        Args:
            detections: List of detection data from drones

        Returns:
            Combined confidence (0-1)
        """
        if not detections:
            return 0.0

        # Use maximum confidence approach
        # (could also use Bayesian fusion for better results)
        confidences = [d.get("confidence", 0.0) for d in detections]
        return max(confidences)

    def calculate_swarm_center(
        self,
        positions: Dict[str, Tuple[float, float, float]]
    ) -> Tuple[float, float, float]:
        """
        Calculate the geometric center of the swarm.

        Args:
            positions: Dictionary of drone positions

        Returns:
            Center position (x, y, z)
        """
        if not positions:
            return (0.0, 0.0, 0.0)

        n = len(positions)
        x_sum = sum(pos[0] for pos in positions.values())
        y_sum = sum(pos[1] for pos in positions.values())
        z_sum = sum(pos[2] for pos in positions.values())

        return (x_sum / n, y_sum / n, z_sum / n)

    def estimate_mission_completion(
        self,
        current_state: MissionState,
        elapsed_time: float
    ) -> float:
        """
        Estimate percentage of mission completion.

        Args:
            current_state: Current mission state
            elapsed_time: Time elapsed since mission start

        Returns:
            Estimated completion percentage (0-100)
        """
        # Expected time for each phase
        phase_times = {
            MissionState.INITIALIZATION: 10,
            MissionState.SAFETY_CHECK: 30,
            MissionState.PATROL_SEARCH: 90,
            MissionState.TARGET_DETECTED: 1,
            MissionState.ROLE_ASSIGNMENT: 1,
            MissionState.APPROACH_TARGET: 30,
            MissionState.JAMMING: 20,
            MissionState.NEUTRALIZATION: 5,
        }

        # Calculate total expected time up to current state
        total_time = 0
        for state, duration in phase_times.items():
            total_time += duration
            if state == current_state:
                break

        # Estimate completion
        if total_time > 0:
            return min(100.0, (elapsed_time / total_time) * 100.0)
        else:
            return 0.0
