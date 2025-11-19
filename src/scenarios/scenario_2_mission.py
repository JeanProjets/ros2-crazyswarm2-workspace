"""
Scenario 2 Mission Sequencer

This module orchestrates the full mission sequence for Scenario 2,
coordinating the state machine, battery management, and boundary monitoring.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import time
import logging

from .scenario_2_fsm import (
    Scenario2StateMachine,
    MissionState,
    SwarmTelemetry
)
from .battery_role_manager import (
    BatteryRoleManager,
    Drone,
    DroneRole,
    MissionPhase
)
from .boundary_guard import (
    SafetyOverride,
    Telemetry,
    ViolationType
)

logger = logging.getLogger(__name__)


class MissionResult(str):
    """Mission outcome results"""
    SUCCESS = "success"
    FAILED = "failed"
    ABORTED = "aborted"
    IN_PROGRESS = "in_progress"


@dataclass
class MissionTelemetryLog:
    """Telemetry logging for mission analysis"""
    timestamp: float
    state: str
    drone_positions: Dict[str, Dict[str, float]]
    drone_voltages: Dict[str, float]
    distance_to_walls: Dict[str, Dict[str, float]]

    def __repr__(self):
        return f"TelemetryLog(t={self.timestamp:.1f}s, state={self.state})"


class Scenario2MissionSequencer:
    """
    Orchestrates the full Scenario 2 mission sequence

    This class coordinates:
    1. Battery-based role assignment
    2. State machine progression
    3. Boundary safety monitoring
    4. Telemetry logging
    5. Mission timing and sequencing
    """

    # Timing parameters (seconds)
    INIT_DURATION = 10
    MAX_PATROL_DURATION = 120  # 2 minutes
    MAX_APPROACH_DURATION = 45
    JAMMING_DURATION = 20
    MAX_ATTACK_DURATION = 5

    # Telemetry logging rate
    TELEMETRY_LOG_RATE_HZ = 10

    def __init__(self, swarm_manager=None):
        """
        Initialize the mission sequencer

        Args:
            swarm_manager: Optional swarm manager for commanding drones
        """
        self.swarm_manager = swarm_manager

        # Core components
        self.fsm = Scenario2StateMachine()
        self.battery_manager = BatteryRoleManager()
        self.safety_override = SafetyOverride(swarm_manager)

        # Mission state
        self.mission_result = MissionResult.IN_PROGRESS
        self.mission_start_time: Optional[float] = None
        self.mission_end_time: Optional[float] = None

        # Telemetry logging
        self.telemetry_logs: List[MissionTelemetryLog] = []
        self.last_log_time: float = 0.0

        # Wall distance tracking
        self.wall_distances: List[Dict] = []

    def initialize_mission(self, available_drones: List[Drone]) -> bool:
        """
        Initialize the mission with role assignment

        Args:
            available_drones: List of available drones with battery info

        Returns:
            True if initialization successful, False otherwise
        """
        logger.info(f"Initializing Scenario 2 mission with {len(available_drones)} drones")

        # Phase 1: Assign roles based on battery
        role_assignments = self.battery_manager.assign_roles(available_drones)

        if not role_assignments or self.battery_manager.mission_aborted:
            logger.error(f"Role assignment failed: {self.battery_manager.abort_reason}")
            self.mission_result = MissionResult.ABORTED
            return False

        # Extract leader and followers
        leader = role_assignments.get(DroneRole.LEADER)
        followers = []
        for role in [DroneRole.FOLLOWER_LEFT, DroneRole.FOLLOWER_RIGHT]:
            follower = role_assignments.get(role)
            if follower:
                followers.append(follower.drone_id)

        if not leader:
            logger.error("No leader assigned")
            self.mission_result = MissionResult.ABORTED
            return False

        logger.info(f"Roles assigned - Leader: {leader.drone_id}({leader.voltage}V), "
                   f"Followers: {followers}")

        # Phase 2: Check OptiTrack visibility (simulated)
        logger.info("Checking OptiTrack visibility in far corner...")
        time.sleep(0.1)  # Simulate check

        # Phase 3: Start state machine
        self.fsm.start_mission(leader.drone_id, followers)
        self.mission_start_time = time.time()

        logger.info("Mission initialization complete")
        return True

    def execute_mission_step(self, current_telemetry: SwarmTelemetry) -> MissionState:
        """
        Execute one step of the mission

        Args:
            current_telemetry: Current telemetry from all drones

        Returns:
            Current mission state
        """
        # Log telemetry
        self._log_telemetry(current_telemetry)

        # Check safety boundaries first
        telemetry_list = self._convert_to_telemetry_list(current_telemetry)
        violations = self.safety_override.monitor_loop(telemetry_list)

        if violations:
            logger.warning(f"Boundary violations detected: {len(violations)} drones")
            # Safety system will trigger emergency stop if needed

        # Check leader battery
        if self.fsm.leader_id:
            leader_voltage = current_telemetry.get_drone_voltage(self.fsm.leader_id)
            if leader_voltage:
                current_phase = self._map_state_to_phase(self.fsm.current_state)
                battery_check = self.battery_manager.check_leader_voltage(
                    leader_voltage,
                    current_phase
                )

                if battery_check['action'] == 'IMMEDIATE_RTH':
                    logger.critical("Leader battery critical - aborting mission")
                    self.mission_result = MissionResult.ABORTED
                    return self.fsm.current_state

        # Update state machine
        new_state = self.fsm.update(current_telemetry)

        # Check for mission completion
        if new_state == MissionState.MISSION_COMPLETE:
            self._complete_mission(MissionResult.SUCCESS)
        elif new_state == MissionState.EMERGENCY_ABORT:
            self._complete_mission(MissionResult.ABORTED)

        return new_state

    def run_patrol_phase(self, telemetry: SwarmTelemetry) -> bool:
        """
        Execute patrol phase with timeout

        Args:
            telemetry: Current swarm telemetry

        Returns:
            True if target found, False if timeout
        """
        patrol_start = time.time()

        while time.time() - patrol_start < self.MAX_PATROL_DURATION:
            # In real implementation, would trigger Agent 2's "CornerBiasPatrol"
            # For now, simulate target detection
            if self.fsm.leader_id:
                leader_pos = telemetry.get_drone_position(self.fsm.leader_id)
                if leader_pos and leader_pos['x'] > 8.0:
                    logger.info("Target found at X > 8.0, switching to Approach")
                    return True

        logger.warning("Patrol timeout - target not found within 2 minutes")
        return False

    def execute_approach_phase(self, telemetry: SwarmTelemetry) -> bool:
        """
        Execute approach phase with formation verification

        Args:
            telemetry: Current swarm telemetry

        Returns:
            True if approach successful
        """
        logger.info("Commanding formation to X=8.5 (Standoff)")

        # In real implementation, would command formation
        # For now, simulate the process
        approach_start = time.time()

        while time.time() - approach_start < self.MAX_APPROACH_DURATION:
            # Check if formation assembled
            if self._verify_formation(telemetry):
                logger.info("Formation verified - proceeding to X=9.5")
                return True

            time.sleep(0.1)

        logger.warning("Approach timeout")
        return False

    def execute_jamming_phase(self, telemetry: SwarmTelemetry) -> bool:
        """
        Execute jamming phase with position variance monitoring

        Args:
            telemetry: Current swarm telemetry

        Returns:
            True if jamming successful
        """
        jamming_start = time.time()
        variance_samples = []

        while time.time() - jamming_start < self.JAMMING_DURATION:
            # Monitor XYZ variance
            variance = self._calculate_position_variance(telemetry)
            variance_samples.append(variance)

            if variance > 0.1:
                logger.warning(f"High position variance: {variance:.3f}m - drone unstable near wall")
                # In real implementation, would pull back
                return False

            time.sleep(0.1)

        logger.info(f"Jamming complete - avg variance: {sum(variance_samples)/len(variance_samples):.3f}m")
        return True

    def execute_attack_phase(self, telemetry: SwarmTelemetry) -> bool:
        """
        Execute vertical strike attack

        Args:
            telemetry: Current swarm telemetry

        Returns:
            True if attack successful
        """
        logger.info("Triggering vertical strike")

        # In real implementation, would trigger Agent 2's "VerticalStrike"
        # Verify P is above C (Z > 5.0)
        attack_start = time.time()

        while time.time() - attack_start < self.MAX_ATTACK_DURATION:
            if self.fsm.leader_id:
                leader_pos = telemetry.get_drone_position(self.fsm.leader_id)
                if leader_pos and leader_pos['z'] > 5.0:
                    logger.info("Vertical strike complete - P above target altitude")
                    return True

            time.sleep(0.1)

        logger.warning("Attack phase timeout")
        return False

    def _log_telemetry(self, telemetry: SwarmTelemetry):
        """Log telemetry at specified rate"""
        current_time = time.time()

        # Check if enough time has passed since last log
        if current_time - self.last_log_time < (1.0 / self.TELEMETRY_LOG_RATE_HZ):
            return

        # Calculate distance to walls for each drone
        distance_to_walls = {}
        for drone_id, drone_data in telemetry.drones.items():
            x = drone_data.get('x', 0.0)
            y = drone_data.get('y', 0.0)

            distance_to_walls[drone_id] = {
                'x_max': 10.0 - x,  # Distance to right wall
                'x_min': x,  # Distance to left wall
                'y_max': 10.0 - y,  # Distance to far wall
                'y_min': y  # Distance to near wall
            }

        # Create log entry
        log_entry = MissionTelemetryLog(
            timestamp=current_time - (self.mission_start_time or current_time),
            state=self.fsm.current_state.value,
            drone_positions={
                drone_id: {'x': d.get('x', 0), 'y': d.get('y', 0), 'z': d.get('z', 0)}
                for drone_id, d in telemetry.drones.items()
            },
            drone_voltages={
                drone_id: d.get('voltage', 0.0)
                for drone_id, d in telemetry.drones.items()
            },
            distance_to_walls=distance_to_walls
        )

        self.telemetry_logs.append(log_entry)
        self.wall_distances.append({
            'timestamp': log_entry.timestamp,
            'distances': distance_to_walls
        })

        self.last_log_time = current_time

    def _convert_to_telemetry_list(self, swarm_telemetry: SwarmTelemetry) -> List[Telemetry]:
        """Convert SwarmTelemetry to list of Telemetry objects"""
        telemetry_list = []

        for drone_id, drone_data in swarm_telemetry.drones.items():
            telem = Telemetry(
                drone_id=drone_id,
                x=drone_data.get('x', 0.0),
                y=drone_data.get('y', 0.0),
                z=drone_data.get('z', 0.0),
                vx=drone_data.get('vx', 0.0),
                vy=drone_data.get('vy', 0.0),
                vz=drone_data.get('vz', 0.0)
            )
            telemetry_list.append(telem)

        return telemetry_list

    def _map_state_to_phase(self, state: MissionState) -> MissionPhase:
        """Map FSM state to battery management phase"""
        mapping = {
            MissionState.IDLE: MissionPhase.INIT,
            MissionState.SAFETY_CHK_AND_TRANSIT: MissionPhase.TRANSIT,
            MissionState.PERIMETER_SWEEP: MissionPhase.PATROL,
            MissionState.CORNER_IDENTIFICATION: MissionPhase.PATROL,
            MissionState.FORMATION_ASSEMBLE_LONG: MissionPhase.APPROACH,
            MissionState.PRECISION_CRAWL: MissionPhase.APPROACH,
            MissionState.CORNER_JAMMING: MissionPhase.JAMMING,
            MissionState.VERTICAL_DROP: MissionPhase.ATTACK,
            MissionState.MISSION_COMPLETE: MissionPhase.RTH,
            MissionState.EMERGENCY_ABORT: MissionPhase.RTH
        }
        return mapping.get(state, MissionPhase.PATROL)

    def _verify_formation(self, telemetry: SwarmTelemetry) -> bool:
        """Verify that drones are in proper formation"""
        # In real implementation, would check Agent 2's "Adaptive Formation"
        # For now, just check if followers are present
        if not self.fsm.follower_ids:
            return True

        # Check if followers are near standoff position
        for follower_id in self.fsm.follower_ids:
            follower_pos = telemetry.get_drone_position(follower_id)
            if not follower_pos:
                return False

        return True

    def _calculate_position_variance(self, telemetry: SwarmTelemetry) -> float:
        """Calculate position variance of leader drone"""
        if not self.fsm.leader_id:
            return 0.0

        # In real implementation, would calculate variance over time window
        # For now, return 0 (stable)
        return 0.0

    def _complete_mission(self, result: MissionResult):
        """Mark mission as complete"""
        self.mission_result = result
        self.mission_end_time = time.time()

        duration = self.mission_end_time - (self.mission_start_time or self.mission_end_time)
        logger.info(f"Mission completed: {result} in {duration:.1f}s")

    def get_mission_summary(self) -> Dict[str, Any]:
        """Get comprehensive mission summary"""
        return {
            'result': self.mission_result,
            'duration': self.mission_end_time - self.mission_start_time if self.mission_end_time and self.mission_start_time else 0.0,
            'state_machine': self.fsm.get_state_summary(),
            'battery_management': self.battery_manager.get_role_summary(),
            'safety_violations': self.safety_override.get_violation_summary(),
            'telemetry_logs_count': len(self.telemetry_logs),
            'wall_distances_logged': len(self.wall_distances)
        }

    def export_telemetry(self, filepath: str):
        """
        Export telemetry logs to file

        Args:
            filepath: Path to export file
        """
        try:
            import json
            with open(filepath, 'w') as f:
                data = {
                    'mission_summary': self.get_mission_summary(),
                    'telemetry_logs': [
                        {
                            'timestamp': log.timestamp,
                            'state': log.state,
                            'positions': log.drone_positions,
                            'voltages': log.drone_voltages,
                            'wall_distances': log.distance_to_walls
                        }
                        for log in self.telemetry_logs
                    ]
                }
                json.dump(data, f, indent=2)
            logger.info(f"Telemetry exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export telemetry: {e}")
