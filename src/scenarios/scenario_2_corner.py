"""
Scenario 2: Corner Target Mission

This module implements the state machine for Scenario 2 where the target
is located in the far corner (9.5, 0.5, 5.0). The mission requires:
- Precision flight near boundaries
- Battery-optimized role selection
- Dynamic formation adjustment to avoid wall collisions
"""

import time
import yaml
import logging
from enum import Enum
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

from src.core.swarm_manager_v2 import SwarmCoordinator, DroneRole
from src.core.safe_drone_controller import SafeDroneController


class MissionState(Enum):
    """States for Scenario 2 mission state machine."""
    INITIALIZATION = "INITIALIZATION"
    LONG_RANGE_PATROL = "LONG_RANGE_PATROL"
    TARGET_LOCK = "TARGET_LOCK"
    PRECISION_APPROACH = "PRECISION_APPROACH"
    CORNER_JAMMING = "CORNER_JAMMING"
    VERTICAL_NEUTRALIZATION = "VERTICAL_NEUTRALIZATION"
    MISSION_COMPLETE = "MISSION_COMPLETE"
    EMERGENCY_ABORT = "EMERGENCY_ABORT"


class Scenario2Mission:
    """
    Scenario 2: Corner Target Mission Controller.

    This mission targets a threat located in the far corner of the arena,
    requiring precise boundary management and battery-efficient operations.
    """

    def __init__(self, config_path: str = "config/scenario_2_config.yaml"):
        """
        Initialize the Scenario 2 mission.

        Args:
            config_path: Path to the configuration YAML file
        """
        # Setup logging
        self.logger = logging.getLogger("Scenario2Mission")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] [Scenario2] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize swarm coordinator
        self.swarm = SwarmCoordinator(config=self.config)
        self.swarm.initialize_swarm(self.config['drone_configs'])

        # Mission state
        self.current_state = MissionState.INITIALIZATION
        self.state_start_time = time.time()
        self.mission_start_time = None

        # Target information
        self.target_pos = (
            self.config['target_position']['x'],
            self.config['target_position']['y'],
            self.config['target_position']['z']
        )
        self.target_detected = False

        # Leader/Follower assignments
        self.leader_id: Optional[str] = None
        self.follower_id: Optional[str] = None

        self.logger.info("=" * 60)
        self.logger.info("Scenario 2: Corner Target Mission Initialized")
        self.logger.info(f"Target: {self.target_pos}")
        self.logger.info("=" * 60)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found, using defaults")
            # Return minimal default config
            return {
                'safety_bounds': {
                    'x_min': 0.3, 'x_max': 9.7,
                    'y_min': 0.3, 'y_max': 5.7,
                    'z_min': 0.2, 'z_max': 6.0
                },
                'target_position': {'x': 9.5, 'y': 0.5, 'z': 5.0},
                'drone_configs': {
                    'cf1': {'role': 'NEUTRAL_1', 'start_pos': [2.5, 2.5, 0]},
                    'cf2': {'role': 'NEUTRAL_2', 'start_pos': [2.5, 3.5, 0]},
                    'cf3': {'role': 'PATROL', 'start_pos': [3.0, 5.0, 0]}
                },
                'mission_parameters': {
                    'approach_speed': 0.7,
                    'precision_speed': 0.5,
                    'total_mission_timeout': 210,
                    'patrol_timeout': 120
                }
            }

    def transition_to_state(self, new_state: MissionState):
        """
        Transition to a new mission state.

        Args:
            new_state: The state to transition to
        """
        self.logger.info(f"State Transition: {self.current_state.value} -> {new_state.value}")
        self.current_state = new_state
        self.state_start_time = time.time()

    def get_time_in_state(self) -> float:
        """
        Get time elapsed in current state.

        Returns:
            Time in seconds
        """
        return time.time() - self.state_start_time

    def get_mission_elapsed_time(self) -> float:
        """
        Get total mission elapsed time.

        Returns:
            Time in seconds since mission start
        """
        if self.mission_start_time is None:
            return 0.0
        return time.time() - self.mission_start_time

    def check_battery_emergency(self) -> bool:
        """
        Check if any drone has critical battery.

        Returns:
            True if emergency abort needed
        """
        for drone_id, controller in self.swarm.drones.items():
            if controller.check_battery_status() == 'CRITICAL':
                self.logger.error(f"🔋 CRITICAL BATTERY on {drone_id}! Emergency abort!")
                return True
        return False

    # ===== STATE HANDLERS =====

    def state_initialization(self):
        """
        INITIALIZATION state: Safety zone setup and preflight checks.
        """
        self.logger.info("=== INITIALIZATION ===")

        # Perform preflight checks
        self.logger.info("Performing preflight checks...")

        # Check all drones are present
        expected_drones = ['cf1', 'cf2', 'cf3']
        for drone_id in expected_drones:
            if drone_id not in self.swarm.drones:
                self.logger.error(f"Drone {drone_id} not found!")
                return

        # Log initial battery status
        self.swarm.log_battery_status()

        # Takeoff all drones to initial height
        takeoff_height = self.config.get('mission_parameters', {}).get('takeoff_height', 1.0)
        self.logger.info(f"Taking off all drones to {takeoff_height}m...")

        for drone_id, controller in self.swarm.drones.items():
            controller.takeoff(takeoff_height)

        # Transition to patrol after initialization
        time.sleep(2.0)  # Simulate takeoff time
        self.transition_to_state(MissionState.LONG_RANGE_PATROL)

    def state_long_range_patrol(self):
        """
        LONG_RANGE_PATROL state: P covers full field to find corner target.
        """
        patrol_drone_id = self.swarm.get_drone_by_role(DroneRole.PATROL)
        if not patrol_drone_id:
            self.logger.error("No PATROL drone found!")
            return

        patrol_drone = self.swarm.drones[patrol_drone_id]

        self.logger.info(f"=== LONG_RANGE_PATROL (Drone: {patrol_drone_id}) ===")

        # Simulate patrol pattern - move toward target area
        patrol_waypoints = [
            (5.0, 3.0, 3.0),
            (7.0, 2.0, 4.0),
            (9.0, 1.0, 5.0),  # Near target
        ]

        for waypoint in patrol_waypoints:
            x, y, z = waypoint
            patrol_drone.clamped_navigate(x, y, z, yaw=0.0, duration=3.0)
            time.sleep(0.5)

            # Simulate target detection when near target
            distance_to_target = (
                (x - self.target_pos[0])**2 +
                (y - self.target_pos[1])**2 +
                (z - self.target_pos[2])**2
            )**0.5

            if distance_to_target < 3.0:
                self.target_detected = True
                self.logger.info(f"🎯 TARGET DETECTED at distance {distance_to_target:.2f}m!")
                break

        if self.target_detected:
            self.transition_to_state(MissionState.TARGET_LOCK)
        elif self.get_time_in_state() > self.config.get('mission_parameters', {}).get('patrol_timeout', 120):
            self.logger.warning("Patrol timeout! Aborting mission.")
            self.transition_to_state(MissionState.EMERGENCY_ABORT)

    def state_target_lock(self):
        """
        TARGET_LOCK state: Detection at distance, prepare for approach.
        """
        self.logger.info("=== TARGET LOCK ===")

        # Select optimal leader from neutral drones
        candidates = ['cf1', 'cf2']  # N1 and N2
        self.leader_id = self.swarm.select_optimal_leader(candidates)

        # Assign the other as follower
        self.follower_id = 'cf2' if self.leader_id == 'cf1' else 'cf1'

        # Update roles
        self.swarm.assign_roles(self.leader_id, self.follower_id)

        self.logger.info(f"Leader: {self.leader_id}, Follower: {self.follower_id}")

        time.sleep(1.0)
        self.transition_to_state(MissionState.PRECISION_APPROACH)

    def state_precision_approach(self):
        """
        PRECISION_APPROACH state: Leader/Follower move to standoff point.
        """
        self.logger.info("=== PRECISION APPROACH ===")

        if not self.leader_id or not self.follower_id:
            self.logger.error("Leader/Follower not assigned!")
            return

        # Standoff point (safer position before final approach)
        standoff_point = (9.0, 1.0, 5.0)

        leader = self.swarm.drones[self.leader_id]
        follower = self.swarm.drones[self.follower_id]

        # Move leader to standoff point (slow speed for safety)
        self.logger.info(f"Moving {self.leader_id} to standoff point {standoff_point}")
        leader.clamped_navigate(
            standoff_point[0],
            standoff_point[1],
            standoff_point[2],
            yaw=0.0,
            duration=5.0
        )

        time.sleep(2.0)

        # Calculate safe formation position for follower
        follower_pos = self.swarm.calculate_safe_formation(
            standoff_point,
            self.follower_id
        )

        self.logger.info(f"Moving {self.follower_id} to formation position {follower_pos}")
        follower.clamped_navigate(
            follower_pos[0],
            follower_pos[1],
            follower_pos[2],
            yaw=0.0,
            duration=5.0
        )

        time.sleep(3.0)
        self.transition_to_state(MissionState.CORNER_JAMMING)

    def state_corner_jamming(self):
        """
        CORNER_JAMMING state: Maintain position without drifting behind target.
        """
        self.logger.info("=== CORNER JAMMING ===")

        # Move to final target position
        leader = self.swarm.drones[self.leader_id]
        follower = self.swarm.drones[self.follower_id]

        # Leader moves to target
        self.logger.info(f"Moving {self.leader_id} to target position {self.target_pos}")
        leader.clamped_navigate(
            self.target_pos[0],
            self.target_pos[1],
            self.target_pos[2],
            yaw=0.0,
            duration=4.0
        )

        time.sleep(2.0)

        # Update follower formation
        follower_pos = self.swarm.calculate_safe_formation(
            self.target_pos,
            self.follower_id
        )

        follower.clamped_navigate(
            follower_pos[0],
            follower_pos[1],
            follower_pos[2],
            yaw=0.0,
            duration=4.0
        )

        # Maintain position (simulate jamming)
        self.logger.info("Jamming target...")
        time.sleep(3.0)

        self.transition_to_state(MissionState.VERTICAL_NEUTRALIZATION)

    def state_vertical_neutralization(self):
        """
        VERTICAL_NEUTRALIZATION state: P attacks from strictly above.
        """
        self.logger.info("=== VERTICAL NEUTRALIZATION ===")

        patrol_drone_id = self.swarm.get_drone_by_role(DroneRole.PATROL)
        if not patrol_drone_id:
            self.logger.error("No PATROL drone for neutralization!")
            return

        patrol_drone = self.swarm.drones[patrol_drone_id]

        # Move directly above target
        above_target = (self.target_pos[0], self.target_pos[1], self.target_pos[2] + 1.0)

        self.logger.info(f"Moving {patrol_drone_id} above target {above_target}")
        patrol_drone.clamped_navigate(
            above_target[0],
            above_target[1],
            above_target[2],
            yaw=0.0,
            duration=3.0
        )

        time.sleep(2.0)

        # Vertical descent to neutralize
        self.logger.info("Executing vertical descent to neutralize target...")
        patrol_drone.clamped_navigate(
            self.target_pos[0],
            self.target_pos[1],
            self.target_pos[2],
            yaw=0.0,
            duration=3.0
        )

        time.sleep(2.0)

        self.logger.info("🎯 TARGET NEUTRALIZED!")
        self.transition_to_state(MissionState.MISSION_COMPLETE)

    def state_mission_complete(self):
        """
        MISSION_COMPLETE state: Return to base.
        """
        self.logger.info("=== MISSION COMPLETE ===")

        # Return all drones to safe height
        rtb_height = 1.5

        self.logger.info(f"Returning all drones to {rtb_height}m...")
        for drone_id, controller in self.swarm.drones.items():
            pos = controller.get_position()
            controller.clamped_navigate(pos[0], pos[1], rtb_height, yaw=0.0, duration=3.0)

        time.sleep(3.0)

        # Land all drones
        self.logger.info("Landing all drones...")
        for drone_id, controller in self.swarm.drones.items():
            controller.land()

        time.sleep(2.0)

        # Final status
        self.swarm.log_battery_status()
        elapsed = self.get_mission_elapsed_time()
        self.logger.info("=" * 60)
        self.logger.info(f"✓ MISSION SUCCESS in {elapsed:.1f}s")
        self.logger.info("=" * 60)

    def state_emergency_abort(self):
        """
        EMERGENCY_ABORT state: Emergency landing of all drones.
        """
        self.logger.error("=== EMERGENCY ABORT ===")

        for drone_id, controller in self.swarm.drones.items():
            self.logger.error(f"Emergency landing {drone_id}")
            controller.land()

        time.sleep(2.0)
        self.logger.error("Mission aborted.")

    def run_mission(self):
        """
        Execute the complete Scenario 2 mission.
        """
        self.mission_start_time = time.time()

        # State machine loop
        state_handlers = {
            MissionState.INITIALIZATION: self.state_initialization,
            MissionState.LONG_RANGE_PATROL: self.state_long_range_patrol,
            MissionState.TARGET_LOCK: self.state_target_lock,
            MissionState.PRECISION_APPROACH: self.state_precision_approach,
            MissionState.CORNER_JAMMING: self.state_corner_jamming,
            MissionState.VERTICAL_NEUTRALIZATION: self.state_vertical_neutralization,
            MissionState.MISSION_COMPLETE: self.state_mission_complete,
            MissionState.EMERGENCY_ABORT: self.state_emergency_abort,
        }

        while self.current_state not in [MissionState.MISSION_COMPLETE, MissionState.EMERGENCY_ABORT]:
            # Check battery emergency
            if self.check_battery_emergency():
                self.transition_to_state(MissionState.EMERGENCY_ABORT)
                continue

            # Check mission timeout
            if self.get_mission_elapsed_time() > self.config.get('mission_parameters', {}).get('total_mission_timeout', 210):
                self.logger.warning("Mission timeout exceeded!")
                self.transition_to_state(MissionState.EMERGENCY_ABORT)
                continue

            # Execute current state handler
            handler = state_handlers.get(self.current_state)
            if handler:
                handler()
            else:
                self.logger.error(f"No handler for state {self.current_state}")
                break

        # Final state execution
        if self.current_state == MissionState.MISSION_COMPLETE:
            self.state_mission_complete()
        elif self.current_state == MissionState.EMERGENCY_ABORT:
            self.state_emergency_abort()


def main():
    """Main entry point for Scenario 2 mission."""
    import sys

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    # Determine config path
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "config/scenario_2_config.yaml"

    # Run mission
    mission = Scenario2Mission(config_path)
    mission.run_mission()


if __name__ == "__main__":
    main()
