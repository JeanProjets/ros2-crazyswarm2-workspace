"""
Scenario 1 Mission Sequencer: Static Target Neutralization.

Implements state machine for coordinated search and neutralization
of a static target using a 3-drone swarm.
"""
import asyncio
import logging
import time
from enum import Enum
from typing import Dict, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.swarm_coordinator import SwarmCoordinator, DroneRole
from core.drone_controller import DroneController


class MissionState(Enum):
    """States in the Scenario 1 mission state machine."""
    INITIALIZATION = "initialization"
    SAFETY_CHECK = "safety_check"
    PATROL_SEARCH = "patrol_search"
    TARGET_DETECTED = "target_detected"
    ROLE_ASSIGNMENT = "role_assignment"
    APPROACH_TARGET = "approach_target"
    JAMMING = "jamming"
    NEUTRALIZATION = "neutralization"
    MISSION_COMPLETE = "mission_complete"
    MISSION_FAILED = "mission_failed"


class Scenario1Mission:
    """
    Mission sequencer for Scenario 1: Static Target Neutralization.

    Coordinates 3 drones to search for and neutralize a static target
    using a multi-phase approach with role-based coordination.
    """

    def __init__(self, config: Dict):
        """
        Initialize Scenario 1 mission.

        Args:
            config: Mission configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("Scenario1Mission")

        # Extract configuration
        self.cage_bounds = (
            config['cage_dimensions']['x'],
            config['cage_dimensions']['y'],
            config['cage_dimensions']['z']
        )
        self.target_position = (
            config['target_position']['x'],
            config['target_position']['y'],
            config['target_position']['z']
        )
        self.safety_zone = config['safety_zone']
        self.mission_params = config['mission_parameters']

        # Initialize swarm coordinator
        self.swarm = SwarmCoordinator(
            drone_ids=['cf1', 'cf2', 'cf3'],
            cage_bounds=self.cage_bounds
        )

        # Mission state
        self.current_state = MissionState.INITIALIZATION
        self.start_time: Optional[float] = None
        self.target_detected = False
        self.leader_id: Optional[str] = None
        self.follower_id: Optional[str] = None
        self.patrol_id: Optional[str] = None

        self.logger.info("Scenario 1 mission initialized")

    async def run_mission(self) -> bool:
        """
        Execute the complete mission sequence.

        Returns:
            True if mission completed successfully, False otherwise
        """
        self.start_time = time.time()
        self.logger.info("=== MISSION START ===")

        try:
            # Execute state machine
            while self.current_state not in [
                MissionState.MISSION_COMPLETE,
                MissionState.MISSION_FAILED
            ]:
                # Check timeout
                if self._check_timeout():
                    self.logger.error("Mission timeout!")
                    self.current_state = MissionState.MISSION_FAILED
                    break

                # Execute current state
                success = await self._execute_current_state()

                if not success:
                    self.logger.error(f"State {self.current_state.value} failed")
                    self.current_state = MissionState.MISSION_FAILED
                    break

                # Small delay between states
                await asyncio.sleep(0.5)

            # Mission complete
            elapsed = time.time() - self.start_time
            self.logger.info(
                f"=== MISSION {self.current_state.value.upper()} "
                f"in {elapsed:.1f}s ==="
            )

            return self.current_state == MissionState.MISSION_COMPLETE

        except Exception as e:
            self.logger.error(f"Mission failed with exception: {e}")
            return False

        finally:
            # Always land all drones
            self.swarm.emergency_land_all()

    async def _execute_current_state(self) -> bool:
        """
        Execute logic for current state and transition to next.

        Returns:
            True if state executed successfully
        """
        self.logger.info(f"Executing state: {self.current_state.value}")

        # State handlers
        handlers = {
            MissionState.INITIALIZATION: self._state_initialization,
            MissionState.SAFETY_CHECK: self._state_safety_check,
            MissionState.PATROL_SEARCH: self._state_patrol_search,
            MissionState.TARGET_DETECTED: self._state_target_detected,
            MissionState.ROLE_ASSIGNMENT: self._state_role_assignment,
            MissionState.APPROACH_TARGET: self._state_approach_target,
            MissionState.JAMMING: self._state_jamming,
            MissionState.NEUTRALIZATION: self._state_neutralization,
        }

        handler = handlers.get(self.current_state)
        if handler is None:
            self.logger.error(f"No handler for state {self.current_state}")
            return False

        return await handler()

    async def _state_initialization(self) -> bool:
        """Initialize swarm in safety zone."""
        try:
            # Initialize swarm with default roles
            if not self.swarm.initialize_swarm():
                return False

            # Assign initial positions
            positions = {
                'cf1': (2.5, 2.5, 0, 4.0),  # NEUTRAL_1
                'cf2': (2.5, 3.5, 0, 4.0),  # NEUTRAL_2
                'cf3': (3.0, 5.0, 0, 4.0),  # PATROL
            }

            if not self.swarm.assign_initial_positions(positions):
                return False

            self.logger.info("Drones initialized and hovering")

            # Transition
            self.current_state = MissionState.SAFETY_CHECK
            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False

    async def _state_safety_check(self) -> bool:
        """Neutral drones scan safety zone."""
        try:
            self.logger.info("Scanning safety zone (3x3m)")

            # Define safety zone scan pattern (rectangular)
            scan_waypoints = [
                (0.5, 0.5, 4.0),
                (2.5, 0.5, 4.0),
                (2.5, 2.5, 4.0),
                (0.5, 2.5, 4.0),
                (0.5, 0.5, 4.0),
            ]

            # Both neutral drones scan in parallel
            tasks = []
            for drone_id in ['cf1', 'cf2']:
                task = self._fly_pattern(drone_id, scan_waypoints)
                tasks.append(task)

            await asyncio.gather(*tasks)

            # Simulate scanning time
            await asyncio.sleep(2.0)

            self.logger.info("Safety zone clear")

            # Transition
            self.current_state = MissionState.PATROL_SEARCH
            return True

        except Exception as e:
            self.logger.error(f"Safety check failed: {e}")
            return False

    async def _state_patrol_search(self) -> bool:
        """Patrol drone searches for target."""
        try:
            self.logger.info("Patrol searching for target")

            # Patrol search pattern (cover area x=3 to x=10)
            search_waypoints = [
                (4.0, 1.0, 4.0),
                (6.0, 1.0, 4.0),
                (6.0, 5.0, 4.0),
                (8.0, 5.0, 4.0),
                (8.0, 1.0, 4.0),
                (9.0, 3.0, 4.0),
            ]

            # Patrol drone searches
            await self._fly_pattern('cf3', search_waypoints)

            # Simulate target detection
            self.logger.info("TARGET DETECTED!")
            self.target_detected = True

            # Transition
            self.current_state = MissionState.TARGET_DETECTED
            return True

        except Exception as e:
            self.logger.error(f"Patrol search failed: {e}")
            return False

    async def _state_target_detected(self) -> bool:
        """Broadcast target detection."""
        try:
            # Update swarm with target position
            self.swarm.target_position = self.target_position

            # Broadcast from patrol drone
            self.swarm.broadcast_drone_status(
                drone_id='cf3',
                position=self.target_position,
                battery=self.swarm.controllers['cf3'].get_battery_percentage(),
                target_found=True
            )

            self.logger.info(f"Target broadcasted at {self.target_position}")

            # Transition
            self.current_state = MissionState.ROLE_ASSIGNMENT
            return True

        except Exception as e:
            self.logger.error(f"Target detection broadcast failed: {e}")
            return False

    async def _state_role_assignment(self) -> bool:
        """Assign Leader and Follower roles."""
        try:
            # Select leader (highest battery among neutrals)
            self.leader_id = self.swarm.select_leader()
            if self.leader_id is None:
                return False

            # Select follower (next highest battery)
            self.follower_id = self.swarm.select_follower(
                exclude_ids=[self.leader_id]
            )
            if self.follower_id is None:
                return False

            # Patrol drone keeps patrol role
            self.patrol_id = 'cf3'

            self.logger.info(
                f"Roles assigned - Leader: {self.leader_id}, "
                f"Follower: {self.follower_id}, Patrol: {self.patrol_id}"
            )

            # Transition
            self.current_state = MissionState.APPROACH_TARGET
            return True

        except Exception as e:
            self.logger.error(f"Role assignment failed: {e}")
            return False

    async def _state_approach_target(self) -> bool:
        """Leader and Follower approach target in formation."""
        try:
            self.logger.info("Approaching target in formation")

            # Coordinate formation movement
            formation_offset = self.mission_params.get('formation_offset', 0.5)
            if not self.swarm.coordinate_formation(
                self.leader_id,
                self.follower_id,
                formation_offset
            ):
                return False

            # Wait for movement
            await asyncio.sleep(3.0)

            self.logger.info("Formation in position")

            # Transition
            self.current_state = MissionState.JAMMING
            return True

        except Exception as e:
            self.logger.error(f"Approach failed: {e}")
            return False

    async def _state_jamming(self) -> bool:
        """Leader and Follower simulate jamming (hover in front)."""
        try:
            self.logger.info("Jamming target communications")

            # Hover in place for jamming duration
            await asyncio.sleep(2.0)

            self.logger.info("Jamming complete")

            # Transition
            self.current_state = MissionState.NEUTRALIZATION
            return True

        except Exception as e:
            self.logger.error(f"Jamming failed: {e}")
            return False

    async def _state_neutralization(self) -> bool:
        """Patrol executes neutralization (kamikaze from above)."""
        try:
            self.logger.info("Patrol executing neutralization")

            # Move patrol to position above target
            target_x, target_y, target_z = self.target_position
            approach_height = target_z + 2.0

            self.swarm.controllers[self.patrol_id].go_to(
                target_x, target_y, approach_height,
                duration=2.0
            )
            await asyncio.sleep(2.5)

            # Simulate kamikaze descent
            self.logger.info("Kamikaze descent!")
            self.swarm.controllers[self.patrol_id].go_to(
                target_x, target_y, target_z,
                duration=1.0
            )
            await asyncio.sleep(1.5)

            self.logger.info("Target neutralized!")

            # Transition
            self.current_state = MissionState.MISSION_COMPLETE
            return True

        except Exception as e:
            self.logger.error(f"Neutralization failed: {e}")
            return False

    async def _fly_pattern(
        self,
        drone_id: str,
        waypoints: list,
        duration_per_waypoint: float = 2.0
    ) -> None:
        """
        Fly drone through a pattern of waypoints.

        Args:
            drone_id: Drone identifier
            waypoints: List of (x, y, z) tuples
            duration_per_waypoint: Time to reach each waypoint
        """
        controller = self.swarm.controllers[drone_id]

        for x, y, z in waypoints:
            controller.go_to(x, y, z, duration=duration_per_waypoint)
            await asyncio.sleep(duration_per_waypoint + 0.5)

    def _check_timeout(self) -> bool:
        """
        Check if mission has exceeded maximum duration.

        Returns:
            True if timeout exceeded
        """
        if self.start_time is None:
            return False

        elapsed = time.time() - self.start_time
        max_duration = self.mission_params.get('max_duration', 180)

        return elapsed > max_duration


async def main():
    """Main entry point for Scenario 1 mission."""
    import yaml

    # Load configuration
    config_path = os.path.join(
        os.path.dirname(__file__),
        '../../config/scenario_1_config.yaml'
    )

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        # Use default config if file not found
        config = {
            'cage_dimensions': {'x': 10.0, 'y': 6.0, 'z': 8.0},
            'safety_zone': {'x_min': 0.0, 'x_max': 3.0, 'y_min': 0.0, 'y_max': 6.0},
            'target_position': {'x': 7.5, 'y': 3.0, 'z': 5.0},
            'mission_parameters': {
                'max_duration': 180,
                'min_battery': 20,
                'detection_range': 2.0,
                'formation_offset': 0.5
            }
        }

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run mission
    mission = Scenario1Mission(config)
    success = await mission.run_mission()

    if success:
        print("\n✓ Mission completed successfully!")
    else:
        print("\n✗ Mission failed!")

    return success


if __name__ == '__main__':
    asyncio.run(main())
