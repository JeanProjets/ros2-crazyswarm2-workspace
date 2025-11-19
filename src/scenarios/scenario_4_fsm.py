"""
Scenario 4 Hybrid State Machine
This module implements the hierarchical state machine for handling obstacles and loss of sight.
"""

import numpy as np
from enum import Enum
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


class MissionState(Enum):
    """Mission states for Scenario 4"""
    GLOBAL_SEARCH = "GLOBAL_SEARCH"
    PURSUIT_DIRECT = "PURSUIT_DIRECT"
    PURSUIT_NAV = "PURSUIT_NAV"
    PREDICTIVE_COAST = "PREDICTIVE_COAST"
    REACQUISITION_SCAN = "REACQUISITION_SCAN"
    MOVING_STRIKE_V4 = "MOVING_STRIKE_V4"


class TargetStatus(Enum):
    """Target visibility status"""
    VISIBLE = "VISIBLE"
    OCCLUDED_PREDICTED = "OCCLUDED_PREDICTED"
    LOST = "LOST"


@dataclass
class Telemetry:
    """Telemetry data from sensors"""
    drone_pos: np.ndarray
    drone_vel: np.ndarray
    target_pos: Optional[np.ndarray] = None
    target_vel: Optional[np.ndarray] = None
    target_status: TargetStatus = TargetStatus.LOST
    target_confidence: float = 0.0
    los_clear: bool = False
    dist_to_obstacle: float = float('inf')
    jamming_timer: float = 0.0
    occlusion_timer: float = 0.0
    stable_track_time: float = 0.0


class Scenario4FSM:
    """
    Hybrid State Machine for Scenario 4
    Handles obstacles and loss of sight with intelligent mode switching
    """

    def __init__(self, map_handler=None, behavior_handler=None, vision_handler=None):
        """
        Initialize the FSM

        Args:
            map_handler: Agent 1 - Map and pathfinding
            behavior_handler: Agent 2 - Behavior execution
            vision_handler: Agent 3 - Vision and tracking
        """
        self.current_state = MissionState.GLOBAL_SEARCH
        self.map_handler = map_handler
        self.behavior_handler = behavior_handler
        self.vision_handler = vision_handler

        # Timing parameters
        self.occlusion_timeout = 2.0  # seconds
        self.jamming_required_time = 20.0  # seconds
        self.stable_track_threshold = 3.0  # seconds

        # State history
        self.state_entry_time = 0.0
        self.last_update_time = 0.0

    def update(self, telemetry: Telemetry, current_time: float) -> MissionState:
        """
        Main update loop - evaluates conditions and transitions states

        Args:
            telemetry: Current sensor data
            current_time: Current timestamp

        Returns:
            New mission state
        """
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # Evaluate state transition
        new_state = self._evaluate_state_transition(telemetry, current_time, dt)

        # Handle state entry
        if new_state != self.current_state:
            self._on_state_entry(new_state, telemetry, current_time)
            self.current_state = new_state
            self.state_entry_time = current_time

        return self.current_state

    def _evaluate_state_transition(self, telemetry: Telemetry,
                                   current_time: float, dt: float) -> MissionState:
        """
        Evaluate state transitions based on current conditions

        Args:
            telemetry: Current sensor data
            current_time: Current timestamp
            dt: Delta time since last update

        Returns:
            Next mission state
        """
        # Priority 1: Check for MOVING_STRIKE_V4 conditions
        if self._check_strike_conditions(telemetry):
            return MissionState.MOVING_STRIKE_V4

        # Priority 2: Handle target visibility states
        if telemetry.target_status == TargetStatus.VISIBLE:
            # Target is visible - decide between direct pursuit and navigation
            if telemetry.los_clear:
                return MissionState.PURSUIT_DIRECT
            else:
                # LOS blocked by obstacle - use pathfinding
                return MissionState.PURSUIT_NAV

        elif telemetry.target_status == TargetStatus.OCCLUDED_PREDICTED:
            # Target is predicted but not visible - coast to emergence point
            if telemetry.occlusion_timer > self.occlusion_timeout:
                # Coasting timed out - start reacquisition scan
                return MissionState.REACQUISITION_SCAN
            else:
                return MissionState.PREDICTIVE_COAST

        else:  # TargetStatus.LOST
            # No target - continue search
            return MissionState.GLOBAL_SEARCH

    def _check_strike_conditions(self, telemetry: Telemetry) -> bool:
        """
        Check if conditions are met for moving strike

        Args:
            telemetry: Current sensor data

        Returns:
            True if strike should be initiated
        """
        # Conditions for strike:
        # 1. Target visible in open space (not near obstacles)
        # 2. Stable track for required time
        # 3. Jamming timer complete
        # 4. LOS clear

        if telemetry.target_status != TargetStatus.VISIBLE:
            return False

        if not telemetry.los_clear:
            return False

        # Check if target is in open space (not near obstacles)
        if telemetry.dist_to_obstacle < 0.5:
            return False

        # Check timing requirements
        if telemetry.stable_track_time < self.stable_track_threshold:
            return False

        if telemetry.jamming_timer < self.jamming_required_time:
            return False

        return True

    def _on_state_entry(self, new_state: MissionState,
                       telemetry: Telemetry, current_time: float):
        """
        Handle state entry actions

        Args:
            new_state: State being entered
            telemetry: Current sensor data
            current_time: Current timestamp
        """
        print(f"[FSM] State transition: {self.current_state.value} -> {new_state.value}")

        # State-specific entry actions
        if new_state == MissionState.GLOBAL_SEARCH:
            self._enter_global_search(telemetry)
        elif new_state == MissionState.PURSUIT_DIRECT:
            self._enter_pursuit_direct(telemetry)
        elif new_state == MissionState.PURSUIT_NAV:
            self._enter_pursuit_nav(telemetry)
        elif new_state == MissionState.PREDICTIVE_COAST:
            self._enter_predictive_coast(telemetry)
        elif new_state == MissionState.REACQUISITION_SCAN:
            self._enter_reacquisition_scan(telemetry)
        elif new_state == MissionState.MOVING_STRIKE_V4:
            self._enter_moving_strike(telemetry)

    def _enter_global_search(self, telemetry: Telemetry):
        """Enter global search patrol mode"""
        if self.behavior_handler:
            self.behavior_handler.start_patrol()

    def _enter_pursuit_direct(self, telemetry: Telemetry):
        """Enter direct high-speed pursuit"""
        if self.behavior_handler:
            self.behavior_handler.start_pure_pursuit(
                target_pos=telemetry.target_pos,
                target_vel=telemetry.target_vel
            )

    def _enter_pursuit_nav(self, telemetry: Telemetry):
        """Enter obstacle-aware navigation pursuit"""
        if self.map_handler and self.behavior_handler:
            # Request path from Agent 1
            path = self.map_handler.plan_path(
                start=telemetry.drone_pos,
                goal=telemetry.target_pos
            )
            # Command Agent 2 to follow path
            self.behavior_handler.follow_path(path)

    def _enter_predictive_coast(self, telemetry: Telemetry):
        """Enter predictive coast to emergence point"""
        if self.behavior_handler:
            # Calculate emergence point (handled by shadow_manager)
            self.behavior_handler.coast_to_emergence()

    def _enter_reacquisition_scan(self, telemetry: Telemetry):
        """Enter reacquisition scan mode"""
        if self.behavior_handler:
            # Rise to higher altitude and scan
            self.behavior_handler.execute_scan(altitude=5.0)

    def _enter_moving_strike(self, telemetry: Telemetry):
        """Enter moving strike mode"""
        if self.behavior_handler:
            self.behavior_handler.execute_moving_strike(
                target_pos=telemetry.target_pos,
                target_vel=telemetry.target_vel
            )

    def evaluate_los(self, drone_pos: np.ndarray, target_pos: np.ndarray,
                    obstacle_map=None) -> bool:
        """
        Evaluate Line-of-Sight between drone and target

        Args:
            drone_pos: Drone position [x, y, z]
            target_pos: Target position [x, y, z]
            obstacle_map: Obstacle map for collision checking

        Returns:
            True if LOS is clear
        """
        if obstacle_map is None and self.map_handler:
            obstacle_map = self.map_handler.get_map()

        if obstacle_map is None:
            # No map available - assume clear
            return True

        # Perform raycast from drone to target
        direction = target_pos - drone_pos
        distance = np.linalg.norm(direction)

        if distance < 1e-6:
            return True

        direction_normalized = direction / distance

        # Sample along ray
        num_samples = int(distance / 0.1) + 1  # 10cm resolution
        for i in range(num_samples):
            sample_dist = (i / num_samples) * distance
            sample_pos = drone_pos + direction_normalized * sample_dist

            # Check if sample point is in obstacle
            if obstacle_map.is_collision(sample_pos[0], sample_pos[1]):
                return False

        return True

    def select_pursuit_mode(self, telemetry: Telemetry) -> MissionState:
        """
        Select pursuit mode based on telemetry

        Args:
            telemetry: Current sensor data

        Returns:
            Recommended pursuit mode
        """
        if telemetry.target_status == TargetStatus.VISIBLE:
            if telemetry.los_clear:
                return MissionState.PURSUIT_DIRECT
            else:
                return MissionState.PURSUIT_NAV
        elif telemetry.target_status == TargetStatus.OCCLUDED_PREDICTED:
            if telemetry.occlusion_timer < self.occlusion_timeout:
                return MissionState.PREDICTIVE_COAST
            else:
                return MissionState.REACQUISITION_SCAN
        else:
            return MissionState.GLOBAL_SEARCH

    def get_state(self) -> MissionState:
        """Get current mission state"""
        return self.current_state

    def get_state_duration(self, current_time: float) -> float:
        """Get time spent in current state"""
        return current_time - self.state_entry_time


class MissionBrain:
    """
    High-level mission coordinator integrating all agents
    Runs at 10Hz
    """

    def __init__(self, fsm: Scenario4FSM, vision_agent, map_agent, behavior_agent):
        """
        Initialize mission brain

        Args:
            fsm: Finite state machine
            vision_agent: Agent 3 - Vision handler
            map_agent: Agent 1 - Map handler
            behavior_agent: Agent 2 - Behavior handler
        """
        self.fsm = fsm
        self.vision = vision_agent
        self.map = map_agent
        self.behaviors = behavior_agent

        self.update_rate = 10.0  # Hz
        self.jamming_timer = 0.0
        self.stable_track_timer = 0.0
        self.occlusion_timer = 0.0

    def update(self, current_time: float, drone_state: Dict[str, Any]) -> MissionState:
        """
        Main brain update loop

        Args:
            current_time: Current timestamp
            drone_state: Current drone state dict with pos, vel, etc.

        Returns:
            Current mission state
        """
        # 1. Get Vision Data (Agent 3)
        target = self.vision.get_target_state() if self.vision else None

        # 2. Get drone state
        drone_pos = drone_state.get('pos', np.zeros(3))
        drone_vel = drone_state.get('vel', np.zeros(3))

        # 3. Check LOS (Agent 1 Map)
        los_clear = False
        dist_to_obstacle = float('inf')

        if target and target.get('pos') is not None and self.map:
            target_pos = target['pos']
            los_clear = self.map.check_line_of_sight(drone_pos, target_pos)

            # Check distance to nearest obstacle at target location
            dist_to_obstacle = self.map.get_distance_to_nearest_obstacle(target_pos)

        # 4. Update timers based on target status
        target_status = target.get('status', TargetStatus.LOST) if target else TargetStatus.LOST

        if target_status == TargetStatus.VISIBLE and los_clear:
            self.stable_track_timer += 1.0 / self.update_rate
            self.occlusion_timer = 0.0

            # Only increment jamming timer if in open space
            if dist_to_obstacle >= 0.5:
                self.jamming_timer += 1.0 / self.update_rate
            # else: pause jamming timer near obstacles
        elif target_status == TargetStatus.OCCLUDED_PREDICTED:
            self.occlusion_timer += 1.0 / self.update_rate
            self.stable_track_timer = 0.0
        else:  # LOST
            self.stable_track_timer = 0.0
            self.occlusion_timer = 0.0
            self.jamming_timer = 0.0

        # 5. Create telemetry object
        telemetry = Telemetry(
            drone_pos=drone_pos,
            drone_vel=drone_vel,
            target_pos=target.get('pos') if target else None,
            target_vel=target.get('vel') if target else None,
            target_status=target_status,
            target_confidence=target.get('confidence', 0.0) if target else 0.0,
            los_clear=los_clear,
            dist_to_obstacle=dist_to_obstacle,
            jamming_timer=self.jamming_timer,
            occlusion_timer=self.occlusion_timer,
            stable_track_time=self.stable_track_timer
        )

        # 6. Update FSM
        mode = self.fsm.update(telemetry, current_time)

        # 7. Execute behaviors
        if self.behaviors:
            self.behaviors.execute(mode)

        return mode
