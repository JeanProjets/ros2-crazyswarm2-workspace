"""
Test suite for Scenario 4 Agent 4 implementation
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scenarios.scenario_4_fsm import (
    Scenario4FSM, MissionState, TargetStatus, Telemetry, MissionBrain
)
from scenarios.shadow_manager import OcclusionStrategy, ShadowHunter, ObstacleInfo
from scenarios.risk_manager import (
    AttackCorridorValidator, DynamicRiskManager, AttackClearance
)
from scenarios.swarm_splitter import (
    FormationManagerV4, FormationMode, SwarmCoordinator
)


# Mock obstacle map for testing
class MockObstacleMap:
    """Mock obstacle map for testing"""

    def __init__(self):
        # Simple obstacle at (5, 5) with 2x2 size
        self.obstacles = [
            {'center': np.array([5.0, 5.0]), 'size': np.array([2.0, 2.0])}
        ]

    def is_collision(self, x, y):
        """Check if point is in collision"""
        for obs in self.obstacles:
            half_size = obs['size'] / 2
            if (abs(x - obs['center'][0]) < half_size[0] and
                abs(y - obs['center'][1]) < half_size[1]):
                return True
        return False

    def get_distance_to_nearest_obstacle(self, x, y):
        """Get distance to nearest obstacle"""
        min_dist = float('inf')
        for obs in self.obstacles:
            dist = np.linalg.norm(np.array([x, y]) - obs['center'])
            # Approximate distance (not exact, but good enough for tests)
            dist = max(0, dist - np.linalg.norm(obs['size']) / 2)
            min_dist = min(min_dist, dist)
        return min_dist

    def check_line_of_sight(self, pos1, pos2):
        """Check if line of sight is clear"""
        direction = pos2 - pos1
        distance = np.linalg.norm(direction)
        if distance < 1e-6:
            return True

        direction_normalized = direction / distance
        num_samples = int(distance / 0.1) + 1

        for i in range(num_samples):
            sample_dist = (i / num_samples) * distance
            sample_pos = pos1 + direction_normalized * sample_dist
            if self.is_collision(sample_pos[0], sample_pos[1]):
                return False
        return True


# Mock handlers
class MockMapHandler:
    """Mock map handler (Agent 1)"""

    def __init__(self):
        self.map = MockObstacleMap()

    def get_map(self):
        return self.map

    def check_line_of_sight(self, pos1, pos2):
        return self.map.check_line_of_sight(pos1, pos2)

    def get_distance_to_nearest_obstacle(self, pos):
        return self.map.get_distance_to_nearest_obstacle(pos[0], pos[1])

    def plan_path(self, start, goal):
        # Simple straight line path
        return [start, goal]


class MockBehaviorHandler:
    """Mock behavior handler (Agent 2)"""

    def __init__(self):
        self.last_command = None

    def start_patrol(self):
        self.last_command = "patrol"

    def start_pure_pursuit(self, target_pos, target_vel):
        self.last_command = "pure_pursuit"

    def follow_path(self, path):
        self.last_command = "follow_path"

    def coast_to_emergence(self):
        self.last_command = "coast"

    def execute_scan(self, altitude):
        self.last_command = "scan"

    def execute_moving_strike(self, target_pos, target_vel):
        self.last_command = "strike"

    def execute(self, mode):
        self.last_command = f"execute_{mode.value}"


class MockVisionHandler:
    """Mock vision handler (Agent 3)"""

    def __init__(self):
        self.target_state = {
            'pos': np.array([3.0, 3.0, 1.0]),
            'vel': np.array([1.0, 0.0, 0.0]),
            'status': TargetStatus.VISIBLE,
            'confidence': 0.9
        }

    def get_target_state(self):
        return self.target_state


# Tests for Scenario4FSM
class TestScenario4FSM:
    """Test the hybrid state machine"""

    def test_initialization(self):
        """Test FSM initialization"""
        fsm = Scenario4FSM()
        assert fsm.current_state == MissionState.GLOBAL_SEARCH

    def test_pursuit_direct_transition(self):
        """Test transition to direct pursuit when LOS is clear"""
        map_handler = MockMapHandler()
        behavior_handler = MockBehaviorHandler()
        fsm = Scenario4FSM(map_handler, behavior_handler)

        telemetry = Telemetry(
            drone_pos=np.array([0.0, 0.0, 2.0]),
            drone_vel=np.array([0.0, 0.0, 0.0]),
            target_pos=np.array([3.0, 3.0, 1.0]),
            target_vel=np.array([1.0, 0.0, 0.0]),
            target_status=TargetStatus.VISIBLE,
            los_clear=True
        )

        state = fsm.update(telemetry, 0.0)
        assert state == MissionState.PURSUIT_DIRECT

    def test_pursuit_nav_transition(self):
        """Test transition to navigation pursuit when LOS is blocked"""
        map_handler = MockMapHandler()
        behavior_handler = MockBehaviorHandler()
        fsm = Scenario4FSM(map_handler, behavior_handler)

        telemetry = Telemetry(
            drone_pos=np.array([0.0, 0.0, 2.0]),
            drone_vel=np.array([0.0, 0.0, 0.0]),
            target_pos=np.array([10.0, 10.0, 1.0]),
            target_vel=np.array([1.0, 0.0, 0.0]),
            target_status=TargetStatus.VISIBLE,
            los_clear=False  # LOS blocked
        )

        state = fsm.update(telemetry, 0.0)
        assert state == MissionState.PURSUIT_NAV

    def test_predictive_coast_transition(self):
        """Test transition to predictive coast when target is occluded"""
        fsm = Scenario4FSM()

        telemetry = Telemetry(
            drone_pos=np.array([0.0, 0.0, 2.0]),
            drone_vel=np.array([0.0, 0.0, 0.0]),
            target_status=TargetStatus.OCCLUDED_PREDICTED,
            occlusion_timer=1.0  # Less than timeout
        )

        state = fsm.update(telemetry, 0.0)
        assert state == MissionState.PREDICTIVE_COAST

    def test_evaluate_los(self):
        """Test line of sight evaluation"""
        map_handler = MockMapHandler()
        fsm = Scenario4FSM(map_handler)

        # Clear LOS
        los_clear = fsm.evaluate_los(
            np.array([0.0, 0.0, 2.0]),
            np.array([3.0, 3.0, 1.0]),
            map_handler.get_map()
        )
        assert los_clear

        # Blocked LOS (through obstacle at 5,5)
        los_blocked = fsm.evaluate_los(
            np.array([4.0, 4.0, 2.0]),
            np.array([6.0, 6.0, 1.0]),
            map_handler.get_map()
        )
        assert not los_blocked


# Tests for Shadow Manager
class TestShadowManager:
    """Test the shadow hunter logic"""

    def test_occlusion_strategy_initialization(self):
        """Test occlusion strategy initialization"""
        strategy = OcclusionStrategy()
        assert strategy is not None

    def test_calculate_emergence_point(self):
        """Test emergence point calculation"""
        obstacle_map = MockObstacleMap()
        strategy = OcclusionStrategy(obstacle_map)

        # Target moving from inside obstacle outward
        last_pos = np.array([5.0, 5.0, 1.0])  # In obstacle
        velocity = np.array([2.0, 0.0, 0.0])  # Moving right

        emergence = strategy.calculate_emergence_point(last_pos, velocity, obstacle_map)

        # Should find emergence point to the right of obstacle
        assert emergence is not None
        assert emergence[0] > 6.0  # Past obstacle

    def test_shadow_hunter_tracking(self):
        """Test shadow hunter tracking logic"""
        strategy = OcclusionStrategy()
        hunter = ShadowHunter(strategy)

        drone_pos = np.array([0.0, 0.0, 2.0])

        # Visible target
        waypoint, status = hunter.update(
            drone_pos=drone_pos,
            target_visible=True,
            target_pos=np.array([3.0, 3.0, 1.0]),
            target_vel=np.array([1.0, 0.0, 0.0]),
            dt=0.1
        )

        assert waypoint is not None
        assert status == "TRACKING_VISIBLE"

    def test_shadow_hunter_coasting(self):
        """Test shadow hunter coasting when target lost"""
        strategy = OcclusionStrategy()
        hunter = ShadowHunter(strategy)

        drone_pos = np.array([0.0, 0.0, 2.0])

        # First, track visible target
        hunter.update(
            drone_pos=drone_pos,
            target_visible=True,
            target_pos=np.array([3.0, 3.0, 1.0]),
            target_vel=np.array([1.0, 0.0, 0.0]),
            dt=0.1
        )

        # Then lose target
        waypoint, status = hunter.update(
            drone_pos=drone_pos,
            target_visible=False,
            target_pos=None,
            target_vel=None,
            dt=0.1
        )

        assert waypoint is not None
        assert status in ["COASTING_PREDICTED", "INTERCEPTING_EMERGENCE"]


# Tests for Risk Manager
class TestRiskManager:
    """Test the dynamic risk manager"""

    def test_attack_corridor_validator_initialization(self):
        """Test validator initialization"""
        validator = AttackCorridorValidator()
        assert validator is not None

    def test_safe_attack_in_open_space(self):
        """Test that attack is safe in open space"""
        obstacle_map = MockObstacleMap()
        validator = AttackCorridorValidator(obstacle_map)

        # Target far from obstacles
        target_pos = np.array([0.0, 0.0, 2.0])
        is_safe = validator.is_attack_safe(target_pos, obstacle_map)

        assert is_safe

    def test_unsafe_attack_near_obstacle(self):
        """Test that attack is unsafe near obstacle"""
        obstacle_map = MockObstacleMap()
        validator = AttackCorridorValidator(obstacle_map)

        # Target close to obstacle at (5, 5)
        target_pos = np.array([5.2, 5.2, 2.0])
        assessment = validator.assess_attack_risk(target_pos, obstacle_map)

        assert assessment.clearance == AttackClearance.UNSAFE_NEAR_OBSTACLE

    def test_risk_manager_evaluation(self):
        """Test risk manager strike evaluation"""
        obstacle_map = MockObstacleMap()
        validator = AttackCorridorValidator(obstacle_map)
        risk_manager = DynamicRiskManager(validator)

        drone_pos = np.array([0.0, 0.0, 3.0])
        target_pos = np.array([1.0, 1.0, 2.0])
        target_vel = np.array([0.5, 0.0, 0.0])

        should_strike, reason = risk_manager.evaluate_strike_conditions(
            drone_pos, target_pos, target_vel, obstacle_map
        )

        # Should be safe in open space
        assert should_strike or "CLEAR" in reason or "SAFE" in reason


# Tests for Swarm Splitter
class TestSwarmSplitter:
    """Test the formation manager"""

    def test_formation_manager_initialization(self):
        """Test formation manager initialization"""
        manager = FormationManagerV4()
        assert manager.current_mode == FormationMode.TIGHT_FORMATION

    def test_formation_mode_switching(self):
        """Test formation mode switching based on environment"""
        obstacle_map = MockObstacleMap()
        manager = FormationManagerV4(obstacle_map)

        # In open space - should use combat spread or tight formation
        leader_pos = np.array([0.0, 0.0, 2.0])
        leader_vel = np.array([1.0, 0.0, 0.0])

        mode = manager.update_formation_mode(leader_pos, leader_vel, obstacle_map)
        assert mode in [FormationMode.TIGHT_FORMATION, FormationMode.COMBAT_SPREAD]

    def test_follower_goal_calculation(self):
        """Test follower goal position calculation"""
        manager = FormationManagerV4()

        leader_pos = np.array([5.0, 5.0, 2.0])
        leader_vel = np.array([1.0, 0.0, 0.0])

        goal = manager.calculate_follower_goal(leader_pos, leader_vel)

        assert goal is not None
        assert len(goal) == 3

        # Goal should be behind/beside leader
        # (exact position depends on formation mode)

    def test_formation_integrity_check(self):
        """Test formation integrity checking"""
        obstacle_map = MockObstacleMap()
        manager = FormationManagerV4(obstacle_map)

        leader_pos = np.array([0.0, 0.0, 2.0])
        follower_pos = np.array([1.0, 0.5, 2.0])

        # Should be intact in open space
        intact = manager.check_formation_integrity(
            leader_pos, follower_pos, obstacle_map
        )
        assert intact

    def test_swarm_coordinator(self):
        """Test swarm coordinator"""
        obstacle_map = MockObstacleMap()
        manager = FormationManagerV4(obstacle_map)
        coordinator = SwarmCoordinator(manager)

        drone_states = [
            {'pos': np.array([0.0, 0.0, 2.0]), 'vel': np.array([1.0, 0.0, 0.0])},
            {'pos': np.array([1.0, 0.5, 2.0]), 'vel': np.array([1.0, 0.0, 0.0])}
        ]

        goals = coordinator.update_swarm(drone_states, obstacle_map)

        assert 1 in goals  # Follower should have a goal
        assert len(goals[1]) == 3


# Integration test
class TestMissionBrain:
    """Test the mission brain integration"""

    def test_mission_brain_initialization(self):
        """Test mission brain initialization"""
        map_handler = MockMapHandler()
        behavior_handler = MockBehaviorHandler()
        vision_handler = MockVisionHandler()
        fsm = Scenario4FSM(map_handler, behavior_handler, vision_handler)

        brain = MissionBrain(fsm, vision_handler, map_handler, behavior_handler)

        assert brain is not None

    def test_mission_brain_update(self):
        """Test mission brain update loop"""
        map_handler = MockMapHandler()
        behavior_handler = MockBehaviorHandler()
        vision_handler = MockVisionHandler()
        fsm = Scenario4FSM(map_handler, behavior_handler, vision_handler)

        brain = MissionBrain(fsm, vision_handler, map_handler, behavior_handler)

        drone_state = {
            'pos': np.array([0.0, 0.0, 2.0]),
            'vel': np.array([0.0, 0.0, 0.0])
        }

        mode = brain.update(0.1, drone_state)

        assert mode is not None
        assert isinstance(mode, MissionState)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
