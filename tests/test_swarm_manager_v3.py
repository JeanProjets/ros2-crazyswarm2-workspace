"""
Tests for swarm_manager_v3 module
"""

import pytest
import numpy as np
from src.core.swarm_manager_v3 import (
    SwarmCoordinator,
    SwarmRole,
    DroneState,
    FormationOffset,
    FormationPresets
)


class TestDroneState:
    """Test DroneState dataclass"""

    def test_initialization(self):
        """Test drone state initialization"""
        state = DroneState(
            drone_id="drone1",
            position=[1.0, 2.0, 3.0],
            velocity=[0.1, 0.2, 0.3],
            role=SwarmRole.LEADER,
            timestamp=100.0
        )

        assert state.drone_id == "drone1"
        assert isinstance(state.position, np.ndarray)
        assert isinstance(state.velocity, np.ndarray)
        assert state.role == SwarmRole.LEADER
        assert state.timestamp == 100.0


class TestFormationOffset:
    """Test FormationOffset dataclass"""

    def test_initialization(self):
        """Test formation offset initialization"""
        offset = FormationOffset(
            offset=[1.0, 0.0, 0.0],
            damping=0.7
        )

        assert isinstance(offset.offset, np.ndarray)
        assert np.allclose(offset.offset, [1.0, 0.0, 0.0])
        assert offset.damping == 0.7

    def test_default_damping(self):
        """Test default damping value"""
        offset = FormationOffset(offset=[0.0, 1.0, 0.0])
        assert offset.damping == 0.8


class TestSwarmCoordinator:
    """Test SwarmCoordinator class"""

    def test_initialization(self):
        """Test coordinator initialization"""
        coordinator = SwarmCoordinator(
            max_velocity=2.0,
            safety_distance=0.5,
            formation_tolerance=0.2
        )

        assert coordinator.max_velocity == 2.0
        assert coordinator.safety_distance == 0.5
        assert coordinator.formation_tolerance == 0.2
        assert len(coordinator.drones) == 0
        assert coordinator.leader_id is None

    def test_register_drone(self):
        """Test drone registration"""
        coordinator = SwarmCoordinator()
        coordinator.register_drone(
            "drone1",
            initial_position=(1.0, 2.0, 3.0),
            role=SwarmRole.LEADER
        )

        assert "drone1" in coordinator.drones
        assert coordinator.drones["drone1"].role == SwarmRole.LEADER
        assert np.allclose(coordinator.drones["drone1"].position, [1.0, 2.0, 3.0])

    def test_set_formation_offset(self):
        """Test setting formation offset"""
        coordinator = SwarmCoordinator()
        coordinator.set_formation_offset(
            "follower1",
            offset=(1.0, 0.0, 0.0),
            damping=0.7
        )

        assert "follower1" in coordinator.formations
        assert np.allclose(coordinator.formations["follower1"].offset, [1.0, 0.0, 0.0])
        assert coordinator.formations["follower1"].damping == 0.7

    def test_assign_leader(self):
        """Test leader assignment"""
        coordinator = SwarmCoordinator()
        coordinator.register_drone("drone1", (0, 0, 1))
        coordinator.assign_leader("drone1")

        assert coordinator.leader_id == "drone1"
        assert coordinator.drones["drone1"].role == SwarmRole.LEADER

    def test_assign_follower(self):
        """Test follower assignment"""
        coordinator = SwarmCoordinator()
        coordinator.register_drone("drone2", (0, 0, 1))
        coordinator.assign_follower("drone2")

        assert "drone2" in coordinator.follower_ids
        assert coordinator.drones["drone2"].role == SwarmRole.FOLLOWER

    def test_update_drone_state(self):
        """Test drone state update"""
        coordinator = SwarmCoordinator()
        coordinator.register_drone("drone1", (0, 0, 1))

        # Update position
        coordinator.update_drone_state("drone1", (1, 1, 1.5), (0.1, 0.1, 0.05))

        drone = coordinator.drones["drone1"]
        assert np.allclose(drone.position, [1, 1, 1.5])
        assert np.allclose(drone.velocity, [0.1, 0.1, 0.05])

    def test_coordinate_moving_formation_basic(self):
        """Test basic moving formation coordination"""
        coordinator = SwarmCoordinator()

        # Register leader and follower
        coordinator.register_drone("leader", (2, 2, 2))
        coordinator.register_drone("follower", (0, 0, 2))

        coordinator.assign_leader("leader")
        coordinator.assign_follower("follower")

        # Set formation offset
        coordinator.set_formation_offset("follower", offset=(1.0, 0.0, 0.0))

        # Update leader with velocity
        coordinator.update_drone_state("leader", (2, 2, 2), (0.5, 0.0, 0.0))

        # Coordinate formation
        result = coordinator.coordinate_moving_formation("leader", "follower")

        assert result is not None
        target_pos, target_vel = result

        # Target position should be leader + offset
        expected_pos = np.array([2, 2, 2]) + np.array([1, 0, 0])
        assert np.allclose(target_pos, expected_pos)

        # Target velocity should include leader velocity (with damping)
        assert target_vel[0] > 0  # Should have forward component

    def test_coordinate_moving_formation_velocity_feedforward(self):
        """Test velocity feedforward in formation"""
        coordinator = SwarmCoordinator()

        # Setup
        coordinator.register_drone("leader", (5, 5, 2))
        coordinator.register_drone("follower", (4, 5, 2))
        coordinator.assign_leader("leader")
        coordinator.assign_follower("follower")
        coordinator.set_formation_offset("follower", offset=(1, 0, 0), damping=0.8)

        # Leader moving forward
        coordinator.update_drone_state("leader", (5, 5, 2), (1.0, 0.0, 0.0))

        result = coordinator.coordinate_moving_formation("leader", "follower")
        assert result is not None
        _, target_vel = result

        # Follower should have forward velocity (feedforward)
        assert target_vel[0] > 0

    def test_coordinate_moving_formation_velocity_limit(self):
        """Test that formation coordination respects velocity limits"""
        coordinator = SwarmCoordinator(max_velocity=1.0)

        # Setup
        coordinator.register_drone("leader", (10, 10, 2))
        coordinator.register_drone("follower", (0, 0, 2))  # Far away
        coordinator.assign_leader("leader")
        coordinator.assign_follower("follower")
        coordinator.set_formation_offset("follower", offset=(1, 0, 0))

        # Leader moving fast
        coordinator.update_drone_state("leader", (10, 10, 2), (2.0, 2.0, 0.0))

        result = coordinator.coordinate_moving_formation("leader", "follower")
        assert result is not None
        _, target_vel = result

        # Should not exceed max velocity
        assert np.linalg.norm(target_vel) <= coordinator.max_velocity + 0.01

    def test_coordinate_moving_formation_invalid_drones(self):
        """Test formation coordination with invalid drones"""
        coordinator = SwarmCoordinator()

        result = coordinator.coordinate_moving_formation("invalid1", "invalid2")
        assert result is None

    def test_check_collision_risk(self):
        """Test collision risk checking"""
        coordinator = SwarmCoordinator(safety_distance=1.0)

        # Register two drones
        coordinator.register_drone("drone1", (0, 0, 1))
        coordinator.register_drone("drone2", (0.5, 0, 1))  # 0.5m away

        at_risk, distance = coordinator.check_collision_risk("drone1", "drone2")

        assert at_risk == True  # Within safety distance
        assert abs(distance - 0.5) < 0.01

    def test_check_collision_risk_safe(self):
        """Test collision risk when drones are safe"""
        coordinator = SwarmCoordinator(safety_distance=0.5)

        coordinator.register_drone("drone1", (0, 0, 1))
        coordinator.register_drone("drone2", (2, 0, 1))  # 2m away

        at_risk, distance = coordinator.check_collision_risk("drone1", "drone2")

        assert at_risk == False
        assert abs(distance - 2.0) < 0.01

    def test_smooth_role_transition(self):
        """Test smooth role transition"""
        coordinator = SwarmCoordinator()

        # Register two drones
        coordinator.register_drone("drone1", (0, 0, 1))
        coordinator.register_drone("drone2", (1, 1, 1))

        # Set drone1 as leader
        coordinator.assign_leader("drone1")
        coordinator.update_drone_state("drone1", (0, 0, 1), (0.5, 0.0, 0.0))

        # Transition to drone2
        success = coordinator.smooth_role_transition("drone1", "drone2")

        assert success is True
        assert coordinator.leader_id == "drone2"
        assert coordinator.drones["drone1"].role == SwarmRole.FOLLOWER

    def test_get_formation_status(self):
        """Test formation status retrieval"""
        coordinator = SwarmCoordinator()

        # Setup formation
        coordinator.register_drone("leader", (2, 2, 2))
        coordinator.register_drone("follower", (3, 2, 2))
        coordinator.assign_leader("leader")
        coordinator.assign_follower("follower")
        coordinator.set_formation_offset("follower", (1, 0, 0))

        coordinator.update_drone_state("leader", (2, 2, 2), (0.5, 0, 0))
        coordinator.update_drone_state("follower", (3, 2, 2), (0.5, 0, 0))

        status = coordinator.get_formation_status()

        assert status['leader_id'] == "leader"
        assert status['num_followers'] == 1
        assert 'leader_position' in status
        assert 'follower' in status['drones']

    def test_emergency_stop(self):
        """Test emergency stop"""
        coordinator = SwarmCoordinator()

        # Register drones with velocities
        coordinator.register_drone("drone1", (0, 0, 1))
        coordinator.update_drone_state("drone1", (1, 1, 1), (1.0, 1.0, 0))

        # Emergency stop
        coordinator.emergency_stop()

        # All velocities should be zero
        for drone in coordinator.drones.values():
            assert np.allclose(drone.velocity, [0, 0, 0])


class TestFormationPresets:
    """Test FormationPresets class"""

    def test_line_formation(self):
        """Test line formation preset"""
        formation = FormationPresets.line_formation(spacing=1.5)

        assert 'N1' in formation
        assert 'N2' in formation
        assert len(formation) == 2

        # Check positions are on a line
        assert formation['N1'][0] == 0.0
        assert formation['N2'][0] == 0.0

    def test_triangle_formation(self):
        """Test triangle formation preset"""
        formation = FormationPresets.triangle_formation(spacing=2.0)

        assert 'N1' in formation
        assert 'N2' in formation
        assert len(formation) == 2

        # Check symmetry
        assert formation['N1'][0] == formation['N2'][0]  # Same x
        assert formation['N1'][1] == -formation['N2'][1]  # Opposite y

    def test_jamming_formation(self):
        """Test jamming formation preset"""
        formation = FormationPresets.jamming_formation(radius=1.0)

        assert 'N1' in formation
        assert 'N2' in formation
        assert 'P' in formation
        assert len(formation) == 3

        # Verify all positions are approximately same distance from center
        for pos in formation.values():
            distance = np.linalg.norm(pos[:2])  # Only x, y
            assert abs(distance - 1.0) < 0.01


class TestIntegration:
    """Integration tests for swarm coordination"""

    def test_complete_formation_flight(self):
        """Test complete formation flight scenario"""
        coordinator = SwarmCoordinator(max_velocity=2.0)

        # Register 3 drones
        coordinator.register_drone("leader", (0, 0, 1.5))
        coordinator.register_drone("follower1", (0, 1, 1.5))
        coordinator.register_drone("follower2", (0, -1, 1.5))

        # Setup formation
        coordinator.assign_leader("leader")
        coordinator.assign_follower("follower1")
        coordinator.assign_follower("follower2")

        coordinator.set_formation_offset("follower1", (1, 0.5, 0))
        coordinator.set_formation_offset("follower2", (1, -0.5, 0))

        # Simulate leader moving
        positions = [
            (0, 0, 1.5),
            (1, 0, 1.5),
            (2, 0, 1.5),
            (3, 1, 1.5),
        ]

        for pos in positions:
            coordinator.update_drone_state("leader", pos, (0.5, 0.2, 0))

            # Coordinate followers
            result1 = coordinator.coordinate_moving_formation("leader", "follower1")
            result2 = coordinator.coordinate_moving_formation("leader", "follower2")

            assert result1 is not None
            assert result2 is not None

            target_pos1, target_vel1 = result1
            target_pos2, target_vel2 = result2

            # Verify reasonable commands
            assert np.linalg.norm(target_vel1) <= coordinator.max_velocity + 0.01
            assert np.linalg.norm(target_vel2) <= coordinator.max_velocity + 0.01

    def test_moving_formation_no_collision(self):
        """Test that moving formation avoids collisions during braking"""
        coordinator = SwarmCoordinator(max_velocity=2.0, safety_distance=0.5)

        # Setup - follower in formation with leader
        coordinator.register_drone("leader", (5, 5, 2))
        coordinator.register_drone("follower", (6, 5, 2))  # Already in formation
        coordinator.assign_leader("leader")
        coordinator.assign_follower("follower")
        coordinator.set_formation_offset("follower", (1, 0, 0), damping=0.8)

        # Update follower position to be in formation
        coordinator.update_drone_state("follower", (6, 5, 2), (0.8, 0, 0))

        # Leader moving
        coordinator.update_drone_state("leader", (5, 5, 2), (1.0, 0, 0))
        result1 = coordinator.coordinate_moving_formation("leader", "follower")

        # Leader stops
        coordinator.update_drone_state("leader", (5.1, 5, 2), (0.0, 0, 0))
        result2 = coordinator.coordinate_moving_formation("leader", "follower")

        assert result1 is not None
        assert result2 is not None

        _, vel1 = result1
        _, vel2 = result2

        # Follower should decelerate too (via feedforward damping)
        # vel1 should have forward component from leader velocity
        # vel2 should have less forward component since leader stopped
        assert vel1[0] > vel2[0]  # Less forward velocity when leader stops


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
