"""
Unit tests for SwarmCoordinator class.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from core.swarm_coordinator import SwarmCoordinator, DroneRole, DroneStatus


class TestSwarmCoordinator:
    """Test suite for SwarmCoordinator."""

    @pytest.fixture
    def coordinator(self):
        """Create a SwarmCoordinator instance for testing."""
        return SwarmCoordinator(
            drone_ids=['cf1', 'cf2', 'cf3'],
            cage_bounds=(10.0, 6.0, 8.0)
        )

    def test_initialization(self, coordinator):
        """Test coordinator initialization."""
        assert coordinator.drone_ids == ['cf1', 'cf2', 'cf3']
        assert len(coordinator.controllers) == 3
        assert 'cf1' in coordinator.controllers
        assert 'cf2' in coordinator.controllers
        assert 'cf3' in coordinator.controllers

    def test_initialize_swarm(self, coordinator):
        """Test swarm initialization with roles."""
        result = coordinator.initialize_swarm()
        assert result is True
        assert len(coordinator.drone_statuses) == 3
        assert coordinator.roles['cf1'] == DroneRole.NEUTRAL_1
        assert coordinator.roles['cf2'] == DroneRole.NEUTRAL_2
        assert coordinator.roles['cf3'] == DroneRole.PATROL

    def test_assign_initial_positions(self, coordinator):
        """Test assigning initial positions."""
        coordinator.initialize_swarm()

        positions = {
            'cf1': (2.5, 2.5, 0, 4.0),
            'cf2': (2.5, 3.5, 0, 4.0),
            'cf3': (3.0, 5.0, 0, 4.0),
        }

        result = coordinator.assign_initial_positions(positions)
        assert result is True

    def test_broadcast_drone_status(self, coordinator):
        """Test broadcasting drone status."""
        coordinator.initialize_swarm()

        coordinator.broadcast_drone_status(
            drone_id='cf1',
            position=(5.0, 3.0, 4.0),
            battery=85.0,
            target_found=False
        )

        status = coordinator.get_drone_status('cf1')
        assert status.position == (5.0, 3.0, 4.0)
        assert status.battery == 85.0
        assert status.target_found is False

    def test_select_leader(self, coordinator):
        """Test leader selection based on battery."""
        coordinator.initialize_swarm()

        # Set different battery levels
        coordinator.drone_statuses['cf1'].battery = 90.0
        coordinator.drone_statuses['cf2'].battery = 85.0
        coordinator.drone_statuses['cf3'].battery = 80.0

        leader_id = coordinator.select_leader()
        assert leader_id == 'cf1'  # Highest battery
        assert coordinator.roles[leader_id] == DroneRole.LEADER

    def test_select_follower(self, coordinator):
        """Test follower selection."""
        coordinator.initialize_swarm()

        # Set battery levels
        coordinator.drone_statuses['cf1'].battery = 90.0
        coordinator.drone_statuses['cf2'].battery = 85.0
        coordinator.drone_statuses['cf3'].battery = 80.0

        # Select leader first
        leader_id = coordinator.select_leader()

        # Select follower
        follower_id = coordinator.select_follower(exclude_ids=[leader_id])
        assert follower_id in ['cf2', 'cf3']
        assert follower_id != leader_id
        assert coordinator.roles[follower_id] == DroneRole.FOLLOWER

    def test_coordinate_formation(self, coordinator):
        """Test formation coordination."""
        coordinator.initialize_swarm()
        coordinator.target_position = (7.5, 3.0, 5.0)

        # Assign roles
        coordinator.roles['cf1'] = DroneRole.LEADER
        coordinator.roles['cf2'] = DroneRole.FOLLOWER

        result = coordinator.coordinate_formation('cf1', 'cf2', formation_offset=0.5)
        assert result is True

    def test_get_all_statuses(self, coordinator):
        """Test getting all drone statuses."""
        coordinator.initialize_swarm()

        statuses = coordinator.get_all_statuses()
        assert len(statuses) == 3
        assert all(isinstance(s, DroneStatus) for s in statuses.values())

    def test_emergency_land_all(self, coordinator):
        """Test emergency landing all drones."""
        coordinator.initialize_swarm()

        # Should not raise exception
        coordinator.emergency_land_all()
