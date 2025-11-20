#!/usr/bin/env python

"""
Hackathon Scenario 1: Multi-Drone Search and Neutralize Mission

Four drones coordinate to locate and neutralize a stationary target:
- cf1 (N_1): Neutral drone 1
- cf2 (N_2): Neutral drone 2
- cf3 (P): Patrol drone
- cf4 (C): Target drone (stationary)

Mission phases:
1. Initialization: All drones takeoff to 4m altitude
2. Patrol: N_1, N_2 check safety zone, P scans area
3. Detection: Proximity-based target detection
4. Role Assignment: Highest battery neutral becomes Leader
5. Engagement: Leader + Follower position, jam, and neutralize target
"""

from crazyflie_py import Crazyswarm
import numpy as np
import rclpy
from rclpy.node import Node
from crazyflie_interfaces.msg import DroneStatus
from geometry_msgs.msg import Point
from std_msgs.msg import Header
import time
from enum import Enum


class DroneRole(Enum):
    """Enum for drone roles in the mission"""
    NEUTRAL = "NEUTRAL"
    PATROL = "PATROL"
    TARGET = "TARGET"
    LEADER = "LEADER"
    FOLLOWER = "FOLLOWER"


class MissionPhase(Enum):
    """Enum for mission phases"""
    INITIALIZATION = "INITIALIZATION"
    PATROL = "PATROL"
    DETECTION = "DETECTION"
    ROLE_ASSIGNMENT = "ROLE_ASSIGNMENT"
    ENGAGEMENT = "ENGAGEMENT"
    COMPLETE = "COMPLETE"


class DroneState:
    """Tracks the state of an individual drone"""

    def __init__(self, cf, name, role, initial_position):
        self.cf = cf
        self.name = name
        self.role = role
        self.initial_position = np.array(initial_position)
        self.current_position = np.array(initial_position)
        self.battery_level = 4.0  # Simulated battery voltage (V)
        self.target_detected = False
        self.target_position = None

    def update_position(self, position):
        """Update current position"""
        self.current_position = np.array(position)

    def simulate_battery_drain(self, time_delta):
        """Simulate battery drain over time (0.01V per second)"""
        self.battery_level -= 0.01 * time_delta
        self.battery_level = max(3.5, self.battery_level)  # Minimum voltage

    def detect_target(self, target_position, threshold=1.5):
        """Check if target is within detection range"""
        distance = np.linalg.norm(self.current_position - target_position)
        return distance < threshold


class ScenarioCoordinator(Node):
    """ROS2 node that coordinates the hackathon scenario"""

    def __init__(self, swarm):
        super().__init__('hackathon_scenario_coordinator')

        self.swarm = swarm
        self.timeHelper = swarm.timeHelper
        self.allcfs = swarm.allcfs

        # Mission parameters from scenario spec
        self.CAGE_SIZE = np.array([10.0, 6.0, 8.0])  # X, Y, Z in meters
        self.SAFETY_ZONE_SIZE = 3.0  # meters
        self.FLIGHT_HEIGHT = 4.0  # meters
        self.DETECTION_THRESHOLD = 1.5  # meters

        # Scenario positions from spec
        self.POSITIONS = {
            'N_1_start': np.array([2.5, 2.5, 0.0]),
            'N_2_start': np.array([2.5, 3.5, 0.0]),
            'P_start': np.array([3.0, 5.0, 0.0]),
            'C_position': np.array([7.5, 3.0, 5.0]),  # Target at 5m height
        }

        # Initialize drone states
        self.drones = {}
        self._initialize_drones()

        # Mission state
        self.current_phase = MissionPhase.INITIALIZATION
        self.target_found = False
        self.target_position = self.POSITIONS['C_position']

        # ROS2 publishers for each drone
        self.status_publishers = {}
        for name in self.drones.keys():
            topic = f'/drone_status/{name}'
            self.status_publishers[name] = self.create_publisher(DroneStatus, topic, 10)

        # Mission phase publisher
        self.phase_pub = self.create_publisher(DroneStatus, '/mission_coordinator/phase', 10)

        # Timer for publishing drone status
        self.status_timer = self.create_timer(0.1, self.publish_status)  # 10 Hz

        self.get_logger().info('Hackathon Scenario Coordinator initialized')

    def _initialize_drones(self):
        """Initialize drone state objects"""
        # Assuming drones are in order: cf1, cf2, cf3, cf4
        drone_configs = [
            ('cf1', DroneRole.NEUTRAL, self.POSITIONS['N_1_start']),
            ('cf2', DroneRole.NEUTRAL, self.POSITIONS['N_2_start']),
            ('cf3', DroneRole.PATROL, self.POSITIONS['P_start']),
            ('cf4', DroneRole.TARGET, self.POSITIONS['C_position']),
        ]

        for idx, (name, role, pos) in enumerate(drone_configs):
            if idx < len(self.allcfs.crazyflies):
                cf = self.allcfs.crazyflies[idx]
                self.drones[name] = DroneState(cf, name, role, pos)
                # Add small random variation to battery levels
                self.drones[name].battery_level += np.random.uniform(-0.1, 0.1)
            else:
                self.get_logger().warn(f'Not enough drones configured for {name}')

    def publish_status(self):
        """Publish status for all drones at 10Hz"""
        current_time = self.get_clock().now().to_msg()

        for name, drone in self.drones.items():
            msg = DroneStatus()
            msg.header = Header()
            msg.header.stamp = current_time
            msg.header.frame_id = 'world'

            msg.drone_name = name
            msg.position = Point(
                x=drone.current_position[0],
                y=drone.current_position[1],
                z=drone.current_position[2]
            )
            msg.battery_level = drone.battery_level
            msg.target_detected = drone.target_detected

            if drone.target_position is not None:
                msg.target_position = Point(
                    x=drone.target_position[0],
                    y=drone.target_position[1],
                    z=drone.target_position[2]
                )
            else:
                msg.target_position = Point(x=0.0, y=0.0, z=0.0)

            msg.role = drone.role.value

            self.status_publishers[name].publish(msg)

    def run_mission(self):
        """Execute the complete mission sequence"""
        self.get_logger().info('=' * 60)
        self.get_logger().info('STARTING HACKATHON SCENARIO 1')
        self.get_logger().info('=' * 60)

        try:
            # Phase 1: Initialization
            self.phase_initialization()

            # Phase 2: Patrol
            self.phase_patrol()

            # Phase 3: Detection & Role Assignment
            self.phase_detection_and_assignment()

            # Phase 4: Engagement
            self.phase_engagement()

            # Phase 5: Mission Complete
            self.phase_complete()

        except Exception as e:
            self.get_logger().error(f'Mission failed: {str(e)}')
            self.emergency_land()

    def phase_initialization(self):
        """Phase 1: Initialize drones and takeoff to flight height"""
        self.current_phase = MissionPhase.INITIALIZATION
        self.get_logger().info('PHASE 1: INITIALIZATION (10s)')

        start_time = time.time()

        # All drones except target takeoff to 4m
        for name in ['cf1', 'cf2', 'cf3']:
            if name in self.drones:
                drone = self.drones[name]
                drone.cf.takeoff(targetHeight=self.FLIGHT_HEIGHT, duration=3.0)

        # Target drone goes to its position at 5m
        if 'cf4' in self.drones:
            target = self.drones['cf4']
            target.cf.takeoff(targetHeight=self.POSITIONS['C_position'][2], duration=3.0)

        self.timeHelper.sleep(4.0)

        # Update drone positions after takeoff
        self.drones['cf1'].update_position([2.5, 2.5, 4.0])
        self.drones['cf2'].update_position([2.5, 3.5, 4.0])
        self.drones['cf3'].update_position([3.0, 5.0, 4.0])
        self.drones['cf4'].update_position(self.POSITIONS['C_position'])

        elapsed = time.time() - start_time
        self.get_logger().info(f'Initialization complete in {elapsed:.1f}s')
        self.timeHelper.sleep(max(0, 10 - elapsed))

    def phase_patrol(self):
        """Phase 2: Patrol phase - search for target"""
        self.current_phase = MissionPhase.PATROL
        self.get_logger().info('PHASE 2: PATROL (max 2 min)')

        start_time = time.time()

        # N_1 and N_2 check safety zone
        self.get_logger().info('N_1 and N_2: Checking safety zone...')

        # N_1 patrol pattern in safety zone
        n1_waypoints = [
            [1.5, 1.5, 4.0],
            [2.5, 1.5, 4.0],
            [2.5, 4.5, 4.0],
            [1.5, 4.5, 4.0],
            [2.5, 2.5, 4.0],  # Return to start
        ]

        # N_2 patrol pattern in safety zone (offset)
        n2_waypoints = [
            [1.5, 2.0, 4.0],
            [2.5, 2.0, 4.0],
            [2.5, 5.0, 4.0],
            [1.5, 5.0, 4.0],
            [2.5, 3.5, 4.0],  # Return to start
        ]

        # P patrol pattern - scan rest of area
        self.get_logger().info('P: Scanning area...')
        p_waypoints = [
            [4.0, 5.0, 4.0],
            [6.0, 5.0, 4.0],
            [8.0, 5.0, 4.0],  # Moving along Y=5
            [8.0, 3.0, 4.0],  # P will detect target near here!
        ]

        # Execute patrol movements
        # Move N_1 and N_2 through safety zone
        for i, (wp1, wp2) in enumerate(zip(n1_waypoints[:2], n2_waypoints[:2])):
            self.drones['cf1'].cf.goTo(wp1, 0, 2.0)
            self.drones['cf2'].cf.goTo(wp2, 0, 2.0)
            self.timeHelper.sleep(2.5)

            self.drones['cf1'].update_position(wp1)
            self.drones['cf2'].update_position(wp2)
            self.drones['cf1'].simulate_battery_drain(2.5)
            self.drones['cf2'].simulate_battery_drain(2.5)

        # P moves through patrol waypoints until target detected
        for i, waypoint in enumerate(p_waypoints):
            self.get_logger().info(f'P moving to waypoint {i+1}/{len(p_waypoints)}: {waypoint}')
            self.drones['cf3'].cf.goTo(waypoint, 0, 3.0)
            self.timeHelper.sleep(3.5)

            # Update position and battery
            self.drones['cf3'].update_position(waypoint)
            self.drones['cf3'].simulate_battery_drain(3.5)

            # Check for target detection
            if self.drones['cf3'].detect_target(self.target_position, self.DETECTION_THRESHOLD):
                self.get_logger().info('!' * 60)
                self.get_logger().info('TARGET DETECTED BY PATROL DRONE!')
                self.get_logger().info('!' * 60)
                self.drones['cf3'].target_detected = True
                self.drones['cf3'].target_position = self.target_position
                self.target_found = True
                break

        elapsed = time.time() - start_time
        self.get_logger().info(f'Patrol phase complete in {elapsed:.1f}s')

    def phase_detection_and_assignment(self):
        """Phase 3: Target detected, assign Leader and Follower roles"""
        self.current_phase = MissionPhase.ROLE_ASSIGNMENT
        self.get_logger().info('PHASE 3: DETECTION & ROLE ASSIGNMENT (7s)')

        start_time = time.time()

        if not self.target_found:
            self.get_logger().warn('Target not detected during patrol!')
            return

        # P broadcasts target position (simulated via shared state)
        self.get_logger().info(f'P broadcasting target position: {self.target_position}')
        self.timeHelper.sleep(1.0)

        # Update N_1 and N_2 with target info
        self.drones['cf1'].target_position = self.target_position
        self.drones['cf2'].target_position = self.target_position
        self.drones['cf1'].target_detected = True
        self.drones['cf2'].target_detected = True

        # Determine Leader and Follower based on battery level
        n1_battery = self.drones['cf1'].battery_level
        n2_battery = self.drones['cf2'].battery_level

        self.get_logger().info(f'N_1 battery: {n1_battery:.2f}V')
        self.get_logger().info(f'N_2 battery: {n2_battery:.2f}V')

        if n1_battery > n2_battery:
            leader_name = 'cf1'
            follower_name = 'cf2'
        else:
            leader_name = 'cf2'
            follower_name = 'cf1'

        self.drones[leader_name].role = DroneRole.LEADER
        self.drones[follower_name].role = DroneRole.FOLLOWER

        self.get_logger().info(f'LEADER: {leader_name} (Battery: {self.drones[leader_name].battery_level:.2f}V)')
        self.get_logger().info(f'FOLLOWER: {follower_name} (Battery: {self.drones[follower_name].battery_level:.2f}V)')

        elapsed = time.time() - start_time
        self.timeHelper.sleep(max(0, 7 - elapsed))

    def phase_engagement(self):
        """Phase 4: Leader and Follower engage target, P performs kamikaze"""
        self.current_phase = MissionPhase.ENGAGEMENT
        self.get_logger().info('PHASE 4: ENGAGEMENT (55s)')

        start_time = time.time()

        # Find Leader and Follower
        leader = None
        follower = None
        for drone in self.drones.values():
            if drone.role == DroneRole.LEADER:
                leader = drone
            elif drone.role == DroneRole.FOLLOWER:
                follower = drone

        if not leader or not follower:
            self.get_logger().error('Leader or Follower not assigned!')
            return

        # Calculate engagement positions
        # Leader and Follower position in front of target
        target_pos = self.target_position
        leader_engage_pos = target_pos + np.array([-1.5, 0, 0])  # 1.5m in front (lower X)
        follower_offset = np.array([-0.5, -0.5, -0.5])  # Offset from leader
        follower_engage_pos = leader_engage_pos + follower_offset

        self.get_logger().info('Leader and Follower moving to engagement positions...')
        self.get_logger().info(f'Leader target: {leader_engage_pos}')
        self.get_logger().info(f'Follower target: {follower_engage_pos}')

        # Move to engagement positions
        leader.cf.goTo(leader_engage_pos, 0, 5.0)
        follower.cf.goTo(follower_engage_pos, 0, 5.0)
        self.timeHelper.sleep(5.5)

        leader.update_position(leader_engage_pos)
        follower.update_position(follower_engage_pos)
        leader.simulate_battery_drain(5.5)
        follower.simulate_battery_drain(5.5)

        # Jamming phase
        self.get_logger().info('=' * 60)
        self.get_logger().info('JAMMING TARGET COMMUNICATIONS (20s)...')
        self.get_logger().info('=' * 60)
        self.timeHelper.sleep(20.0)

        leader.simulate_battery_drain(20.0)
        follower.simulate_battery_drain(20.0)
        self.drones['cf3'].simulate_battery_drain(20.0)

        # Kamikaze attack - P moves above target
        self.get_logger().info('=' * 60)
        self.get_logger().info('PATROL DRONE: INITIATING KAMIKAZE ATTACK!')
        self.get_logger().info('=' * 60)

        kamikaze_pos = target_pos + np.array([0, 0, 0.5])  # 0.5m above target
        self.drones['cf3'].cf.goTo(kamikaze_pos, 0, 5.0)
        self.timeHelper.sleep(5.5)

        self.drones['cf3'].update_position(kamikaze_pos)
        self.drones['cf3'].simulate_battery_drain(5.5)

        self.get_logger().info('*' * 60)
        self.get_logger().info('BOOM! TARGET NEUTRALIZED!')
        self.get_logger().info('*' * 60)

        elapsed = time.time() - start_time
        self.get_logger().info(f'Engagement phase complete in {elapsed:.1f}s')

    def phase_complete(self):
        """Phase 5: Mission complete, land all drones"""
        self.current_phase = MissionPhase.COMPLETE
        self.get_logger().info('=' * 60)
        self.get_logger().info('MISSION COMPLETE - LANDING ALL DRONES')
        self.get_logger().info('=' * 60)

        # Land all drones
        self.allcfs.land(targetHeight=0.06, duration=3.0)
        self.timeHelper.sleep(4.0)

        # Print mission summary
        self.print_mission_summary()

    def print_mission_summary(self):
        """Print summary of mission results"""
        self.get_logger().info('')
        self.get_logger().info('=' * 60)
        self.get_logger().info('MISSION SUMMARY')
        self.get_logger().info('=' * 60)

        for name, drone in self.drones.items():
            self.get_logger().info(f'{name}:')
            self.get_logger().info(f'  Role: {drone.role.value}')
            self.get_logger().info(f'  Final Battery: {drone.battery_level:.2f}V')
            self.get_logger().info(f'  Target Detected: {drone.target_detected}')

        self.get_logger().info('=' * 60)
        self.get_logger().info('HACKATHON SCENARIO 1: SUCCESS')
        self.get_logger().info('=' * 60)

    def emergency_land(self):
        """Emergency landing for all drones"""
        self.get_logger().error('EMERGENCY LANDING!')
        try:
            self.allcfs.land(targetHeight=0.06, duration=2.0)
            self.timeHelper.sleep(3.0)
        except Exception as e:
            self.get_logger().error(f'Emergency landing failed: {str(e)}')


def main():
    """Main entry point for hackathon scenario 1"""
    # Initialize Crazyswarm (this also initializes ROS2)
    swarm = Crazyswarm()

    # Create scenario coordinator
    coordinator = ScenarioCoordinator(swarm)

    # Run the mission
    coordinator.run_mission()

    # Cleanup
    coordinator.destroy_node()


if __name__ == '__main__':
    main()
