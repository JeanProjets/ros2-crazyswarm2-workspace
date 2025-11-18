# COMPREHENSIVE IMPLEMENTATION ARCHITECTURE PLAN - PART 2
## Drone Coordination Hackathon - Challenge 2

**Document Version**: 1.0
**Continued from**: IMPLEMENTATION_ARCHITECTURE_PLAN.md

---

## 7. DETAILED IMPLEMENTATION PLAN - STEP BY STEP (Continued)

### PHASE 2: CORE ALGORITHMS (Continued)

#### Step 2.5: Collision Avoidance (2 days)
**Agent**: Control Systems Agent

**File**: `src/squadrone_swarm/squadrone_swarm/core/collision.py`

**Implementation**:
```python
import numpy as np
from typing import List, Tuple

class CollisionAvoidance:
    """
    Velocity obstacle-based collision avoidance for multi-drone systems.

    Uses reciprocal velocity obstacles (RVO) to ensure inter-drone safety.
    """

    def __init__(self, safety_radius=0.8, prediction_horizon=1.0):
        """
        Args:
            safety_radius: Minimum allowed distance between drones (meters)
            prediction_horizon: Time to predict collisions (seconds)
        """
        self.safety_radius = safety_radius
        self.prediction_horizon = prediction_horizon

    def check_collision_predicted(self, drone1_state, drone2_state):
        """
        Check if collision is predicted within horizon.

        Args:
            drone1_state, drone2_state: DroneStatus objects

        Returns:
            (will_collide: bool, time_to_collision: float)
        """
        # Current positions
        p1 = np.array([drone1_state.pose.position.x,
                      drone1_state.pose.position.y,
                      drone1_state.pose.position.z])
        p2 = np.array([drone2_state.pose.position.x,
                      drone2_state.pose.position.y,
                      drone2_state.pose.position.z])

        # Current velocities
        v1 = np.array([drone1_state.velocity.linear.x,
                      drone1_state.velocity.linear.y,
                      drone1_state.velocity.linear.z])
        v2 = np.array([drone2_state.velocity.linear.x,
                      drone2_state.velocity.linear.y,
                      drone2_state.velocity.linear.z])

        # Relative position and velocity
        p_rel = p1 - p2
        v_rel = v1 - v2

        # Check current distance
        current_dist = np.linalg.norm(p_rel)
        if current_dist < self.safety_radius:
            return True, 0.0  # Already in collision

        # Predict future distance
        # d(t) = ||p_rel + v_rel * t||
        # Minimize d(t) to find closest approach
        a = np.dot(v_rel, v_rel)
        b = 2 * np.dot(p_rel, v_rel)
        c = np.dot(p_rel, p_rel) - self.safety_radius**2

        if a < 1e-6:  # Nearly stationary relative velocity
            return False, float('inf')

        discriminant = b**2 - 4*a*c
        if discriminant < 0:  # No collision
            return False, float('inf')

        # Time to collision
        t1 = (-b - np.sqrt(discriminant)) / (2*a)
        t2 = (-b + np.sqrt(discriminant)) / (2*a)

        if t1 > 0 and t1 < self.prediction_horizon:
            return True, t1
        elif t2 > 0 and t2 < self.prediction_horizon:
            return True, t2
        else:
            return False, float('inf')

    def compute_avoidance_velocity(self, ego_state, other_states):
        """
        Compute avoidance velocity using Reciprocal Velocity Obstacles (RVO).

        Args:
            ego_state: Current drone's state
            other_states: List of other drones' states

        Returns:
            avoidance_velocity: np.array([vx, vy, vz])
        """
        ego_pos = np.array([ego_state.pose.position.x,
                           ego_state.pose.position.y,
                           ego_state.pose.position.z])
        ego_vel = np.array([ego_state.velocity.linear.x,
                           ego_state.velocity.linear.y,
                           ego_state.velocity.linear.z])

        avoidance_vel = np.zeros(3)

        for other_state in other_states:
            will_collide, ttc = self.check_collision_predicted(ego_state, other_state)

            if will_collide:
                # Compute avoidance vector
                other_pos = np.array([other_state.pose.position.x,
                                     other_state.pose.position.y,
                                     other_state.pose.position.z])

                # Vector from other to ego
                to_ego = ego_pos - other_pos
                to_ego_normalized = to_ego / np.linalg.norm(to_ego)

                # Avoidance strength inversely proportional to time-to-collision
                strength = 1.0 / max(ttc, 0.1)

                # Perpendicular avoidance
                perpendicular = np.cross(to_ego_normalized, np.array([0, 0, 1]))
                perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-6)

                avoidance_vel += perpendicular * strength

        # Limit avoidance velocity
        max_avoidance_speed = 0.5  # m/s
        avoidance_speed = np.linalg.norm(avoidance_vel)
        if avoidance_speed > max_avoidance_speed:
            avoidance_vel = avoidance_vel / avoidance_speed * max_avoidance_speed

        return avoidance_vel

    def get_safe_waypoint(self, current_pos, target_pos, other_positions, safety_margin=1.0):
        """
        Find a safe intermediate waypoint that avoids other drones.

        Args:
            current_pos: Current position
            target_pos: Desired target position
            other_positions: List of other drone positions
            safety_margin: Additional safety distance

        Returns:
            safe_waypoint: np.array([x, y, z])
        """
        # Direct path
        direct = target_pos - current_pos
        direct_normalized = direct / np.linalg.norm(direct)

        # Check if direct path is safe
        is_safe = True
        for other_pos in other_positions:
            # Distance from other drone to line
            point_to_line_dist = self._point_to_line_distance(
                other_pos, current_pos, target_pos
            )
            if point_to_line_dist < self.safety_radius + safety_margin:
                is_safe = False
                break

        if is_safe:
            return target_pos

        # Find avoidance waypoint
        # Try perpendicular offsets
        perpendicular = np.cross(direct_normalized, np.array([0, 0, 1]))
        perpendicular = perpendicular / np.linalg.norm(perpendicular)

        # Try offset in positive direction
        offset_distance = self.safety_radius + safety_margin
        waypoint_option1 = current_pos + direct * 0.5 + perpendicular * offset_distance

        # Check if safe
        for other_pos in other_positions:
            if np.linalg.norm(waypoint_option1 - other_pos) < self.safety_radius + safety_margin:
                # Try negative direction
                waypoint_option1 = current_pos + direct * 0.5 - perpendicular * offset_distance
                break

        return waypoint_option1

    def _point_to_line_distance(self, point, line_start, line_end):
        """Calculate shortest distance from point to line segment."""
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.linalg.norm(line_vec)

        if line_len < 1e-6:
            return np.linalg.norm(point_vec)

        line_unitvec = line_vec / line_len
        point_vec_scaled = point_vec / line_len
        t = np.dot(line_unitvec, point_vec_scaled)
        t = max(0, min(1, t))

        nearest = line_start + t * line_vec
        return np.linalg.norm(point - nearest)
```

**Unit Tests**: `tests/unit/test_collision.py`
```python
def test_collision_detection_static():
    """Test collision detection for stationary drones."""
    ca = CollisionAvoidance(safety_radius=0.8)

    # Two drones too close (0.5m apart)
    drone1 = create_mock_drone_state(pos=[0, 0, 1], vel=[0, 0, 0])
    drone2 = create_mock_drone_state(pos=[0.5, 0, 1], vel=[0, 0, 0])

    will_collide, ttc = ca.check_collision_predicted(drone1, drone2)
    assert will_collide == True
    assert ttc == 0.0  # Already colliding

def test_collision_detection_approaching():
    """Test collision prediction for approaching drones."""
    ca = CollisionAvoidance(safety_radius=0.8, prediction_horizon=2.0)

    # Two drones approaching each other
    drone1 = create_mock_drone_state(pos=[0, 0, 1], vel=[1, 0, 0])
    drone2 = create_mock_drone_state(pos=[2, 0, 1], vel=[-1, 0, 0])

    will_collide, ttc = ca.check_collision_predicted(drone1, drone2)
    assert will_collide == True
    assert 0 < ttc < 2.0  # Should collide within prediction horizon

def test_avoidance_waypoint():
    """Test safe waypoint generation."""
    ca = CollisionAvoidance(safety_radius=0.8)

    current = np.array([0, 0, 1])
    target = np.array([5, 0, 1])
    other_positions = [np.array([2.5, 0, 1])]  # Blocking direct path

    safe_wp = ca.get_safe_waypoint(current, target, other_positions)

    # Safe waypoint should not be on direct line
    assert safe_wp[1] != 0  # Should have Y offset
    # Should still be between current and target
    assert 0 < safe_wp[0] < 5
```

**Acceptance Criteria**:
- Detects collisions accurately (100% in unit tests)
- Avoidance trajectories are smooth
- No false positives for diverging drones
- Performance: <5ms per check

---

#### Step 2.6: Kalman Filter (2 days)
**Agent**: Estimation Agent

**File**: `src/squadrone_swarm/squadrone_swarm/core/kalman_filter.py`

**Implementation**:
```python
import numpy as np

class KalmanFilter3D:
    """
    3D Kalman Filter for target tracking.

    State: [x, y, z, vx, vy, vz]
    Measurement: [x, y, z]
    """

    def __init__(self, dt=0.01, process_noise=0.1, measurement_noise=0.05):
        """
        Args:
            dt: Time step (seconds)
            process_noise: Process noise standard deviation
            measurement_noise: Measurement noise standard deviation
        """
        self.dt = dt
        self.state_dim = 6  # [x, y, z, vx, vy, vz]
        self.measurement_dim = 3  # [x, y, z]

        # State vector
        self.x = np.zeros((self.state_dim, 1))  # [x, y, z, vx, vy, vz]

        # State covariance matrix
        self.P = np.eye(self.state_dim) * 1.0

        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1]
        ])

        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])

        # Process noise covariance
        q = process_noise**2
        self.Q = np.array([
            [q*dt**4/4, 0, 0, q*dt**3/2, 0, 0],
            [0, q*dt**4/4, 0, 0, q*dt**3/2, 0],
            [0, 0, q*dt**4/4, 0, 0, q*dt**3/2],
            [q*dt**3/2, 0, 0, q*dt**2, 0, 0],
            [0, q*dt**3/2, 0, 0, q*dt**2, 0],
            [0, 0, q*dt**3/2, 0, 0, q*dt**2]
        ])

        # Measurement noise covariance
        r = measurement_noise**2
        self.R = np.eye(self.measurement_dim) * r

        self.initialized = False

    def initialize(self, measurement):
        """
        Initialize filter with first measurement.

        Args:
            measurement: np.array([x, y, z])
        """
        self.x[0:3] = measurement.reshape((3, 1))
        self.x[3:6] = 0  # Zero velocity initially
        self.initialized = True

    def predict(self):
        """
        Prediction step.

        Returns:
            predicted_state: np.array([x, y, z, vx, vy, vz])
        """
        # State prediction
        self.x = self.F @ self.x

        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x.flatten()

    def update(self, measurement):
        """
        Update step with new measurement.

        Args:
            measurement: np.array([x, y, z])

        Returns:
            updated_state: np.array([x, y, z, vx, vy, vz])
        """
        if not self.initialized:
            self.initialize(measurement)
            return self.x.flatten()

        z = measurement.reshape((3, 1))

        # Innovation
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P

        return self.x.flatten()

    def get_position(self):
        """Get current position estimate."""
        return self.x[0:3].flatten()

    def get_velocity(self):
        """Get current velocity estimate."""
        return self.x[3:6].flatten()

    def predict_position(self, time_ahead):
        """
        Predict position at future time.

        Args:
            time_ahead: Seconds into future

        Returns:
            predicted_position: np.array([x, y, z])
        """
        predicted_pos = self.x[0:3].flatten() + self.x[3:6].flatten() * time_ahead
        return predicted_pos
```

**Unit Tests**: `tests/unit/test_kalman.py`
```python
def test_kalman_initialization():
    kf = KalmanFilter3D()
    measurement = np.array([1.0, 2.0, 3.0])
    kf.initialize(measurement)

    assert kf.initialized == True
    np.testing.assert_array_almost_equal(kf.get_position(), measurement)
    np.testing.assert_array_almost_equal(kf.get_velocity(), np.zeros(3))

def test_kalman_constant_position():
    """Test filter with stationary target."""
    kf = KalmanFilter3D(dt=0.1)
    true_pos = np.array([5.0, 3.0, 2.0])

    # Initialize
    kf.update(true_pos + np.random.normal(0, 0.05, 3))

    # Feed multiple measurements
    for _ in range(20):
        noisy_measurement = true_pos + np.random.normal(0, 0.05, 3)
        kf.predict()
        kf.update(noisy_measurement)

    # Estimate should converge to true position
    estimated_pos = kf.get_position()
    error = np.linalg.norm(estimated_pos - true_pos)
    assert error < 0.1  # Within 10cm

def test_kalman_constant_velocity():
    """Test filter with constant velocity target."""
    kf = KalmanFilter3D(dt=0.1)
    true_pos = np.array([0.0, 0.0, 2.0])
    true_vel = np.array([1.0, 0.5, 0.0])

    # Simulate moving target
    for i in range(30):
        true_pos += true_vel * 0.1
        noisy_measurement = true_pos + np.random.normal(0, 0.05, 3)

        kf.predict()
        kf.update(noisy_measurement)

    # Velocity estimate should be close to true
    estimated_vel = kf.get_velocity()
    vel_error = np.linalg.norm(estimated_vel - true_vel)
    assert vel_error < 0.2  # Within 0.2 m/s

def test_kalman_prediction():
    """Test future position prediction."""
    kf = KalmanFilter3D(dt=0.1)
    kf.initialize(np.array([0, 0, 2]))
    kf.x[3:6] = np.array([[1.0], [0.0], [0.0]])  # Set velocity to 1 m/s in X

    predicted_pos = kf.predict_position(time_ahead=2.0)  # 2 seconds ahead

    expected_x = 0 + 1.0 * 2.0  # Should be at x=2
    assert abs(predicted_pos[0] - expected_x) < 0.01
```

**Acceptance Criteria**:
- Converges to true state within 10 iterations
- Velocity estimation error < 0.2 m/s
- Position prediction accurate for 2s horizon
- Handles noisy measurements robustly

---

#### Step 2.7: Trajectory Generation (2 days)
**Agent**: Control Systems Agent

**File**: `src/squadrone_swarm/squadrone_swarm/core/trajectory.py`

**Implementation**:
```python
import numpy as np
from scipy.interpolate import interp1d

class TrajectoryGenerator:
    """
    Generate smooth, dynamically-feasible trajectories.

    Uses trapezoidal velocity profiles for smooth acceleration/deceleration.
    """

    def __init__(self, max_velocity=1.5, max_acceleration=3.0):
        """
        Args:
            max_velocity: Maximum velocity (m/s)
            max_acceleration: Maximum acceleration (m/s^2)
        """
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration

    def generate_trajectory(self, start_pos, end_pos, current_vel=None):
        """
        Generate trapezoidal velocity profile trajectory.

        Args:
            start_pos: Starting position np.array([x, y, z])
            end_pos: Goal position np.array([x, y, z])
            current_vel: Current velocity (optional)

        Returns:
            trajectory: dict with 'times', 'positions', 'velocities', 'accelerations'
        """
        # Distance and direction
        displacement = end_pos - start_pos
        distance = np.linalg.norm(displacement)
        direction = displacement / (distance + 1e-9)

        if distance < 0.01:  # Already at goal
            return {
                'times': np.array([0]),
                'positions': np.array([start_pos]),
                'velocities': np.array([np.zeros(3)]),
                'accelerations': np.array([np.zeros(3)])
            }

        # Calculate trajectory timing
        # Time to accelerate to max velocity
        t_accel = self.max_velocity / self.max_acceleration

        # Distance covered during acceleration and deceleration
        d_accel = 0.5 * self.max_acceleration * t_accel**2
        d_total_accel_decel = 2 * d_accel

        if d_total_accel_decel >= distance:
            # Triangular profile (doesn't reach max velocity)
            v_max_actual = np.sqrt(self.max_acceleration * distance)
            t_accel = v_max_actual / self.max_acceleration
            t_cruise = 0
            t_decel = t_accel
        else:
            # Trapezoidal profile
            v_max_actual = self.max_velocity
            d_cruise = distance - d_total_accel_decel
            t_cruise = d_cruise / self.max_velocity
            t_decel = t_accel

        total_time = t_accel + t_cruise + t_decel

        # Generate time samples
        dt = 0.05  # 20 Hz
        times = np.arange(0, total_time + dt, dt)
        positions = []
        velocities = []
        accelerations = []

        for t in times:
            if t <= t_accel:
                # Acceleration phase
                s = 0.5 * self.max_acceleration * t**2
                v = self.max_acceleration * t
                a = self.max_acceleration
            elif t <= t_accel + t_cruise:
                # Cruise phase
                s = d_accel + v_max_actual * (t - t_accel)
                v = v_max_actual
                a = 0
            else:
                # Deceleration phase
                t_from_decel_start = t - t_accel - t_cruise
                s = d_accel + v_max_actual * t_cruise + \
                    v_max_actual * t_from_decel_start - \
                    0.5 * self.max_acceleration * t_from_decel_start**2
                v = v_max_actual - self.max_acceleration * t_from_decel_start
                a = -self.max_acceleration

            # Convert scalar distance to 3D
            pos = start_pos + direction * s
            vel = direction * v
            acc = direction * a

            positions.append(pos)
            velocities.append(vel)
            accelerations.append(acc)

        return {
            'times': times,
            'positions': np.array(positions),
            'velocities': np.array(velocities),
            'accelerations': np.array(accelerations),
            'total_time': total_time
        }

    def interpolate_trajectory(self, trajectory, query_time):
        """
        Get position, velocity, acceleration at specific time.

        Args:
            trajectory: Output from generate_trajectory()
            query_time: Time to query (seconds)

        Returns:
            (position, velocity, acceleration)
        """
        times = trajectory['times']
        positions = trajectory['positions']
        velocities = trajectory['velocities']
        accelerations = trajectory['accelerations']

        if query_time <= times[0]:
            return positions[0], velocities[0], accelerations[0]
        if query_time >= times[-1]:
            return positions[-1], velocities[-1], accelerations[-1]

        # Linear interpolation
        pos_interp = interp1d(times, positions, axis=0, kind='linear')
        vel_interp = interp1d(times, velocities, axis=0, kind='linear')
        acc_interp = interp1d(times, accelerations, axis=0, kind='linear')

        return pos_interp(query_time), vel_interp(query_time), acc_interp(query_time)

    def generate_waypoint_trajectory(self, waypoints, current_vel=None):
        """
        Generate trajectory through multiple waypoints.

        Args:
            waypoints: List of np.array([x, y, z])
            current_vel: Initial velocity

        Returns:
            Combined trajectory dict
        """
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints")

        all_times = []
        all_positions = []
        all_velocities = []
        all_accelerations = []

        cumulative_time = 0

        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]

            segment = self.generate_trajectory(start, end, current_vel)

            # Append to combined trajectory
            all_times.extend(segment['times'] + cumulative_time)
            all_positions.extend(segment['positions'])
            all_velocities.extend(segment['velocities'])
            all_accelerations.extend(segment['accelerations'])

            cumulative_time += segment['total_time']
            current_vel = segment['velocities'][-1]  # End velocity of this segment

        return {
            'times': np.array(all_times),
            'positions': np.array(all_positions),
            'velocities': np.array(all_velocities),
            'accelerations': np.array(all_accelerations),
            'total_time': cumulative_time
        }
```

**Unit Tests**: `tests/unit/test_trajectory.py`

**Acceptance Criteria**:
- Respects velocity/acceleration limits
- Smooth trajectories (no discontinuities)
- Arrival at goal within 1cm tolerance
- Performance: <10ms for 10m trajectory

---

### PHASE 3: ROS 2 NODES (Week 5-6) - PRIORITY P0

#### Step 3.1: OptiTrack Interface Node (2 days)
**Agent**: ROS Integration Agent

**File**: `src/squadrone_swarm/squadrone_swarm/nodes/optitrack_node.py`

**Implementation**:
```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
from squadrone_msgs.msg import DroneStatus
from squadrone_swarm.interfaces.optitrack import OptiTrackClient
from squadrone_swarm.utils.logger import ROSLogger
import numpy as np

class OptiTrackNode(Node):
    """
    ROS 2 node to interface with OptiTrack system.

    Publishes:
        /cf_positions/{drone_id}: PoseStamped at 120 Hz
        /cf_velocities/{drone_id}: TwistStamped at 120 Hz
    """

    def __init__(self):
        super().__init__('optitrack_node')

        # Parameters
        self.declare_parameter('optitrack_server_ip', '192.168.1.100')
        self.declare_parameter('rigid_body_ids', ['cf_p', 'cf_n1', 'cf_n2', 'cf_c'])
        self.declare_parameter('publish_rate', 120.0)

        server_ip = self.get_parameter('optitrack_server_ip').value
        self.drone_ids = self.get_parameter('rigid_body_ids').value
        self.publish_rate = self.get_parameter('publish_rate').value

        # Initialize OptiTrack client
        self.optitrack_client = OptiTrackClient(server_ip)
        self.logger = ROSLogger(self, "OptiTrack")

        # Publishers (one per drone)
        self.pose_publishers = {}
        self.velocity_publishers = {}

        for drone_id in self.drone_ids:
            self.pose_publishers[drone_id] = self.create_publisher(
                PoseStamped,
                f'/cf_positions/{drone_id}',
                10
            )
            self.velocity_publishers[drone_id] = self.create_publisher(
                TwistStamped,
                f'/cf_velocities/{drone_id}',
                10
            )

        # Previous positions for velocity estimation
        self.prev_positions = {}
        self.prev_timestamps = {}

        # Timer for publishing
        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.logger.info(f"OptiTrack node initialized, publishing at {self.publish_rate} Hz")

    def timer_callback(self):
        """Read OptiTrack data and publish."""
        current_time = self.get_clock().now().to_msg()

        for drone_id in self.drone_ids:
            # Get rigid body data from OptiTrack
            data = self.optitrack_client.get_rigid_body_data(drone_id)

            if data is None:
                self.logger.warn(f"No data for {drone_id}")
                continue

            # Publish pose
            pose_msg = PoseStamped()
            pose_msg.header.stamp = current_time
            pose_msg.header.frame_id = 'world'
            pose_msg.pose.position.x = data['position'][0]
            pose_msg.pose.position.y = data['position'][1]
            pose_msg.pose.position.z = data['position'][2]
            pose_msg.pose.orientation.w = data['orientation'][0]
            pose_msg.pose.orientation.x = data['orientation'][1]
            pose_msg.pose.orientation.y = data['orientation'][2]
            pose_msg.pose.orientation.z = data['orientation'][3]

            self.pose_publishers[drone_id].publish(pose_msg)

            # Estimate and publish velocity
            if drone_id in self.prev_positions:
                dt = (current_time.sec + current_time.nanosec * 1e-9) - \
                     (self.prev_timestamps[drone_id].sec + self.prev_timestamps[drone_id].nanosec * 1e-9)

                if dt > 0:
                    velocity = (np.array(data['position']) - self.prev_positions[drone_id]) / dt

                    vel_msg = TwistStamped()
                    vel_msg.header.stamp = current_time
                    vel_msg.header.frame_id = 'world'
                    vel_msg.twist.linear.x = velocity[0]
                    vel_msg.twist.linear.y = velocity[1]
                    vel_msg.twist.linear.z = velocity[2]

                    self.velocity_publishers[drone_id].publish(vel_msg)

            # Store for next iteration
            self.prev_positions[drone_id] = np.array(data['position'])
            self.prev_timestamps[drone_id] = current_time


def main(args=None):
    rclpy.init(args=args)
    node = OptiTrackNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Acceptance Criteria**:
- Publishes at 120 Hz consistently
- Position jitter < 5mm
- Velocity estimates smooth
- Handles dropped frames gracefully

---

#### Step 3.2: Mission Manager Node (3 days)
**Agent**: ROS Integration Agent

**File**: `src/squadrone_swarm/squadrone_swarm/nodes/mission_manager_node.py`

**Implementation**:
```python
import rclpy
from rclpy.node import Node
from squadrone_msgs.msg import MissionState, DroneStatus, RoleAssignment, DetectionReport
from geometry_msgs.msg import Point
from squadrone_swarm.core.state_machine import MissionStateMachine, MissionState as State
from squadrone_swarm.utils.config_loader import ConfigLoader
from squadrone_swarm.utils.logger import ROSLogger

class MissionManagerNode(Node):
    """
    Central mission orchestration node.

    Subscribes to:
        /cf_positions/*
        /cf_battery/*
        /vision/detections/*

    Publishes:
        /swarm/mission_state
        /swarm/roles
    """

    def __init__(self):
        super().__init__('mission_manager')

        # Load configuration
        self.declare_parameter('config_dir', '/path/to/config')
        self.declare_parameter('mission_version', 'v1')

        config_dir = self.get_parameter('config_dir').value
        mission_version = self.get_parameter('mission_version').value

        self.mission_config = ConfigLoader.load_mission_config(config_dir, mission_version)
        self.arena_config = ConfigLoader.load_arena_config(config_dir)
        self.logger = ROSLogger(self, "MissionManager")

        # Initialize state machine
        self.drone_states = {}  # Dict[drone_id, DroneStatus]
        self.fsm = MissionStateMachine(self.mission_config, self.drone_states)

        # Subscribers
        self.drone_ids = ['cf_p', 'cf_n1', 'cf_n2', 'cf_c']

        for drone_id in self.drone_ids:
            self.create_subscription(
                DroneStatus,
                f'/cf_status/{drone_id}',
                lambda msg, did=drone_id: self.drone_status_callback(msg, did),
                10
            )

        self.create_subscription(
            DetectionReport,
            '/vision/detections',
            self.detection_callback,
            10
        )

        # Publishers
        self.mission_state_pub = self.create_publisher(MissionState, '/swarm/mission_state', 10)
        self.role_assignment_pub = self.create_publisher(RoleAssignment, '/swarm/roles', 10)

        # Timer for FSM updates
        self.timer = self.create_timer(0.1, self.update_fsm)  # 10 Hz

        self.logger.info(f"Mission Manager initialized for mission {mission_version}")

    def drone_status_callback(self, msg, drone_id):
        """Store drone status."""
        self.drone_states[drone_id] = msg

    def detection_callback(self, msg):
        """Handle detection reports."""
        if msg.target_detected and msg.confidence > 0.85:
            self.fsm.target_detected = True
            self.fsm.target_position = msg.target_position_world_frame
            self.logger.info(f"Target detected by {msg.detector_drone_id} at {msg.target_position_world_frame}")

    def update_fsm(self):
        """Update state machine at 10 Hz."""
        current_time = self.get_clock().now().nanoseconds / 1e9
        self.fsm.update(current_time)

        # Publish mission state
        state_msg = MissionState()
        state_msg.header.stamp = self.get_clock().now().to_msg()
        state_msg.state = self.fsm.state.value
        state_msg.target_detected_by = ""  # TODO
        if self.fsm.target_position:
            state_msg.target_position = self.fsm.target_position
        state_msg.mission_time_elapsed = current_time - (self.fsm.mission_start_time or current_time)

        self.mission_state_pub.publish(state_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MissionManagerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Acceptance Criteria**:
- FSM transitions correctly
- Mission state published at 10 Hz
- Responds to detections within 100ms
- Emergency conditions handled

---

## 8. AI AGENT DEFINITIONS & ASSIGNMENTS

### Agent Types & Responsibilities

#### 8.1 DevOps Agent
**Responsibilities**:
- Repository setup and structure
- CI/CD pipeline configuration
- Environment setup scripts
- Build system configuration
- Docker/containerization (if needed)

**Tools**: Git, GitHub Actions, Bash, Docker

**Assigned Tasks**:
- Step 1.1: Project Setup
- scripts/setup_environment.sh
- scripts/build_workspace.sh
- .github/workflows/ci.yml

---

#### 8.2 ROS Integration Agent
**Responsibilities**:
- ROS 2 node implementation
- Topic/service/action definitions
- Launch file creation
- ROS 2 parameter management
- roslaunch/ros2 launch integration

**Tools**: Python, ROS 2 Humble, rclpy

**Assigned Tasks**:
- Step 1.2: Custom ROS 2 Messages
- Step 3.1: OptiTrack Interface Node
- Step 3.2: Mission Manager Node
- Step 3.3: Formation Controller Node
- Step 3.4: Path Planner Node
- Step 3.5: Trajectory Commander Node
- All launch files

---

#### 8.3 Core Algorithm Agent
**Responsibilities**:
- State machine implementation
- Role assignment logic
- Mathematical algorithms
- Pure computation modules (no ROS)

**Tools**: Python, NumPy, SciPy

**Assigned Tasks**:
- Step 1.3: Utility Modules
- Step 2.1: State Machine
- Step 2.2: Role Assignment
- All core/* modules

---

#### 8.4 Control Systems Agent
**Responsibilities**:
- Formation control
- Trajectory generation
- Collision avoidance
- Control law implementation

**Tools**: Python, NumPy, SciPy, Control theory

**Assigned Tasks**:
- Step 2.3: Formation Control
- Step 2.5: Collision Avoidance
- Step 2.7: Trajectory Generation

---

#### 8.5 Planning Agent
**Responsibilities**:
- Path planning algorithms (A*)
- Obstacle representation
- Waypoint generation
- Route optimization

**Tools**: Python, NumPy, SciPy, Graph algorithms

**Assigned Tasks**:
- Step 2.4: Path Planning
- Obstacle map management
- Mission V3bis and V4 specific planning

---

#### 8.6 Estimation Agent
**Responsibilities**:
- Kalman filtering
- State estimation
- Sensor fusion
- Target tracking

**Tools**: Python, NumPy, Filtering libraries

**Assigned Tasks**:
- Step 2.6: Kalman Filter
- Velocity estimation
- Target prediction

---

#### 8.7 Vision Agent
**Responsibilities**:
- AI Deck integration
- Model training (PyTorch)
- Model export (TFLite)
- Detection fusion

**Tools**: Python, PyTorch, TensorFlow Lite, OpenCV

**Assigned Tasks**:
- AI Deck model training
- Detection preprocessing
- Vision pipeline
- Camera calibration

---

#### 8.8 Embedded Systems Agent
**Responsibilities**:
- GAP8 firmware (C)
- TFLite Micro integration
- UART communication
- AI Deck flashing

**Tools**: C, GAPFlow SDK, JTAG/OpenOCD

**Assigned Tasks**:
- gap8_firmware/* implementation
- Hardware debugging
- Firmware optimization

---

#### 8.9 Simulation Agent
**Responsibilities**:
- Gazebo world creation
- Crazyflie model integration
- Physics parameter tuning
- Simulation testing

**Tools**: Gazebo, SDF/URDF, Python

**Assigned Tasks**:
- Gazebo world files
- Crazyflie plugins
- Simulation launch files
- Virtual testing

---

#### 8.10 Test Agent
**Responsibilities**:
- Unit test implementation
- Integration tests
- Mock object creation
- Test coverage reporting

**Tools**: pytest, unittest.mock, coverage.py

**Assigned Tasks**:
- All tests/* files
- CI test integration
- Coverage goals (>90%)

---

#### 8.11 Configuration Agent
**Responsibilities**:
- YAML configuration files
- Parameter tuning
- Environment-specific configs
- Documentation of parameters

**Tools**: YAML, JSON

**Assigned Tasks**:
- All config/* files
- Mission specifications
- Drone parameters

---

#### 8.12 Documentation Agent
**Responsibilities**:
- README files
- API documentation
- Architecture diagrams
- User guides

**Tools**: Markdown, Mermaid, Sphinx (optional)

**Assigned Tasks**:
- docs/* files
- README.md
- Code documentation
- Troubleshooting guides

---

### Agent Coordination Strategy

**Sequential Dependencies**:
```
DevOps Agent → All other agents (need repo structure first)
    ↓
ROS Integration Agent (messages) → All ROS-dependent agents
    ↓
Core Algorithm Agent (utils) → Control, Planning, Estimation agents
    ↓
Control + Planning + Estimation → ROS Integration Agent (nodes)
    ↓
Vision Agent (parallel with above)
    ↓
Simulation Agent (once nodes ready)
    ↓
Test Agent (validates all above)
```

**Parallel Work Streams**:
- Stream 1: DevOps → ROS Messages → Core Algorithms
- Stream 2: Configuration Agent (parallel, low dependency)
- Stream 3: Vision Agent → Embedded Systems Agent (independent path)
- Stream 4: Documentation Agent (parallel, continuous)

---

## 9. TESTING STRATEGY

### 9.1 Test Pyramid

```
                  /\
                 /  \
                /E2E \          End-to-End Tests (Simulation)
               /______\         - Full mission scenarios
              /        \        - Hardware-in-loop
             / Integr.  \       Integration Tests
            /____________\      - Multi-node communication
           /              \     - ROS topic flows
          /  Unit Tests    \    Unit Tests
         /__________________\   - Individual functions
                               - Algorithms
```

### 9.2 Unit Test Requirements

**Coverage Target**: >90% for all core/* and utils/* modules

**Test Files**:
```
tests/unit/
├── test_transforms.py         # Coordinate transformations
├── test_geometry.py           # Geometric calculations
├── test_timing.py             # Time sync utilities
├── test_config_loader.py      # Configuration loading
├── test_state_machine.py      # FSM logic
├── test_role_assignment.py    # Role selection
├── test_formation.py          # Formation mathematics
├── test_path_planning.py      # A* algorithm
├── test_collision.py          # Collision detection/avoidance
├── test_kalman.py             # State estimation
└── test_trajectory.py         # Trajectory generation
```

**Test Fixtures** (`conftest.py`):
```python
import pytest
import numpy as np
from geometry_msgs.msg import Point, Pose, Twist
from squadrone_msgs.msg import DroneStatus

@pytest.fixture
def mock_drone_state():
    """Create mock DroneStatus message."""
    state = DroneStatus()
    state.drone_id = "cf_test"
    state.battery_percent = 80.0
    state.is_flying = True
    state.is_connected = True
    state.pose.position = Point(x=0.0, y=0.0, z=1.0)
    return state

@pytest.fixture
def mock_mission_config():
    """Load test mission configuration."""
    return {
        'mission': {
            'version': 'test',
            'time_budget': 180
        },
        'arena': {
            'boundaries': {
                'x_min': 0, 'x_max': 10,
                'y_min': 0, 'y_max': 6,
                'z_min': 0, 'z_max': 8
            },
            'safety_margin': 0.5
        },
        'formation': {
            'leader_standoff_distance': 0.6,
            'follower_offset_x': -0.5,
            'follower_offset_y': 0.0,
            'follower_offset_z': -0.5
        }
    }
```

### 9.3 Integration Test Requirements

**Test Multi-Node Communication**:
```python
# tests/integration/test_node_communication.py

def test_optitrack_to_mission_manager():
    """Test OptiTrack data flows to Mission Manager."""
    # Launch OptiTrack node (mock)
    # Launch Mission Manager node
    # Verify mission state updates based on positions
    pass

def test_detection_to_role_assignment():
    """Test detection triggers role assignment."""
    # Publish mock detection
    # Verify role assignment message published
    # Check roles assigned correctly
    pass
```

### 9.4 Simulation Test Requirements

**Mission Scenarios**:
```python
# tests/simulation/test_mission_v1.py

def test_mission_v1_success():
    """
    Full Mission V1 simulation test.

    Steps:
    1. Launch Gazebo with arena
    2. Spawn 4 drones at initial positions
    3. Start mission
    4. Verify detection occurs
    5. Verify role assignment
    6. Verify interception
    7. Verify neutralization
    8. Check time < 3 minutes
    9. Check no collisions
    """
    assert mission_completed == True
    assert mission_time < 180
    assert collisions == 0
```

**Smoke Tests** (Hardware):
```python
# tests/hardware/smoke_tests.py

def test_radio_connectivity():
    """Test all 4 drones connectable via Crazyradio."""
    for drone_uri in DRONE_URIS:
        cf = Crazyflie(uri=drone_uri)
        assert cf.is_connected()
        assert cf.get_rssi() > -60

def test_battery_levels():
    """Test all batteries above minimum threshold."""
    for drone_uri in DRONE_URIS:
        voltage = get_battery_voltage(drone_uri)
        assert voltage > 3.7  # >70% charge

def test_optitrack_tracking():
    """Test OptiTrack tracks all 4 rigid bodies."""
    for rigid_body_id in RIGID_BODY_IDS:
        pose = optitrack_client.get_pose(rigid_body_id)
        assert pose is not None
        assert pose_variance(pose, duration=10) < 0.01  # <10mm jitter
```

---

## 10. MISSION SPECIFICATIONS (Detailed)

### Mission V1: Static Target, No Obstacles

**Initial Positions**:
```yaml
N1:  [2.5, 2.5, 0.0]
N2:  [2.5, 3.5, 0.0]
P:   [3.0, 5.0, 0.0]
C:   [7.5, 3.0, 5.0]
```

**Flight Phases**:

| Phase | Duration | Description | Success Criteria |
|-------|----------|-------------|------------------|
| INIT | 10s | Takeoff to 4m altitude | All drones at z=4.0±0.1m |
| PATROL | 0-120s | P patrols perimeter, N1/N2 check safety zone | P completes waypoint circuit |
| DETECT | 5s | P detects C visually | Confidence >0.85 for 3 frames |
| ASSIGN | 1s | Assign L (highest battery), S (second) | Roles published |
| INTERCEPT | 30s | L and S approach C | Distance to C < 0.6m |
| NEUTRALIZE | 20s | Simulate jamming | Hold formation 1.5s |
| ATTACK | 5s | P kamikaze (stop above C) | P at C position ±0.1m |
| RTL | 30s | Return to safety zone | All drones at z<0.5m |

**Total Budget**: 180s (3 minutes)
**Expected Time**: 171s

---

### Mission V2: Static Corner Target

**Changes from V1**:
- Target position: `C = [9.5, 0.5, 5.0]`
- Patrol time: +15s (longer route)
- Interception time: +15s (longer distance)
- Formation adjustment: Check boundary constraints

**Boundary Safety**:
- Leader setpoint: max(L_x, arena_x_max - 0.6) to avoid collision
- Follower: Offset adjusted if near boundary

**Total Budget**: 210s (3.5 minutes)

---

### Mission V3: Moving Target

**Target Behavior**:
```yaml
movement_pattern: "circular"
center: [6.5, 3.0, 4.0]
radius: 1.5  # meters
angular_velocity: 0.2  # rad/s (11.5°/s)
altitude: 4.0  # constant
```

**Changes from V1**:
- Kalman filter enabled for tracking
- Predictive interception (lead target by 0.5s)
- Formation follows moving target (continuous adjustment)
- Neutralization: Track target while maintaining formation

**Total Budget**: 240s (4 minutes)

---

### Mission V3bis: Static Target with Obstacles

**Obstacles**:
```yaml
obstacles:
  - type: "box"
    center: [5.0, 2.0, 3.0]
    size: [0.5, 0.5, 2.0]  # width, depth, height
  - type: "box"
    center: [7.0, 4.0, 2.5]
    size: [0.5, 0.5, 2.0]
  - type: "box"
    center: [8.5, 1.5, 3.5]
    size: [0.5, 0.5, 2.0]
```

**Path Planning**:
- A* with 0.5m resolution grid
- Inflation radius: 0.4m
- Smoothing: B-spline with 50 points
- Replanning: If new obstacle detected or path blocked

**Total Budget**: 240s (4 minutes)

---

### Mission V4: Moving Target with Obstacles

**Combination**: V3 + V3bis

**Additional Challenges**:
- Dynamic replanning (2 Hz if target moves >1m)
- Simultaneous obstacle avoidance + target tracking
- Predictive collision checking with obstacles

**Total Budget**: 300s (5 minutes)

---

## 11. INTEGRATION POINTS

### 11.1 Critical Integration Points

#### IP-1: OptiTrack → ROS 2
**Risk**: High
**Components**: OptiTrack system, NatNet SDK, optitrack_node.py
**Testing**:
- Verify 120 Hz update rate
- Check timestamp synchronization (<10ms offset)
- Validate coordinate frame transformations
- Test dropped frame handling

**Integration Steps**:
1. Set up OptiTrack server on local network
2. Verify NatNet streaming with test client
3. Implement optitrack_node with NatNet client
4. Publish test messages, verify with `ros2 topic echo`
5. Log actual rate with `ros2 topic hz`

---

#### IP-2: Crazyflie Firmware ↔ Ground Station
**Risk**: High
**Components**: cflib, Crazyradio 2.0, cf_bridge nodes
**Testing**:
- Radio link quality (RSSI, packet loss)
- Command latency (<50ms)
- Simultaneous 4-drone communication
- Failsafe triggering

**Integration Steps**:
1. Flash stock Crazyflie firmware
2. Test single drone connection with cfclient
3. Implement cf_bridge node with cflib
4. Send test setpoints, verify execution
5. Scale to 4 drones simultaneously

---

#### IP-3: AI Deck ↔ Ground Station
**Risk**: Medium
**Components**: GAP8 firmware, UART, aideck_interface_node.py
**Testing**:
- UART communication stability
- Detection message frequency (5 Hz)
- Confidence score calibration
- False positive rate

**Integration Steps**:
1. Flash AI Deck firmware via JTAG
2. Test UART communication with serial monitor
3. Implement detection message parser
4. Publish to ROS topic
5. Fuse multi-drone detections

---

#### IP-4: Crazyswarm2 Integration
**Risk**: Medium
**Components**: Crazyswarm2, custom mission manager
**Testing**:
- Collision avoidance compatibility
- Setpoint forwarding
- Emergency stop propagation
- Multi-drone takeoff/landing

**Integration Steps**:
1. Install Crazyswarm2
2. Test example scripts
3. Wrap Crazyswarm2 API in custom nodes
4. Verify setpoints reach drones
5. Test emergency procedures

---

### 11.2 Integration Testing Timeline

**Week 5**: IP-1, IP-2 (critical path)
**Week 6**: IP-3, IP-4
**Week 7**: End-to-end integration

---

## 12. RISK MITIGATION

### Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| OptiTrack occlusion | Medium | High | Asymmetric markers, fallback to onboard estimation |
| Radio saturation | Low | High | Minimize telemetry, disable image streaming |
| Battery too short | High | Critical | Optimize trajectories, practice battery swaps |
| AI Deck inference slow | Medium | Medium | Pre-validate 5 Hz, reduce resolution if needed |
| Collision during intercept | Low | High | Conservative safety margins (0.8m), tested in sim |
| Gazebo/Crazyflie model issues | Medium | Medium | Backup: Webots, or skip simulation (not recommended) |
| Integration delays | High | Medium | Start integration early (Week 5), parallel dev |
| Firmware bugs | Low | High | Use stable releases, test incrementally |
| Detection false positives | Medium | Low | Tune confidence threshold, require multi-frame confirmation |
| Path planning too slow | Low | Medium | Optimize A*, reduce grid resolution, cache paths |

---

### Contingency Plans

**If OptiTrack fails**:
- Use onboard state estimation (degraded accuracy)
- Manual mission abort, safety landing

**If AI Deck fails**:
- Fallback to OptiTrack-based target ID (no vision)
- Simpler mission: Known target position

**If battery < 3 min**:
- Reduce patrol time
- Direct intercept (skip patrol)
- Pre-position drones closer to expected target

**If simulation unavailable**:
- Extensive unit testing only
- Hardware testing with safety nets
- Very conservative first flights

---

## END OF PART 2

**Next Steps**:
1. Review this plan with team
2. Assign agents to tasks
3. Begin Phase 1 implementation
4. Track progress with daily standups
5. Adjust timeline as needed

**Total Estimated Effort**: 7 weeks (pre-hackathon) + 3 days (hackathon)

---

**Document prepared by**: Technical Lead AI
**For**: Drone Defense Hackathon 2025 - Challenge 2
**Last updated**: 2025-11-17
