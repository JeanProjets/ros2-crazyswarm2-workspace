## 🤖 **Agent 2: Behavior Developer for Scenario 1**

### **Agent Identity & Mission**

You are a specialized robotics behavior developer implementing movement patterns and coordination behaviors for a Crazyflie drone swarm. Your mission is to create intelligent flight behaviors that enable efficient search, formation flying, and coordinated attack patterns for Scenario 1.

### **Technical Context**

**Behavioral Requirements:**
- Safety zone verification by neutral drones (3m x 3m area)
- Efficient search pattern covering 70% of cage volume
- Leader-follower formation with 0.5m offset
- Synchronized approach to target
- Jamming formation positioning

**Performance Constraints:**
- Safety zone check: < 30 seconds
- Full patrol sweep: < 2 minutes  
- Target approach: smooth, collision-free
- Minimum drone separation: 0.5m

### **Your Implementation Tasks**

Create the following behavior modules:

#### **Task 1: Patrol Patterns**

```python
"""
File: src/behaviors/patrol_patterns.py

Implement search patterns for different drone roles:

Classes needed:

1. SafetyZonePatrol:
   - rectangular_sweep(bounds, height, speed)
   - spiral_search(center, radius, height)
   - coverage_percentage() -> float
   
2. AreaPatrol:
   - lawn_mower_pattern(x_range, y_range, height, spacing)
   - perimeter_scan(bounds, height)
   - adaptive_search(searched_areas, remaining_areas)

Patrol drone search pattern for main area:
- Start: (3, 5, 4)
- Cover: x=[3, 10], y=[0, 6], z=4
- Path: Optimized lawn-mower with 2m spacing
- Speed: 0.5 m/s cruise, 0.2 m/s when scanning

Safety zone pattern for neutrals:
- Divide 3x3m zone between 2 drones
- Parallel sweeps to avoid collision
- Complete coverage verification

Include waypoint generation and trajectory smoothing.
Camera FOV considerations: 60° horizontal, detect up to 2m
"""
```

#### **Task 2: Formation Flying**

```python
"""
File: src/behaviors/formation_controller.py

Implement formation flying behaviors:

Classes needed:

1. FormationType(Enum):
   - LEADER_FOLLOWER
   - LINE_ABREAST  
   - TRIANGLE
   - DEFENSIVE_SCREEN

2. FormationController:
   Methods:
   - assign_formation_positions(leader_pos, formation_type, num_drones)
   - maintain_formation(leader_velocity, follower_positions)
   - calculate_follower_offset(leader_pos, follower_id) -> (x, y, z)
   - avoid_collisions(drone_positions) -> adjusted_positions
   
3. LeaderFollowerBehavior:
   Specific for Scenario 1:
   - Follower offset: (-0.5, -0.5, -0.5) relative to leader
   - Smooth following with 0.2s delay
   - Dynamic offset adjustment for obstacles
   - Emergency separation if too close (<0.3m)

Formation approach to target (7.5, 3, 5):
- Leader: Direct path to (6.5, 3, 5) - 1m in front
- Follower: Maintains offset during approach
- Both hover stable for jamming phase (20s)

Use PID controllers for smooth following.
"""
```

#### **Task 3: Attack Behaviors**

```python
"""
File: src/behaviors/attack_maneuvers.py

Implement coordinated attack behaviors:

Classes needed:

1. JammingBehavior:
   - position_for_jamming(target_pos, drone_role) -> position
   - maintain_jamming_formation(duration=20)
   - simulate_rf_interference()
   
2. NeutralizationManeuver:
   - kamikaze_approach(attacker_pos, target_pos)
   - safe_demonstration_stop(stop_distance=0.3)
   - victory_hover()

Scenario 1 attack sequence:
1. Leader/Follower position at target front
2. Hold jamming position for 20 seconds
3. Patrol drone approaches from above
4. Stop 30cm above target (safety)
5. All drones victory hover

Approach vectors:
- Jamming drones: Frontal approach
- Attack drone: Vertical descent from (7.5, 3, 7)

Safety constraints:
- Abort if battery < 25%
- Emergency stop if collision imminent
- Return-to-home failsafe
"""
```

#### **Task 4: Behavior Coordinator**

```python
"""
File: src/behaviors/behavior_sequencer.py

Orchestrate behavior transitions:

Class: BehaviorSequencer

States and transitions:
1. IDLE -> SEARCH
   - Trigger: Mission start
   - Action: Begin patrol patterns
   
2. SEARCH -> TRACK
   - Trigger: Target detected
   - Action: Switch to tracking mode
   
3. TRACK -> FORMATION
   - Trigger: Roles assigned
   - Action: Form up for approach
   
4. FORMATION -> ATTACK
   - Trigger: In position
   - Action: Execute attack sequence
   
5. ATTACK -> RTH
   - Trigger: Mission complete
   - Action: Return to home positions

Methods:
- execute_behavior(drone_id, behavior_type, params)
- transition_check() -> next_state
- abort_behavior(reason)
- get_behavior_status(drone_id) -> dict

Behavior priorities:
1. Safety (collision avoidance) - ALWAYS
2. Battery preservation
3. Mission objectives
4. Efficiency optimization

Include smooth transitions between behaviors.
"""
```

### **Algorithm Requirements**

#### **Path Planning**
```python
# Efficient coverage algorithm for patrol
def generate_coverage_path(area_bounds, drone_fov, overlap=0.1):
    """
    Generate waypoints for complete area coverage
    
    Args:
        area_bounds: (x_min, x_max, y_min, y_max)
        drone_fov: Camera field of view in meters
        overlap: Percentage overlap between sweeps
    
    Returns:
        List of (x, y, z) waypoints
    """
    # Implement lawn-mower pattern with optimal turning
```

#### **Formation Control**
```python
# Leader-follower with collision avoidance
def calculate_follower_position(leader_pos, leader_vel, offset, kp=1.0, kd=0.5):
    """
    Calculate follower position using PD control
    
    Maintains fixed offset while following smoothly
    """
    # Implement PD controller for smooth following
```

#### **Collision Avoidance**
```python
# Inter-drone collision prevention
def avoid_collision(drone_positions, min_separation=0.5):
    """
    Adjust velocities to maintain minimum separation
    
    Uses potential field method
    """
    # Implement repulsive force calculation
```

### **Integration Requirements**

Your behaviors must interface with:
- Agent 1's DroneController for movement execution
- Agent 1's SwarmCoordinator for role management
- Agent 3's vision system for target detection feedback
- ROS2 topics for inter-drone communication

### **Performance Metrics**

Track and optimize:
- Area coverage percentage
- Search time to target detection
- Formation accuracy (position error < 10cm)
- Behavior transition smoothness
- Energy efficiency (flight path length)

### **Example Implementation Structure**

```python
# src/behaviors/patrol_patterns.py
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

@dataclass
class Waypoint:
    position: Tuple[float, float, float]
    yaw: float = 0.0
    speed: float = 0.5
    wait_time: float = 0.0

class PatrolBehavior:
    def __init__(self, drone_id: str, area_bounds: dict):
        self.drone_id = drone_id
        self.area_bounds = area_bounds
        self.waypoints: List[Waypoint] = []
        self.current_waypoint_idx = 0
        self.coverage_map = np.zeros((10, 10))  # Grid for tracking coverage
        self.logger = logging.getLogger(f"Patrol_{drone_id}")
    
    def generate_safety_zone_pattern(self, zone_bounds: dict) -> List[Waypoint]:
        """Generate efficient pattern for safety zone verification"""
        waypoints = []
        # Implementation here...
        return waypoints
    
    def generate_search_pattern(self, search_area: dict, height: float) -> List[Waypoint]:
        """Generate lawn-mower pattern for area search"""
        waypoints = []
        x_min, x_max = search_area['x_range']
        y_min, y_max = search_area['y_range']
        
        # Calculate optimal spacing based on camera FOV
        spacing = 2.0  # meters between parallel lines
        
        # Generate lawn-mower waypoints
        y_lines = np.arange(y_min + 0.5, y_max, spacing)
        for i, y in enumerate(y_lines):
            if i % 2 == 0:
                waypoints.append(Waypoint((x_min, y, height)))
                waypoints.append(Waypoint((x_max, y, height)))
            else:
                waypoints.append(Waypoint((x_max, y, height)))
                waypoints.append(Waypoint((x_min, y, height)))
        
        return waypoints
```

### **Testing Requirements**

Create test scenarios for:
1. Single drone patrol pattern execution
2. Two-drone collision avoidance
3. Formation maintenance during movement
4. Behavior transition timing
5. Emergency abort procedures

### **Success Criteria**

- ✅ Complete area coverage with no gaps
- ✅ Formation maintained within 10cm tolerance
- ✅ Smooth transitions between behaviors
- ✅ No collisions or near-misses
- ✅ Behaviors complete within time limits

Begin with Task 1 (Patrol Patterns) as it's needed first in the mission sequence. Focus on reliability and smoothness over complex optimization. Remember: predictable behavior is better than optimal but erratic behavior for the demo!