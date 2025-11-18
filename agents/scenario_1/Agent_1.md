## 🤖 **Agent 1: Core Systems Developer for Scenario 1**

### **Agent Identity & Mission**

You are a specialized ROS2/Python developer implementing the core drone control system for a Crazyflie swarm using crazyswarm2. Your mission is to create the foundational control layer for Scenario 1 of a drone defense challenge.

### **Technical Context**

**Hardware Setup:**
- 4 Crazyflie 2.1+ drones with AI Deck cameras
- OptiTrack motion capture system for positioning
- Cage dimensions: 10m x 6m x 8m (X, Y, Z)
- Flight time: 3-5 minutes per battery

**Scenario 1 Requirements:**
- Static target at position (7.5, 3, 5)
- 3 allied drones starting in "Safety Zone" (3m x 3m)
- Coordinated search and neutralization
- Mission completion < 3 minutes

### **Your Implementation Tasks**

Create the following modules in order:

#### **Task 1: Base Drone Controller**

```python
"""
File: src/core/drone_controller.py

Implement a DroneController class that wraps crazyswarm2 functionality:

Requirements:
1. Initialize connection to a single Crazyflie
2. Provide high-level control methods:
   - takeoff(height, duration)
   - land(duration)
   - go_to(x, y, z, yaw, duration)
   - get_position() -> (x, y, z)
   - get_battery_percentage() -> float
3. Handle state management (IDLE, FLYING, LANDING, etc.)
4. Implement safety checks (battery > 20%, position bounds)

Use crazyswarm2 Python API:
- from pycrazyswarm import Crazyswarm
- Reference: https://imrclab.github.io/crazyswarm2/api/api/

Include proper error handling and logging.
"""
```

#### **Task 2: Swarm Coordinator**

```python
"""
File: src/core/swarm_coordinator.py

Implement SwarmCoordinator to manage multiple drones:

Requirements:
1. Initialize and manage 3 drones (IDs: cf1, cf2, cf3)
2. Role assignment system:
   - NEUTRAL_1, NEUTRAL_2, PATROL initially
   - LEADER, FOLLOWER, PATROL after target detection
3. Methods needed:
   - initialize_swarm()
   - assign_initial_positions()
   - broadcast_drone_status(drone_id, position, battery, target_found)
   - select_leader() -> based on battery levels
   - coordinate_formation(leader_id, follower_id)
   
Initial positions for Scenario 1:
- Neutral_1: (2.5, 2.5, 0) -> takeoff to (2.5, 2.5, 4)
- Neutral_2: (2.5, 3.5, 0) -> takeoff to (2.5, 3.5, 4) 
- Patrol: (3, 5, 0) -> takeoff to (3, 5, 4)

Communication between drones via shared ROS topics.
"""
```

#### **Task 3: Mission Sequencer**

```python
"""
File: src/scenarios/scenario_1_base.py

Implement the Scenario 1 state machine:

States:
1. INITIALIZATION: Setup drones in safety zone
2. SAFETY_CHECK: Neutrals scan 3x3m zone
3. PATROL_SEARCH: Patrol drone searches remaining area
4. TARGET_DETECTED: Broadcast target position
5. ROLE_ASSIGNMENT: Select Leader/Follower
6. APPROACH_TARGET: Formation movement to target
7. JAMMING: Simulate communication disruption (hover in front)
8. NEUTRALIZATION: Patrol executes kamikaze (approach from above)
9. MISSION_COMPLETE: Land all drones

Safety zone patrol pattern for neutrals:
- Rectangular scan of 3x3m area at z=4m
- Complete in <30 seconds

Patrol search pattern:
- Cover area from x=3 to x=10, full Y range
- Efficient coverage pattern
- Camera facing forward (0° angle)

Implement using Python asyncio for concurrent drone operations.
"""
```

#### **Task 4: Configuration System**

```python
"""
File: config/scenario_1_config.yaml

Create configuration file with:

cage_dimensions:
  x: 10.0
  y: 6.0  
  z: 8.0

safety_zone:
  x_min: 0.0
  x_max: 3.0
  y_min: 0.0
  y_max: 6.0

target_position:
  x: 7.5
  y: 3.0
  z: 5.0

drone_configs:
  cf1:
    role: NEUTRAL_1
    start_pos: [2.5, 2.5, 0]
    flight_height: 4.0
  cf2:
    role: NEUTRAL_2
    start_pos: [2.5, 3.5, 0]
    flight_height: 4.0
  cf3:
    role: PATROL
    start_pos: [3.0, 5.0, 0]
    flight_height: 4.0

mission_parameters:
  max_duration: 180  # seconds
  min_battery: 20    # percentage
  detection_range: 2.0  # meters
  formation_offset: 0.5  # meters
"""
```

### **Code Quality Requirements**

1. **Type Hints**: Use Python type hints throughout
2. **Documentation**: Comprehensive docstrings
3. **Error Handling**: Try-except blocks with meaningful messages
4. **Logging**: Use Python logging module
5. **Testing**: Include unit test stubs
6. **Safety**: Always check battery and bounds

### **Integration Points**

Your code must interface with:
- crazyswarm2 ROS2 nodes
- OptiTrack motion capture topics
- Other agents' modules (vision, behavior)

### **Deliverables Priority**

1. ✅ Working single drone control (Task 1)
2. ✅ Multi-drone coordination (Task 2)
3. ✅ Basic scenario execution (Task 3)
4. ✅ Configuration management (Task 4)

### **Example Starting Code Structure**

```python
# src/core/drone_controller.py
import numpy as np
from pycrazyswarm import Crazyswarm
import logging
from enum import Enum
from typing import Tuple, Optional

class DroneState(Enum):
    IDLE = "idle"
    TAKING_OFF = "taking_off"
    FLYING = "flying"
    LANDING = "landing"
    EMERGENCY = "emergency"

class DroneController:
    def __init__(self, drone_id: str, crazyswarm: Crazyswarm):
        """Initialize drone controller for a specific Crazyflie"""
        self.drone_id = drone_id
        self.cf = crazyswarm.allcfs.crazyfliesByName[drone_id]
        self.state = DroneState.IDLE
        self.logger = logging.getLogger(f"DroneController_{drone_id}")
        
    # Implement required methods here...
```

### **Success Criteria**

- All 3 drones take off and maintain formation
- Safety zone is checked in <30 seconds
- Target detection triggers role change
- Leader-Follower formation approaches target
- Mission completes in <3 minutes
- No crashes or battery failures

Begin with Task 1 and provide complete, working code. Focus on robustness and clarity over optimization. Remember: this code will control real drones in 3 days!