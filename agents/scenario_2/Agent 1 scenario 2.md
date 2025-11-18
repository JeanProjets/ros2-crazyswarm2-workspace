Here is the detailed specification for **Agent 1 (Core Systems Developer)** tailored specifically for **Scenario 2**.

The main differences from Scenario 1 are the **tight constraints near cage boundaries** (target is in a corner) and the **increased importance of battery management** due to the longer flight distance.

-----

## 🤖 **Agent 1: Core Systems Developer for Scenario 2**

### **Agent Identity & Mission**

You are a specialized ROS2/Python developer implementing the core drone control system for a Crazyflie swarm. Your mission is to adapt the foundational control layer for **Scenario 2**, where precision flight near boundaries and strict battery management are critical.

### **Technical Context**

**Scenario 2 Specifics:**

  - **Target Position:** Far corner at **(9.5, 0.5, 5)**.
  - **Constraint:** **Strict boundary avoidance.** The target is 50cm from the walls ($X=10, Y=0$). Standard GPS/OptiTrack jitter could push drones into the net.
  - **Battery Criticality:** Longer flight path requires efficient role assignment.
  - **Formation Constraint:** The standard offset formation must not push the "Follower" drone into the wall.

**Hardware Setup:**

  - Same as Scenario 1 (4x Crazyflie 2.1+, OptiTrack, AI Decks).
  - **Safety Margin:** Hardware limit is wall contact. Software limit must be defined as 0.3m from mesh.

### **Your Implementation Tasks**

#### **Task 1: Precision Drone Controller**

```python
"""
File: src/core/safe_drone_controller.py

Extend the basic DroneController to include strict boundary enforcement:

Requirements:
1. Inherit from base DroneController (or extend it).
2. Implement `clamped_navigate(x, y, z, yaw)`:
   - Before sending command, check against safety bounds.
   - If target is outside bounds, clamp to nearest safe coordinate.
   - Log warning if clamping occurs.
   
3. Implement `precision_hover(height)`:
   - Tighter PID gains for station keeping (if accessible via params) or reduced velocity limits for final approach.

4. Implement `get_battery_voltage()`:
   - Voltage is more accurate than percentage for this scenario.
   - Warning threshold: 3.5V (Return to Home immediately).

Boundaries for Scenario 2:
- X: [0.3, 9.7] (Target is at 9.5, very close to limit)
- Y: [0.3, 5.7] (Target is at 0.5, very close to limit)
- Z: [0.2, 5.8]

Code must prevent the high-level logic from accidentally commanding a crash.
"""
```

#### **Task 2: Battery-Optimized Swarm Manager**

```python
"""
File: src/core/swarm_manager_v2.py

Implement SwarmCoordinator with advanced role logic:

Requirements:
1. Initialize 4 drones (P, N1, N2, C - though C is virtual/static).
2. `select_optimal_leader(candidates: List[str]) -> str`:
   - Query voltage from N1 and N2.
   - **Strict Rule:** Highest voltage becomes LEADER.
   - If difference < 0.1V, prefer drone closest to X center.
   
3. `calculate_safe_formation(leader_pos, follower_id)`:
   - Standard offset is (-0.5, -0.5, -0.5).
   - **Scenario 2 Check:** If Leader is at (9.5, 0.5), a Y offset of -0.5 puts Follower at 0.0 (WALL COLLISION).
   - Logic: Dynamically invert Y-offset if Leader is near Y=0 boundary.
     - New offset: (-0.5, +0.5, -0.5).

This ensures the "Follower" doesn't crash while maintaining formation.
"""
```

#### **Task 3: Scenario 2 Sequencer**

```python
"""
File: src/scenarios/scenario_2_corner.py

Implement the Scenario 2 state machine:

States:
1. INITIALIZATION: Safety Zone (Same as Scen 1).
2. LONG_RANGE_PATROL: P covers full field to find corner target.
3. TARGET_LOCK: Detection at distance.
4. PRECISION_APPROACH: 
   - Leader/Follower move to standoff point (9.0, 1.0, 5.0).
   - Slow speed (0.5 m/s) to prevent overshoot into wall.
5. CORNER_JAMMING: 
   - Drones must not drift behind target (X > 9.5).
6. VERTICAL_NEUTRALIZATION: 
   - P must attack from strictly above (Z-axis descent).
   - No swooping maneuvers (room too tight).
7. MISSION_COMPLETE: RTB.

Timeouts:
- Total Mission: 3m 30s (Estimated flight time 1m 45s).
- Patrol: 2 min max.
"""
```

#### **Task 4: Configuration System**

```python
"""
File: config/scenario_2_config.yaml

Create configuration file with tighter constraints:

cage_dimensions:
  x: 10.0
  y: 6.0
  z: 8.0

safety_bounds:
  x_min: 0.3
  x_max: 9.7  # Critical for target at 9.5
  y_min: 0.3
  y_max: 5.7
  z_max: 6.0

target_position:
  x: 9.5
  y: 0.5
  z: 5.0

drone_configs:
  # Same start positions as Scen 1
  cf1: {role: NEUTRAL_1, start_pos: [2.5, 2.5, 0]}
  cf2: {role: NEUTRAL_2, start_pos: [2.5, 3.5, 0]}
  cf3: {role: PATROL,    start_pos: [3.0, 5.0, 0]}

mission_parameters:
  approach_speed: 0.7  # m/s (Slower than Scen 1)
  formation_offset_y: 0.5 # Dynamic
  battery_abort_voltage: 3.4
"""
```

### **Code Quality Requirements**

1.  **Safety First**: The `clamped_navigate` method is the most important function you will write.
2.  **Explicit offsets**: Do not hardcode offsets in the sequencer; calculate them based on Leader position to avoid walls.
3.  **Battery Logging**: Log voltage every 5 seconds.

### **Example Starting Code Structure**

```python
# src/core/safe_drone_controller.py
import numpy as np
from src.core.drone_controller import DroneController

class SafeDroneController(DroneController):
    def __init__(self, drone_id, crazyswarm, config):
        super().__init__(drone_id, crazyswarm)
        self.bounds = config['safety_bounds']

    def clamp_position(self, x, y, z):
        """Ensure coordinates are within cage safety margins"""
        safe_x = np.clip(x, self.bounds['x_min'], self.bounds['x_max'])
        safe_y = np.clip(y, self.bounds['y_min'], self.bounds['y_max'])
        safe_z = np.clip(z, 0.2, self.bounds['z_max'])
        
        if safe_x != x or safe_y != y:
            self.logger.warning(f"CLIPPING CMD: ({x},{y}) -> ({safe_x},{safe_y})")
            
        return safe_x, safe_y, safe_z

    def go_to(self, x, y, z, yaw, duration):
        """Override go_to with safety checks"""
        sx, sy, sz = self.clamp_position(x, y, z)
        super().go_to(sx, sy, sz, yaw, duration)
```

### **Success Criteria**

  - ✅ Drones never command a position outside `[0.3, 9.7]` in X.
  - ✅ Leader drone is always the one with higher voltage.
  - ✅ Formation automatically inverts Y-offset when near the $Y=0$ wall.
  - ✅ Mission completes without hitting the cage mesh.

**Deliverables:**

1.  `SafeDroneController` class.
2.  `SwarmManagerV2` with dynamic offset logic.
3.  `scenario_2_config.yaml`.
4.  `scenario_2_corner.py` script.