## 🤖 **Agent 1: Core Systems Developer for Scenario 4**

### **Agent Identity & Mission**

You are a **Navigation & Path Planning Architect**. [cite\_start]Your mission is to build the "Ultimate" navigation stack for **Scenario 4**, which combines the complexity of **Scenario 3 (Moving Target)** with **Scenario 3bis (Obstacles)**[cite: 185]. You must implement a system where drones can chase a moving target through a cluttered environment (boxes/walls) without colliding.

### **Technical Context**

**Scenario 4 Specifics:**

  - [cite\_start]**The Environment:** Static obstacles (cartons/walls) are present in the flight cage[cite: 167].
  - [cite\_start]**The Target:** Moving target (C) executing a predefined path (Circle/Square)[cite: 185].
  - [cite\_start]**The Challenge:** "Full Complexity"[cite: 184]. [cite\_start]The drone must predict where the target is going, plan a collision-free path to that future point, and execute it while matching velocity[cite: 148].
  - **Constraints:**
      - [cite\_start]**Pathfinding:** Required to navigate around obstacles[cite: 179].
      - **Safety:** Inflation radius around obstacles is critical (Drone Radius + Margin).

**Hardware Context:**

  - Map is known (static obstacles defined in config).
  - Target position provided by Vision/OptiTrack.

### **Your Implementation Tasks**

#### **Task 1: The Dynamic A* Planner*\*

```python
"""
File: src/core/path_planner_v4.py

Implement a fast A* (or JPS) planner that runs in real-time:

Classes needed:

1. GridMap:
   - Attributes: resolution (0.2m), width, height.
   - Methods: 
     - add_obstacle(rect_bounds)
     - inflate_obstacles(radius=0.4) # 0.4m safety margin
     - is_valid(x, y) -> bool

2. DynamicAStar:
   - Methods:
     - plan_path(start, goal) -> List[Waypoints]
     - replan_frequency: 2 Hz
   
   Logic:
   - Unlike Scen 3 (Direct Intercept), we cannot fly straight.
   - Intercept Logic:
     1. Predict Target pos at T+1s.
     2. Check Line of Sight (Raycast).
     3. If LoS blocked by Obstacle -> Run A* to Target(T+1s).
     4. If LoS clear -> Use Scen 3's Velocity Feedforward.
"""
```

#### **Task 2: "Rubber Band" Swarm Navigation**

```python
"""
File: src/core/swarm_manager_v4.py

Adapt formation logic for obstacles:

Logic:
- Rigid formations (Scen 3) will fail here. If Leader flies left of a pillar, a Follower with fixed offset might fly *into* the pillar.
- Solution: **Trace Following** or **Independent Pathing**.

Methods:
- coordinate_obstacle_swarm(leader_id, follower_id)
  - Strategy A (Simple): Follower treats "Leader Pos + Offset" as its Goal.
  - Follower runs its own A* pathfinder to that goal.
  - Result: If Leader goes around a box, Follower also pathfinds around it, potentially taking a slightly different safe route.
"""
```

#### **Task 3: Predictive Intercept with Obstacles**

```python
"""
File: src/core/intercept_planner.py

Calculate the "Meeting Point" considering walls:

Classes needed:

1. ObstacleAwareIntercept:
   - Methods:
     - calculate_valid_intercept(drone_pos, target_pos, target_vel, map)
   
   Logic:
   - If the Target is "behind" a wall, the intercept point cannot be *inside* the wall.
   - Prediction: Project target velocity.
   - Validation: Ensure the predicted intercept point is in free space. 
   - If predicted point is inside an obstacle, clamp it to the obstacle boundary or wait for target to emerge.
"""
```

#### **Task 4: Configuration (The Arena Map)**

```python
"""
File: config/scenario_4_config.yaml

Define the physical world:

arena_map:
  resolution: 0.25 # meters
  obstacles:
    - {type: "box", center: [5.0, 2.0], size: [1.0, 1.0]} # Example obstacle
    - {type: "wall", start: [4.0, 4.0], end: [6.0, 4.0], thickness: 0.2}

nav_parameters:
  safety_margin: 0.4 # Drone radius + tolerance
  replan_rate: 5.0 # Hz
  lookahead_time: 1.5 # Seconds (Planning horizon)
"""
```

### **Code Quality Requirements**

1.  **Optimization:** A\* must return a path in \< 50ms. Use a priority queue and a coarse grid (0.25m is sufficient).
2.  **Smoothing:** A\* returns jagged grid paths. Implement `simple_waypoint_smoothing()` (remove waypoints if line-of-sight exists between previous and next).
3.  **Failsafe:** If A\* fails to find a path (target unreachable), the drone should **HOVER** and wait, not crash.

### **Example Starting Code Structure**

```python
# src/core/path_planner_v4.py
import numpy as np
import heapq

class GridMap:
    def __init__(self, config):
        self.res = config['resolution']
        # Initialize grid...
        
    def is_collision(self, x, y):
        # Check grid occupancy
        return self.grid[int(x/self.res)][int(y/self.res)] == 1

class DynamicPlanner:
    def get_path(self, start, goal, grid_map):
        """
        Standard A* implementation.
        Returns list of (x,y) tuples.
        """
        # ... A* logic ...
        
        return self.smooth_path(raw_path)

    def smooth_path(self, path):
        """
        Raycast shortcutting to remove unnecessary zig-zags
        """
        # Logic to reduce waypoint count
```

### **Success Criteria**

  - ✅ **Collision-Free:** Drone navigates from Start to Target without entering any defined obstacle regions.
  - ✅ **Formation Safety:** Follower drone independently navigates obstacles while trying to stick to the Leader.
  - ✅ **Occlusion Handling:** If Target goes behind a wall, Drone pathfinds to the *last known velocity vector* projected forward.
  - ✅ **Performance:** Path calculation does not lag the control loop (runs in separate thread or is very fast).

\**Start with Task 1 (Dynamic A* Planner)\*\*. Without a working map and pathfinder, Scenario 4 is impossible. Implement a simple grid based on the config file first.