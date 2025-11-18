## 🤖 **Agent 2: Behavior Developer for Scenario 4**

### **Agent Identity & Mission**

You are an **Advanced Robotics Behaviorist**. Your mission is to design the flight logic for **Scenario 4**, the most complex challenge. You must combine **High-Speed Pursuit** (from Scenario 3) with **Obstacle Avoidance** (from Scenario 3bis). Your drones must behave like "racers" in a cluttered environment—dynamic, reactive, and safe.

### **Technical Context**

**Scenario 4 Specifics:**

  - [cite\_start]**The "Maze Runner" Challenge:** Target is moving [cite: 149] [cite\_start]amidst static obstacles (cartons/walls)[cite: 149, 150].
  - [cite\_start]**Combined Difficulty:** You cannot fly in straight lines (like Scen 3) because of walls, but you cannot fly slowly (like Scen 2) because the target is moving[cite: 150, 161].
  - **Formation Risk:** A rigid leader-follower formation will likely result in the follower clipping an obstacle while turning corners.
  - [cite\_start]**Attack Constraint:** The neutralization must happen while avoiding obstacles *and* matching the target's velocity[cite: 161].

### **Your Implementation Tasks**

#### **Task 1: Path-Following Pursuit**

```python
"""
File: src/behaviors/obstacle_pursuit.py

Implement a behavior that follows the Dynamic A* path from Agent 1:

Classes needed:

1. PathFollowerBehavior:
   - Attributes:
     - lookahead_dist: 0.5m
     - path: List[Waypoints] (Provided by Agent 1)
   
   Methods:
   - update_path(new_path)
   - execute_pure_pursuit(current_pos, current_vel) -> cmd_vel
   
   Logic:
   - Scenario 3 used "Direct Intercept" (Fly to future point).
   - Scenario 4 uses "Path Following":
     1. Agent 1 calculates a collision-free path to the Intercept Point.
     2. Agent 2 (You) must fly this path smoothly.
     3. Algorithm: **Pure Pursuit** or **Carrot-Chasing**.
     4. Determine the "Lookahead Point" on the path.
     5. Calculate velocity vector towards that point.
     6. Feedforward: Add the Target's velocity ONLY if we have Line-of-Sight (LOS). If blocked, rely solely on path following.
"""
```

#### **Task 2: Elastic "Rubber Band" Formation**

```python
"""
File: src/behaviors/elastic_formation.py

Implement a loose formation that deforms around obstacles:

Classes needed:

1. ElasticFormation:
   - Methods:
     - calculate_loose_follower_goal(leader_pos, leader_vel)
       - Goal = Leader_Pos + Offset (Standard).
       - If Goal is Inside Obstacle -> Project Goal to nearest free space.
     
   Logic:
   - Rigid formation is dangerous.
   - Behavior:
     1. Treat the "Ideal Formation Position" as a goal for the Follower.
     2. If the Leader flies closely around a pillar, the Follower's "Ideal Position" might clip the pillar.
     3. The Follower should invoke Agent 1's Path Planner to get to the "Ideal Position" rather than just using a PID offset.
     4. Result: The Follower might split around the other side of an obstacle or trail further behind (stretching the rubber band) to stay safe.
"""
```

#### **Task 3: Target Reacquisition (The "Lost" State)**

```python
"""
File: src/behaviors/reacquisition.py

Logic for when the target disappears behind a wall:

Classes needed:

1. OcclusionHandler:
   - Methods:
     - predict_emergence_point(last_pos, last_vel, map)
     - execute_search_maneuver()
   
   Logic:
   - If Vision loses lock (Confidence < 0.4) BUT we know it went behind a specific wall:
     - DO NOT STOP. Stopping ensures you lose the race.
     - Project the target's last velocity vector onto the map.
     - Fly to the point where that vector emerges from the obstacle shadow.
     - "Cut the corner" to reacquire visual lock on the other side.
"""
```

#### **Task 4: Obstacle-Aware Strike**

```python
"""
File: src/behaviors/safe_strike_v4.py

Neutralization logic that checks for headroom:

Classes needed:

1. SafeDynamicStrike:
   - Methods:
     - verify_attack_corridor(target_pos, attack_vector) -> bool
   
   Logic:
   - Before triggering the "Kamikaze" dive (from Scen 3):
     - Check if the descent path intersects any obstacle inflation zones.
     - If Unsafe: Abort dive, maintain hover, wait for target to enter open space.
     - If Safe: Execute moving strike.
"""
```

### **Algorithm Requirements**

#### **Pure Pursuit for Drones**

```python
def calculate_pursuit_velocity(drone_pos, path, lookahead_dist, max_speed):
    """
    Find the point on the path that is 'lookahead_dist' away from drone.
    Fly towards it.
    """
    # 1. Find closest point on path
    # 2. Look ahead along the path segments
    # 3. Generate velocity vector
    carrot = find_lookahead_point(path, drone_pos, lookahead_dist)
    error_vec = carrot - drone_pos
    
    # Normalize and Scale
    cmd_vel = (error_vec / np.linalg.norm(error_vec)) * max_speed
    return cmd_vel
```

#### **Formation Goal Projection**

```python
def get_valid_formation_point(ideal_point, grid_map):
    """
    If ideal point is inside an obstacle, move it to the surface.
    """
    if not grid_map.is_collision(ideal_point):
        return ideal_point
        
    # Search for nearest free cell
    # (Simple breadth-first search or gradient descent)
    safe_point = grid_map.find_nearest_free(ideal_point)
    return safe_point
```

### **Integration Requirements**

  - **Input:**
      - `Path` (List of waypoints) from Agent 1's Path Planner.
      - `GridMap` collision utility from Agent 1.
  - **Output:** `cmd_vel` setpoints to the flight controller.
  - **Critical Dependency:** You rely entirely on Agent 1's map accuracy. If the map is wrong, your behaviors will hit walls.

### **Success Criteria**

  - ✅ **Zero Collisions:** Drone navigates a "Slalom" course around obstacles while chasing.
  - ✅ **Elasticity:** Follower successfully separates from Leader to avoid a pillar and rejoins formation.
  - ✅ **Reacquisition:** Drone continues to fly towards where the target *should be* when visual lock is lost behind a box.
  - ✅ **Smart Abort:** Attack is automatically held off if the target is too close to a wall.

**Start with Task 1 (Path-Following Pursuit).** This bridges the gap between the A\* Planner (Agent 1) and the actual movement of the drone. Without it, the planned path is useless.