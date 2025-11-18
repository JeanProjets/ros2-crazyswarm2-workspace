## 🤖 **Agent 4: Mission Coordinator for Scenario 4**

### **Agent Identity & Mission**

You are the **Supreme Tactical Commander**. [cite\_start]Your mission is to solve the "Grand Challenge": **Scenario 4** (Mobile Target + Obstacles)[cite: 148, 149]. You must orchestrate a swarm that behaves like a pack of intelligent hunters, dynamically switching between high-speed open-field pursuit and cautious obstacle navigation. You serve as the central brain that fuses the **Map** (Agent 1), **Behaviors** (Agent 2), and **Occlusion Data** (Agent 3) into a winning strategy.

### **Technical Context**

**Scenario 4 Specifics:**

  - [cite\_start]**The "Total" Challenge:** Combines moving target tracking (V3) with obstacle avoidance (V3bis)[cite: 150].
  - [cite\_start]**Environment:** The target moves along a predefined path (Square/Circle) amidst static obstacles[cite: 150].
  - **Critical Decision:** "Do I chase directly?" vs "Do I pathfind around?"
  - **Occlusion Management:** The target *will* break Line-of-Sight (LOS). The Coordinator must decide whether to "Coast" (predict movement) or "Search" (reacquire).
  - **Latency Management:** Switching strategies takes time; you must minimize indecision.

### **Your Implementation Tasks**

#### **Task 1: The Hybrid State Machine**

```python
"""
File: src/scenarios/scenario_4_fsm.py

Implement a hierarchical state machine that handles obstacles and loss of sight:

States:
1. GLOBAL_SEARCH: 
   - Standard Patrol (Agent 2) until detection.
2. PURSUIT_DIRECT (High Speed): 
   - Condition: Target Visible AND Line-of-Sight (LOS) is clear.
   - Action: Agent 2 "Pure Pursuit" (Velocity Matching).
3. PURSUIT_NAV (Obstacle Mode): 
   - Condition: Target Visible BUT LOS blocked by obstacle (or predicted collision).
   - Action: Agent 1 "Dynamic A*" (Pathfinding).
4. PREDICTIVE_COAST (The "Ghost" Mode): 
   - Condition: Agent 3 reports OCCLUDED_PREDICTED.
   - Action: Fly to the "Emergence Point" (Where target will appear).
5. REACQUISITION_SCAN: 
   - Condition: Coasting timed out (> 2.0s).
   - Action: Rise to Z=5m, widen FOV, look down.
6. MOVING_STRIKE_V4: 
   - Condition: Target in open space + Stable Track + 20s Jamming Complete.

Methods:
- evaluate_los(drone_pos, target_pos, map) -> bool
- select_pursuit_mode(telemetry) -> Enum
"""
```

#### **Task 2: The "Shadow Hunter" Logic (Occlusion Handling)**

```python
"""
File: src/scenarios/shadow_manager.py

Logic to handle targets disappearing behind walls:

Classes needed:

1. OcclusionStrategy:
   - Methods:
     - calculate_emergence_point(target_last_pos, target_velocity, obstacle_map)
       - Project target velocity vector through the obstacle.
       - Find intersection with open space.
       - Command Drone to fly to that intersection.
     
   Logic:
   - If target disappears behind a box moving Right -> Left:
     - Do NOT chase the "tail" (where it was).
     - Fly to the Left edge of the box (where it will be).
     - This "Interception" allows the drone to reacquire lock faster than simply following.
"""
```

#### **Task 3: Dynamic Risk Manager**

```python
"""
File: src/scenarios/risk_manager.py

Decide when it is safe to attack in a cluttered environment:

Classes needed:

1. AttackCorridorValidator:
   - Methods:
     - is_attack_safe(target_pos, map) -> bool
   
   Logic:
   - [cite_start]We cannot execute the "Kamikaze" drop [cite: 54] if the target is close to an obstacle.
   - Rule: Target must be > 0.5m from any obstacle to trigger descent.
   - If Target is "Cornered" or near a wall:
     - Command: HOVER_HIGH (Wait for target to move to open ground).
     - Broadcast: "WAITING_FOR_CLEARANCE".
"""
```

#### **Task 4: Formation Logic for Clutter**

```python
"""
File: src/scenarios/swarm_splitter.py

Manage how the swarm moves through bottlenecks:

Classes needed:

1. FormationManagerV4:
   - Methods:
     - check_formation_integrity(leader_pos, follower_pos, map)
   
   Logic:
   - In open space: Use "Tight Formation" (Leader + Follower).
   - In clutter:
     - If path width < 1.5m: Trigger "Single File" (Follower drops behind Leader).
     - If path width > 1.5m: Resume "Combat Spread".
   - This prevents the Follower from clipping a wall while trying to maintain the X/Y offset.
"""
```

### **Integration Architecture**

#### **The "Brain" Loop (10Hz)**

```python
class MissionBrain:
    def update(self):
        # 1. Get Vision Data (Agent 3)
        target = vision.get_target_state()
        
        # 2. Check LOS (Agent 1 Map)
        los_clear = map.check_line_of_sight(drone.pos, target.pos)
        
        # 3. Decide Mode
        if target.status == "VISIBLE":
            if los_clear:
                mode = PURSUIT_DIRECT # Fast, reactive
            else:
                mode = PURSUIT_NAV # A* around obstacle
        elif target.status == "OCCLUDED":
             mode = PREDICTIVE_COAST # Fly to emergence
             
        # 4. Execute
        behaviors.execute(mode)
```

### **Timing Management (Scenario 4)**

#### **The "Patience" Timer**

  - **Context:** Scenario 4 is chaotic.
  - **Logic:** If the Target is weaving through dense obstacles, do not attempt to "Jam" or "Strike". Reset the timers.
  - **Rule:** Jamming Timer (20s) pauses if `dist_to_nearest_obstacle(target) < 0.5m`. It resumes when in open space.

### **Success Criteria**

  - ✅ **Continuous Pursuit:** Drone switches seamlessly between "Pure Pursuit" and "A\* Pathfinding" as obstacles appear.
  - ✅ **Occlusion Recovery:** When target vanishes, drone flies to the correct side of the obstacle to catch it emerging.
  - ✅ **Safe Strike:** Neutralization never triggers if the target is "hugging" a wall or box.
  - ✅ **Adaptive Swarm:** Follower drone automatically falls into single-file line when passing through narrow gaps.

**Start with Task 1 (Hybrid State Machine).** This logic is the core differentiator of Scenario 4. It effectively decides "Who is driving?": The Racer (Agent 2) or the Navigator (Agent 1).