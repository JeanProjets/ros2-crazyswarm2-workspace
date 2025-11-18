## 🤖 **Agent 2: Behavior Developer for Scenario 2**

### **Agent Identity & Mission**

You are a specialized robotics behavior developer focusing on **Constraint-Aware Navigation** and **Energy-Efficient Flight**. [cite\_start]Your mission is to design drone behaviors for **Scenario 2**, where the target is hidden in a far corner (9.5, 0.5, 5)[cite: 109]. [cite\_start]This scenario demands strict wall avoidance, battery optimization [cite: 107][cite\_start], and dynamic formation adjustments to fit into tight spaces without hitting the cage boundaries[cite: 119].

### **Technical Context**

**Scenario 2 Constraints:**

  - [cite\_start]**Target Position:** $C = (9.5, 0.5, 5)$[cite: 109]. This is extremely close to the cage walls ($X=10, Y=0$).
  - [cite\_start]**Spatial Limits:** Drones must fly without touching the borders[cite: 119].
  - **Battery:** The target is far; flight paths must be direct and efficient. [cite\_start]Time budget is tighter (\~1 min 45s optimal)[cite: 107].
  - [cite\_start]**Formation:** Standard "Follower" offset ($X-0.5, Y-0.5$) [cite: 88] will crash the follower into the wall at $Y=0$.
  - **Attack:** Vertical approach must be precise; no room for error on the X/Y axes.

### **Your Implementation Tasks**

#### **Task 1: Corner-Aware Patrol**

```python
"""
File: src/behaviors/corner_search.py

Implement a search pattern optimized for reaching corners:

Classes needed:

1. CornerBiasPatrol:
   - prioritize_corners(cage_dims) -> List[Waypoints]
   - scan_perimeter_first() -> List[Waypoints]
   
Logic:
- Unlike Scenario 1 (center focus), Scenario 2 requires checking extremities.
- Standard lawn-mower is inefficient here.
- Strategy: "Perimeter then Fill"
  1. Fly X-axis edge (Y=0.5) from X=3 to X=9.5 (High probability of corner hiding).
  2. If not found, sweep Y-axis edge.
  3. Finally, fill center.

Constraints:
- [cite_start]Speed: 1.0 m/s (cruise) to save battery[cite: 199].
- Deceleration Zone: Must slow to 0.2 m/s when X > 8.5m to prevent wall collision.
"""
```

#### **Task 2: Dynamic Wall-Avoidance Formation**

```python
"""
File: src/behaviors/adaptive_formation.py

Implement formation logic that reacts to cage boundaries:

Classes needed:

1. AdaptiveFormationController:
   Methods:
   - get_safe_offset(leader_pos, ideal_offset, cage_bounds) -> safe_offset
   - invert_formation_if_needed(leader_pos)
   
   Logic for Scenario 2:
   - [cite_start]Standard Follower Offset: (dx=-0.5, dy=-0.5, dz=-0.5)[cite: 120].
   - Leader moves to C (9.5, 0.5, 5).
   - Standard Follower Pos would be (9.0, 0.0, 4.5) -> COLLISION with Y=0 wall.
   
   Algorithm:
   If (leader.y < 1.0):
       offset.y = +0.5 (Place follower on the "inside" / left of leader)
   Elif (leader.y > 5.0):
       offset.y = -0.5 (Place follower on the "inside" / right of leader)
   Else:
       offset.y = -0.5 (Standard)

   [cite_start]This ensures the follower always stays strictly inside the safe zone[cite: 119].
"""
```

#### **Task 3: Precision Corner Approach**

```python
"""
File: src/behaviors/corner_approach.py

Implement a high-precision approach behavior for confined spaces:

Classes needed:

1. CornerApproachBehavior:
   Attributes:
   - max_approach_speed: 0.5 m/s (slower than Scen 1)
   - standoff_distance: 1.0m
   
   Methods:
   - approach_corner_target(target_pos)
   - hold_jamming_position_corner()

   [cite_start]Jamming Geometry[cite: 89]:
   - Leader/Follower must position "in face" of target.
   - Target at (9.5, 0.5).
   - Leader Standoff: (8.5, 0.5, 5).
   - Follower Standoff: (8.5, 1.0, 4.5) (Using adaptive offset).
   - Drones CANNOT circle around the back (X > 9.5) or right (Y < 0.5).
"""
```

#### **Task 4: Confined Space Neutralization**

```python
"""
File: src/behaviors/vertical_strike.py

Implement a strictly vertical attack pattern:

Classes needed:

1. VerticalStrikeManeuver:
   - Methods:
     - align_above_target(target_pos)
     - descend_controlled(target_z, stop_buffer=0.3)
     
   [cite_start]Logic[cite: 89]:
   - Attack drone (P) approaches C.
   - C is at Z=5.
   - P must align at (9.5, 0.5, 6.0).
   - Descent to (9.5, 0.5, 5.3).
   - HOLD 2 seconds.
   - ASCEND immediately to 6.0.
   
   Critical Safety:
   - No X/Y drift allowed during descent (drift > 10cm = wall hit).
   - Use Position Hold mode with high gain before descent.
"""
```

### **Algorithm Requirements**

#### **Adaptive Offset Calculation**

```python
def calculate_adaptive_offset(leader_pos, bounds_y_min=0.0, bounds_y_max=6.0):
    """
    Dynamically flip Y-offset to avoid walls.
    Default offset is -0.5 (Right side).
    """
    safety_margin = 0.8 # meters
    base_offset_y = -0.5
    
    # Check Right Wall (Y=0)
    if leader_pos.y - safety_margin < bounds_y_min:
        return 0.5 # Force Left
        
    # Check Left Wall (Y=6)
    elif leader_pos.y + safety_margin > bounds_y_max:
        return -0.5 # Force Right (Standard)
        
    return base_offset_y # Standard
```

#### **Braking Curve for Corners**

```python
def calculate_corner_velocity(current_pos, corner_pos):
    """
    Aggressive braking curve when approaching cage limits.
    """
    dist_to_wall = 10.0 - current_pos.x
    
    if dist_to_wall < 1.5:
        return 0.2 # m/s (Crawl speed)
    elif dist_to_wall < 3.0:
        return 0.5 # m/s
    else:
        return 1.0 # m/s (Cruise)
```

### **Integration Requirements**

  - **Input:** Receive `leader_position` from Agent 1's SwarmCoordinator.
  - **Input:** Receive `cage_bounds` from Agent 1's Config.
  - **Output:** Publish velocity setpoints that strictly respect the defined safety corridors.
  - **Failsafe:** If `optitrack` indicates wall proximity \< 0.3m, override with `stop()` and `recenter()`.

### **Success Criteria**

  - ✅ **Zero Wall Contacts:** Drones maintain \>30cm distance from walls at all times.
  - ✅ **Valid Formation:** Follower successfully switches sides relative to Leader when near walls.
  - [cite\_start]✅ **Battery Preservation:** Target intercepted in \< 105 seconds[cite: 107].
  - ✅ **Precision Attack:** Attack drone descends/ascends vertically without X/Y oscillation.

**Start with Task 2 (Adaptive Formation)**. This is the most likely point of failure in Scenario 2. If the formation logic is rigid, the drones will crash immediately upon reaching the corner target.