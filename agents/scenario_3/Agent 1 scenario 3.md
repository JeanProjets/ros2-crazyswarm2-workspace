## 🤖 **Agent 1: Core Systems Developer for Scenario 3**

### **Agent Identity & Mission**

You are a **Control Systems & Dynamics Specialist**. Your mission is to upgrade the core flight stack for **Scenario 3** (Mobile Target). [cite\_start]The target is no longer static; it moves in a \~3m circle/square pattern in the center of the arena[cite: 102]. [cite\_start]You must implement **predictive tracking**, **velocity feedforward control**, and **dynamic formation keeping** to ensure the swarm can jam and neutralize a moving enemy[cite: 112, 113].

### **Technical Context**

**Scenario 3 Specifics:**

  - [cite\_start]**Target Behavior:** Moving target (C) executing circles/squares in the center[cite: 100, 102].
  - [cite\_start]**Mission Constraint:** "Jamming must be done in movement"[cite: 113].
  - **Latency Challenge:** Targeting where the drone *was* results in chasing the tail. You must target where it *will be*.
  - [cite\_start]**Fallback:** If target is not found during patrol, drones must line up at the 3m mark to scan[cite: 115].

**Hardware Context:**

  - Crazyflie 2.1+ dynamics.
  - Input: Target position $(x, y, z)$ updated at \~10Hz via Vision/OptiTrack.
  - Output: Smooth velocity/position commands to match target speed.

### **Your Implementation Tasks**

#### **Task 1: Dynamic Tracking Controller**

```python
"""
File: src/core/tracking_controller.py

Implement a controller capable of smooth pursuit:

Classes needed:

1. DynamicTracker(DroneController):
   Methods:
   - update_target_state(pos, vel)
   - compute_intercept_vector(drone_pos, target_pos, target_vel) -> velocity_cmd
   - match_velocity_hover(target_vel) 
   
   Logic:
   - Standard 'go_to' is insufficient for moving targets (causes lag).
   - Implement a Proportional Navigation (PN) or "Lead Pursuit" logic.
   - Lead Point Calculation: 
     target_future = target_pos + (target_vel * lookahead_time)
     lookahead_time = distance / drone_speed
   
   Control Mode:
   - Use `cmdVelocityWorld` (vx, vy, vz, yaw_rate) for smoother tracking than `cmdPosition`.
"""
```

#### **Task 2: Motion-Compensated Swarm Manager**

```python
"""
File: src/core/swarm_manager_v3.py

Upgrade SwarmCoordinator to handle a moving frame of reference:

Requirements:
1. Relative Formation on the Move:
   - In Scen 1/2, Leader was static. Now Leader is moving.
   - Follower Setpoint = Leader_Pos(t) + Offset.
   - PROBLEM: If Leader brakes, Follower might crash into it due to comms lag.
   - SOLUTION: Feedforward Leader's velocity to Follower.
   
2. `coordinate_moving_formation(leader_id, follower_id)`:
   - Get Leader Velocity (vx, vy).
   - Follower Cmd = (Leader_Pos + Offset) + (Leader_Vel * damping).
   
3. Role Handoff in Motion:
   - Ensure role switching (Leader/Follower) doesn't cause a sudden stop.
"""
```

#### **Task 3: Scenario 3 Sequencer (Search & Intercept)**

```python
"""
File: src/scenarios/scenario_3_mobile.py

Implement the state machine for dynamic interception:

States:
1. [cite_start]PATROL_SEARCH: Standard Lawnmower[cite: 101].
2. FALLBACK_SCAN: 
   - If Patrol finishes with no detection:
   - [cite_start]Line up 3 drones at X=3m (N1, P, N2)[cite: 115].
   - [cite_start]Hover and rotate yaw to scan "Hostile Zone"[cite: 115].
3. DYNAMIC_APPROACH: 
   - Leader intercepts Target trajectory. 
   - [cite_start]Follower maintains relative position (moving jamming)[cite: 113].
4. MOVING_JAM: 
   - [cite_start]Match velocity with target for 20s[cite: 113].
   - Tolerance: Keep within 1.0m radius of moving target.
5. INTERCEPTION_STRIKE: 
   - [cite_start]"Semi-kamikaze" on moving target[cite: 113].
   - Predict impact point, dive, and pull up.

"""
```

#### **Task 4: Configuration & Physics**

```python
"""
File: config/scenario_3_config.yaml

Define physics constraints for moving targets:

target_dynamics:
  estimated_speed: 0.5 # m/s (Assumed based on 3m circle)
  [cite_start]pattern_type: "CIRCLE_OR_SQUARE" [cite: 102]
  prediction_lookahead: 0.5 # seconds

tracking_gains:
  kp_pos: 1.5
  kd_vel: 0.5
  max_vel: 1.5 # m/s (Must be faster than target)

[cite_start]fallback_positions: [cite: 117, 118, 119]
  # "Les 3 drones se placent sur la ligne des 3 m"
  N1: [3.0, 1.5, 4.0] 
  N2: [3.0, 3.0, 4.0]
  P:  [3.0, 4.5, 4.0]
"""
```

### **Code Quality Requirements**

1.  **Velocity Feedforward:** Your control loop must use velocity estimates. Sending static waypoints to a moving target will fail.
2.  **Safety Limits:** Cap max velocity to 2.0 m/s to prevent loss of control during aggressive pursuit.
3.  **Smoothness:** Use low-pass filters on the Target Velocity estimate to prevent jittery drone movements.

### **Example Starting Code Structure**

```python
# src/core/tracking_controller.py
import numpy as np

class TrackingController:
    def compute_lead_pursuit(self, drone_pos, target_pos, target_vel, speed_gain=1.0):
        """
        Calculate velocity command to intercept moving target
        """
        # 1. Distance to target
        dist_vector = np.array(target_pos) - np.array(drone_pos)
        dist = np.linalg.norm(dist_vector)
        
        # 2. Estimate time to intercept
        time_to_go = dist / speed_gain
        
        # 3. Predict future target position
        # Cap prediction time to avoid overshooting
        pred_time = np.clip(time_to_go, 0.1, 1.0)
        future_target = np.array(target_pos) + (np.array(target_vel) * pred_time)
        
        # 4. Calculate command vector
        cmd_vector = future_target - np.array(drone_pos)
        
        # Normalize and scale
        if np.linalg.norm(cmd_vector) > 0:
            cmd_vel = (cmd_vector / np.linalg.norm(cmd_vector)) * speed_gain
        else:
            cmd_vel = [0, 0, 0]
            
        return cmd_vel
```

### **Success Criteria**

  - ✅ **No Oscillations:** Drones track the target smoothly without "orbiting" or wobbling.
  - [cite\_start]✅ **Moving Jam:** Leader stays within 1m of the target while both are moving at \~0.5 m/s[cite: 112, 113].
  - [cite\_start]✅ **Fallback Trigger:** If target is invisible for \> 60s, drones correctly assemble at the X=3m line[cite: 115].
  - ✅ **Collision Avoidance:** Follower does not crash into Leader when Leader brakes to match target speed.