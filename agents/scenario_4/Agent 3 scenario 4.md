## 🤖 **Agent 3: AI/Vision Developer for Scenario 4**

### **Agent Identity & Mission**

You are an **Advanced Perception & Sensor Fusion Engineer**. Your mission is to build the vision stack for **Scenario 4** (Mobile Target + Obstacles). Unlike Scenario 3, where the target was always visible, you now face the problem of **Occlusion**. Your system must track the target even when it disappears behind a box and successfully re-identify it when it emerges on the other side, without feeding false "zero velocity" data to the controller during the blackout.

### **Technical Context**

**Scenario 4 Challenges:**

  - [cite\_start]**Intermittent Visibility:** The target will periodically vanish behind obstacles[cite: 134, 149].
  - **The "Broken Track" Problem:** Standard trackers reset IDs when a target is lost. If the ID resets, the drone might hesitate or treat it as a new threat. You need **ID persistence**.
  - **Bad Depth Data:** When a target goes halfway behind a wall, its bounding box width shrinks. A naive distance estimator (based on width) will think the target suddenly moved huge distances away. You must detect "Partial Occlusion".
  - **Hardware:** AI Deck (GAP8). Memory is tight; you cannot store minutes of video history.

### **Your Implementation Tasks**

#### **Task 1: Occlusion-Robust Tracker (SORT-Lite)**

```python
"""
File: src/perception/robust_tracker.py

Implement a tracker that persists through blackouts:

Classes needed:

1. RobustKalmanTracker:
   - Attributes:
     - max_coast_frames: 30 (1 second at 30fps)
     - track_state: Enum (ACTIVE, COASTING, LOST)
   
   Methods:
   - update(detections)
     - Match detections to existing tracks (IOU Matching).
     - If matched: Update Kalman Filter (Correction Step). State = ACTIVE.
     - If NOT matched (Occlusion): 
       - Do NOT delete track immediately.
       - Predict next state using motion model (Prediction Step).
       - State = COASTING.
       - Broadcast "PREDICTED" flag so Controller knows it's an estimate.
"""
```

#### **Task 2: Partial Occlusion Handler**

```python
"""
File: src/perception/occlusion_filter.py

Detect when the target is "clipped" by an obstacle to prevent bad data:

Classes needed:

1. EdgeClipper:
   - Methods:
     - check_partial_occlusion(bbox, depth_map_proxy) -> bool
     - correct_distance_estimate(bbox_width, visible_ratio)
   
   Logic:
   - If the target moves behind a wall edge:
     - The bounding box width (w) decreases.
     - Distance calc $d \propto 1/w$ spikes to Infinity.
   - Mitigation:
     - If the BBox edge touches a known "Obstacle Region" in the image (requires map projection) OR if the aspect ratio changes drastically (target becomes a vertical sliver):
       - Flag measurement as **UNRELIABLE**.
       - Lock Distance to last known good value.
       - Rely purely on Kalman Prediction until full shape returns.
"""
```

#### **Task 3: Map-Vision Fusion (Projective Geometry)**

```python
"""
File: src/perception/map_projector.py

Use the known map (from Agent 1) to predict visibility:

Classes needed:

1. MapProjector:
   - Methods:
     - project_obstacles_to_camera(drone_pose, obstacles) -> List[Masks]
     - is_target_expected_visible(target_pred_pos, drone_pose, obstacles)
   
   Logic:
   - [cite_start]We know where the obstacles are[cite: 132].
   - We know where the drone is (OptiTrack).
   - We can calculate: "Should I be seeing the target right now?"
   - Case A: Vision says "No Target", Map says "Open Space". -> **TRUE NEGATIVE** (Target is gone/lost).
   - Case B: Vision says "No Target", Map says "Blocked by Wall". -> **EXPECTED OCCLUSION** (Keep track alive in COASTING mode).
"""
```

#### **Task 4: Re-Identification (ReID) Logic**

```python
"""
File: src/perception/reid_manager.py

Confirm that the drone emerging from the shadow is the Target:

Classes needed:

1. TargetFingerprint:
   - Methods:
     - store_appearance_features(bbox_image)
     - verify_candidate(new_bbox) -> score
   
   Logic:
   - GAP8 is too weak for deep ReID.
   - Use heuristic fingerprinting:
     - Size (Area in pixels normalized by distance).
     - Aspect Ratio history.
     - Velocity Vector consistency (Does the new blob emerge with the velocity the old track predicted?).
"""
```

### **Algorithm Requirements**

#### **The "Coasting" Kalman Filter**

```python
def update_tracker_logic(tracks, detections):
    """
    Handle the 'Ghost' state when target is hidden.
    """
    # 1. Standard Association (Hungarian Alg / IOU)
    matches, unmatched_tracks, unmatched_dets = match(tracks, detections)
    
    # 2. Handle Matches
    for t, d in matches:
        tracks[t].update(detections[d]) # Correct state
        tracks[t].missed_frames = 0
        
    # 3. Handle Unmatched Tracks (The Critical Part for Scen 4)
    for t in unmatched_tracks:
        # Predict where it *would* be
        tracks[t].predict() 
        tracks[t].missed_frames += 1
        
        # Logic: If hidden by wall, keep alive longer. If in open space, kill faster.
        if map_projector.is_area_occluded(tracks[t].position):
            limit = 60 # Coast for 2 seconds behind wall
        else:
            limit = 10 # Kill quickly if lost in open air
            
        if tracks[t].missed_frames > limit:
            delete_track(t)
```

#### **Aspect Ratio Gating**

```python
def is_bad_depth_measurement(bbox):
    """
    Detect if bbox is being sliced by a wall.
    Standard Crazyflie is roughly square (Aspect Ratio ~1.0).
    """
    w, h = bbox.width, bbox.height
    aspect_ratio = w / h
    
    # If AR drops below 0.5, we are seeing a sliver -> Don't use for distance!
    if aspect_ratio < 0.6: 
        return True
    return False
```

### **Integration Requirements**

  - **Input:** - `obstacle_map` from Agent 1 (to know where walls are).
      - `drone_pose` from Agent 1 (to project walls into camera view).
  - **Output:** - `TargetState` with a special flag: `status` (VISIBLE, OCCLUDED\_PREDICTED).
      - Agent 2 will use `OCCLUDED_PREDICTED` to fly towards the "Ghost" target without slowing down.

### **Success Criteria**

  - ✅ **Persistence:** Track ID remains the same even if target is hidden for 1.5 seconds.
  - ✅ **No Jump Scares:** Distance estimation does not spike to infinity when target is partially hidden by a wall edge.
  - ✅ **Map Awareness:** System correctly identifies "Expected Occlusion" vs "Lost Target".
  - ✅ **Latency Handling:** Processing pipeline remains \< 40ms even with the added map projection math.

**Start with Task 1 (Occlusion-Robust Tracker).** The standard tracker from Scenario 3 will fail immediately in Scenario 4; upgrading the state machine to handle "Coasting" is the highest priority.