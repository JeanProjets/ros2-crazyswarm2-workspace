## 🤖 **Agent 4: Mission Coordinator for Scenario 1**

### **Agent Identity & Mission**

You are a specialized mission orchestration developer implementing high-level coordination and state management for a Crazyflie drone swarm. Your mission is to create the intelligent decision-making layer that sequences mission phases, manages role assignments, and ensures successful completion of Scenario 1.

### **Technical Context**

**Mission Requirements:**
- Coordinate 3 drones through 9 mission phases
- Dynamic role assignment based on battery levels
- Real-time state transitions based on sensor feedback
- Mission completion in < 3 minutes
- Graceful handling of failures and contingencies

**Scenario 1 Timeline:**
- Initialization: 10 seconds
- Safety zone check: 30 seconds
- Patrol search: 60-90 seconds
- Target approach: 30 seconds
- Jamming: 20 seconds
- Neutralization: 5 seconds
- Total: < 180 seconds

### **Your Implementation Tasks**

#### **Task 1: Mission State Machine**

```python
"""
File: src/scenarios/mission_state_machine.py

Implement the core mission state machine:

Classes needed:

1. MissionState(Enum):
   - INITIALIZATION = "init"
   - SAFETY_CHECK = "safety_check"
   - PATROL_SEARCH = "patrol_search"
   - TARGET_DETECTED = "target_detected"
   - ROLE_ASSIGNMENT = "role_assignment"
   - APPROACH_TARGET = "approach_target"
   - JAMMING = "jamming"
   - NEUTRALIZATION = "neutralization"
   - MISSION_COMPLETE = "complete"
   - MISSION_ABORT = "abort"

2. MissionStateMachine:
   Attributes:
   - current_state: MissionState
   - start_time: float
   - phase_timers: Dict[MissionState, float]
   - drone_roles: Dict[str, str]
   
   Methods:
   - transition_to(new_state: MissionState) -> bool
   - check_transition_conditions() -> Optional[MissionState]
   - execute_state_actions(state: MissionState)
   - get_mission_elapsed_time() -> float
   - is_mission_timeout() -> bool
   - abort_mission(reason: str)
   
State transition conditions:
- INIT -> SAFETY_CHECK: All drones airborne
- SAFETY_CHECK -> PATROL_SEARCH: Zone clear (30s timeout)
- PATROL_SEARCH -> TARGET_DETECTED: Vision detection
- TARGET_DETECTED -> ROLE_ASSIGNMENT: Automatic (1s)
- ROLE_ASSIGNMENT -> APPROACH_TARGET: Roles assigned
- APPROACH_TARGET -> JAMMING: Formation in position
- JAMMING -> NEUTRALIZATION: 20s elapsed
- NEUTRALIZATION -> MISSION_COMPLETE: Attack complete

Include timeout handling for each state.
"""
```

#### **Task 2: Role Assignment System**

```python
"""
File: src/scenarios/role_manager.py

Implement dynamic role assignment:

Classes needed:

1. DroneRole(Enum):
   - NEUTRAL_1 = "neutral_1"
   - NEUTRAL_2 = "neutral_2"
   - PATROL = "patrol"
   - LEADER = "leader"
   - FOLLOWER = "follower"
   - ATTACKER = "attacker"

2. RoleManager:
   Attributes:
   - drone_registry: Dict[str, DroneInfo]
   - current_assignments: Dict[str, DroneRole]
   - role_history: List[RoleAssignment]
   
   Methods:
   - register_drone(drone_id, capabilities)
   - assign_initial_roles() -> Dict[str, DroneRole]
   - reassign_roles_on_detection(detector_id) -> Dict[str, DroneRole]
   - select_leader(drone_states) -> str
   - get_role_position(role, mission_phase) -> Tuple[float, float, float]
   - validate_role_assignment() -> bool
   
3. DroneInfo(dataclass):
   - drone_id: str
   - battery_level: float
   - position: Tuple[float, float, float]
   - status: str
   - has_detected_target: bool
   - last_update: float

Role assignment logic for Scenario 1:
1. Initial assignment:
   - cf1: NEUTRAL_1
   - cf2: NEUTRAL_2  
   - cf3: PATROL

2. After target detection:
   - Highest battery neutral -> LEADER
   - Other neutral -> FOLLOWER
   - Patrol -> ATTACKER

Battery-based selection:
- Query battery levels from all drones
- Select leader with max(battery) from neutrals
- Ensure leader has > 40% battery
- Fallback if batteries too low

Include role transition timing and coordination.
"""
```

#### **Task 3: Mission Sequencer**

```python
"""
File: src/scenarios/scenario_1_mission.py

Implement the complete mission sequence:

Classes needed:

1. Scenario1Mission:
   Attributes:
   - state_machine: MissionStateMachine
   - role_manager: RoleManager
   - drone_controllers: Dict[str, DroneController]
   - behavior_executors: Dict[str, BehaviorExecutor]
   - vision_systems: Dict[str, VisionSystem]
   
   Methods:
   - initialize_mission(config_file)
   - execute_mission() -> MissionResult
   - execute_phase(phase: MissionState)
   - coordinate_drone_actions(actions: List[DroneAction])
   - monitor_mission_health() -> HealthStatus
   - handle_contingency(event_type: str)

2. PhaseExecutor:
   Methods for each phase:
   - execute_initialization()
   - execute_safety_check()  
   - execute_patrol_search()
   - execute_target_approach()
   - execute_jamming()
   - execute_neutralization()

Phase implementations:

INITIALIZATION (10s):
- Launch all drones to hover positions
- Verify OptiTrack tracking
- Confirm communication links
- Start mission timer

SAFETY_CHECK (30s):
- Neutrals sweep safety zone
- Parallel paths to avoid collision
- Report zone status

PATROL_SEARCH (60-90s):
- Patrol executes search pattern
- Neutrals maintain ready positions
- Monitor for target detection

TARGET_APPROACH (30s):
- Leader navigates to jamming position
- Follower maintains formation
- Attacker moves to attack position

JAMMING (20s):
- Hold positions in front of target
- Simulate RF interference
- Prepare for final attack

NEUTRALIZATION (5s):
- Attacker descends on target
- Stop 30cm above (safety)
- Signal mission success

Include telemetry logging for post-mission analysis.
"""
```

#### **Task 4: Coordination & Communication**

```python
"""
File: src/scenarios/mission_coordinator.py

Implement swarm coordination layer:

Classes needed:

1. MissionCoordinator:
   Methods:
   - synchronize_drones(sync_point: str)
   - broadcast_mission_state(state: MissionState)
   - collect_drone_telemetry() -> Dict[str, DroneTelemetry]
   - aggregate_sensor_data() -> SensorFusion
   - make_tactical_decision(situation: dict) -> Decision
   
2. DecisionEngine:
   Methods:
   - evaluate_mission_progress() -> float
   - determine_next_action(current_state) -> Action
   - assess_risk_level() -> RiskLevel
   - recommend_abort() -> Tuple[bool, str]
   
3. TelemetryAggregator:
   Methods:
   - merge_position_data(positions: List) -> ConsolidatedPosition
   - combine_detection_confidence(detections: List) -> float
   - calculate_swarm_center() -> Position
   - estimate_mission_completion() -> float

Communication protocols:
- State broadcasts: 5 Hz
- Telemetry collection: 10 Hz
- Emergency signals: Immediate
- Role changes: Synchronized

ROS2 Topics:
- /mission/state
- /mission/role_assignments
- /mission/abort
- /swarm/telemetry
- /swarm/synchronization

Decision rules:
- Abort if any drone battery < 20%
- Abort if drone loses tracking > 5s
- Abort if collision imminent
- Continue if 2/3 drones operational

Synchronization points:
- After takeoff
- Before role change
- Start of approach
- Start of jamming
"""
```

### **Integration Architecture**

#### **Component Orchestration**
```python
# How Agent 4 coordinates other agents' components

class MissionOrchestrator:
    def __init__(self):
        # Agent 1 components
        self.drone_controllers = {}  # DroneController instances
        self.swarm_coordinator = None  # SwarmCoordinator instance
        
        # Agent 2 components  
        self.patrol_behaviors = {}  # PatrolBehavior instances
        self.formation_controller = None  # FormationController
        
        # Agent 3 components
        self.vision_systems = {}  # TargetDetector instances
        self.detection_broadcaster = None  # DetectionBroadcaster
        
        # Agent 4 components
        self.state_machine = MissionStateMachine()
        self.role_manager = RoleManager()
        
    def orchestrate_mission_phase(self, phase: MissionState):
        """Coordinate all agents for current phase"""
        if phase == MissionState.PATROL_SEARCH:
            # Use Agent 2's patrol behavior
            self.patrol_behaviors['cf3'].execute_search_pattern()
            
            # Monitor with Agent 3's vision
            detections = self.vision_systems['cf3'].detect_drone()
            
            # Control with Agent 1's controller
            if detections:
                self.swarm_coordinator.broadcast_drone_status(...)
```

### **Timing Management**

#### **Mission Timeline Controller**
```python
class MissionTimer:
    """Precise timing for mission phases"""
    
    PHASE_TIMEOUTS = {
        MissionState.INITIALIZATION: 10,
        MissionState.SAFETY_CHECK: 30,
        MissionState.PATROL_SEARCH: 90,
        MissionState.APPROACH_TARGET: 30,
        MissionState.JAMMING: 20,
        MissionState.NEUTRALIZATION: 5,
    }
    
    def __init__(self):
        self.mission_start = None
        self.phase_start = None
        self.phase_history = []
        
    def start_phase(self, phase: MissionState):
        self.phase_start = time.time()
        
    def is_phase_timeout(self, phase: MissionState) -> bool:
        elapsed = time.time() - self.phase_start
        return elapsed > self.PHASE_TIMEOUTS.get(phase, float('inf'))
```

### **Contingency Handling**

#### **Failure Recovery**
```python
class ContingencyHandler:
    """Handle mission failures gracefully"""
    
    def handle_drone_failure(self, drone_id: str, failure_type: str):
        """Adapt mission with reduced drones"""
        if failure_type == "battery_critical":
            # Land immediately
            self.emergency_land(drone_id)
            # Reassign roles if possible
            self.adapt_roles_for_remaining_drones()
            
    def handle_detection_failure(self):
        """If target not found in time"""
        # Move all drones to line formation
        # Sweep forward together
        # Last resort search pattern
        
    def handle_communication_loss(self, drone_id: str):
        """Lost contact with drone"""
        # Wait 5 seconds
        # If not recovered, abort mission
        # Land all drones safely
```

### **Example Implementation**

```python
# src/scenarios/scenario_1_mission.py
import asyncio
import time
from enum import Enum
from typing import Dict, Optional, List
import logging

class Scenario1Mission:
    def __init__(self, config_path: str):
        self.logger = logging.getLogger("Scenario1Mission")
        self.config = self.load_config(config_path)
        
        # Initialize all components
        self.state_machine = MissionStateMachine()
        self.role_manager = RoleManager()
        self.mission_start_time = None
        
        # Component references (injected)
        self.drone_controllers = {}
        self.behavior_executors = {}
        self.vision_systems = {}
        
    async def execute_mission(self) -> MissionResult:
        """Main mission execution loop"""
        self.mission_start_time = time.time()
        self.logger.info("Starting Scenario 1 mission")
        
        try:
            # Initialize
            await self.execute_initialization()
            
            # Main mission loop
            while not self.is_mission_complete():
                current_state = self.state_machine.current_state
                
                # Check for state transition
                next_state = self.state_machine.check_transition_conditions()
                if next_state:
                    self.state_machine.transition_to(next_state)
                    
                # Execute current state actions
                await self.execute_phase(current_state)
                
                # Monitor health
                if not self.is_mission_healthy():
                    self.abort_mission("Health check failed")
                    break
                    
                # Small delay to prevent CPU overload
                await asyncio.sleep(0.1)
                
            return self.generate_mission_result()
            
        except Exception as e:
            self.logger.error(f"Mission failed: {e}")
            self.abort_mission(str(e))
            return MissionResult(success=False, reason=str(e))
    
    async def execute_phase(self, phase: MissionState):
        """Execute actions for current phase"""
        if phase == MissionState.SAFETY_CHECK:
            await self.execute_safety_check()
        elif phase == MissionState.PATROL_SEARCH:
            await self.execute_patrol_search()
        elif phase == MissionState.APPROACH_TARGET:
            await self.execute_target_approach()
        # ... other phases
        
    async def execute_safety_check(self):
        """Coordinate safety zone check"""
        # Get neutral drones
        neutrals = self.role_manager.get_drones_by_role([
            DroneRole.NEUTRAL_1, 
            DroneRole.NEUTRAL_2
        ])
        
        # Execute parallel sweeps
        tasks = []
        for drone_id in neutrals:
            behavior = self.behavior_executors[drone_id]
            task = behavior.execute_safety_sweep()
            tasks.append(task)
            
        # Wait for completion
        await asyncio.gather(*tasks)
        
        # Check if zone is clear
        if self.is_safety_zone_clear():
            self.logger.info("Safety zone clear")
```

### **Success Criteria**

- ✅ All mission phases execute in sequence
- ✅ Smooth transitions between states (< 1s)
- ✅ Dynamic role assignment works correctly
- ✅ Mission completes in < 3 minutes
- ✅ Graceful handling of failures
- ✅ Clear telemetry and logging

### **Testing Requirements**

1. Unit tests for state machine transitions
2. Integration tests with all agent components
3. Simulated failure scenarios
4. Timing verification for each phase
5. Role assignment under various battery levels

Begin with Task 1 (State Machine) as it forms the backbone of mission control. Focus on clear state transitions and robust error handling. Remember: in the demo, recovering gracefully from errors is better than perfect execution that breaks catastrophically!