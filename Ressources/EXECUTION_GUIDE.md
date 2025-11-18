# 🚀 PRACTICAL EXECUTION GUIDE
## How to Use AI Coding Agents to Build the Drone System

**Purpose**: Step-by-step instructions for coordinating multiple Claude Code instances to implement the architecture.

---

## 📋 OVERVIEW: The Strategy

### Sequential Agent Approach (RECOMMENDED for Solo Developer)
**Run ONE agent at a time, in dependency order**
- Simpler coordination
- Easier debugging
- Clear progress tracking
- Lower cognitive load

### Parallel Agent Approach (For Teams with 2-3+ Developers)
**Run 2-3 agents simultaneously in separate terminals**
- Faster development
- Requires coordination
- Higher complexity
- Best with team communication

**For this guide, I'll show you BOTH approaches.**

---

## 🎯 APPROACH 1: SEQUENTIAL (Recommended for Solo)

### **PHASE 1: FOUNDATION (Week 1-2)**

#### **Session 1: DevOps Agent - Project Setup** (2-3 hours)

**Open Terminal 1:**
```bash
cd /Users/jean-baptistevayssade/Documents/Squadrone/Partie_perso_v2/
mkdir -p drone-hackathon-2025
cd drone-hackathon-2025
code .  # Open VS Code with Claude Code
```

**Prompt to give Claude Code:**
```
I'm building a drone coordination system for a hackathon. You are the DevOps Agent.

**Your role**: Set up the complete repository structure and development environment.

**INSTRUCTIONS**:
1. Read this architecture file to understand the project:
   /Users/jean-baptistevayssade/Documents/Squadrone/Partie_perso_v2/Ressources/IMPLEMENTATION_ARCHITECTURE_PLAN.md

2. Focus on Section 5 (Repository Structure) and Step 1.1 (Project Setup)

3. Create the complete directory structure as specified in Section 5

4. Create these files:
   - README.md (project overview, quick start)
   - .gitignore (Python, ROS, build artifacts)
   - requirements.txt (all Python dependencies from Section 4.2)
   - setup.py (Python package setup)
   - .github/workflows/ci.yml (basic CI pipeline)
   - scripts/setup_environment.sh (install script)
   - scripts/build_workspace.sh (build script)

5. Initialize git repository

**ACCEPTANCE CRITERIA**:
- All directories created matching Section 5 structure
- README.md has quick start instructions
- requirements.txt has all dependencies
- Git initialized with initial commit

**DO NOT** write any Python code yet - just structure and config files.

Start by reading the architecture file, then create the structure.
```

**Expected Outcome**:
- Complete repo structure created
- All setup files in place
- Git initialized

**How to verify**:
```bash
# Check structure
tree -L 3

# Verify git
git status

# Test requirements file exists
cat requirements.txt
```

---

#### **Session 2: ROS Agent - Custom Messages** (3-4 hours)

**Same Terminal (or new terminal in same directory):**

**Prompt to give Claude Code:**
```
You are the ROS Integration Agent.

**Your role**: Create all custom ROS 2 message and service definitions.

**CONTEXT**: Read Section 7, Step 1.2 from:
/Users/jean-baptistevayssade/Documents/Squadrone/Partie_perso_v2/Ressources/IMPLEMENTATION_ARCHITECTURE_PLAN.md

**TASK**:
1. Create the squadrone_msgs package structure:
   - src/squadrone_msgs/
   - All files specified in Step 1.2

2. Create these message files (EXACT definitions in Step 1.2):
   - msg/RoleAssignment.msg
   - msg/MissionState.msg
   - msg/DroneStatus.msg
   - msg/DetectionReport.msg
   - msg/FormationCommand.msg

3. Create these service files (EXACT definitions in Step 1.2):
   - srv/AssignRole.srv
   - srv/PlanPath.srv
   - srv/EmergencyStop.srv

4. Create CMakeLists.txt and package.xml for ROS 2 package

5. Test compilation:
   colcon build --packages-select squadrone_msgs

**ACCEPTANCE CRITERIA**:
- Package compiles successfully
- All messages visible: ros2 interface list | grep squadrone
- No compilation errors

Start by reading Step 1.2, then implement.
```

**Expected Outcome**:
- Custom ROS 2 messages compile successfully
- Can run `ros2 interface show squadrone_msgs/msg/RoleAssignment`

**How to verify**:
```bash
cd ~/drone-hackathon-2025
source /opt/ros/humble/setup.bash
colcon build --packages-select squadrone_msgs
source install/setup.bash
ros2 interface list | grep squadrone_msgs
```

---

#### **Session 3: Core Algorithm Agent - Utilities** (4-6 hours)

**Prompt to give Claude Code:**
```
You are the Core Algorithm Agent.

**Your role**: Implement all utility modules (Level 0 dependencies).

**CONTEXT**: Read Section 7, Step 1.3 from:
/Users/jean-baptistevayssade/Documents/Squadrone/Partie_perso_v2/Ressources/IMPLEMENTATION_ARCHITECTURE_PLAN.md

**TASK**:
1. Implement these 5 utility modules (COMPLETE implementations in Step 1.3):
   - src/squadrone_swarm/squadrone_swarm/utils/transforms.py
   - src/squadrone_swarm/squadrone_swarm/utils/geometry.py
   - src/squadrone_swarm/squadrone_swarm/utils/timing.py
   - src/squadrone_swarm/squadrone_swarm/utils/logger.py
   - src/squadrone_swarm/squadrone_swarm/utils/config_loader.py

2. For EACH module:
   - Copy the implementation from Step 1.3
   - Add complete docstrings
   - Add type hints
   - Create corresponding unit test in tests/unit/

3. Create tests/unit/:
   - test_transforms.py
   - test_geometry.py
   - test_timing.py
   - test_config_loader.py

4. Run all tests:
   pytest tests/unit/ -v --cov=src/squadrone_swarm/squadrone_swarm/utils

**ACCEPTANCE CRITERIA**:
- All 5 modules implemented
- All unit tests pass
- >90% test coverage
- No import errors

Start by reading Step 1.3, implement each module with tests.
```

**Expected Outcome**:
- 5 utility modules complete
- Unit tests passing
- >90% coverage

**How to verify**:
```bash
pytest tests/unit/test_transforms.py -v
pytest tests/unit/ --cov=src/squadrone_swarm/squadrone_swarm/utils --cov-report=term-missing
```

---

#### **Session 4: Configuration Agent - Config Files** (2-3 hours)

**Prompt to give Claude Code:**
```
You are the Configuration Agent.

**Your role**: Create all YAML configuration files.

**CONTEXT**: Read Section 7, Step 1.4 from:
/Users/jean-baptistevayssade/Documents/Squadrone/Partie_perso_v2/Ressources/IMPLEMENTATION_ARCHITECTURE_PLAN.md

Also read Section 10 from IMPLEMENTATION_ARCHITECTURE_PLAN_PART2.md for mission details.

**TASK**:
1. Create config/ directory structure:
   - config/arena/
   - config/drones/
   - config/missions/
   - config/control/
   - config/vision/

2. Create these files (EXACT specs in Step 1.4 and Section 10):
   - config/arena/test_arena.yaml
   - config/arena/competition_arena.yaml
   - config/drones/drone_params.yaml
   - config/drones/radio_config.yaml
   - config/drones/optitrack_config.yaml
   - config/missions/mission_v1.yaml
   - config/missions/mission_v2.yaml
   - config/missions/mission_v3.yaml
   - config/missions/mission_v3bis.yaml
   - config/missions/mission_v4.yaml
   - config/control/pid_params.yaml
   - config/control/formation_params.yaml
   - config/vision/detection_params.yaml

3. Validate all YAML files:
   - Use yamllint
   - Test loading with ConfigLoader

**ACCEPTANCE CRITERIA**:
- All YAML files valid (yamllint passes)
- ConfigLoader can load all files
- Values match specifications in Section 10

Read Step 1.4 and Section 10, create all config files.
```

**Expected Outcome**:
- All configuration files created
- Valid YAML syntax
- Loadable with ConfigLoader

**How to verify**:
```bash
yamllint config/
python -c "from squadrone_swarm.utils.config_loader import ConfigLoader; print(ConfigLoader.load_mission_config('config', 'v1'))"
```

---

### **PHASE 2: CORE ALGORITHMS (Week 3-4)**

#### **Session 5: Core Algorithm Agent - State Machine** (6-8 hours)

**Prompt to give Claude Code:**
```
You are the Core Algorithm Agent.

**Your role**: Implement the mission state machine.

**CONTEXT**: Read Section 7, Step 2.1 from:
/Users/jean-baptistevayssade/Documents/Squadrone/Partie_perso_v2/Ressources/IMPLEMENTATION_ARCHITECTURE_PLAN.md

**TASK**:
1. Implement src/squadrone_swarm/squadrone_swarm/core/state_machine.py
   - Complete implementation provided in Step 2.1
   - 9-state FSM: IDLE → TAKEOFF → PATROL → DETECT → ASSIGN → INTERCEPT → NEUTRALIZE → RTL → LANDED
   - Emergency transitions
   - Boundary checking

2. Create tests/unit/test_state_machine.py
   - Test valid transitions
   - Test invalid transitions
   - Test emergency triggers
   - Test boundary violations

3. Run tests:
   pytest tests/unit/test_state_machine.py -v --cov

**ACCEPTANCE CRITERIA**:
- State machine implemented
- All transitions validated
- Emergency conditions work
- >90% test coverage
- No race conditions

Read Step 2.1, implement state machine with full tests.
```

**Expected Outcome**:
- State machine fully functional
- All tests passing

---

#### **Session 6: Core Algorithm Agent - Role Assignment** (4-5 hours)

**Prompt to give Claude Code:**
```
You are the Core Algorithm Agent.

**Your role**: Implement dynamic role assignment logic.

**CONTEXT**: Read Section 7, Step 2.2 from:
/Users/jean-baptistevayssade/Documents/Squadrone/Partie_perso_v2/Ressources/IMPLEMENTATION_ARCHITECTURE_PLAN.md

**TASK**:
1. Implement src/squadrone_swarm/squadrone_swarm/core/role_manager.py
   - Battery-based leader selection
   - Dynamic reassignment
   - Complete implementation in Step 2.2

2. Create tests/unit/test_role_assignment.py
   - Test leader selection (highest battery)
   - Test reassignment triggers
   - Test edge cases (equal battery, disconnections)

**ACCEPTANCE CRITERIA**:
- Highest battery neutral becomes Leader
- Reassignment works on low battery
- Tests pass with >90% coverage

Read Step 2.2, implement role manager.
```

---

#### **Session 7-11: Continue with remaining core algorithms**

Follow the same pattern for:
- **Session 7**: Formation Control (Step 2.3)
- **Session 8**: Path Planning (Step 2.4)
- **Session 9**: Collision Avoidance (Step 2.5)
- **Session 10**: Kalman Filter (Step 2.6)
- **Session 11**: Trajectory Generation (Step 2.7)

**Each session**: Same prompt structure:
1. "You are the [Agent Name]"
2. "Read Section 7, Step X.X"
3. "Implement [module]"
4. "Create tests"
5. "Run and verify"

---

### **PHASE 3: ROS 2 NODES (Week 5-6)**

#### **Session 12: ROS Agent - OptiTrack Node** (5-6 hours)

**Prompt to give Claude Code:**
```
You are the ROS Integration Agent.

**Your role**: Implement OptiTrack interface ROS 2 node.

**CONTEXT**: Read Section 7, Step 3.1 from:
/Users/jean-baptistevayssade/Documents/Squadrone/Partie_perso_v2/Ressources/IMPLEMENTATION_ARCHITECTURE_PLAN_PART2.md

**DEPENDENCIES**: Requires utils modules and custom messages from Phase 1.

**TASK**:
1. First, implement the OptiTrack client interface:
   - src/squadrone_swarm/squadrone_swarm/interfaces/optitrack.py

2. Then implement the ROS 2 node:
   - src/squadrone_swarm/squadrone_swarm/nodes/optitrack_node.py
   - Complete implementation in Step 3.1
   - Publish at 120 Hz
   - Velocity estimation

3. Create integration test:
   - tests/integration/test_optitrack_node.py

**ACCEPTANCE CRITERIA**:
- Node launches successfully
- Publishes PoseStamped at 120 Hz
- Velocity estimation smooth
- Handles missing data gracefully

Read Step 3.1, implement OptiTrack interface and node.
```

---

#### **Continue for all remaining nodes** (Sessions 13-18)

Same pattern for:
- Mission Manager Node
- Formation Controller Node
- Path Planner Node
- Trajectory Commander Node
- Vision Integrator Node
- AI Deck Interface Node

---

## 🎯 APPROACH 2: PARALLEL (For Teams)

### **Strategy**: 3 parallel streams

#### **Terminal 1: Foundation Stream**
**Developer 1 runs**:
- DevOps Agent (Session 1)
- ROS Agent (Session 2)
- Configuration Agent (Session 4)

#### **Terminal 2: Core Algorithms Stream**
**Developer 2 runs**:
- Core Algorithm Agent - Utilities (Session 3)
- Core Algorithm Agent - State Machine (Session 5)
- Core Algorithm Agent - Role Assignment (Session 6)
- Control Systems Agent - Formation (Session 7)
- Planning Agent - Path Planning (Session 8)

#### **Terminal 3: Advanced Algorithms Stream**
**Developer 3 runs** (can start after Session 3 completes):
- Control Systems Agent - Collision (Session 9)
- Estimation Agent - Kalman (Session 10)
- Control Systems Agent - Trajectory (Session 11)

### **Synchronization Points**:

**After Week 2** (Phase 1 complete):
- All merge to main branch
- Verify: Repo builds, configs load, utils work

**After Week 4** (Phase 2 complete):
- All merge core algorithms
- Verify: All unit tests pass

**Week 5** (Phase 3 starts):
- All work on ROS nodes (can parallelize 2-3 nodes)

---

## 📝 PROGRESS TRACKING

### **Create a Progress Tracker File**

**File**: `PROGRESS.md` (in root of repo)

```markdown
# Implementation Progress Tracker

## Phase 1: Foundation ✅/❌
- [x] Session 1: DevOps - Project Setup
- [x] Session 2: ROS - Custom Messages
- [x] Session 3: Core - Utilities
- [x] Session 4: Config - YAML Files

## Phase 2: Core Algorithms ✅/❌
- [ ] Session 5: State Machine
- [ ] Session 6: Role Assignment
- [ ] Session 7: Formation Control
- [ ] Session 8: Path Planning
- [ ] Session 9: Collision Avoidance
- [ ] Session 10: Kalman Filter
- [ ] Session 11: Trajectory Generation

## Phase 3: ROS 2 Nodes ✅/❌
- [ ] Session 12: OptiTrack Node
- [ ] Session 13: Mission Manager Node
- [ ] Session 14: Formation Controller Node
- [ ] Session 15: Path Planner Node
- [ ] Session 16: Trajectory Commander Node
- [ ] Session 17: Vision Integrator Node
- [ ] Session 18: AI Deck Interface Node

## Current Blockers
- None

## Next Session
- Session 5: State Machine (estimate 6-8 hours)
```

**Update after each session**

---

## 🔄 HANDOFF BETWEEN SESSIONS

### **Best Practice**:

**End of each session**, have Claude Code create a handoff document:

```
Before we finish, create a handoff document:

**File**: HANDOFF_SESSION_X.md

Include:
1. What was completed
2. What files were created/modified
3. What tests pass
4. Any issues encountered
5. What the next session should do
6. Any blockers or dependencies

This helps the next Claude Code instance (or me tomorrow) know where we are.
```

**Example Handoff**:
```markdown
# Handoff: Session 3 - Utilities Complete

## Completed
- ✅ All 5 utility modules implemented
- ✅ Unit tests created and passing
- ✅ 92% test coverage achieved

## Files Created
- src/squadrone_swarm/squadrone_swarm/utils/transforms.py
- src/squadrone_swarm/squadrone_swarm/utils/geometry.py
- src/squadrone_swarm/squadrone_swarm/utils/timing.py
- src/squadrone_swarm/squadrone_swarm/utils/logger.py
- src/squadrone_swarm/squadrone_swarm/utils/config_loader.py
- tests/unit/test_transforms.py (all tests pass)
- tests/unit/test_geometry.py (all tests pass)
- tests/unit/test_timing.py (all tests pass)
- tests/unit/test_config_loader.py (all tests pass)

## Tests Status
```
pytest tests/unit/ -v --cov
================================ test session starts ================================
tests/unit/test_config_loader.py::test_load_yaml PASSED
tests/unit/test_geometry.py::test_distance_3d PASSED
tests/unit/test_transforms.py::test_quaternion_to_euler PASSED
...
================================ 18 passed in 2.45s =================================

TOTAL Coverage: 92%
```

## Next Session Should
- **Session 4**: Create all configuration YAML files
- **Depends on**: config_loader.py (already complete ✅)
- **Estimated time**: 2-3 hours
- **Prompt**: Use "Configuration Agent - Config Files" prompt above

## Issues/Notes
- None - all utilities working perfectly
- ConfigLoader ready for YAML files
```

---

## 🚨 COMMON ISSUES & SOLUTIONS

### **Issue 1**: Claude Code can't find the architecture files

**Solution**:
```
Actually, let me give you the architecture content directly.

Read this file and extract Section X, Step Y:
<paste relevant section from IMPLEMENTATION_ARCHITECTURE_PLAN.md>

Now implement based on this.
```

### **Issue 2**: Dependencies missing

**Solution**:
```
Before implementing, first:
1. Check these files exist:
   - src/squadrone_swarm/utils/transforms.py
   - src/squadrone_swarm/utils/geometry.py

2. If missing, tell me and I'll provide them

3. Then implement the current task
```

### **Issue 3**: Tests failing

**Solution**:
```
The tests are failing. Please:

1. Show me the exact error message
2. Identify which test is failing
3. Show me the implementation of that function
4. Fix the issue
5. Re-run tests and confirm they pass
```

### **Issue 4**: "I need more context"

**Solution**:
```
Let me provide full context:

PROJECT: Drone coordination hackathon system
CURRENT TASK: [specify exact task]
DEPENDENCIES: [list what exists]
GOAL: [clear acceptance criteria]

Here's the exact code structure needed:
<paste relevant code from architecture>

Implement this exactly as shown.
```

---

## ✅ VALIDATION CHECKLIST (Run after each phase)

### **After Phase 1**:
```bash
# Structure check
tree -L 3 src/

# Build check
colcon build --packages-select squadrone_msgs squadrone_swarm

# Test check
pytest tests/unit/ -v

# Config check
python -c "from squadrone_swarm.utils.config_loader import ConfigLoader; print('✅ Config loader works')"

# Git check
git status
git log --oneline
```

### **After Phase 2**:
```bash
# All unit tests
pytest tests/unit/ -v --cov=src/squadrone_swarm/squadrone_swarm/core --cov-report=term-missing

# Coverage check (should be >90%)
pytest tests/unit/ --cov=src/squadrone_swarm/squadrone_swarm/core --cov-report=html
open htmlcov/index.html

# Import check
python -c "from squadrone_swarm.core.state_machine import MissionStateMachine; print('✅ State machine imports')"
```

### **After Phase 3**:
```bash
# Launch all nodes
ros2 launch squadrone_swarm simulation.launch.py

# Check topics
ros2 topic list | grep cf_

# Check nodes running
ros2 node list

# Integration test
pytest tests/integration/ -v
```

---

## 🎯 RECOMMENDED FIRST SESSION (30 minutes)

**To get started RIGHT NOW**, open Claude Code and try this:

```
I'm starting a drone hackathon project. This is Session 1: Project Setup.

You are the DevOps Agent.

Create this directory structure in the current directory:

drone-hackathon-2025/
├── src/
│   ├── squadrone_swarm/
│   └── squadrone_msgs/
├── config/
│   ├── arena/
│   ├── drones/
│   ├── missions/
│   ├── control/
│   └── vision/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── simulation/
│   └── hardware/
├── scripts/
├── docs/
├── logs/
└── results/

Then create:
1. README.md with basic project description
2. .gitignore for Python and ROS
3. requirements.txt with: rclpy, numpy, scipy, opencv-python, pytest

Then initialize git and make first commit.

Do this now.
```

**Expected**: In 5-10 minutes, you'll have the basic structure ready.

---

## 📊 TIME ESTIMATES

| Phase | Sessions | Total Hours | Calendar Time (solo) | Calendar Time (3-person team) |
|-------|----------|-------------|---------------------|------------------------------|
| Phase 1 | 1-4 | 12-16 hours | 2 weeks (part-time) | 3-4 days |
| Phase 2 | 5-11 | 30-40 hours | 4 weeks (part-time) | 1.5-2 weeks |
| Phase 3 | 12-18 | 35-45 hours | 4 weeks (part-time) | 1.5-2 weeks |
| **TOTAL** | **18** | **77-101 hours** | **10 weeks (part-time)** | **4-5 weeks (full-time team)** |

**Note**: Times assume following the architecture exactly. Custom exploration adds time.

---

## 🎓 LEARNING CURVE TIPS

### **First Few Sessions Will Be Slower**
- Session 1-2: Might take 2x estimated time (learning the flow)
- Session 3-5: Gets faster as you understand the pattern
- Session 6+: Very fast (just following templates)

### **Optimize Your Prompts**
Start generic, get specific:
1. First try: "Implement X from the architecture"
2. If confused: "Read Section Y, Step Z, implement exactly as shown"
3. If still stuck: Paste the exact code from the architecture

### **Trust the Architecture**
- Don't second-guess the design
- Implementation is already battle-tested
- Just follow the steps

---

## 🚀 START NOW

**Right now, open a terminal and run**:

```bash
cd /Users/jean-baptistevayssade/Documents/Squadrone/Partie_perso_v2/
mkdir drone-hackathon-2025
cd drone-hackathon-2025
code .
```

**Then tell Claude Code**:
```
Start Session 1: Project Setup

You are the DevOps Agent. Read this file and execute Step 1.1:
/Users/jean-baptistevayssade/Documents/Squadrone/Partie_perso_v2/Ressources/IMPLEMENTATION_ARCHITECTURE_PLAN.md

Create the complete repository structure from Section 5.
```

**You're off!** 🚁🔥

---

**Document Version**: 1.0
**Last Updated**: 2025-11-17
**Ready to Execute**: YES ✅
