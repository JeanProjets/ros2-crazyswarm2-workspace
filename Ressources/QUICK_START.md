# ⚡ QUICK START - Copy-Paste Ready

## 🎯 IMMEDIATE ACTION (Do This Right Now)

### **Step 1: Create Project** (2 minutes)

```bash
cd /Users/jean-baptistevayssade/Documents/Squadrone/Partie_perso_v2/
mkdir drone-hackathon-2025
cd drone-hackathon-2025
code .
```

### **Step 2: First Prompt to Claude Code** (Copy-paste this)

```
START: Session 1 - DevOps Agent

You are the DevOps Agent for a drone hackathon project.

TASK: Create complete repository structure and initial files.

ARCHITECTURE: Read Section 5 and Step 1.1 from this file:
/Users/jean-baptistevayssade/Documents/Squadrone/Partie_perso_v2/Ressources/IMPLEMENTATION_ARCHITECTURE_PLAN.md

ACTIONS:
1. Create the complete directory structure from Section 5
2. Create README.md with project overview
3. Create .gitignore (Python, ROS, __pycache__, build/, install/, log/)
4. Create requirements.txt with all dependencies from Section 4.2
5. Create setup.py for Python package
6. Create scripts/setup_environment.sh (install dependencies)
7. Initialize git and make first commit

ACCEPTANCE CRITERIA:
- All directories created matching Section 5
- README.md exists with quick start
- requirements.txt has all dependencies
- Git initialized

Start by reading the architecture file, then create everything.
```

**Expected Time**: 10-15 minutes
**Expected Outcome**: Complete repo structure ready

---

## 📋 SESSION PROMPTS (Copy-Paste Ready)

### **Session 2: ROS Messages** (3-4 hours)

```
START: Session 2 - ROS Integration Agent

You are the ROS Integration Agent.

TASK: Create all custom ROS 2 message and service definitions.

ARCHITECTURE: Read Section 7, Step 1.2 from:
/Users/jean-baptistevayssade/Documents/Squadrone/Partie_perso_v2/Ressources/IMPLEMENTATION_ARCHITECTURE_PLAN.md

ACTIONS:
1. Create src/squadrone_msgs/ package structure
2. Create 5 message files (exact definitions in Step 1.2):
   - msg/RoleAssignment.msg
   - msg/MissionState.msg
   - msg/DroneStatus.msg
   - msg/DetectionReport.msg
   - msg/FormationCommand.msg
3. Create 3 service files:
   - srv/AssignRole.srv
   - srv/PlanPath.srv
   - srv/EmergencyStop.srv
4. Create CMakeLists.txt and package.xml
5. Build and test: colcon build --packages-select squadrone_msgs

ACCEPTANCE CRITERIA:
- Package compiles successfully
- All messages visible with: ros2 interface list | grep squadrone_msgs
- No errors

Start by reading Step 1.2, then implement exactly as specified.
```

---

### **Session 3: Utility Modules** (4-6 hours)

```
START: Session 3 - Core Algorithm Agent

You are the Core Algorithm Agent.

TASK: Implement all utility modules with unit tests.

ARCHITECTURE: Read Section 7, Step 1.3 from:
/Users/jean-baptistevayssade/Documents/Squadrone/Partie_perso_v2/Ressources/IMPLEMENTATION_ARCHITECTURE_PLAN.md

ACTIONS:
1. Implement these 5 modules (complete code in Step 1.3):
   - src/squadrone_swarm/squadrone_swarm/utils/transforms.py
   - src/squadrone_swarm/squadrone_swarm/utils/geometry.py
   - src/squadrone_swarm/squadrone_swarm/utils/timing.py
   - src/squadrone_swarm/squadrone_swarm/utils/logger.py
   - src/squadrone_swarm/squadrone_swarm/utils/config_loader.py

2. For EACH module, also create unit test:
   - tests/unit/test_transforms.py
   - tests/unit/test_geometry.py
   - tests/unit/test_timing.py
   - tests/unit/test_config_loader.py

3. Run: pytest tests/unit/ -v --cov=src/squadrone_swarm/squadrone_swarm/utils

ACCEPTANCE CRITERIA:
- All 5 modules implemented with docstrings and type hints
- All tests pass
- Coverage >90%
- No import errors

Read Step 1.3, implement each module with its test.
```

---

### **Session 4: Configuration Files** (2-3 hours)

```
START: Session 4 - Configuration Agent

You are the Configuration Agent.

TASK: Create all YAML configuration files.

ARCHITECTURE: Read Section 7, Step 1.4 AND Section 10 from:
- /Users/jean-baptistevayssade/Documents/Squadrone/Partie_perso_v2/Ressources/IMPLEMENTATION_ARCHITECTURE_PLAN.md
- /Users/jean-baptistevayssade/Documents/Squadrone/Partie_perso_v2/Ressources/IMPLEMENTATION_ARCHITECTURE_PLAN_PART2.md

ACTIONS:
1. Create all config files (exact specs in Step 1.4 and Section 10):
   - config/arena/test_arena.yaml
   - config/arena/competition_arena.yaml
   - config/drones/drone_params.yaml
   - config/missions/mission_v1.yaml (through mission_v4.yaml)
   - config/control/pid_params.yaml
   - config/control/formation_params.yaml
   - config/vision/detection_params.yaml

2. Validate: yamllint config/

3. Test loading: ConfigLoader.load_mission_config('config', 'v1')

ACCEPTANCE CRITERIA:
- All YAML files valid
- ConfigLoader can load all files
- Values match Section 10 specifications

Read Step 1.4 and Section 10, create all configs.
```

---

### **Session 5: State Machine** (6-8 hours)

```
START: Session 5 - Core Algorithm Agent

You are the Core Algorithm Agent.

TASK: Implement the mission finite state machine.

ARCHITECTURE: Read Section 7, Step 2.1 from:
/Users/jean-baptistevayssade/Documents/Squadrone/Partie_perso_v2/Ressources/IMPLEMENTATION_ARCHITECTURE_PLAN.md

ACTIONS:
1. Implement src/squadrone_swarm/squadrone_swarm/core/state_machine.py
   - Complete implementation in Step 2.1
   - 9 states: IDLE → TAKEOFF → PATROL → DETECT → ASSIGN → INTERCEPT → NEUTRALIZE → RTL → LANDED
   - Emergency transitions
   - Boundary checks

2. Create tests/unit/test_state_machine.py:
   - Test valid transitions
   - Test invalid transitions (should raise ValueError)
   - Test emergency triggers
   - Test boundary violations

3. Run: pytest tests/unit/test_state_machine.py -v --cov

ACCEPTANCE CRITERIA:
- All transitions validated
- Emergency conditions trigger correctly
- >90% coverage
- No race conditions

Read Step 2.1, implement state machine with full test coverage.
```

---

### **Session 6: Role Assignment** (4-5 hours)

```
START: Session 6 - Core Algorithm Agent

TASK: Implement dynamic role assignment.

ARCHITECTURE: Read Section 7, Step 2.2 from:
/Users/jean-baptistevayssade/Documents/Squadrone/Partie_perso_v2/Ressources/IMPLEMENTATION_ARCHITECTURE_PLAN.md

ACTIONS:
1. Implement src/squadrone_swarm/squadrone_swarm/core/role_manager.py
2. Create tests/unit/test_role_assignment.py
3. Run tests: pytest tests/unit/test_role_assignment.py -v

ACCEPTANCE CRITERIA:
- Highest battery neutral becomes Leader
- Reassignment works correctly
- Edge cases handled

Read Step 2.2, implement role manager.
```

---

### **Session 7: Formation Control** (6-8 hours)

```
START: Session 7 - Control Systems Agent

You are the Control Systems Agent.

TASK: Implement leader-follower formation control.

ARCHITECTURE: Read Section 7, Step 2.3 from:
/Users/jean-baptistevayssade/Documents/Squadrone/Partie_perso_v2/Ressources/IMPLEMENTATION_ARCHITECTURE_PLAN.md

ACTIONS:
1. Implement src/squadrone_swarm/squadrone_swarm/core/formation.py
2. Create tests/unit/test_formation.py
3. Validate formation geometry

ACCEPTANCE CRITERIA:
- Leader positions at standoff distance
- Follower maintains offset
- Moving target tracking works

Read Step 2.3, implement formation controller.
```

---

### **Sessions 8-18: Continue Pattern**

**Same structure for**:
- Session 8: Path Planning (Step 2.4)
- Session 9: Collision Avoidance (Step 2.5)
- Session 10: Kalman Filter (Step 2.6)
- Session 11: Trajectory Generation (Step 2.7)
- Session 12: OptiTrack Node (Part 2, Step 3.1)
- Session 13: Mission Manager Node (Part 2, Step 3.2)
- Session 14-18: Remaining ROS nodes

---

## 🔄 BETWEEN SESSIONS

### **End of Session Checklist**:

```
Before finishing this session:

1. Run final tests:
   pytest tests/unit/ -v --cov

2. Check git status:
   git status
   git diff

3. Commit work:
   git add .
   git commit -m "Session X: [description]"

4. Create handoff document:
   Create HANDOFF_SESSION_X.md with:
   - What was completed
   - What files were created
   - Test results
   - Next session instructions
   - Any blockers

5. Update PROGRESS.md:
   Mark session as complete [x]
```

---

## ✅ VALIDATION COMMANDS

### **After Phase 1**:
```bash
# Build check
colcon build --packages-select squadrone_msgs squadrone_swarm

# Source
source install/setup.bash

# Test
pytest tests/unit/ -v

# Config test
python -c "from squadrone_swarm.utils.config_loader import ConfigLoader; print(ConfigLoader.load_mission_config('config', 'v1'))"
```

### **After Phase 2**:
```bash
# All core tests
pytest tests/unit/ -v --cov=src/squadrone_swarm/squadrone_swarm/core

# Coverage report
pytest tests/unit/ --cov=src/squadrone_swarm/squadrone_swarm/core --cov-report=html

# Import check
python -c "
from squadrone_swarm.core.state_machine import MissionStateMachine
from squadrone_swarm.core.role_manager import RoleManager
from squadrone_swarm.core.formation import FormationController
print('✅ All core modules import successfully')
"
```

### **After Phase 3**:
```bash
# Launch nodes (simulation mode)
ros2 launch squadrone_swarm simulation.launch.py

# In another terminal, check:
ros2 topic list
ros2 node list
ros2 topic hz /cf_positions/cf_p

# Integration tests
pytest tests/integration/ -v
```

---

## 📊 PROGRESS TRACKER (Update Daily)

**File**: `PROGRESS.md` (root directory)

```markdown
# Progress Tracker

Updated: 2025-11-17

## Phase 1: Foundation
- [ ] Session 1: DevOps Setup (2-3h) - Target: Day 1
- [ ] Session 2: ROS Messages (3-4h) - Target: Day 2
- [ ] Session 3: Utilities (4-6h) - Target: Day 3-4
- [ ] Session 4: Configs (2-3h) - Target: Day 5

**Phase 1 Target**: Week 1-2 complete

## Phase 2: Core Algorithms
- [ ] Session 5: State Machine (6-8h)
- [ ] Session 6: Role Assignment (4-5h)
- [ ] Session 7: Formation (6-8h)
- [ ] Session 8: Path Planning (6-8h)
- [ ] Session 9: Collision (4-6h)
- [ ] Session 10: Kalman (4-6h)
- [ ] Session 11: Trajectory (4-6h)

**Phase 2 Target**: Week 3-4 complete

## Phase 3: ROS Nodes
- [ ] Session 12-18: All nodes

**Phase 3 Target**: Week 5-6 complete

## Current Session
Session 1 (not started)

## Blockers
None

## Notes
Starting fresh on 2025-11-17
```

---

## 🚨 TROUBLESHOOTING

### **Problem**: Claude Code can't find architecture file

**Solution**:
```
The file is at this exact path:
/Users/jean-baptistevayssade/Documents/Squadrone/Partie_perso_v2/Ressources/IMPLEMENTATION_ARCHITECTURE_PLAN.md

Please read it and extract Section X, Step Y.

If you still can't access it, I'll paste the relevant section directly.
```

### **Problem**: Import errors

**Solution**:
```
First check if these exist:
ls -la src/squadrone_swarm/squadrone_swarm/utils/

If files are missing, we need to implement them first.
The dependency order is in Section 6 of the architecture.

What files are currently missing?
```

### **Problem**: Tests failing

**Solution**:
```
Show me:
1. The exact error message
2. Which test is failing
3. The implementation of that function

Then fix the issue and re-run tests.
```

---

## 🎯 SUCCESS METRICS

Track these after each session:

```markdown
## Session X Complete

✅ Files created: X files
✅ Lines of code: ~X LOC
✅ Tests written: X tests
✅ Tests passing: X/X (100%)
✅ Coverage: X%
✅ Time taken: X hours
✅ Blockers: None / [list]

Next session: Session Y (estimated Xh)
```

---

## 🚀 FIRST 30 MINUTES

**Right now, do this**:

1. **Open terminal** (2 min):
   ```bash
   cd /Users/jean-baptistevayssade/Documents/Squadrone/Partie_perso_v2/
   mkdir drone-hackathon-2025
   cd drone-hackathon-2025
   code .
   ```

2. **Copy Session 1 prompt** (1 min): From above

3. **Paste into Claude Code** (1 min)

4. **Wait for structure to be created** (10-15 min)

5. **Verify**:
   ```bash
   tree -L 2
   git log --oneline
   cat README.md
   ```

6. **You're rolling!** 🚁🔥

---

## 📞 QUICK HELP

**Stuck?** Check these in order:
1. EXECUTION_GUIDE.md (detailed explanations)
2. IMPLEMENTATION_ARCHITECTURE_PLAN.md (technical details)
3. ARCHITECTURE_SUMMARY.md (big picture)
4. README_ARCHITECTURE.md (navigation)

**Fast answer?** Use this prompt:
```
I'm stuck on [problem].

Current context:
- Phase: X
- Session: Y
- Task: [description]
- Error: [if any]

Please help me:
1. Understand the issue
2. Find the relevant architecture section
3. Get back on track
```

---

**Document Version**: 1.0
**Ready to Use**: YES ✅
**Time to First Result**: 30 minutes

**START NOW** 🚀
