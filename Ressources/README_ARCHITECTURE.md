# 🚁 DRONE HACKATHON ARCHITECTURE DOCUMENTATION

## Complete Implementation Guide for AI Coding Agents

**Project**: Drone Defense Hackathon 2025 - Challenge 2: Cooperate to Succeed
**Team**: Squadrone
**Created**: 2025-11-17

---

## 📂 Documentation Files

### 1. **START HERE** → `ARCHITECTURE_SUMMARY.md`
**Purpose**: Executive overview and quick reference
**Read Time**: 10 minutes
**Audience**: Everyone

**Contents**:
- Document structure overview
- Key numbers and metrics
- Quick start guide for AI agents
- Critical path timeline
- Success criteria for each phase
- Common pitfalls to avoid

---

### 2. **DETAILED PLAN PART 1** → `IMPLEMENTATION_ARCHITECTURE_PLAN.md`
**Purpose**: Foundation and core algorithms (Sections 1-7)
**Read Time**: 60 minutes
**Audience**: All AI coding agents

**Contents**:

#### Section 1: Executive Summary
- Mission objective
- Critical success factors
- Development timeline

#### Section 2: Project Context & Constraints
- Hardware specifications (Crazyflie, AI Deck, OptiTrack)
- Arena dimensions and constraints
- Drone roles (P, N1, N2, C, L, S)
- 4 mission versions (V1→V2→V3→V4)
- Battery and timing constraints

#### Section 3: System Architecture Overview
- High-level architecture diagram
- Data flow architecture
- Communication architecture (ROS 2 topics)
- Ground station ↔ Drones interaction

#### Section 4: Technology Stack Decisions
- Core technologies (ROS 2 Humble, Python 3.10, Gazebo)
- Libraries and dependencies
- Simulation strategy (Gazebo primary, Webots backup)

#### Section 5: Repository Structure
- **Complete file tree** with 100+ files
- Directory organization
- Package structure
- Configuration hierarchy

#### Section 6: Module Breakdown & Dependencies
- 6-level dependency graph
- Critical path analysis
- Parallel development opportunities
- Integration risk points

#### Section 7: Detailed Implementation Plan (Phases 1-2)

**PHASE 1: FOUNDATION** (Weeks 1-2)
- Step 1.1: Project Setup (DevOps Agent)
  - Repository structure
  - CI/CD pipeline
  - Build system
- Step 1.2: Custom ROS 2 Messages (ROS Agent)
  - 5 message types defined
  - 3 service types defined
- Step 1.3: Utility Modules (Core Algorithm Agent)
  - `utils/transforms.py` - Coordinate transformations
  - `utils/geometry.py` - Geometric calculations
  - `utils/timing.py` - Time synchronization
  - `utils/logger.py` - Logging utilities
  - `utils/config_loader.py` - YAML configuration
- Step 1.4: Configuration Files (Configuration Agent)
  - Arena configs (test and competition)
  - Drone parameters
  - Mission specifications (V1-V4)
  - Control parameters

**PHASE 2: CORE ALGORITHMS** (Weeks 3-4)
- Step 2.1: State Machine (Core Algorithm Agent)
  - 9-state FSM implementation
  - Transition validation
  - Emergency handling
- Step 2.2: Role Assignment (Core Algorithm Agent)
  - Battery-based leader selection
  - Dynamic reassignment
- Step 2.3: Formation Control (Control Systems Agent)
  - Leader-Follower mathematics
  - Standoff distance calculation
  - Moving target tracking
- Step 2.4: Path Planning (Planning Agent)
  - A* implementation
  - 3D grid-based planning
  - B-spline smoothing
- Step 2.5: Collision Avoidance (Control Systems Agent)
  - Reciprocal Velocity Obstacles (RVO)
  - Prediction horizon
  - Safe waypoint generation
- Step 2.6: Kalman Filter (Estimation Agent)
  - 3D state estimation
  - Position + velocity tracking
  - Future position prediction
- Step 2.7: Trajectory Generation (Control Systems Agent)
  - Trapezoidal velocity profiles
  - Multi-waypoint trajectories
  - Dynamic feasibility

**Implementation Details**:
- Complete Python code for each module
- Unit test specifications
- Acceptance criteria
- Performance requirements

---

### 3. **DETAILED PLAN PART 2** → `IMPLEMENTATION_ARCHITECTURE_PLAN_PART2.md`
**Purpose**: ROS nodes, agents, testing, deployment (Sections 7-12)
**Read Time**: 45 minutes
**Audience**: Integration specialists, testers, deployment engineers

**Contents**:

#### Section 7 (Continued): Detailed Implementation Plan (Phase 3)

**PHASE 3: ROS 2 NODES** (Weeks 5-6)
- Step 3.1: OptiTrack Interface Node
  - NatNet SDK integration
  - 120 Hz position publishing
  - Velocity estimation
- Step 3.2: Mission Manager Node
  - FSM orchestration
  - Detection handling
  - State broadcasting
- Additional nodes:
  - Formation Controller Node
  - Path Planner Node
  - Trajectory Commander Node
  - Vision Integrator Node
  - AI Deck Interface Node

#### Section 8: AI Agent Definitions & Assignments
**12 Specialized Agents**:

1. **DevOps Agent** - Repo, CI/CD, build systems
2. **ROS Integration Agent** - All ROS 2 nodes and launch files
3. **Core Algorithm Agent** - State machine, utilities, algorithms
4. **Control Systems Agent** - Formation, collision, trajectories
5. **Planning Agent** - A* path planning, obstacles
6. **Estimation Agent** - Kalman filtering, tracking
7. **Vision Agent** - AI Deck, PyTorch training, TFLite
8. **Embedded Systems Agent** - GAP8 firmware (C code)
9. **Simulation Agent** - Gazebo worlds, plugins
10. **Test Agent** - pytest, coverage, validation
11. **Configuration Agent** - YAML files, parameters
12. **Documentation Agent** - README, guides, docs

**For Each Agent**:
- Clear responsibilities
- Tool requirements
- Assigned tasks with file paths
- Dependencies on other agents

**Coordination Strategy**:
- Sequential dependencies
- Parallel work streams
- Integration timeline

#### Section 9: Testing Strategy

**Test Pyramid**:
- Unit Tests (>90% coverage target)
- Integration Tests (multi-node)
- Simulation Tests (full missions)
- Hardware Smoke Tests

**Test Files**:
- 30+ test modules specified
- Mock fixtures and utilities
- Acceptance criteria per module

**Testing Timeline**:
- Continuous unit testing (every commit)
- Integration tests (end of each phase)
- Simulation validation (Week 6)
- Hardware smoke tests (Week 7)

#### Section 10: Mission Specifications (Detailed)

**Mission V1**: Static Center Target
- Time budget: 180s (3 min)
- 8 phases with durations
- Initial positions
- Success criteria

**Mission V2**: Static Corner Target
- Time budget: 210s (3.5 min)
- Boundary constraint handling
- Adjusted formation

**Mission V3**: Moving Circular Target
- Time budget: 240s (4 min)
- Kalman filter tracking
- Predictive interception
- Target movement pattern

**Mission V3bis**: Static with Obstacles
- Time budget: 240s (4 min)
- 3 obstacle specifications
- A* path planning
- Replanning triggers

**Mission V4**: Moving + Obstacles
- Time budget: 300s (5 min)
- Full system complexity
- Dynamic replanning

#### Section 11: Integration Points

**4 Critical Integration Points**:
- **IP-1**: OptiTrack → ROS 2 (High risk)
- **IP-2**: Crazyflie ↔ Ground Station (High risk)
- **IP-3**: AI Deck ↔ Ground Station (Medium risk)
- **IP-4**: Crazyswarm2 Integration (Medium risk)

**For Each IP**:
- Risk assessment
- Testing procedures
- Integration steps
- Validation criteria

#### Section 12: Risk Mitigation

**Risk Matrix**:
- 10 key risks identified
- Probability and impact ratings
- Specific mitigation strategies

**Contingency Plans**:
- OptiTrack failure fallbacks
- AI Deck failure fallbacks
- Battery optimization strategies
- Simulation unavailability plan

---

## 🎯 How to Use This Documentation

### For AI Coding Agents:

**Step 1**: Read `ARCHITECTURE_SUMMARY.md` (10 min)
- Understand overall project
- Find your agent role
- See the big picture

**Step 2**: Read relevant sections from Part 1
- Section 5: Know what files to create
- Section 6: Understand dependencies
- Section 7 (your phase): Implementation details

**Step 3**: Read Section 8 in Part 2
- Your specific assignments
- Tools you'll need
- Dependencies on other agents

**Step 4**: Follow implementation plan
- Create files in dependency order
- Write unit tests (>90% coverage)
- Validate before moving to next task

**Step 5**: Integrate (Section 11)
- Test integration points
- Validate end-to-end
- Document any issues

---

## 📊 Document Statistics

| Metric | Value |
|--------|-------|
| **Total Pages** | ~50 pages |
| **Total Words** | ~25,000 words |
| **Code Examples** | 30+ complete implementations |
| **File Specifications** | 100+ files defined |
| **Test Specs** | 30+ test modules |
| **Diagrams** | 10+ architecture diagrams |
| **Configuration Files** | 15+ YAML specs |

---

## 🚀 Quick Reference

### Critical Files to Read First:
1. ✅ `ARCHITECTURE_SUMMARY.md` - Start here (everyone)
2. ✅ `IMPLEMENTATION_ARCHITECTURE_PLAN.md` Section 5 - Repository structure
3. ✅ `IMPLEMENTATION_ARCHITECTURE_PLAN.md` Section 7 - Your phase
4. ✅ `IMPLEMENTATION_ARCHITECTURE_PLAN_PART2.md` Section 8 - Your agent role

### Key Diagrams:
- Section 3.1: High-Level Architecture
- Section 3.2: Data Flow Architecture
- Section 3.3: Communication Architecture (ROS topics)
- Section 6.1: Module Dependency Graph

### Important Tables:
- Section 2.1: Hardware Specifications
- Section 2.4: Mission Version Comparison
- Section 4.1: Technology Stack
- Section 8: Agent Assignments
- Section 10: Mission Phase Timings
- Section 12: Risk Matrix

---

## 📁 Additional Resources (From Original Docs)

Located in same directory:

1. **`Hackathon_CR_scenario (1).pdf`**
   - Official mission scenarios
   - Hardware specifications
   - Arena dimensions
   - Mission timing requirements

2. **`Criteres_evalutation.pdf`**
   - Evaluation criteria (4×5 points)
   - Sub-challenges list
   - Judging guidelines

3. **`Documentation_mentor.md`**
   - Mentor guidance
   - Recommended approaches
   - Crazyswarm2 references

4. **`deroule_hackathon.md`**
   - Hackathon schedule
   - Logistics
   - Key deadlines

5. **`Tech_lead_role_no_agents.md`**
   - Original technical lead role definition
   - Architecture principles

---

## ⚡ Quick Start Commands

### Setup Environment:
```bash
# Clone repository
git clone https://github.com/squadrone-naval/hackathon.git
cd hackathon

# Run setup script
./scripts/setup_environment.sh

# Build workspace
./scripts/build_workspace.sh

# Run tests
pytest tests/unit -v --cov
```

### Simulation:
```bash
# Launch Mission V1 simulation
ros2 launch squadrone_swarm mission_v1.launch.py

# Monitor topics
ros2 topic list
ros2 topic echo /swarm/mission_state

# Check rates
ros2 topic hz /cf_positions/cf_p
```

### Hardware:
```bash
# Flash AI Deck
./scripts/flash_aideck.sh

# Run smoke tests
pytest tests/hardware/smoke_tests.py

# Launch hardware mission
ros2 launch squadrone_swarm hardware.launch.py mission:=v1
```

---

## ✅ Validation Checklist

Before proceeding to next phase:

### Phase 1 Complete:
- [ ] Repository builds successfully
- [ ] All custom messages compile
- [ ] Utility modules have >90% test coverage
- [ ] Configuration files load without errors
- [ ] CI pipeline passes

### Phase 2 Complete:
- [ ] All core algorithm unit tests pass
- [ ] State machine transitions correctly
- [ ] Formation control validated with mock data
- [ ] Path planning finds solutions in <50ms
- [ ] Collision avoidance prevents collisions
- [ ] Kalman filter tracks with <0.3m error

### Phase 3 Complete:
- [ ] All ROS 2 nodes launch successfully
- [ ] Topics publish at correct rates
- [ ] Mission state machine operates automatically
- [ ] OptiTrack integration working
- [ ] Crazyflie commands sent successfully

### Integration Complete:
- [ ] Mission V1 succeeds in simulation (8/10)
- [ ] No collisions in 100 sim runs
- [ ] Time budget satisfied
- [ ] All integration points validated

### Hardware Ready:
- [ ] Smoke tests pass
- [ ] OptiTrack tracks all drones
- [ ] AI Deck detects at 5 Hz
- [ ] Hardware Mission V1 succeeds (3/5)

---

## 🎓 Learning Path

### Week 1-2: Learn ROS 2 + Crazyflie Basics
- ROS 2 tutorials
- Crazyflie documentation
- Crazyswarm2 examples

### Week 3-4: Study Control Theory
- Formation control
- Collision avoidance
- Path planning algorithms

### Week 5-6: Master Integration
- Multi-node ROS systems
- Hardware interfacing
- Debugging distributed systems

### Week 7: Practice, Practice, Practice
- Simulation testing
- Hardware validation
- Emergency procedures

---

## 📞 Support

**Questions about architecture?**
- Review relevant sections in detail
- Check diagrams in Section 3
- Consult dependency graph in Section 6

**Questions about implementation?**
- See code examples in Section 7
- Check acceptance criteria
- Review unit tests

**Questions about your role?**
- Read your agent definition in Section 8
- Check assigned tasks
- Review dependencies

**Stuck on integration?**
- See Section 11 for integration points
- Check risk mitigation in Section 12
- Test incrementally

---

## 🏆 Success Factors

**Critical Success Factors** (must achieve):
1. ✅ Follow dependency order strictly
2. ✅ Write tests for everything (>90%)
3. ✅ Validate in simulation before hardware
4. ✅ Start integration early (Week 5)
5. ✅ Battery-efficient trajectories
6. ✅ Robust collision avoidance
7. ✅ Emergency failsafes working

**Nice-to-Have Factors** (competitive edge):
- Reach Mission V4
- AI Deck detection working
- Smooth, professional demonstration
- Fast execution times
- Creative scenario storytelling

---

## 📅 Timeline At-a-Glance

```
NOW → Week 2:  Foundation (Utils, Messages, Config)
Week 3-4:      Core Algorithms (FSM, Formation, Planning)
Week 5-6:      ROS 2 Nodes + Vision
Week 7:        Integration + Hardware Testing
Hackathon:     Deploy, Test, Demonstrate, WIN!
```

---

**Document Version**: 1.0
**Last Updated**: 2025-11-17
**Prepared For**: Squadrone Team - Hackathon 2025

**Good Luck! 🚁✨**
