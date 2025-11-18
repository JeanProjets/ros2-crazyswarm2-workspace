# COMPREHENSIVE IMPLEMENTATION ARCHITECTURE PLAN
## Drone Coordination Hackathon - Challenge 2

**Document Version**: 1.0
**Last Updated**: 2025-11-17
**Target**: AI Coding Agents Implementation Guide

---

## TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [Project Context & Constraints](#2-project-context--constraints)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Technology Stack Decisions](#4-technology-stack-decisions)
5. [Repository Structure](#5-repository-structure)
6. [Module Breakdown & Dependencies](#6-module-breakdown--dependencies)
7. [Detailed Implementation Plan - Step by Step](#7-detailed-implementation-plan---step-by-step)
8. [AI Agent Definitions & Assignments](#8-ai-agent-definitions--assignments)
9. [Testing Strategy](#9-testing-strategy)
10. [Mission Specifications](#10-mission-specifications)
11. [Integration Points](#11-integration-points)
12. [Risk Mitigation](#12-risk-mitigation)

---

## 1. EXECUTIVE SUMMARY

### 1.1 Mission Objective
Develop a multi-drone coordination system where 4 Crazyflie drones cooperate to detect, intercept, and neutralize a hostile target drone through coordinated autonomous flight, vision-based detection, dynamic role assignment, and formation flying.

### 1.2 Critical Success Factors
- **Simulation-first**: 100% functionality validated in simulation before hardware
- **Battery efficiency**: Complete mission in < 3 minutes (5-minute battery limit)
- **Progressive complexity**: 4 mission versions (V1→V2→V3→V4)
- **Robust coordination**: Collision avoidance, role assignment, formation control
- **Vision integration**: AI Deck camera for target detection

### 1.3 Development Timeline
- **Phase 1 (Weeks 1-2)**: Core infrastructure + V1 simulation
- **Phase 2 (Weeks 3-4)**: V2 + Vision integration
- **Phase 3 (Weeks 5-6)**: V3 (moving target) + V3bis (obstacles)
- **Phase 4 (Week 7)**: V4 (full complexity) + hardware testing
- **Hackathon (3 days)**: Hardware integration, tuning, demonstration

---

## 2. PROJECT CONTEXT & CONSTRAINTS

### 2.1 Hardware Specifications

| Component | Quantity | Specifications | Constraints |
|-----------|----------|----------------|-------------|
| Crazyflie 2.1+ | 4 | 27g nano-drone, programmable Python/C | Limited compute, low payload |
| Crazyradio 2.0 | 1 | USB radio dongle, ~1km range | Single radio for 4 drones, bandwidth limit |
| AI Deck 1.1 | 4 | GAP8 processor + Himax camera | Limited inference capability, low resolution |
| Motion Capture Deck | 4 | OptiTrack marker support | Requires OptiTrack system |
| Batteries (350mAh LiPo) | 4+ | 3-5 min flight time with payload | Critical constraint |
| JTAG Debugger | 1 | Olimex ARM-USB-TINY-H | For AI Deck flashing |

### 2.2 Arena Specifications

**Test Environment** (Initial Development):
- Dimensions: 10m (X) × 6m (Y) × 8m (Z)
- Safety Zone: 3m × 6m (starting area)
- Coordinate system: Origin at (0, 0, 0)

**Competition Environment** (Grand Palais):
- Dimensions: 20m (X) × 8m (Y) × 6m (Z)
- Safety Zone: 6m × 8m (doubled)
- OptiTrack system: 120 Hz localization

### 2.3 Drone Roles

| Role | Symbol | Description | Count |
|------|--------|-------------|-------|
| Target | C | Hostile drone to neutralize | 1 |
| Patroller | P | Search/reconnaissance drone | 1 |
| Neutral 1 | N1 | Becomes Leader (L) or Follower (S) | 1 |
| Neutral 2 | N2 | Becomes Leader (L) or Follower (S) | 1 |
| Leader | L | Dynamic role - highest battery neutral | 1 (dynamic) |
| Follower | S | Dynamic role - second neutral | 1 (dynamic) |

### 2.4 Mission Versions (Progressive Complexity)

#### Version 1: Static Target, No Obstacles
- **Target Position**: (7.5, 3, 5)
- **Time Budget**: < 3 minutes
- **Complexity**: Baseline - patrol, detect, coordinate, neutralize
- **Success Criteria**: Neutralization achieved, no collisions

#### Version 2: Static Corner Target, No Obstacles
- **Target Position**: (9.5, 0.5, 5)
- **Time Budget**: < 3.5 minutes
- **Complexity**: Edge case handling, boundary constraints
- **Success Criteria**: Safe positioning near boundaries, neutralization

#### Version 3: Moving Target, No Obstacles
- **Target Behavior**: Circular path, 3m radius, center of arena
- **Time Budget**: < 4 minutes
- **Complexity**: Prediction, tracking, dynamic interception
- **Success Criteria**: Intercept moving target, maintain formation while tracking

#### Version 3bis: Static Target, With Obstacles
- **Obstacles**: 3 static obstacles (cardboard boxes)
- **Time Budget**: < 4 minutes
- **Complexity**: Path planning, obstacle avoidance
- **Success Criteria**: Navigate around obstacles, reach target

#### Version 4: Moving Target, With Obstacles (ULTIMATE)
- **Combination**: V3 + V3bis
- **Time Budget**: < 5 minutes
- **Complexity**: Full system integration
- **Success Criteria**: Dynamic obstacle avoidance + moving target interception

### 2.5 Key Technical Constraints

1. **Battery Life**: 3-5 minutes effective flight time
2. **Radio Bandwidth**: Shared among 4 drones via single Crazyradio
3. **Camera Limitations**:
   - Fixed horizontal angle (0°)
   - Blind spot < 30cm from camera
   - Optimal detection altitude: 4m
4. **Processing**:
   - Onboard: Limited (STM32 + GAP8)
   - Ground station: Main coordination logic
5. **Localization**: Dependent on OptiTrack (no GPS, limited onboard state estimation)

---

## 3. SYSTEM ARCHITECTURE OVERVIEW

### 3.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GROUND STATION                              │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                    ROS 2 MIDDLEWARE (Humble)                  │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐              │
│  │  OptiTrack  │  │   Mission    │  │   Vision    │              │
│  │  Interface  │→ │   Manager    │← │  Integrator │              │
│  │   Node      │  │   (FSM)      │  │    Node     │              │
│  └─────────────┘  └──────────────┘  └─────────────┘              │
│         ↓               ↓                   ↑                      │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐              │
│  │  Position   │  │     Role     │  │  Detection  │              │
│  │   Fusion    │→ │  Assignment  │  │   Fusion    │              │
│  └─────────────┘  └──────────────┘  └─────────────┘              │
│         ↓               ↓                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐              │
│  │    Path     │  │  Formation   │  │  Collision  │              │
│  │   Planner   │← │   Controller │→ │  Avoidance  │              │
│  └─────────────┘  └──────────────┘  └─────────────┘              │
│         ↓               ↓                   ↓                      │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │          Trajectory Generator & Commander                 │   │
│  └───────────────────────────────────────────────────────────┘   │
│         ↓               ↓               ↓               ↓         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ CF_Swarm │    │ CF_Swarm │    │ CF_Swarm │    │ CF_Swarm │  │
│  │  Bridge  │    │  Bridge  │    │  Bridge  │    │  Bridge  │  │
│  │   (P)    │    │   (N1)   │    │   (N2)   │    │   (C)    │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│         ↓               ↓               ↓               ↓         │
└─────────┼───────────────┼───────────────┼───────────────┼─────────┘
          │               │               │               │
          │        CRAZYRADIO 2.0 (USB)   │               │
          ↓               ↓               ↓               ↓
   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
   │  DRONE P │    │ DRONE N1 │    │ DRONE N2 │    │  DRONE C │
   │  ┌────┐  │    │  ┌────┐  │    │  ┌────┐  │    │  ┌────┐  │
   │  │STM │  │    │  │STM │  │    │  │STM │  │    │  │STM │  │
   │  │32  │  │    │  │32  │  │    │  │32  │  │    │  │32  │  │
   │  └────┘  │    │  └────┘  │    │  └────┘  │    │  └────┘  │
   │  ┌────┐  │    │  ┌────┐  │    │  ┌────┐  │    │          │
   │  │AI  │  │    │  │AI  │  │    │  │AI  │  │    │          │
   │  │Deck│  │    │  │Deck│  │    │  │Deck│  │    │          │
   │  └────┘  │    │  └────┘  │    │  └────┘  │    │          │
   │ OptiTrack│    │ OptiTrack│    │ OptiTrack│    │ OptiTrack│
   │  Markers │    │  Markers │    │  Markers │    │  Markers │
   └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

### 3.2 Data Flow Architecture

```
OptiTrack System (120 Hz)
    ↓
[Position Data: x, y, z, qw, qx, qy, qz]
    ↓
ROS 2 Topic: /cf_positions (PoseStamped)
    ↓
Position Fusion Node
    ↓
[Fused State: position, velocity, orientation]
    ↓
Mission Manager (FSM)
    ├→ State: IDLE, TAKEOFF, PATROL, DETECT, ASSIGN, INTERCEPT, NEUTRALIZE, RTL
    ↓
Role Assignment
    ├→ Roles: P, N1, N2 → L, S based on battery
    ↓
Formation Controller
    ├→ Leader position calculation
    ├→ Follower offset calculation
    ↓
Path Planner (A* for obstacles)
    ├→ Waypoint generation
    ├→ Collision-free paths
    ↓
Trajectory Generator
    ├→ Smooth trajectories (velocity profiles)
    ↓
Crazyflie Commander
    ├→ High-level setpoints (position + yaw)
    ↓
Crazyswarm2 Bridge
    ↓
Crazyflie Firmware
    ↓
Motor Commands
```

### 3.3 Communication Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ROS 2 TOPICS                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  /cf_positions/{drone_id}                                  │
│    Type: geometry_msgs/PoseStamped                         │
│    Rate: 120 Hz                                            │
│    Publisher: optitrack_node                               │
│    Subscribers: mission_manager, formation_controller      │
│                                                             │
│  /cf_velocities/{drone_id}                                 │
│    Type: geometry_msgs/TwistStamped                        │
│    Rate: 120 Hz                                            │
│    Publisher: position_fusion                              │
│    Subscribers: path_planner, collision_avoidance          │
│                                                             │
│  /cf_battery/{drone_id}                                    │
│    Type: sensor_msgs/BatteryState                          │
│    Rate: 1 Hz                                              │
│    Publisher: cf_bridge                                    │
│    Subscribers: role_assignment, mission_manager           │
│                                                             │
│  /vision/detections/{drone_id}                             │
│    Type: vision_msgs/Detection2DArray                      │
│    Rate: 5 Hz                                              │
│    Publisher: aideck_node                                  │
│    Subscribers: detection_fusion, mission_manager          │
│                                                             │
│  /swarm/roles                                              │
│    Type: custom_msgs/RoleAssignment                        │
│    Rate: 1 Hz                                              │
│    Publisher: role_assignment                              │
│    Subscribers: ALL nodes                                  │
│                                                             │
│  /swarm/mission_state                                      │
│    Type: custom_msgs/MissionState                          │
│    Rate: 10 Hz                                             │
│    Publisher: mission_manager                              │
│    Subscribers: ALL nodes                                  │
│                                                             │
│  /cf_cmd/{drone_id}                                        │
│    Type: crazyflie_msgs/FullState                          │
│    Rate: 20 Hz                                             │
│    Publisher: trajectory_commander                         │
│    Subscribers: cf_bridge                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. TECHNOLOGY STACK DECISIONS

### 4.1 Core Technologies

| Layer | Technology | Version | Rationale |
|-------|-----------|---------|-----------|
| **OS** | Ubuntu | 22.04 LTS | ROS 2 Humble compatibility |
| **Middleware** | ROS 2 | Humble | Industry standard, Crazyswarm2 support |
| **Language (Ground)** | Python | 3.10+ | Rapid development, cflib support |
| **Language (Drone)** | C | C11 | Crazyflie firmware standard |
| **Simulation** | Gazebo | Classic 11 | ROS 2 Humble compatibility |
| **Alternative Sim** | Webots | R2023b | Better Crazyflie support (backup) |
| **Swarm Framework** | Crazyswarm2 | Latest | Pre-built collision avoidance |
| **Crazyflie Lib** | cflib | Latest | Official Python library |
| **Vision Framework** | OpenCV | 4.8+ | Image processing |
| **AI Framework** | PyTorch | 2.0+ | Model training |
| **AI Deployment** | TFLite Micro | Latest | GAP8 deployment |
| **Testing** | pytest | Latest | Python unit testing |
| **CI/CD** | GitHub Actions | N/A | Automated testing |

### 4.2 Key Libraries & Dependencies

```yaml
# Python Dependencies (requirements.txt)
ros2:
  - rclpy>=3.3.7
  - geometry-msgs
  - sensor-msgs
  - vision-msgs
  - tf2-ros
  - tf2-geometry-msgs

crazyflie:
  - cflib>=0.1.22
  - crazyswarm2

simulation:
  - gazebo-ros-pkgs
  - webots-ros2 (optional)

vision:
  - opencv-python>=4.8.0
  - numpy>=1.24.0
  - pillow>=10.0.0

ml:
  - torch>=2.0.0
  - torchvision>=0.15.0
  - tensorflow-lite>=2.13.0
  - onnx>=1.14.0

planning:
  - scipy>=1.10.0
  - matplotlib>=3.7.0
  - numpy-quaternion>=2023.0.0

utils:
  - pyyaml>=6.0
  - dataclasses-json>=0.5.9
  - python-dotenv>=1.0.0

testing:
  - pytest>=7.4.0
  - pytest-cov>=4.1.0
  - pytest-mock>=3.11.0
```

### 4.3 Simulation Strategy

**Primary**: Gazebo Classic with Crazyflie model
- Physics: ODE
- Sensor simulation: IMU, Camera, GPS (mock OptiTrack)
- Multi-robot support: Native
- ROS 2 integration: gazebo_ros_pkgs

**Secondary**: Webots (if Gazebo issues)
- Better Crazyflie community support
- More realistic physics for micro-drones
- Built-in camera/sensor models

**Testing Pyramid**:
1. Unit tests (pytest) - Individual functions
2. Integration tests (ROS 2 test) - Node interactions
3. Simulation tests (Gazebo) - Full mission scenarios
4. Hardware-in-loop - Final validation

---

## 5. REPOSITORY STRUCTURE

```
drone-hackathon-2025/
├── README.md                          # Project overview, quick start
├── LICENSE                            # MIT License
├── .gitignore                         # Python, ROS, build artifacts
├── requirements.txt                   # Python dependencies
├── setup.py                           # Python package setup
├── .github/
│   └── workflows/
│       ├── ci.yml                     # Continuous integration
│       └── simulation_tests.yml       # Automated sim testing
│
├── config/                            # Configuration files
│   ├── arena/
│   │   ├── test_arena.yaml           # 10×6×8m test environment
│   │   └── competition_arena.yaml    # 20×8×6m competition environment
│   ├── drones/
│   │   ├── drone_params.yaml         # Physical parameters
│   │   ├── radio_config.yaml         # Crazyradio URIs
│   │   └── optitrack_config.yaml     # Rigid body IDs
│   ├── missions/
│   │   ├── mission_v1.yaml           # Version 1 parameters
│   │   ├── mission_v2.yaml           # Version 2 parameters
│   │   ├── mission_v3.yaml           # Version 3 parameters
│   │   ├── mission_v3bis.yaml        # Version 3bis parameters
│   │   └── mission_v4.yaml           # Version 4 parameters
│   ├── control/
│   │   ├── pid_params.yaml           # PID tuning
│   │   └── formation_params.yaml     # Formation geometry
│   └── vision/
│       ├── detection_params.yaml     # Detection thresholds
│       └── camera_calibration.yaml   # AI Deck calibration
│
├── src/                               # Main source code
│   ├── squadrone_swarm/              # Main ROS 2 package
│   │   ├── __init__.py
│   │   ├── setup.py
│   │   ├── package.xml
│   │   │
│   │   ├── squadrone_swarm/          # Python package
│   │   │   ├── __init__.py
│   │   │   │
│   │   │   ├── nodes/                # ROS 2 nodes
│   │   │   │   ├── __init__.py
│   │   │   │   ├── optitrack_node.py
│   │   │   │   ├── mission_manager_node.py
│   │   │   │   ├── role_assignment_node.py
│   │   │   │   ├── formation_controller_node.py
│   │   │   │   ├── path_planner_node.py
│   │   │   │   ├── collision_avoidance_node.py
│   │   │   │   ├── trajectory_commander_node.py
│   │   │   │   ├── vision_integrator_node.py
│   │   │   │   ├── detection_fusion_node.py
│   │   │   │   └── aideck_interface_node.py
│   │   │   │
│   │   │   ├── core/                 # Core algorithms
│   │   │   │   ├── __init__.py
│   │   │   │   ├── state_machine.py  # Mission FSM
│   │   │   │   ├── role_manager.py   # Role assignment logic
│   │   │   │   ├── formation.py      # Formation mathematics
│   │   │   │   ├── path_planning.py  # A* implementation
│   │   │   │   ├── collision.py      # Collision detection/avoidance
│   │   │   │   ├── trajectory.py     # Trajectory generation
│   │   │   │   └── kalman_filter.py  # State estimation
│   │   │   │
│   │   │   ├── vision/               # Vision processing
│   │   │   │   ├── __init__.py
│   │   │   │   ├── detector.py       # Target detection
│   │   │   │   ├── tracker.py        # Object tracking
│   │   │   │   ├── fusion.py         # Multi-drone fusion
│   │   │   │   └── preprocessor.py   # Image preprocessing
│   │   │   │
│   │   │   ├── utils/                # Utility functions
│   │   │   │   ├── __init__.py
│   │   │   │   ├── transforms.py     # Coordinate transforms
│   │   │   │   ├── geometry.py       # Geometric calculations
│   │   │   │   ├── timing.py         # Time synchronization
│   │   │   │   ├── logger.py         # Logging utilities
│   │   │   │   └── config_loader.py  # YAML config parsing
│   │   │   │
│   │   │   └── interfaces/           # External interfaces
│   │   │       ├── __init__.py
│   │   │       ├── optitrack.py      # OptiTrack interface
│   │   │       ├── crazyflie.py      # cflib wrapper
│   │   │       └── crazyswarm2.py    # Crazyswarm2 bridge
│   │   │
│   │   ├── launch/                   # ROS 2 launch files
│   │   │   ├── simulation.launch.py  # Gazebo simulation
│   │   │   ├── hardware.launch.py    # Hardware deployment
│   │   │   ├── mission_v1.launch.py  # Mission V1
│   │   │   ├── mission_v2.launch.py  # Mission V2
│   │   │   ├── mission_v3.launch.py  # Mission V3
│   │   │   ├── mission_v3bis.launch.py # Mission V3bis
│   │   │   └── mission_v4.launch.py  # Mission V4
│   │   │
│   │   └── resource/                 # ROS 2 resources
│   │       └── squadrone_swarm
│   │
│   └── squadrone_msgs/               # Custom ROS 2 messages
│       ├── CMakeLists.txt
│       ├── package.xml
│       ├── msg/
│       │   ├── RoleAssignment.msg
│       │   ├── MissionState.msg
│       │   ├── DroneStatus.msg
│       │   ├── DetectionReport.msg
│       │   └── FormationCommand.msg
│       └── srv/
│           ├── AssignRole.srv
│           ├── PlanPath.srv
│           └── EmergencyStop.srv
│
├── aideck/                            # AI Deck code
│   ├── README.md
│   ├── training/                      # Model training
│   │   ├── dataset/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── train_detector.py         # PyTorch training
│   │   ├── export_tflite.py          # Model export
│   │   ├── quantize_model.py         # INT8 quantization
│   │   └── validate_model.py         # Accuracy validation
│   │
│   └── gap8_firmware/                 # GAP8 embedded code
│       ├── Makefile
│       ├── main.c                     # Main loop
│       ├── model.tflite              # Quantized model
│       ├── inference.c               # TFLite inference
│       ├── camera.c                  # Himax camera driver
│       ├── preprocessing.c           # Image preprocessing
│       └── uart_comm.c               # UART communication
│
├── simulation/                        # Simulation worlds
│   ├── gazebo/
│   │   ├── worlds/
│   │   │   ├── test_arena.world
│   │   │   ├── competition_arena.world
│   │   │   └── obstacles.world
│   │   ├── models/
│   │   │   ├── crazyflie/
│   │   │   └── obstacles/
│   │   └── plugins/
│   │       └── crazyflie_plugin.cpp
│   │
│   └── webots/                        # Webots (backup)
│       ├── worlds/
│       │   └── arena.wbt
│       └── controllers/
│           └── crazyflie_controller.py
│
├── tests/                             # Test suite
│   ├── __init__.py
│   ├── conftest.py                   # Pytest configuration
│   │
│   ├── unit/                          # Unit tests
│   │   ├── test_formation.py
│   │   ├── test_path_planning.py
│   │   ├── test_collision.py
│   │   ├── test_kalman.py
│   │   ├── test_role_assignment.py
│   │   └── test_state_machine.py
│   │
│   ├── integration/                   # Integration tests
│   │   ├── test_node_communication.py
│   │   ├── test_mission_flow.py
│   │   └── test_vision_pipeline.py
│   │
│   └── simulation/                    # Simulation tests
│       ├── test_mission_v1.py
│       ├── test_mission_v2.py
│       ├── test_mission_v3.py
│       ├── test_mission_v3bis.py
│       └── test_mission_v4.py
│
├── scripts/                           # Utility scripts
│   ├── setup_environment.sh          # Install dependencies
│   ├── build_workspace.sh            # Build ROS 2 workspace
│   ├── flash_aideck.sh               # Flash AI Deck firmware
│   ├── calibrate_optitrack.sh        # OptiTrack calibration
│   ├── run_simulation.sh             # Launch simulation
│   ├── run_hardware.sh               # Launch hardware
│   ├── emergency_stop.py             # Emergency land all
│   └── battery_check.py              # Battery status checker
│
├── docs/                              # Documentation
│   ├── architecture.md               # System architecture
│   ├── api_reference.md              # API documentation
│   ├── mission_specifications.md     # Mission details
│   ├── hardware_setup.md             # Hardware guide
│   ├── troubleshooting.md            # Common issues
│   └── images/                        # Diagrams/screenshots
│
├── logs/                              # Log files (gitignored)
│   ├── .gitkeep
│   └── README.md
│
└── results/                           # Test results (gitignored)
    ├── .gitkeep
    ├── rosbags/
    ├── videos/
    └── plots/
```

---

## 6. MODULE BREAKDOWN & DEPENDENCIES

### 6.1 Module Dependency Graph

```
Level 0 (No Dependencies):
├── utils/transforms.py
├── utils/geometry.py
├── utils/timing.py
├── utils/logger.py
└── utils/config_loader.py

Level 1 (Depends on Level 0):
├── core/kalman_filter.py         [utils/geometry, utils/transforms]
├── core/trajectory.py             [utils/geometry, utils/transforms]
├── vision/preprocessor.py         [utils/logger]
├── interfaces/optitrack.py        [utils/transforms, utils/logger]
└── interfaces/crazyflie.py        [utils/logger, utils/config_loader]

Level 2 (Depends on Level 0-1):
├── core/collision.py              [utils/geometry, core/trajectory]
├── core/formation.py              [utils/geometry, utils/transforms]
├── core/path_planning.py          [utils/geometry, core/collision]
├── vision/detector.py             [vision/preprocessor, utils/logger]
└── vision/tracker.py              [vision/detector, core/kalman_filter]

Level 3 (Depends on Level 0-2):
├── core/role_manager.py           [core/formation, utils/logger]
├── vision/fusion.py               [vision/detector, vision/tracker]
└── interfaces/crazyswarm2.py      [interfaces/crazyflie, utils/config_loader]

Level 4 (Depends on Level 0-3):
├── core/state_machine.py          [core/role_manager, core/path_planning, core/formation]
├── nodes/optitrack_node.py        [interfaces/optitrack]
├── nodes/vision_integrator_node.py [vision/fusion]
└── nodes/collision_avoidance_node.py [core/collision]

Level 5 (Depends on Level 0-4):
├── nodes/mission_manager_node.py   [core/state_machine, nodes/optitrack_node]
├── nodes/role_assignment_node.py   [core/role_manager]
├── nodes/formation_controller_node.py [core/formation, core/trajectory]
├── nodes/path_planner_node.py      [core/path_planning]
└── nodes/trajectory_commander_node.py [core/trajectory, interfaces/crazyswarm2]

Level 6 (Top Level):
└── launch/*.launch.py              [All nodes]
```

### 6.2 Critical Path Analysis

**Must Complete First** (Blocking):
1. `utils/*` - Foundation for everything
2. `interfaces/optitrack.py` - Position data source
3. `interfaces/crazyflie.py` - Drone communication
4. `core/state_machine.py` - Mission orchestration

**Can Develop in Parallel**:
- Vision pipeline (`vision/*`) - Independent from control
- Path planning (`core/path_planning.py`) - Can use mock positions
- Formation control (`core/formation.py`) - Mathematical, standalone

**Integration Points** (High Risk):
- OptiTrack → ROS 2 bridge
- Crazyswarm2 integration
- AI Deck → Ground station communication
- Multi-node coordination

---

## 7. DETAILED IMPLEMENTATION PLAN - STEP BY STEP

### PHASE 1: FOUNDATION (Week 1-2) - PRIORITY P0

#### Step 1.1: Project Setup (2 days)
**Agent**: DevOps Agent

**Checklist**:
- [ ] Create GitHub repository with proper structure
- [ ] Set up .gitignore (Python, ROS, build artifacts)
- [ ] Create virtual environment (Python 3.10)
- [ ] Install ROS 2 Humble
- [ ] Install Crazyswarm2
- [ ] Install cflib
- [ ] Create requirements.txt
- [ ] Set up GitHub Actions CI
- [ ] Write README.md with quick start

**Files to Create**:
```
drone-hackathon-2025/
├── README.md
├── .gitignore
├── requirements.txt
├── setup.py
├── .github/workflows/ci.yml
└── scripts/setup_environment.sh
```

**Acceptance Criteria**:
- `colcon build` succeeds
- `pytest` runs (even with no tests yet)
- CI pipeline passes

---

#### Step 1.2: Custom ROS 2 Messages (1 day)
**Agent**: ROS Agent

**Implementation**:
Create custom message definitions for drone coordination.

**Files to Create**:
```
src/squadrone_msgs/
├── CMakeLists.txt
├── package.xml
├── msg/
│   ├── RoleAssignment.msg
│   ├── MissionState.msg
│   ├── DroneStatus.msg
│   ├── DetectionReport.msg
│   └── FormationCommand.msg
└── srv/
    ├── AssignRole.srv
    ├── PlanPath.srv
    └── EmergencyStop.srv
```

**Message Definitions**:

`RoleAssignment.msg`:
```
std_msgs/Header header
string drone_id
string role  # P, N1, N2, L, S, C
float32 battery_percent
geometry_msgs/Point position
```

`MissionState.msg`:
```
std_msgs/Header header
string state  # IDLE, TAKEOFF, PATROL, DETECT, ASSIGN, INTERCEPT, NEUTRALIZE, RTL, EMERGENCY
string target_detected_by
geometry_msgs/Point target_position
float32 mission_time_elapsed
```

`DroneStatus.msg`:
```
std_msgs/Header header
string drone_id
float32 battery_voltage
float32 battery_percent
bool is_flying
bool is_connected
geometry_msgs/Pose pose
geometry_msgs/Twist velocity
```

`DetectionReport.msg`:
```
std_msgs/Header header
string detector_drone_id
bool target_detected
float32 confidence
geometry_msgs/Point target_position_camera_frame
geometry_msgs/Point target_position_world_frame
float32 distance_to_target
```

`FormationCommand.msg`:
```
std_msgs/Header header
string leader_id
string follower_id
geometry_msgs/Point leader_setpoint
geometry_msgs/Point follower_setpoint
float32 formation_offset_x
float32 formation_offset_y
float32 formation_offset_z
```

**Services**:

`AssignRole.srv`:
```
# Request
string drone_id
---
# Response
bool success
string role
string message
```

`PlanPath.srv`:
```
# Request
geometry_msgs/Point start
geometry_msgs/Point goal
bool avoid_obstacles
---
# Response
bool success
geometry_msgs/Point[] waypoints
float32 estimated_time
string message
```

`EmergencyStop.srv`:
```
# Request
string reason
---
# Response
bool success
int32 drones_stopped
string message
```

**Acceptance Criteria**:
- Messages compile successfully
- `ros2 interface list` shows custom messages
- `ros2 interface show squadrone_msgs/msg/RoleAssignment` works

---

#### Step 1.3: Utility Modules (2 days)
**Agent**: Core Algorithm Agent

**Implementation Order**:

**1.3.1: `utils/transforms.py`**
```python
"""
Coordinate transformation utilities.
- World ↔ Drone body frames
- Quaternion ↔ Euler angles
- OptiTrack ↔ ROS coordinate systems
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternion_to_euler(qw, qx, qy, qz):
    """Convert quaternion to Euler angles (roll, pitch, yaw)."""
    pass

def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles to quaternion."""
    pass

def transform_point_world_to_body(point_world, drone_pose):
    """Transform point from world frame to drone body frame."""
    pass

def transform_point_body_to_world(point_body, drone_pose):
    """Transform point from drone body frame to world frame."""
    pass

def optitrack_to_ros_frame(position, orientation):
    """Convert OptiTrack coordinates to ROS (if axis conventions differ)."""
    pass
```

**1.3.2: `utils/geometry.py`**
```python
"""
Geometric calculation utilities.
- Distance, angle calculations
- Line-line, point-line distances
- Circle-sphere intersections
"""

import numpy as np

def distance_3d(point1, point2):
    """Euclidean distance between two 3D points."""
    pass

def angle_between_vectors(vec1, vec2):
    """Angle in radians between two vectors."""
    pass

def point_to_line_distance(point, line_start, line_end):
    """Shortest distance from point to line segment."""
    pass

def check_sphere_collision(center1, radius1, center2, radius2):
    """Check if two spheres collide."""
    pass

def normalize_vector(vec):
    """Normalize vector to unit length."""
    pass

def project_point_on_line(point, line_start, line_end):
    """Project point onto line segment."""
    pass
```

**1.3.3: `utils/timing.py`**
```python
"""
Time synchronization and timing utilities.
"""

import time
from rclpy.time import Time

def get_timestamp_sec():
    """Get current timestamp in seconds (float)."""
    return time.time()

def ros_time_to_sec(ros_time):
    """Convert ROS Time to seconds."""
    return ros_time.nanoseconds / 1e9

def sec_to_ros_time(seconds):
    """Convert seconds to ROS Time."""
    pass

class RateCounter:
    """Count actual rate of events."""
    def __init__(self, window_size=100):
        self.timestamps = []
        self.window_size = window_size

    def tick(self):
        """Record an event."""
        pass

    def get_rate(self):
        """Get current rate in Hz."""
        pass
```

**1.3.4: `utils/logger.py`**
```python
"""
Logging utilities with ROS 2 integration.
"""

import rclpy
from rclpy.node import Node

class ROSLogger:
    """Unified logging across ROS and Python."""
    def __init__(self, node: Node, name: str):
        self.node = node
        self.name = name

    def info(self, message):
        self.node.get_logger().info(f"[{self.name}] {message}")

    def warn(self, message):
        self.node.get_logger().warn(f"[{self.name}] {message}")

    def error(self, message):
        self.node.get_logger().error(f"[{self.name}] {message}")

    def debug(self, message):
        self.node.get_logger().debug(f"[{self.name}] {message}")
```

**1.3.5: `utils/config_loader.py`**
```python
"""
YAML configuration file loader.
"""

import yaml
from pathlib import Path

class ConfigLoader:
    """Load and validate YAML configuration files."""

    @staticmethod
    def load_yaml(file_path):
        """Load YAML file and return dict."""
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def load_arena_config(config_dir, arena_name="test_arena"):
        """Load arena configuration."""
        path = Path(config_dir) / "arena" / f"{arena_name}.yaml"
        return ConfigLoader.load_yaml(path)

    @staticmethod
    def load_mission_config(config_dir, mission_version="v1"):
        """Load mission configuration."""
        path = Path(config_dir) / "missions" / f"mission_{mission_version}.yaml"
        return ConfigLoader.load_yaml(path)

    @staticmethod
    def load_drone_config(config_dir):
        """Load drone parameters."""
        path = Path(config_dir) / "drones" / "drone_params.yaml"
        return ConfigLoader.load_yaml(path)
```

**Unit Tests**:
```
tests/unit/
├── test_transforms.py
├── test_geometry.py
├── test_timing.py
└── test_config_loader.py
```

**Acceptance Criteria**:
- All unit tests pass (>95% coverage)
- No circular dependencies
- Type hints for all functions
- Docstrings in NumPy format

---

#### Step 1.4: Configuration Files (1 day)
**Agent**: Configuration Agent

**Files to Create**:

`config/arena/test_arena.yaml`:
```yaml
# Test arena dimensions (10×6×8m)
arena:
  name: "test_arena"
  dimensions:
    x: 10.0  # meters
    y: 6.0
    z: 8.0
  origin: [0.0, 0.0, 0.0]

  safety_zone:
    x_min: 0.0
    x_max: 3.0
    y_min: 0.0
    y_max: 6.0
    z_min: 0.0
    z_max: 8.0

  boundaries:
    x_min: 0.0
    x_max: 10.0
    y_min: 0.0
    y_max: 6.0
    z_min: 0.0
    z_max: 8.0

  safety_margin: 0.5  # meters from boundaries

  obstacles: []  # No obstacles in test arena
```

`config/drones/drone_params.yaml`:
```yaml
# Crazyflie 2.1 parameters
physical:
  mass: 0.027  # kg
  max_thrust: 0.16  # N
  max_velocity: 2.0  # m/s
  max_acceleration: 4.0  # m/s^2
  max_angular_velocity: 720  # deg/s

control:
  update_rate: 20  # Hz (setpoint frequency)
  position_tolerance: 0.1  # meters
  velocity_tolerance: 0.2  # m/s

safety:
  min_battery_voltage: 3.0  # Volts
  critical_battery_voltage: 3.3  # Volts
  low_battery_threshold: 0.20  # 20%
  collision_radius: 0.4  # meters (safety bubble)
  min_inter_drone_distance: 0.8  # meters

camera:
  horizontal_fov: 160  # degrees
  vertical_fov: 120  # degrees
  angle_offset: 0  # degrees (horizontal)
  blind_spot_distance: 0.3  # meters
  optimal_detection_altitude: 4.0  # meters
```

`config/missions/mission_v1.yaml`:
```yaml
# Mission Version 1: Static target, no obstacles, center position
mission:
  version: "v1"
  description: "Static target in center, no obstacles"
  time_budget: 180  # seconds (3 minutes)

  initial_positions:
    N1: [2.5, 2.5, 0.0]
    N2: [2.5, 3.5, 0.0]
    P:  [3.0, 5.0, 0.0]
    C:  [7.5, 3.0, 5.0]

  target:
    type: "static"
    behavior: "hover"
    position: [7.5, 3.0, 5.0]

  patrol:
    altitude: 4.0  # meters
    path_type: "rectangular"  # Safety zone check + arena perimeter
    velocity: 1.2  # m/s
    waypoints:
      - [2.5, 2.5, 4.0]
      - [2.5, 5.5, 4.0]
      - [9.5, 5.5, 4.0]
      - [9.5, 0.5, 4.0]
      - [3.5, 0.5, 4.0]

  formation:
    leader_standoff_distance: 0.6  # meters from target
    follower_offset_x: -0.5  # meters behind leader
    follower_offset_y: 0.0
    follower_offset_z: -0.5  # slightly below leader

  neutralization:
    proximity_threshold: 0.6  # meters
    hold_time: 1.5  # seconds
    approach_velocity: 0.5  # m/s (slow and controlled)

  timeouts:
    initialization: 10  # seconds
    patrol_max_time: 120  # seconds
    detection_confidence_time: 5  # seconds (stable detection)
    role_assignment_time: 1  # second
    interception_max_time: 30  # seconds
    jamming_time: 20  # seconds
    attack_time: 5  # seconds
```

**Similar files for V2, V3, V3bis, V4** (see full specs in Section 10)

**Acceptance Criteria**:
- All YAML files valid (yamllint passes)
- ConfigLoader can load all files
- Values match mission specifications

---

### PHASE 2: CORE ALGORITHMS (Week 3-4) - PRIORITY P0

#### Step 2.1: State Machine (3 days)
**Agent**: Core Algorithm Agent

**File**: `src/squadrone_swarm/squadrone_swarm/core/state_machine.py`

**State Definitions**:
```python
from enum import Enum

class MissionState(Enum):
    IDLE = "IDLE"
    TAKEOFF = "TAKEOFF"
    PATROL = "PATROL"
    DETECT = "DETECT"
    ASSIGN = "ASSIGN"
    INTERCEPT = "INTERCEPT"
    NEUTRALIZE = "NEUTRALIZE"
    RTL = "RTL"  # Return to Launch
    EMERGENCY = "EMERGENCY"
    LANDED = "LANDED"

class DroneRole(Enum):
    PATROLLER = "P"
    NEUTRAL_1 = "N1"
    NEUTRAL_2 = "N2"
    LEADER = "L"
    FOLLOWER = "S"
    TARGET = "C"
```

**FSM Implementation**:
```python
class MissionStateMachine:
    """
    Finite State Machine for mission orchestration.

    State Transitions:
    IDLE → TAKEOFF → PATROL → DETECT → ASSIGN → INTERCEPT → NEUTRALIZE → RTL → LANDED
                ↓ (emergency)
              EMERGENCY → RTL → LANDED
    """

    def __init__(self, mission_config, drone_states):
        self.state = MissionState.IDLE
        self.mission_config = mission_config
        self.drone_states = drone_states  # Dict[drone_id, DroneStatus]
        self.target_detected = False
        self.target_position = None
        self.mission_start_time = None

    def transition_to(self, new_state):
        """Transition to new state with validation."""
        if self._is_valid_transition(self.state, new_state):
            print(f"State transition: {self.state.value} → {new_state.value}")
            self.state = new_state
            self._on_enter_state(new_state)
        else:
            raise ValueError(f"Invalid transition: {self.state} → {new_state}")

    def _is_valid_transition(self, from_state, to_state):
        """Check if state transition is allowed."""
        # Define valid transitions
        transitions = {
            MissionState.IDLE: [MissionState.TAKEOFF],
            MissionState.TAKEOFF: [MissionState.PATROL, MissionState.EMERGENCY],
            MissionState.PATROL: [MissionState.DETECT, MissionState.EMERGENCY],
            MissionState.DETECT: [MissionState.ASSIGN, MissionState.PATROL, MissionState.EMERGENCY],
            MissionState.ASSIGN: [MissionState.INTERCEPT, MissionState.EMERGENCY],
            MissionState.INTERCEPT: [MissionState.NEUTRALIZE, MissionState.DETECT, MissionState.EMERGENCY],
            MissionState.NEUTRALIZE: [MissionState.RTL, MissionState.EMERGENCY],
            MissionState.RTL: [MissionState.LANDED, MissionState.EMERGENCY],
            MissionState.EMERGENCY: [MissionState.RTL],
            MissionState.LANDED: [MissionState.IDLE],
        }
        return to_state in transitions.get(from_state, [])

    def _on_enter_state(self, state):
        """Actions to perform when entering a state."""
        if state == MissionState.TAKEOFF:
            self.mission_start_time = time.time()
        elif state == MissionState.PATROL:
            # Set patrol waypoints for P
            pass
        elif state == MissionState.ASSIGN:
            # Trigger role assignment
            pass
        # ... etc for each state

    def update(self, current_time):
        """Update FSM based on current conditions."""
        # Check for emergency conditions
        if self._check_emergency():
            self.transition_to(MissionState.EMERGENCY)
            return

        # State-specific logic
        if self.state == MissionState.TAKEOFF:
            if self._all_drones_at_altitude(4.0):
                self.transition_to(MissionState.PATROL)

        elif self.state == MissionState.PATROL:
            if self.target_detected:
                self.transition_to(MissionState.DETECT)

        elif self.state == MissionState.DETECT:
            if self._detection_stable():
                self.transition_to(MissionState.ASSIGN)

        elif self.state == MissionState.ASSIGN:
            if self._roles_assigned():
                self.transition_to(MissionState.INTERCEPT)

        elif self.state == MissionState.INTERCEPT:
            if self._target_in_neutralization_range():
                self.transition_to(MissionState.NEUTRALIZE)

        elif self.state == MissionState.NEUTRALIZE:
            if self._neutralization_complete():
                self.transition_to(MissionState.RTL)

        elif self.state == MissionState.RTL:
            if self._all_drones_landed():
                self.transition_to(MissionState.LANDED)

    def _check_emergency(self):
        """Check for emergency conditions."""
        for drone_id, status in self.drone_states.items():
            if status.battery_percent < 15:
                return True
            if not status.is_connected:
                return True
            if self._out_of_bounds(status.pose.position):
                return True
        return False

    def _out_of_bounds(self, position):
        """Check if position is outside arena + safety margin."""
        bounds = self.mission_config['arena']['boundaries']
        margin = self.mission_config['arena']['safety_margin']
        return (position.x < bounds['x_min'] + margin or
                position.x > bounds['x_max'] - margin or
                position.y < bounds['y_min'] + margin or
                position.y > bounds['y_max'] - margin or
                position.z < bounds['z_min'] + margin or
                position.z > bounds['z_max'] - margin)

    # ... other helper methods
```

**Unit Tests**: `tests/unit/test_state_machine.py`
```python
def test_valid_transition():
    fsm = MissionStateMachine(mock_config, {})
    fsm.transition_to(MissionState.TAKEOFF)
    assert fsm.state == MissionState.TAKEOFF

def test_invalid_transition():
    fsm = MissionStateMachine(mock_config, {})
    with pytest.raises(ValueError):
        fsm.transition_to(MissionState.NEUTRALIZE)  # Can't go from IDLE to NEUTRALIZE

def test_emergency_transition():
    fsm = MissionStateMachine(mock_config, mock_low_battery_states)
    fsm.update(time.time())
    assert fsm.state == MissionState.EMERGENCY
```

**Acceptance Criteria**:
- All state transitions validated
- Emergency conditions trigger properly
- Unit tests >90% coverage
- No race conditions

---

#### Step 2.2: Role Assignment (2 days)
**Agent**: Core Algorithm Agent

**File**: `src/squadrone_swarm/squadrone_swarm/core/role_manager.py`

**Implementation**:
```python
class RoleManager:
    """
    Dynamic role assignment based on battery and position.

    Rules:
    - P (Patroller): Always the designated patrol drone
    - N1, N2 (Neutrals): Start as neutrals
    - When target detected:
      - Highest battery neutral → Leader (L)
      - Second neutral → Follower (S)
    """

    def __init__(self, initial_roles):
        self.roles = initial_roles.copy()  # Dict[drone_id, role]
        self.previous_roles = {}

    def assign_pursuit_roles(self, drone_states, target_position):
        """
        Assign Leader and Follower roles to neutrals.

        Args:
            drone_states: Dict[drone_id, DroneStatus]
            target_position: geometry_msgs/Point

        Returns:
            Dict[drone_id, new_role]
        """
        # Find neutral drones
        neutrals = [(did, state) for did, state in drone_states.items()
                   if self.roles[did] in [DroneRole.NEUTRAL_1, DroneRole.NEUTRAL_2]]

        if len(neutrals) < 2:
            raise ValueError("Need at least 2 neutral drones for role assignment")

        # Sort by battery (highest first)
        neutrals_sorted = sorted(neutrals, key=lambda x: x[1].battery_percent, reverse=True)

        # Assign roles
        leader_id, leader_state = neutrals_sorted[0]
        follower_id, follower_state = neutrals_sorted[1]

        self.previous_roles = self.roles.copy()
        self.roles[leader_id] = DroneRole.LEADER
        self.roles[follower_id] = DroneRole.FOLLOWER

        print(f"Role assignment: {leader_id} → LEADER (battery: {leader_state.battery_percent}%), "
              f"{follower_id} → FOLLOWER (battery: {follower_state.battery_percent}%)")

        return self.roles

    def reassign_if_needed(self, drone_states, target_position):
        """
        Reassign roles if:
        - Leader battery drops below threshold
        - Leader disconnects
        - Target moves significantly
        """
        leader_id = self.get_drone_with_role(DroneRole.LEADER)

        if leader_id is None:
            return False  # No leader assigned yet

        leader_state = drone_states[leader_id]

        # Check if reassignment needed
        if leader_state.battery_percent < 20:
            print(f"Leader {leader_id} battery low ({leader_state.battery_percent}%), reassigning")
            return True

        if not leader_state.is_connected:
            print(f"Leader {leader_id} disconnected, reassigning")
            return True

        return False

    def get_drone_with_role(self, role):
        """Get drone ID with specified role."""
        for drone_id, drone_role in self.roles.items():
            if drone_role == role:
                return drone_id
        return None

    def get_role(self, drone_id):
        """Get role of specified drone."""
        return self.roles.get(drone_id, None)
```

**Unit Tests**: `tests/unit/test_role_assignment.py`

**Acceptance Criteria**:
- Highest battery neutral becomes Leader
- Reassignment works correctly
- Handles edge cases (equal battery, disconnections)

---

#### Step 2.3: Formation Control (3 days)
**Agent**: Control Systems Agent

**File**: `src/squadrone_swarm/squadrone_swarm/core/formation.py`

**Implementation**:
```python
class FormationController:
    """
    Leader-Follower formation control.

    Formation geometry:
    - Leader (L): Positioned at standoff distance from target
    - Follower (S): Offset behind and below Leader
    """

    def __init__(self, config):
        self.standoff_distance = config['formation']['leader_standoff_distance']
        self.follower_offset = np.array([
            config['formation']['follower_offset_x'],
            config['formation']['follower_offset_y'],
            config['formation']['follower_offset_z']
        ])

    def compute_leader_setpoint(self, target_position, target_velocity=None):
        """
        Compute Leader setpoint at standoff distance from target.

        For static target: Position on approach vector
        For moving target: Predictive intercept point
        """
        if target_velocity is None or np.linalg.norm(target_velocity) < 0.1:
            # Static target: Approach from front
            # Leader positions at standoff distance on X-axis
            leader_pos = np.array([
                target_position.x - self.standoff_distance,
                target_position.y,
                target_position.z
            ])
        else:
            # Moving target: Intercept ahead of current position
            target_vel = np.array([target_velocity.x, target_velocity.y, target_velocity.z])
            target_vel_normalized = target_vel / np.linalg.norm(target_vel)

            # Position at standoff distance opposite to velocity
            leader_pos = np.array([target_position.x, target_position.y, target_position.z])
            leader_pos -= target_vel_normalized * self.standoff_distance

        return leader_pos

    def compute_follower_setpoint(self, leader_position, target_position):
        """
        Compute Follower setpoint relative to Leader.

        Follower maintains offset in target-relative frame.
        """
        # Get target-to-leader vector
        target_to_leader = np.array([
            leader_position.x - target_position.x,
            leader_position.y - target_position.y,
            leader_position.z - target_position.z
        ])

        # Compute rotation from world to target-relative frame
        # (simplified: assume target heading aligns with target-to-leader)
        forward = target_to_leader / np.linalg.norm(target_to_leader)

        # Follower position: Leader + offset (in world frame)
        follower_pos = np.array([leader_position.x, leader_position.y, leader_position.z])
        follower_pos += self.follower_offset

        return follower_pos

    def compute_formation_setpoints(self, target_state, leader_state, follower_state):
        """
        Compute setpoints for both Leader and Follower.

        Returns:
            (leader_setpoint, follower_setpoint)
        """
        target_pos = target_state.pose.position
        target_vel = target_state.velocity if hasattr(target_state, 'velocity') else None

        leader_setpoint = self.compute_leader_setpoint(target_pos, target_vel)
        follower_setpoint = self.compute_follower_setpoint(
            type('obj', (object,), {'x': leader_setpoint[0], 'y': leader_setpoint[1], 'z': leader_setpoint[2]})(),
            target_pos
        )

        return leader_setpoint, follower_setpoint
```

**Unit Tests**: `tests/unit/test_formation.py`
```python
def test_leader_static_target():
    controller = FormationController(mock_config)
    target_pos = Point(x=7.5, y=3.0, z=5.0)
    leader_setpoint = controller.compute_leader_setpoint(target_pos)

    # Leader should be at standoff distance (0.6m) in front of target
    expected_x = 7.5 - 0.6
    assert np.isclose(leader_setpoint[0], expected_x, atol=0.01)
    assert np.isclose(leader_setpoint[1], 3.0, atol=0.01)
    assert np.isclose(leader_setpoint[2], 5.0, atol=0.01)

def test_follower_offset():
    controller = FormationController(mock_config)
    leader_pos = type('obj', (object,), {'x': 7.0, 'y': 3.0, 'z': 5.0})()
    target_pos = Point(x=7.5, y=3.0, z=5.0)
    follower_setpoint = controller.compute_follower_setpoint(leader_pos, target_pos)

    # Follower should be offset by (-0.5, 0, -0.5) from leader
    assert np.isclose(follower_setpoint[0], 6.5, atol=0.01)
    assert np.isclose(follower_setpoint[1], 3.0, atol=0.01)
    assert np.isclose(follower_setpoint[2], 4.5, atol=0.01)
```

**Acceptance Criteria**:
- Formation holds for static target
- Formation tracks moving target
- Follower maintains offset
- Unit tests >90% coverage

---

#### Step 2.4: Path Planning (3 days)
**Agent**: Planning Agent

**File**: `src/squadrone_swarm/squadrone_swarm/core/path_planning.py`

**Implementation**:
```python
from scipy.spatial import distance
from scipy.interpolate import splprep, splev
import numpy as np

class AStarPlanner:
    """
    A* path planner for 3D grid-based obstacle avoidance.
    """

    def __init__(self, arena_config, resolution=0.5):
        self.arena_bounds = arena_config['boundaries']
        self.resolution = resolution  # meters per grid cell

        # Create 3D occupancy grid
        x_cells = int((self.arena_bounds['x_max'] - self.arena_bounds['x_min']) / resolution)
        y_cells = int((self.arena_bounds['y_max'] - self.arena_bounds['y_min']) / resolution)
        z_cells = int((self.arena_bounds['z_max'] - self.arena_bounds['z_min']) / resolution)

        self.grid = np.zeros((x_cells, y_cells, z_cells), dtype=np.uint8)
        self.grid_shape = (x_cells, y_cells, z_cells)

    def world_to_grid(self, point):
        """Convert world coordinates to grid indices."""
        x_idx = int((point[0] - self.arena_bounds['x_min']) / self.resolution)
        y_idx = int((point[1] - self.arena_bounds['y_min']) / self.resolution)
        z_idx = int((point[2] - self.arena_bounds['z_min']) / self.resolution)
        return (x_idx, y_idx, z_idx)

    def grid_to_world(self, grid_idx):
        """Convert grid indices to world coordinates."""
        x = grid_idx[0] * self.resolution + self.arena_bounds['x_min']
        y = grid_idx[1] * self.resolution + self.arena_bounds['y_min']
        z = grid_idx[2] * self.resolution + self.arena_bounds['z_min']
        return np.array([x, y, z])

    def add_obstacle(self, center, size, inflation_radius=0.4):
        """
        Add rectangular obstacle to grid with inflation.

        Args:
            center: (x, y, z) in world coordinates
            size: (width, depth, height) in meters
            inflation_radius: Safety margin in meters
        """
        # Convert to grid coordinates
        center_grid = self.world_to_grid(center)
        size_cells = (int(size[0] / self.resolution),
                     int(size[1] / self.resolution),
                     int(size[2] / self.resolution))
        inflation_cells = int(inflation_radius / self.resolution)

        # Mark occupied cells
        for dx in range(-size_cells[0]//2 - inflation_cells, size_cells[0]//2 + inflation_cells + 1):
            for dy in range(-size_cells[1]//2 - inflation_cells, size_cells[1]//2 + inflation_cells + 1):
                for dz in range(-size_cells[2]//2 - inflation_cells, size_cells[2]//2 + inflation_cells + 1):
                    x, y, z = center_grid[0] + dx, center_grid[1] + dy, center_grid[2] + dz
                    if 0 <= x < self.grid_shape[0] and 0 <= y < self.grid_shape[1] and 0 <= z < self.grid_shape[2]:
                        self.grid[x, y, z] = 1  # Occupied

    def plan(self, start, goal):
        """
        A* path planning.

        Args:
            start: (x, y, z) in world coordinates
            goal: (x, y, z) in world coordinates

        Returns:
            List of waypoints or None if no path found
        """
        start_grid = self.world_to_grid(start)
        goal_grid = self.world_to_grid(goal)

        # Check if start/goal are valid
        if self.grid[start_grid] == 1:
            print(f"ERROR: Start position {start} is in obstacle")
            return None
        if self.grid[goal_grid] == 1:
            print(f"ERROR: Goal position {goal} is in obstacle")
            return None

        # A* search
        open_set = {start_grid}
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self._heuristic(start_grid, goal_grid)}

        while open_set:
            # Get node with lowest f_score
            current = min(open_set, key=lambda node: f_score.get(node, float('inf')))

            if current == goal_grid:
                # Reconstruct path
                path_grid = self._reconstruct_path(came_from, current)
                path_world = [self.grid_to_world(p) for p in path_grid]
                return path_world

            open_set.remove(current)

            # Check neighbors
            for neighbor in self._get_neighbors(current):
                if self.grid[neighbor] == 1:  # Obstacle
                    continue

                tentative_g = g_score[current] + self._distance(current, neighbor)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_grid)
                    open_set.add(neighbor)

        print(f"ERROR: No path found from {start} to {goal}")
        return None

    def _heuristic(self, node1, node2):
        """Euclidean distance heuristic."""
        return np.linalg.norm(np.array(node1) - np.array(node2))

    def _distance(self, node1, node2):
        """Actual distance between adjacent nodes."""
        return np.linalg.norm(np.array(node1) - np.array(node2))

    def _get_neighbors(self, node):
        """Get valid 26-connected neighbors (3D)."""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbor = (node[0] + dx, node[1] + dy, node[2] + dz)
                    if (0 <= neighbor[0] < self.grid_shape[0] and
                        0 <= neighbor[1] < self.grid_shape[1] and
                        0 <= neighbor[2] < self.grid_shape[2]):
                        neighbors.append(neighbor)
        return neighbors

    def _reconstruct_path(self, came_from, current):
        """Reconstruct path from came_from dict."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def smooth_path(self, path, num_points=50):
        """
        Smooth path using B-spline interpolation.

        Args:
            path: List of waypoints
            num_points: Number of points in smoothed path

        Returns:
            Smoothed path
        """
        if len(path) < 4:
            return path  # Need at least 4 points for spline

        path_array = np.array(path)
        tck, u = splprep([path_array[:, 0], path_array[:, 1], path_array[:, 2]], s=0, k=3)
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new, z_new = splev(u_new, tck)

        smoothed_path = np.column_stack((x_new, y_new, z_new))
        return smoothed_path.tolist()
```

**Unit Tests**: `tests/unit/test_path_planning.py`

**Acceptance Criteria**:
- Finds collision-free paths
- Handles no-solution cases gracefully
- Smoothing works without introducing collisions
- Performance: <50ms for typical 20×8×6m arena

---

(Continuing in next part due to length...)

Would you like me to continue with the remaining phases (2.5-7), or would you like me to save this first part and create separate documents for each phase?
