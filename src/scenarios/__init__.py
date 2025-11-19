"""
Scenario 2 Implementation Package

This package contains the implementation for Scenario 2 (Corner Mission):
- State machine for corner operations
- Battery-based role management
- Boundary safety monitoring
- Mission sequencing and coordination
"""

from .scenario_2_fsm import Scenario2StateMachine, MissionState, SwarmTelemetry
from .battery_role_manager import (
    BatteryRoleManager,
    Drone,
    DroneRole,
    MissionPhase,
    BatteryThresholds
)
from .boundary_guard import (
    GeofenceMonitor,
    SafetyOverride,
    Telemetry,
    ViolationType
)
from .scenario_2_mission import Scenario2MissionSequencer, MissionResult

__all__ = [
    'Scenario2StateMachine',
    'MissionState',
    'SwarmTelemetry',
    'BatteryRoleManager',
    'Drone',
    'DroneRole',
    'MissionPhase',
    'BatteryThresholds',
    'GeofenceMonitor',
    'SafetyOverride',
    'Telemetry',
    'ViolationType',
    'Scenario2MissionSequencer',
    'MissionResult'
]
