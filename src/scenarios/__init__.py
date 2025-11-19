"""
Scenario 4 Mission Coordinator Package
Implements the Supreme Tactical Commander for mobile target + obstacles
"""

from .scenario_4_fsm import Scenario4FSM, MissionState, TargetStatus, Telemetry, MissionBrain
from .shadow_manager import OcclusionStrategy, ShadowHunter, ObstacleInfo
from .risk_manager import AttackCorridorValidator, DynamicRiskManager, AttackClearance
from .swarm_splitter import FormationManagerV4, FormationMode, SwarmCoordinator

__all__ = [
    'Scenario4FSM',
    'MissionState',
    'TargetStatus',
    'Telemetry',
    'MissionBrain',
    'OcclusionStrategy',
    'ShadowHunter',
    'ObstacleInfo',
    'AttackCorridorValidator',
    'DynamicRiskManager',
    'AttackClearance',
    'FormationManagerV4',
    'FormationMode',
    'SwarmCoordinator',
]
