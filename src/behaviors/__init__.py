"""
Behaviors package for Crazyflie drone swarm.

This package contains behavior modules for autonomous drone operations:
- patrol_patterns: Search and coverage patterns
- formation_controller: Formation flying and coordination
- attack_maneuvers: Attack behaviors and jamming
- behavior_sequencer: State machine and behavior orchestration
"""

from .patrol_patterns import (
    Waypoint,
    SafetyZonePatrol,
    AreaPatrol,
    PatternType,
    generate_coverage_path,
    smooth_trajectory
)

from .formation_controller import (
    FormationType,
    FormationController,
    FormationOffset,
    LeaderFollowerBehavior,
    PIDController,
    DronePosition,
    calculate_follower_position,
    avoid_collision
)

from .attack_maneuvers import (
    AttackRole,
    AttackPhase,
    AttackWaypoint,
    JammingBehavior,
    NeutralizationManeuver,
    AttackCoordinator,
    calculate_approach_vector
)

from .behavior_sequencer import (
    BehaviorState,
    BehaviorType,
    BehaviorPriority,
    BehaviorStatus,
    StateTransition,
    BehaviorSequencer,
    SwarmBehaviorCoordinator
)

__all__ = [
    # patrol_patterns
    'Waypoint',
    'SafetyZonePatrol',
    'AreaPatrol',
    'PatternType',
    'generate_coverage_path',
    'smooth_trajectory',
    # formation_controller
    'FormationType',
    'FormationController',
    'FormationOffset',
    'LeaderFollowerBehavior',
    'PIDController',
    'DronePosition',
    'calculate_follower_position',
    'avoid_collision',
    # attack_maneuvers
    'AttackRole',
    'AttackPhase',
    'AttackWaypoint',
    'JammingBehavior',
    'NeutralizationManeuver',
    'AttackCoordinator',
    'calculate_approach_vector',
    # behavior_sequencer
    'BehaviorState',
    'BehaviorType',
    'BehaviorPriority',
    'BehaviorStatus',
    'StateTransition',
    'BehaviorSequencer',
    'SwarmBehaviorCoordinator',
]

__version__ = '1.0.0'
