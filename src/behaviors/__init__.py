"""
Scenario 4 Behavior Modules

Implements advanced behaviors for high-speed pursuit with obstacle avoidance.
"""

from .obstacle_pursuit import PathFollowerBehavior, Waypoint, calculate_pursuit_velocity
from .elastic_formation import ElasticFormation, FormationOffset, GridMap, get_valid_formation_point
from .reacquisition import OcclusionHandler, TargetState, SearchState
from .safe_strike_v4 import SafeDynamicStrike, StrikeParameters, StrikeState

__all__ = [
    # Obstacle Pursuit
    'PathFollowerBehavior',
    'Waypoint',
    'calculate_pursuit_velocity',

    # Elastic Formation
    'ElasticFormation',
    'FormationOffset',
    'GridMap',
    'get_valid_formation_point',

    # Reacquisition
    'OcclusionHandler',
    'TargetState',
    'SearchState',

    # Safe Strike
    'SafeDynamicStrike',
    'StrikeParameters',
    'StrikeState',
]
