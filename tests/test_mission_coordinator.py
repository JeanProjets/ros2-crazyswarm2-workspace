"""
Unit tests for the Mission Coordinator
"""

import pytest
import time
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "scenarios"))

from mission_coordinator import (
    MissionCoordinator,
    DecisionEngine,
    TelemetryAggregator,
    DroneTelemetry,
    RiskLevel,
)
from mission_state_machine import MissionState


class TestMissionCoordinator:
    """Test the MissionCoordinator class"""

    def test_initialization(self):
        """Test coordinator initialization"""
        mc = MissionCoordinator()
        assert mc.current_mission_state is None
        assert len(mc.drone_telemetry) == 0

    def test_synchronize_drones(self):
        """Test drone synchronization"""
        mc = MissionCoordinator()
        result = mc.synchronize_drones("takeoff", ["cf1", "cf2", "cf3"])
        assert result is True
        assert "takeoff" in mc.sync_points_reached
        assert len(mc.sync_points_reached["takeoff"]) == 3

    def test_broadcast_mission_state(self):
        """Test broadcasting mission state"""
        mc = MissionCoordinator()
        result = mc.broadcast_mission_state(MissionState.PATROL_SEARCH)
        assert result is True
        assert mc.current_mission_state == MissionState.PATROL_SEARCH

    def test_broadcast_rate_limiting(self):
        """Test that broadcasts are rate-limited"""
        mc = MissionCoordinator()

        # First broadcast should succeed
        result1 = mc.broadcast_mission_state(MissionState.PATROL_SEARCH)
        assert result1 is True

        # Immediate second broadcast should be rate-limited
        result2 = mc.broadcast_mission_state(MissionState.JAMMING)
        assert result2 is False

    def test_collect_drone_telemetry(self):
        """Test collecting drone telemetry"""
        mc = MissionCoordinator()

        drone_states = {
            "cf1": {
                "position": (1.0, 2.0, 3.0),
                "velocity": (0.1, 0.2, 0.0),
                "battery": 85.0,
                "status": "flying"
            },
            "cf2": {
                "position": (4.0, 5.0, 6.0),
                "velocity": (0.0, 0.1, 0.0),
                "battery": 90.0,
                "status": "flying"
            }
        }

        telemetry = mc.collect_drone_telemetry(drone_states)

        assert len(telemetry) == 2
        assert "cf1" in telemetry
        assert telemetry["cf1"].battery_level == 85.0
        assert telemetry["cf1"].position == (1.0, 2.0, 3.0)

    def test_aggregate_sensor_data(self):
        """Test aggregating sensor data"""
        mc = MissionCoordinator()

        # Create some telemetry
        mc.drone_telemetry = {
            "cf1": DroneTelemetry(
                drone_id="cf1",
                position=(1.0, 0.0, 1.0),
                velocity=(0.0, 0.0, 0.0),
                battery_level=80.0,
                status="flying",
                timestamp=time.time()
            ),
            "cf2": DroneTelemetry(
                drone_id="cf2",
                position=(3.0, 0.0, 1.0),
                velocity=(0.0, 0.0, 0.0),
                battery_level=90.0,
                status="flying",
                timestamp=time.time()
            )
        }

        fusion = mc.aggregate_sensor_data()

        assert fusion.swarm_center == (2.0, 0.0, 1.0)  # Average of positions
        assert fusion.average_battery == 85.0  # Average of batteries


class TestDecisionEngine:
    """Test the DecisionEngine class"""

    def test_initialization(self):
        """Test decision engine initialization"""
        de = DecisionEngine()
        assert len(de.decision_history) == 0

    def test_evaluate_mission_progress(self):
        """Test mission progress evaluation"""
        de = DecisionEngine()

        # Test different states
        progress_init = de.evaluate_mission_progress(
            MissionState.INITIALIZATION, 5.0, {}
        )
        assert progress_init == 10

        progress_complete = de.evaluate_mission_progress(
            MissionState.MISSION_COMPLETE, 150.0, {}
        )
        assert progress_complete == 100

    def test_assess_risk_level_low(self):
        """Test risk assessment with good conditions"""
        de = DecisionEngine()

        telemetry = {
            "cf1": DroneTelemetry(
                drone_id="cf1",
                position=(0.0, 0.0, 1.0),
                velocity=(0.0, 0.0, 0.0),
                battery_level=80.0,
                status="flying",
                timestamp=time.time(),
                tracking_quality=0.95
            )
        }

        risk = de.assess_risk_level(telemetry)
        assert risk == RiskLevel.LOW

    def test_assess_risk_level_critical_battery(self):
        """Test risk assessment with critical battery"""
        de = DecisionEngine()

        telemetry = {
            "cf1": DroneTelemetry(
                drone_id="cf1",
                position=(0.0, 0.0, 1.0),
                velocity=(0.0, 0.0, 0.0),
                battery_level=15.0,  # Critical
                status="flying",
                timestamp=time.time(),
                tracking_quality=0.95
            )
        }

        risk = de.assess_risk_level(telemetry)
        assert risk == RiskLevel.CRITICAL

    def test_assess_risk_level_high_battery(self):
        """Test risk assessment with low battery"""
        de = DecisionEngine()

        telemetry = {
            "cf1": DroneTelemetry(
                drone_id="cf1",
                position=(0.0, 0.0, 1.0),
                velocity=(0.0, 0.0, 0.0),
                battery_level=35.0,  # Low but not critical
                status="flying",
                timestamp=time.time(),
                tracking_quality=0.95
            )
        }

        risk = de.assess_risk_level(telemetry)
        assert risk == RiskLevel.HIGH

    def test_recommend_abort_critical_battery(self):
        """Test abort recommendation for critical battery"""
        de = DecisionEngine()

        telemetry = {
            "cf1": DroneTelemetry(
                drone_id="cf1",
                position=(0.0, 0.0, 1.0),
                velocity=(0.0, 0.0, 0.0),
                battery_level=15.0,
                status="flying",
                timestamp=time.time()
            )
        }

        should_abort, reason = de.recommend_abort(telemetry, 50.0)
        assert should_abort is True
        assert "battery" in reason.lower()

    def test_recommend_abort_timeout(self):
        """Test abort recommendation for mission timeout"""
        de = DecisionEngine()

        telemetry = {
            "cf1": DroneTelemetry(
                drone_id="cf1",
                position=(0.0, 0.0, 1.0),
                velocity=(0.0, 0.0, 0.0),
                battery_level=80.0,
                status="flying",
                timestamp=time.time()
            )
        }

        should_abort, reason = de.recommend_abort(telemetry, 250.0)  # Over 200s
        assert should_abort is True
        assert "timeout" in reason.lower()

    def test_recommend_abort_insufficient_drones(self):
        """Test abort recommendation for insufficient operational drones"""
        de = DecisionEngine()

        telemetry = {
            "cf1": DroneTelemetry(
                drone_id="cf1",
                position=(0.0, 0.0, 1.0),
                velocity=(0.0, 0.0, 0.0),
                battery_level=15.0,  # Critical
                status="failed",
                timestamp=time.time()
            )
        }

        should_abort, reason = de.recommend_abort(telemetry, 50.0)
        assert should_abort is True

    def test_no_abort_recommended(self):
        """Test that no abort is recommended in good conditions"""
        de = DecisionEngine()

        telemetry = {
            "cf1": DroneTelemetry(
                drone_id="cf1",
                position=(0.0, 0.0, 1.0),
                velocity=(0.0, 0.0, 0.0),
                battery_level=80.0,
                status="flying",
                timestamp=time.time()
            ),
            "cf2": DroneTelemetry(
                drone_id="cf2",
                position=(1.0, 1.0, 1.0),
                velocity=(0.0, 0.0, 0.0),
                battery_level=85.0,
                status="flying",
                timestamp=time.time()
            )
        }

        should_abort, reason = de.recommend_abort(telemetry, 50.0)
        assert should_abort is False
        assert reason == ""


class TestTelemetryAggregator:
    """Test the TelemetryAggregator class"""

    def test_initialization(self):
        """Test aggregator initialization"""
        ta = TelemetryAggregator()
        assert ta is not None

    def test_merge_telemetry(self):
        """Test merging telemetry from multiple drones"""
        ta = TelemetryAggregator()

        telemetry_list = [
            DroneTelemetry(
                drone_id="cf1",
                position=(0.0, 0.0, 1.0),
                velocity=(0.0, 0.0, 0.0),
                battery_level=80.0,
                status="flying",
                timestamp=time.time()
            ),
            DroneTelemetry(
                drone_id="cf2",
                position=(2.0, 0.0, 1.0),
                velocity=(0.0, 0.0, 0.0),
                battery_level=90.0,
                status="flying",
                timestamp=time.time()
            )
        ]

        fusion = ta.merge_telemetry(telemetry_list)

        assert len(fusion.all_positions) == 2
        assert fusion.average_battery == 85.0
        assert fusion.swarm_center == (1.0, 0.0, 1.0)

    def test_merge_empty_telemetry(self):
        """Test merging empty telemetry list"""
        ta = TelemetryAggregator()
        fusion = ta.merge_telemetry([])

        assert len(fusion.all_positions) == 0
        assert fusion.average_battery == 0.0

    def test_calculate_swarm_center(self):
        """Test swarm center calculation"""
        ta = TelemetryAggregator()

        positions = {
            "cf1": (0.0, 0.0, 1.0),
            "cf2": (2.0, 0.0, 1.0),
            "cf3": (1.0, 3.0, 1.0)
        }

        center = ta.calculate_swarm_center(positions)

        assert center == (1.0, 1.0, 1.0)  # Average of all positions

    def test_calculate_swarm_center_empty(self):
        """Test swarm center with no drones"""
        ta = TelemetryAggregator()
        center = ta.calculate_swarm_center({})
        assert center == (0.0, 0.0, 0.0)

    def test_combine_detection_confidence(self):
        """Test combining detection confidence"""
        ta = TelemetryAggregator()

        detections = [
            {"confidence": 0.7},
            {"confidence": 0.85},
            {"confidence": 0.6}
        ]

        combined = ta.combine_detection_confidence(detections)
        assert combined == 0.85  # Maximum confidence

    def test_combine_detection_confidence_empty(self):
        """Test combining with no detections"""
        ta = TelemetryAggregator()
        combined = ta.combine_detection_confidence([])
        assert combined == 0.0

    def test_estimate_mission_completion(self):
        """Test mission completion estimation"""
        ta = TelemetryAggregator()

        # Should estimate based on time
        completion = ta.estimate_mission_completion(
            MissionState.INITIALIZATION,
            5.0
        )

        assert completion >= 0.0
        assert completion <= 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
