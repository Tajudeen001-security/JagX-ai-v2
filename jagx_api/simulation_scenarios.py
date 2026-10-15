"""Deterministic fault-injection cases for advisory autonomy simulations."""
from __future__ import annotations

from typing import Any

from jagx_api.autonomy_safety import evaluate_simulation_scenario

# Synthetic inputs only. These cases never interact with vehicles or actuators.
_SCENARIOS: dict[str, dict[str, Any]] = {
    "nominal": {
        "description": "Nominal synthetic readings with human override ready.",
        "sensor_confidence": 0.99, "control_latency_ms": 20,
        "human_override_ready": True, "obstacle_distance_m": 10.0,
    },
    "sensor_uncertainty": {
        "description": "Inject low sensor confidence.",
        "sensor_confidence": 0.70, "control_latency_ms": 20,
        "human_override_ready": True, "obstacle_distance_m": 10.0,
    },
    "latency_spike": {
        "description": "Inject a high control-loop latency reading.",
        "sensor_confidence": 0.99, "control_latency_ms": 250,
        "human_override_ready": True, "obstacle_distance_m": 10.0,
    },
    "override_unavailable": {
        "description": "Simulate an unavailable human override.",
        "sensor_confidence": 0.99, "control_latency_ms": 20,
        "human_override_ready": False, "obstacle_distance_m": 10.0,
    },
    "obstacle_too_close": {
        "description": "Simulate an obstacle inside the illustrative caution zone.",
        "sensor_confidence": 0.99, "control_latency_ms": 20,
        "human_override_ready": True, "obstacle_distance_m": 1.0,
    },
    "missing_obstacle_data": {
        "description": "Simulate missing obstacle-distance data.",
        "sensor_confidence": 0.99, "control_latency_ms": 20,
        "human_override_ready": True, "obstacle_distance_m": None,
    },
}

_SUPPORTED_DOMAINS = {"self_driving_car", "drone", "aviation", "industrial_robot"}


def list_simulation_scenarios() -> dict[str, Any]:
    """Return safe, synthetic fault-injection cases and their intended purpose."""
    return {
        "mode": "simulation_only",
        "scenarios": [
            {"id": scenario_id, "description": values["description"]}
            for scenario_id, values in _SCENARIOS.items()
        ],
        "warning": "Synthetic screening only; not validated safety limits or real-world control.",
    }


def run_simulation_checks(domain: str, scenario_ids: list[str] | None = None) -> dict[str, Any]:
    """Run selected synthetic cases through the conservative risk gate."""
    if domain not in _SUPPORTED_DOMAINS:
        raise ValueError("Unsupported simulation domain")
    selected = scenario_ids if scenario_ids is not None else list(_SCENARIOS)
    if not selected:
        raise ValueError("Select at least one scenario")
    if len(selected) > len(_SCENARIOS):
        raise ValueError("Too many scenarios requested")
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate scenario IDs are not allowed")
    unknown = [item for item in selected if item not in _SCENARIOS]
    if unknown:
        raise ValueError("Unknown scenario ID(s): " + ", ".join(unknown))

    results = []
    for scenario_id in selected:
        case = _SCENARIOS[scenario_id]
        assessment = evaluate_simulation_scenario(
            domain=domain,
            sensor_confidence=case["sensor_confidence"],
            control_latency_ms=case["control_latency_ms"],
            human_override_ready=case["human_override_ready"],
            obstacle_distance_m=case["obstacle_distance_m"],
        )
        results.append({
            "scenario_id": scenario_id,
            "description": case["description"],
            "expected_review_required": scenario_id != "nominal" or domain in {"self_driving_car", "drone"} and scenario_id == "missing_obstacle_data",
            "assessment": assessment,
        })
    return {
        "domain": domain,
        "mode": "simulation_only",
        "scenario_count": len(results),
        "review_required_count": sum(1 for item in results if item["assessment"]["status"] == "review_required"),
        "results": results,
        "warning": "This is deterministic fault injection with synthetic inputs. Passing is not proof of safety and never authorizes real-world action.",
    }
