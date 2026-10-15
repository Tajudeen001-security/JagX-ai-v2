"""Safety-first autonomy helpers for simulation and advisory use only.

This module intentionally cannot command real vehicles, aircraft, weapons, or actuators.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any

@dataclass
class SafetyAssessment:
    mode: str
    status: str
    permitted: bool
    reasons: list[str]
    required_checks: list[str]

def assess_scenario(mode: str, telemetry: dict[str, Any]) -> dict[str, Any]:
    """Assess a simulated/advisory scenario; missing telemetry fails closed."""
    allowed_modes = {"car_simulation", "drone_simulation", "aviation_advisory", "robotics_simulation"}
    if mode not in allowed_modes:
        return asdict(SafetyAssessment(mode, "unsupported_mode", False,
            ["This module supports simulation/advisory modes only."],
            ["Use a certified controller and independent safety system for real vehicles."]))
    reasons=[]
    checks=[]
    if telemetry.get("sensor_health") != "ok":
        reasons.append("Sensor health is missing or not confirmed healthy.")
    if telemetry.get("communications") != "ok":
        reasons.append("Communications status is missing or degraded.")
    if telemetry.get("emergency_stop_available") is not True:
        reasons.append("Independent emergency-stop capability is not confirmed.")
    if telemetry.get("operator_authorized") is not True:
        reasons.append("Authorized human oversight is not confirmed.")
    checks.extend(["Validate sensor freshness and cross-sensor agreement",
                   "Check legal operating area and geofence",
                   "Verify certified controller and independent safety interlocks",
                   "Require human authorization before any real-world action"])
    return asdict(SafetyAssessment(mode, "blocked" if reasons else "simulation_checks_passed",
        not reasons, reasons, checks))

def evaluate_drive_conditions(speed_mps: float, obstacle_distance_m: float,
                              reaction_time_s: float = 1.0, deceleration_mps2: float = 4.0) -> dict[str, Any]:
    """Idealized educational stopping-distance estimate, not a driving controller."""
    vals=[speed_mps, obstacle_distance_m, reaction_time_s, deceleration_mps2]
    if any(v < 0 for v in vals) or deceleration_mps2 <= 0:
        raise ValueError("Inputs must be non-negative and deceleration must be greater than zero")
    reaction=max(0.0,speed_mps)*reaction_time_s
    braking=(speed_mps**2)/(2*deceleration_mps2)
    stopping=reaction+braking
    margin=obstacle_distance_m-stopping
    return {"estimate_only":True,"reaction_distance_m":round(reaction,2),
            "braking_distance_m":round(braking,2),"total_stopping_distance_m":round(stopping,2),
            "obstacle_distance_m":round(obstacle_distance_m,2),"estimated_margin_m":round(margin,2),
            "warning":"Idealized calculation only. Never use as a real-time driving or braking command."}

def checklist(domain: str) -> dict[str, Any]:
    checklists={
      "self_driving_car":["Use simulation and closed-course testing first","Verify redundant perception and localization","Check sensor freshness, uncertainty and disagreement","Validate pedestrian/cyclist detection across conditions","Test safe minimal-risk stop on faults","Use independent emergency stop and safety driver","Follow applicable vehicle safety and road regulations"],
      "drone":["Test all flight plans and fault responses in simulation before flight","Confirm legal airspace and operator authorization","Validate battery and return-to-home reserve","Check weather and link quality","Confirm geofence and lost-link behavior","Keep collision avoidance independent of language-model output","Test in simulation before flight"],
      "aviation":["Use approved aircraft performance data","Verify weather, NOTAMs and route constraints","Cross-check fuel/endurance and reserves","Confirm pilot review and approved procedures","Never use generated advice as sole basis for flight"],
      "industrial_robot":["Validate guarding and emergency stops","Use certified safety PLC/interlocks","Verify workspace and human-presence sensing","Test safe-state behavior after communications loss","Require authorized commissioning and change control"]}
    if domain not in checklists: raise ValueError("Unsupported domain")
    return {"domain":domain,"mode":"advisory","checklist":checklists[domain],
            "warning":"Checklist support only; use applicable standards and qualified professionals."}
