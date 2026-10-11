"""Engineering decision-support helpers. Estimates are not safety certification or control commands."""
from __future__ import annotations

def capability_map() -> dict:
    return {"areas":["vehicle diagnostics and maintenance planning","drone maintenance and pre-flight checklists",
                     "aircraft maintenance documentation and hazard analysis","industrial equipment preventive maintenance",
                     "robotics test planning","battery/endurance estimates","FMEA worksheet generation"],
            "boundary":"Decision support only. Never use these outputs as direct safety-critical control commands.",
            "requires_independent_review":True}

def endurance_estimate(battery_wh: float, average_watts: float, usable_fraction: float = .8) -> dict:
    if battery_wh <= 0 or average_watts <= 0 or not 0.1 <= usable_fraction <= 1:
        raise ValueError("Battery and load must be positive; usable_fraction must be between 0.1 and 1")
    minutes = battery_wh * usable_fraction / average_watts * 60
    return {"estimated_minutes":round(minutes,2),"estimated_hours":round(minutes/60,3),
            "warning":"Idealized estimate only. Real performance depends on temperature, ageing, peak loads, conversion losses, reserves, and operating conditions."}

def fmea_template(system: str, components: list[str] | None = None) -> dict:
    if not system.strip():
        raise ValueError("System name is required")
    names = [str(x)[:100] for x in (components or ["Power supply","Sensors","Controller","Actuators","Communications"])]
    return {"system":system[:160],"rows":[{"component":n,"failure_mode":"","effect":"","severity_1_10":None,
             "occurrence_1_10":None,"detection_1_10":None,"risk_priority_number":None,
             "mitigation":"","verification_owner":""} for n in names[:50]],
            "note":"Template only. Ratings and mitigations require qualified domain review."}
