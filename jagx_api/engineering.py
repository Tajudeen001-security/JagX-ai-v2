"""Engineering decision-support helpers; never sends control commands to equipment."""
from typing import Any
DOMAINS={"vehicle":["maintenance triage","fault-code explanation","inspection checklist","simulation planning"],"drone":["pre-flight checklist","battery/endurance estimation","maintenance","mission risk review"],"aviation":["maintenance planning","hazard analysis","simulation and checklist support"],"industrial":["FMEA worksheet","root-cause analysis","preventive maintenance","process risk review"],"robotics":["test planning","sensor plausibility review","fault-tree analysis","simulation"]}
def capability_map()->dict:
    return {"domains":DOMAINS,"safety_boundary":"Decision support only. No direct actuator/flight/vehicle control, no certification, and no replacement for qualified engineering review."}
def endurance_estimate(battery_wh:float,average_watts:float,usable_fraction:float=0.8)->dict[str,Any]:
    if battery_wh<=0 or average_watts<=0: raise ValueError("battery_wh and average_watts must be positive")
    if not 0.1<=usable_fraction<=1: raise ValueError("usable_fraction must be between 0.1 and 1.0")
    hours=battery_wh*usable_fraction/average_watts
    return {"estimated_hours":round(hours,3),"estimated_minutes":round(hours*60,1),"assumptions":{"battery_wh":battery_wh,"average_watts":average_watts,"usable_fraction":usable_fraction},"warning":"Idealized estimate only. Validate battery aging, temperature, reserve policy, payload, wind, and manufacturer limits."}
def fmea_template(system:str,components:list[str]|None=None)->dict:
    system=(system or "").strip()[:160]
    if not system: raise ValueError("system is required")
    names=[str(x)[:120] for x in (components or [])[:40] if str(x).strip()]
    rows=[{"component":c,"failure_mode":"","effect":"","cause":"","severity_1_10":None,"occurrence_1_10":None,"detection_1_10":None,"risk_priority_number":None,"mitigation":"","verification_evidence":""} for c in names]
    return {"system":system,"fmea":rows,"guidance":"A qualified team must score and validate each row. Blank scores are not evidence of safety or certification."}
