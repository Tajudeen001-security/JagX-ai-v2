"""API routes for safe autonomous-systems simulation and decision support."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any
from jagx_api import autonomy_safety, simulation_scenarios

router=APIRouter(prefix="/autonomy",tags=["Autonomy & Safety"])

class AssessmentRequest(BaseModel):
    mode: str=Field(...,min_length=2,max_length=50)
    telemetry: dict[str,Any]=Field(default_factory=dict)

class DrivingEstimateRequest(BaseModel):
    speed_mps: float=Field(...,ge=0,le=100)
    obstacle_distance_m: float=Field(...,ge=0,le=10000)
    reaction_time_s: float=Field(default=1.0,ge=0,le=10)
    deceleration_mps2: float=Field(default=4.0,gt=0,le=15)

class ChecklistRequest(BaseModel):
    domain: str

class SimulationRunRequest(BaseModel):
    domain: str = Field(..., min_length=2, max_length=40)
    scenario_ids: list[str] | None = Field(default=None, max_length=6)


class SimulationRiskRequest(BaseModel):
    domain: str = Field(..., min_length=2, max_length=40)
    sensor_confidence: float = Field(..., ge=0, le=1)
    control_latency_ms: float = Field(..., ge=0, le=60000)
    human_override_ready: bool
    obstacle_distance_m: float | None = Field(default=None, ge=0, le=100000)


@router.get("/simulation/scenarios")
def simulation_scenarios_list():
    return simulation_scenarios.list_simulation_scenarios()

@router.post("/simulation/run-checks")
def simulation_run_checks(req: SimulationRunRequest):
    try:
        return simulation_scenarios.run_simulation_checks(req.domain, req.scenario_ids)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@router.post("/simulation/risk-check")
def simulation_risk_check(req: SimulationRiskRequest):
    try:
        return autonomy_safety.evaluate_simulation_scenario(
            req.domain, req.sensor_confidence, req.control_latency_ms,
            req.human_override_ready, req.obstacle_distance_m
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@router.post("/assess")
def assess(req:AssessmentRequest):
    return autonomy_safety.assess_scenario(req.mode,req.telemetry)

@router.post("/driving/stopping-distance")
def stopping_distance(req:DrivingEstimateRequest):
    try: return autonomy_safety.evaluate_drive_conditions(req.speed_mps,req.obstacle_distance_m,req.reaction_time_s,req.deceleration_mps2)
    except ValueError as exc: raise HTTPException(status_code=400,detail=str(exc))

@router.get("/checklists/{domain}")
def get_checklist(domain:str):
    try: return autonomy_safety.checklist(domain)
    except ValueError as exc: raise HTTPException(status_code=404,detail=str(exc))
