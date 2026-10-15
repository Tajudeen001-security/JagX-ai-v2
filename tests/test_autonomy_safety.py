from jagx_api.autonomy_safety import assess_scenario, evaluate_drive_conditions, checklist, evaluate_simulation_scenario

def test_autonomy_fails_closed_when_sensors_missing():
    result=assess_scenario("car_simulation",{})
    assert result["permitted"] is False
    assert result["status"]=="blocked"

def test_autonomy_requires_operator_and_emergency_stop():
    telemetry={"sensor_health":"ok","communications":"ok","emergency_stop_available":True,"operator_authorized":True}
    result=assess_scenario("drone_simulation",telemetry)
    assert result["permitted"] is True
    telemetry["operator_authorized"]=False
    assert assess_scenario("drone_simulation",telemetry)["permitted"] is False

def test_real_vehicle_control_mode_is_not_supported():
    result=assess_scenario("live_weapon_targeting",{})
    assert result["permitted"] is False

def test_stopping_distance_estimate_is_labeled_not_control():
    result=evaluate_drive_conditions(10,50)
    assert result["estimate_only"] is True
    assert result["total_stopping_distance_m"] > 0
    assert "Never use" in result["warning"]

def test_stopping_distance_rejects_bad_deceleration():
    try:
        evaluate_drive_conditions(10,20,deceleration_mps2=0)
        assert False, "expected ValueError"
    except ValueError:
        pass

def test_checklists_are_advisory():
    assert checklist("self_driving_car")["mode"]=="advisory"
    assert "simulation" in checklist("drone")["checklist"][0].lower()


def test_simulation_risk_gate_blocks_uncertain_inputs():
    result = evaluate_simulation_scenario("self_driving_car", 0.7, 250, False, 1.0)
    assert result["permitted"] is False
    assert result["status"] == "review_required"
    assert len(result["reasons"]) >= 3


def test_simulation_risk_gate_accepts_only_complete_illustrative_inputs():
    result = evaluate_simulation_scenario("self_driving_car", 0.99, 20, True, 10.0)
    assert result["permitted"] is True
    assert result["mode"] == "simulation_only"
    assert "not validated safety limits" in result["warning"]


def test_simulation_risk_gate_rejects_unknown_domain():
    try:
        evaluate_simulation_scenario("weapon_targeting", 0.99, 20, True, 10.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


from jagx_api.simulation_scenarios import list_simulation_scenarios, run_simulation_checks


def test_simulation_scenarios_include_fault_injection_cases():
    result = list_simulation_scenarios()
    ids = {item["id"] for item in result["scenarios"]}
    assert result["mode"] == "simulation_only"
    assert {"nominal", "sensor_uncertainty", "latency_spike", "override_unavailable",
            "obstacle_too_close", "missing_obstacle_data"} <= ids


def test_simulation_runner_flags_faults_and_keeps_nominal_case_separate():
    result = run_simulation_checks("self_driving_car")
    by_id = {item["scenario_id"]: item for item in result["results"]}
    assert result["scenario_count"] == 6
    assert by_id["nominal"]["assessment"]["status"] == "simulation_checks_passed"
    for scenario_id in ("sensor_uncertainty", "latency_spike", "override_unavailable",
                        "obstacle_too_close", "missing_obstacle_data"):
        assert by_id[scenario_id]["assessment"]["status"] == "review_required"


def test_simulation_runner_rejects_unknown_or_duplicate_cases():
    for selected in (["not-a-scenario"], ["nominal", "nominal"]):
        try:
            run_simulation_checks("drone", selected)
            assert False, "expected ValueError"
        except ValueError:
            pass


def test_simulation_runner_rejects_unsupported_domain():
    try:
        run_simulation_checks("weapon_targeting")
        assert False, "expected ValueError"
    except ValueError:
        pass
