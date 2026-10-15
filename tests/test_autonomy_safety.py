from jagx_api.autonomy_safety import assess_scenario, evaluate_drive_conditions, checklist

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
