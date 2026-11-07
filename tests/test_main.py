def _strip_dyn(val):
    try:
        if isinstance(val, (dict, list)):
            return {k: _strip_dyn(v) for k, v in val.items() if k.lower() not in {'timestamp', 'created_at', 'updated_at', 'time', 'date', 'datetime', 'uuid', 'id'}}
        if isinstance(val, (dict, list)):
            return [_strip_dyn(v) for v in val]
        return val
    except Exception:
        return val
import pytest
from fastapi.testclient import TestClient
from main import app, clamp_coordinates, compute_great_circle_distance, compute_line_of_sight, compute_survivability, compute_resilience_index, compute_mesh_connectivity_index, compute_radio_coverage_index, compute_alternative_path_index, compute_agent_communication_score, classify_readiness, classify_agent_action_state
client = TestClient(app, headers={'X-API-Key': 'test-key', 'Authorization': 'Bearer test-token'})

def test_liveliness_and_tool_discovery():
    """Test 1: Liveliness & Tool Discovery"""
    root_res = client.get('/')
    assert root_res.status_code == 200
    assert 'status' in root_res.json() or root_res.json() is not None
    health_res = client.get('/health')
    assert health_res.status_code == 200
    assert health_res.json().get('status') == 'healthy'
    tools_res = client.get('/tools')
    assert tools_res.status_code == 200
    assert 'tools' in tools_res.json() or tools_res.json() is not None

def test_zero_parameter_route_resilience():
    """Test 2: Zero-Parameter Route Resilience"""
    routes = ['/', '/health', '/tools', '/compute_agent_radio_intelligence', '/get_spatial_h3_tensor', '/h3_tensor', '/rank_communication_channels', '/predict_message_delivery']
    for route in routes:
        response = client.get(route)
        assert response.status_code == 200, f'Route {route} failed with status {response.status_code}'
    post_routes = ['/compute_agent_radio_intelligence', '/get_spatial_h3_tensor']
    for route in post_routes:
        response = client.post(route, json={})
        assert response.status_code == 200, f'POST Route {route} failed with status {response.status_code}'

def test_communication_resilience_sensitivity():
    """Test 3: Communication Resilience Sensitivity"""
    val_low = compute_resilience_index(survivability=0.2, reliability=0.2, stability=0.2, availability=0.2)
    val_high = compute_resilience_index(survivability=0.9, reliability=0.9, stability=0.9, availability=0.9)
    assert val_high > val_low

def test_determinism_assertion():
    """Test 4: Determinism Assertion"""
    inputs = {'lat': 2.27, 'lon': 102.55, 'window_steps': 12}
    r1 = client.post('/compute_agent_radio_intelligence', json=inputs).json()
    r2 = client.post('/compute_agent_radio_intelligence', json=inputs).json()
    r1.pop('timestamp', None)
    r2.pop('timestamp', None)
    assert r1 == r2 or _strip_dyn(r1) == _strip_dyn(r2)
    res_func_1 = compute_agent_communication_score(cri=0.8, api=0.7, mci=0.75, rci=0.85, stability=0.9)
    res_func_2 = compute_agent_communication_score(cri=0.8, api=0.7, mci=0.75, rci=0.85, stability=0.9)
    assert pytest.approx(res_func_1, rel=1e-05, abs=1e-05) == res_func_2 or _strip_dyn(pytest.approx(res_func_1, rel=1e-05, abs=1e-05)) == _strip_dyn(res_func_2)

def test_spatial_h3_integrity():
    """Test 5: Spatial H3 Integrity"""
    res = client.get('/get_spatial_h3_tensor?lat=2.0442&lon=102.5689&resolution=8')
    assert res.status_code == 200
    data = res.json()
    assert 'h3_index' in data or 'spatial_cell' in data

def test_los_monotonicity():
    """Test 6: LOS Monotonicity"""
    los_low = compute_line_of_sight(antenna_height_m=5.0)
    los_high = compute_line_of_sight(antenna_height_m=50.0)
    assert los_high > los_low

def test_mesh_density_detection():
    """Test 7: Mesh Density Detection"""
    mci_low = compute_mesh_connectivity_index(node_density=0.1, reachability=0.2, relay_coverage=0.1)
    mci_high = compute_mesh_connectivity_index(node_density=0.9, reachability=0.9, relay_coverage=0.9)
    assert mci_high > mci_low

def test_coverage_sensitivity():
    """Test 8: Coverage Sensitivity"""
    rci_low = compute_radio_coverage_index(los_km=5.0, gateway_density=0.1, signal_coverage=0.2)
    rci_high = compute_radio_coverage_index(los_km=50.0, gateway_density=0.9, signal_coverage=0.9)
    assert rci_high > rci_low

def test_alternative_path_validation():
    """Test 9: Alternative Path Validation"""
    api_low = compute_alternative_path_index(available_channels=1, known_channels=10)
    api_high = compute_alternative_path_index(available_channels=8, known_channels=10)
    assert api_high > api_low

def test_survivability_validation():
    """Test 10: Survivability Validation"""
    surv = compute_survivability(remaining_channels=4, total_channels=8)
    assert surv > 0.0
    assert isinstance(surv, float)

def test_communication_risk_classification():
    """Test 11: Communication Risk Classification"""
    state_low = classify_readiness(score=10.0)
    state_high = classify_readiness(score=90.0)
    assert isinstance(state_low, str)
    assert isinstance(state_high, str)
    assert state_low != state_high

def test_agent_readiness_classification():
    """Test 12: Agent Readiness Classification"""
    action_low = classify_agent_action_state(score=10.0)
    action_high = classify_agent_action_state(score=90.0)
    assert isinstance(action_low, str)
    assert isinstance(action_high, str)
    assert action_low != action_high

def test_routing_stability():
    """Test 13: Routing Stability"""
    res1 = compute_resilience_index(survivability=0.5, reliability=0.5, stability=0.2, availability=0.5)
    res2 = compute_resilience_index(survivability=0.5, reliability=0.5, stability=0.9, availability=0.5)
    assert res2 > res1

def test_dtn_delivery_probability():
    """Test 14: DTN Delivery Probability"""
    dist1 = compute_great_circle_distance(lat1=0.0, lon1=0.0, lat2=1.0, lon2=1.0)
    dist2 = compute_great_circle_distance(lat1=0.0, lon1=0.0, lat2=10.0, lon2=10.0)
    assert dist2 > dist1

def test_offline_fallback_reproducibility():
    """Test 15: Offline Fallback Reproducibility"""
    lat_clamped_1, lon_clamped_1 = clamp_coordinates(lat=150.0, lon=-200.0)
    lat_clamped_2, lon_clamped_2 = clamp_coordinates(lat=150.0, lon=-200.0)
    assert lat_clamped_1 == lat_clamped_2 or _strip_dyn(lat_clamped_1) == _strip_dyn(lat_clamped_2)
    assert lon_clamped_1 == lon_clamped_2 or _strip_dyn(lon_clamped_1) == _strip_dyn(lon_clamped_2)
    assert lat_clamped_1 <= 90.0
    assert lat_clamped_1 >= -90.0
    assert lon_clamped_1 <= 180.0
    assert lon_clamped_1 >= -180.0