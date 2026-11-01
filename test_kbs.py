"""Unit tests for the KBS (run: python -m pytest tests -q   or   python tests/test_kbs.py)."""
import sys, math; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from kbs.engine import KnowledgeBase, InferenceEngine, combine_cf, level_of
from kbs.recommender import recommend

kb = KnowledgeBase(); eng = InferenceEngine(kb)
NORMAL = dict(type="M", air_temp_k=300.0, process_temp_k=310.0, rpm=1520, torque_nm=41.0, tool_wear_min=60, observations="")

def ev(**kw): return {**NORMAL, **kw}

def test_combine_cf():
    assert math.isclose(combine_cf(0.6, 0.5), 0.8)
    assert math.isclose(combine_cf(-0.6, -0.5), -0.8)
    assert math.isclose(combine_cf(0.8, -0.4), 0.4 / 0.6)

def test_normal_has_no_fault():
    d = eng.diagnose(NORMAL); assert all(b < 0.1 for b in d.belief.values())
    assert recommend(kb, d)["status"] == "NORMAL"

def test_hdf_confirmed_and_override():
    d = eng.diagnose(ev(process_temp_k=308.2, rpm=1300))
    assert d.level("HDF") == "CONFIRMED"
    assert any(r["id"] == "R02" for r, by in d.suppressed)          # early-warning rule overridden by R01

def test_hdf_boundary_rule():
    d = eng.diagnose(ev(air_temp_k=300.0, process_temp_k=308.6, rpm=1300))      # gap exactly 8.6 K
    assert 0.3 < d.belief["HDF"] < 0.7                                           # possible, not confirmed
    assert eng.diagnose(ev(air_temp_k=300.0, process_temp_k=308.6, rpm=1500)).belief["HDF"] < 0.1

def test_pwf_both_directions():
    assert eng.diagnose(ev(torque_nm=20.0, rpm=1500)).level("PWF") == "CONFIRMED"      # ~3140 W
    assert eng.diagnose(ev(torque_nm=62.0, rpm=1500)).level("PWF") == "CONFIRMED"      # ~9740 W

def test_osf_depends_on_type():
    e = ev(tool_wear_min=200, torque_nm=58.0, rpm=1500)       # strain 11600
    assert eng.diagnose({**e, "type": "L"}).level("OSF") == "CONFIRMED"      # limit 11000
    assert eng.diagnose({**e, "type": "H"}).belief["OSF"] < 0.1               # limit 13000

def test_conflict_resolution_by_disconfirming_rule():
    d = eng.diagnose(ev(observations="S_BURN"))               # operator smells burning, power normal
    assert d.belief["PWF"] < 0.1                              # contradicted by the power sensor
    d2 = InferenceEngine(kb.without(["disconfirming"])).diagnose(ev(observations="S_BURN"))
    assert d2.belief["PWF"] >= 0.1                            # without conflict handling the report stays at WATCH level
    d3 = InferenceEngine(KnowledgeBase(calibrated=False).without(["disconfirming"])).diagnose(ev(observations="S_BURN"))
    assert d3.belief["PWF"] >= 0.4                            # with uncalibrated expert priors it would be a full false alarm

def test_missing_data_does_not_fire_and_is_reported():
    d = eng.diagnose(ev(air_temp_k=None, process_temp_k=308.0, rpm=1300))
    assert d.belief["HDF"] < 0.1
    assert any("air_temp_k" in miss for r, miss in d.undetermined)
    bw = eng.prove("HDF", ev(air_temp_k=None, process_temp_k=308.0, rpm=1300))
    assert "air_temp_k" in bw["needs"] and bw["best_case"] >= 0.9

def test_numpy_inputs_give_same_result():
    e = ev(process_temp_k=308.2, rpm=1300)
    en = {k: (np.float64(v) if isinstance(v, float) else (np.int64(v) if isinstance(v, int) else v)) for k, v in e.items()}
    assert eng.diagnose(e).belief == eng.diagnose(en).belief

def test_safety_constraint_cannot_be_deferred():
    d = eng.diagnose(ev(torque_nm=62.0, rpm=1500))             # PWF confirmed, safety risk 5
    for p in kb.profiles:
        item = [i for i in recommend(kb, d, p)["items"] if i["fault"] == "PWF"][0]
        assert item["mode"] == "IMMEDIATE" and item["safety_override"]

def test_profile_changes_mode_for_moderate_severity():
    d = eng.diagnose(ev(process_temp_k=308.2, rpm=1300))       # HDF confirmed, safety risk 3
    assert recommend(kb, d, "safety_first")["items"][0]["mode"] == "IMMEDIATE"
    assert recommend(kb, d, "production_first")["items"][0]["mode"] == "PLANNED"

if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns: f(); print("PASS", f.__name__)
    print(f"{len(fns)} tests passed")
