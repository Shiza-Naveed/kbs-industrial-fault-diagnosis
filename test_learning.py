"""Tests for the learning component (python tests/test_learning.py). All files go to a temp folder; real knowledge is never touched."""
import sys, json, tempfile, hashlib; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kbs import learning
from kbs.config import KB_DIR
from kbs.engine import KnowledgeBase, InferenceEngine
base = KB_DIR / "rules_calibrated.json"
eng = InferenceEngine(KnowledgeBase())
EV = dict(type="M", air_temp_k=300.0, process_temp_k=310.0, rpm=1520, torque_nm=41.0, tool_wear_min=60, observations="S_COOLANT_LOW")   # operator-only report (R19)
def cf(path, rid): return {r["id"]: r for r in json.loads(Path(path).read_text())}[rid]["cf"]
def run(n_conf, n_rej, tmp):
    fb, out = Path(tmp) / "fb.jsonl", Path(tmp) / "learned.json"; d = eng.diagnose(EV)
    assert "R19" in [r["id"] for r in d.fired]
    for _ in range(n_conf): learning.record(EV, d, ["HDF"], fb)
    for _ in range(n_rej): learning.record(EV, d, [], fb)
    return learning.relearn(base, fb, out), out
def test_confirmations_raise_cf():
    with tempfile.TemporaryDirectory() as t:
        ch, out = run(60, 0, t); assert cf(out, "R19") > cf(base, "R19") and any(c["rule"] == "R19" for c in ch)
def test_rejections_lower_cf_with_floor():
    with tempfile.TemporaryDirectory() as t:
        ch, out = run(0, 200, t); assert 0.05 <= cf(out, "R19") < cf(base, "R19")
def test_one_case_moves_cf_only_slightly():
    with tempfile.TemporaryDirectory() as t:
        ch, out = run(1, 0, t); assert abs(cf(out, "R19") - cf(base, "R19")) <= 0.03
def test_disconfirming_rules_not_learned_and_originals_untouched():
    h = hashlib.md5(base.read_bytes()).hexdigest(); rules0 = {r["id"]: r["cf"] for r in json.loads(base.read_text())}
    with tempfile.TemporaryDirectory() as t:
        ch, out = run(30, 30, t); new = {r["id"]: r["cf"] for r in json.loads(out.read_text())}
        assert all(new[k] == v for k, v in rules0.items() if k in ("R28", "R29", "R30", "R31"))
    assert hashlib.md5(base.read_bytes()).hexdigest() == h
def test_reset_and_status():
    with tempfile.TemporaryDirectory() as t:
        fb, out = Path(t) / "fb.jsonl", Path(t) / "l.json"; d = eng.diagnose(EV); learning.record(EV, d, ["HDF"], fb); learning.relearn(base, fb, out)
        assert learning.status(fb, out)["feedback_cases"] == 1 and learning.status(fb, out)["learned_rules_file"]
        learning.reset(fb, out); assert learning.status(fb, out)["feedback_cases"] == 0 and not out.exists()
def test_loader_uses_learned_only_when_asked():
    assert KnowledgeBase().rules_file == "rules_calibrated.json"      # evaluation scripts are never affected by feedback
if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns: f(); print("PASS", f.__name__)
    print(f"{len(fns)} learning tests passed")
