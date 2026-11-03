"""Golden-state tests: every scripted session must reproduce its recording.

These pin OrbitApp's observable behavior (physics state, on-screen text, bound
events, camera) so the app.py split can prove it changed nothing. A failure means
the refactor changed behavior -- fix the code, never the golden.
"""
import json
import math
import os
import subprocess
import sys

import pytest

pytest.importorskip("panda3d")

from tests.render.app_harness.record import GOLDEN_DIR, run_driver  # noqa: E402
from tests.render.app_harness.scenarios import SCENARIOS  # noqa: E402


def _assert_matches(actual, expected, path="$"):
    if isinstance(expected, float) or isinstance(actual, float):
        assert isinstance(actual, (int, float)) and isinstance(expected, (int, float)), path
        # rel 1e-9: cross-platform last-bit noise only; a real change is far larger.
        assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6), (
            f"{path}: {actual!r} != golden {expected!r}"
        )
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected), (
            f"{path}: keys {sorted(actual) if isinstance(actual, dict) else actual!r} "
            f"!= golden {sorted(expected)}"
        )
        for key in expected:
            _assert_matches(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: {actual!r} is not a list"
        if all(not isinstance(x, float) for x in expected + actual):
            assert actual == expected, f"{path}: {actual!r} != golden {expected!r}"
            return
        assert len(actual) == len(expected), f"{path}: length {len(actual)} != {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_matches(a, e, f"{path}[{i}]")
    else:
        assert actual == expected, f"{path}: {actual!r} != golden {expected!r}"


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_scenario_matches_golden(scenario, tmp_path):
    out = tmp_path / f"{scenario}.json"
    proc = run_driver(scenario, str(out), timeout_s=180.0)
    assert proc.returncode == 0, proc.stderr[-3000:]
    actual = json.loads(out.read_text(encoding="utf-8"))
    with open(os.path.join(GOLDEN_DIR, f"{scenario}.json"), encoding="utf-8") as f:
        expected = json.load(f)
    _assert_matches(actual, expected)


def test_app_boots_and_destroys_cleanly(tmp_path):
    # The driver ends every scenario with app.destroy(); a leaked task, future or
    # pool would hang the process or exit non-zero.
    proc = run_driver("boot_idle", str(tmp_path / "boot.json"), timeout_s=60.0)
    assert proc.returncode == 0, proc.stderr[-3000:]


def test_golden_comparison_catches_a_changed_value():
    # Guard against a comparator that accepts anything.
    with open(os.path.join(GOLDEN_DIR, "burn.json"), encoding="utf-8") as f:
        golden = json.load(f)
    tampered = json.loads(json.dumps(golden))
    tampered["vessel"]["fuel_mass_kg"] += 0.01
    with pytest.raises(AssertionError):
        _assert_matches(tampered, golden)
    tampered = json.loads(json.dumps(golden))
    tampered["texts"] = tampered["texts"][1:]
    with pytest.raises(AssertionError):
        _assert_matches(tampered, golden)
