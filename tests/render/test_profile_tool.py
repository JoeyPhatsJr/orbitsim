"""The headless profiler reports frame-time stats and hotspots for a fixed workload."""
import json
import pathlib
import subprocess
import sys

import pytest

pytest.importorskip("panda3d")

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_profile_reports_frame_stats_and_hotspots(tmp_path):
    out = tmp_path / "idle.json"
    proc = subprocess.run(
        [sys.executable, "-m", "tests.render.app_harness.profile", "idle",
         "--frames", "5", "--warmup", "5", "--json", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["workload"] == "idle"
    assert report["frames"] == 5
    for key in ("mean_ms", "p50_ms", "p95_ms", "max_ms"):
        assert report["frame_ms"][key] > 0.0
    assert report["frame_ms"]["max_ms"] >= report["frame_ms"]["p50_ms"]
    assert report["hotspots"], "no hotspots recorded"
    top = report["hotspots"][0]
    assert {"function", "calls", "tottime_ms", "cumtime_ms"} <= set(top)
    assert "frame ms" in proc.stdout   # human-readable summary too


def test_profile_rejects_unknown_workload():
    proc = subprocess.run(
        [sys.executable, "-m", "tests.render.app_harness.profile", "no_such_workload"],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode != 0
    assert "unknown workload" in (proc.stdout + proc.stderr)
