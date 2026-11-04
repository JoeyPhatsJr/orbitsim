"""Deterministic screenshots for eyeballing the app.py split (local tool, not a pytest test).

    python -m tests.render.app_harness.screens --record    # baseline, before refactoring
    python -m tests.render.app_harness.screens --compare   # after each step

Images land in .refactor-screens/{baseline,current}/{sandbox,solar}.png (git-ignored).
GPU output is machine-specific, so these are compared only against a baseline made
on the same machine.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SHOTS_DIR = os.path.join(REPO_ROOT, ".refactor-screens")
MODES = ("sandbox", "solar")
DIFF_LEVEL = 8  # per-channel difference (0-255) that counts as a changed pixel


def _shoot(mode: str, out_path: str) -> None:
    """Runs inside a child process: one ShowBase per process."""
    from tests.render.app_harness.driver import Harness, build_app, install_determinism
    install_determinism()
    from panda3d.core import Filename
    from tests.render.app_harness import scenarios
    app = build_app(mode)
    harness = Harness(app)
    if mode == "solar":
        scenarios.solar_boot_warp(harness)
    else:
        scenarios.boot_idle(harness)
        app.rig.set_distance(1.3e9)   # whole Earth-Moon system in frame
        harness.step(20)
    app.win.save_screenshot(Filename.from_os_specific(out_path))
    app.destroy()


def _diff_fraction(a_path: str, b_path: str) -> float:
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(a_path).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(b_path).convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        return 1.0
    changed = np.abs(a - b).max(axis=2) > DIFF_LEVEL
    return float(changed.mean())


def main(argv) -> int:
    if len(argv) >= 4 and argv[1] == "--shoot":
        _shoot(argv[2], argv[3])
        return 0
    which = "baseline" if "--record" in argv else "current"
    out_dir = os.path.join(SHOTS_DIR, which)
    os.makedirs(out_dir, exist_ok=True)
    status = 0
    for mode in MODES:
        out = os.path.join(out_dir, f"{mode}.png")
        proc = subprocess.run(
            [sys.executable, "-m", "tests.render.app_harness.screens", "--shoot", mode, out],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            print(f"{mode}: screenshot FAILED\n{proc.stderr[-2000:]}")
            status = 1
            continue
        if which == "baseline":
            print(f"{mode}: baseline recorded -> {out}")
            continue
        frac = _diff_fraction(os.path.join(SHOTS_DIR, "baseline", f"{mode}.png"), out)
        print(f"{mode}: identical" if frac == 0.0 else f"{mode}: {frac:.4%} of pixels differ -> {out}")
        if frac >= 0.001:
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv))
