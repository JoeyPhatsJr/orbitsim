"""Deterministic headless OrbitApp driver.

Run one scenario in a fresh process and write a JSON snapshot:

    python -m tests.render.app_harness.driver <scenario> <out.json>

Determinism (installed *before* orbitsim.render.app is imported, so it holds no
matter which module ends up constructing executors or reading the wall clock):
- ThreadPoolExecutor / ProcessPoolExecutor -> InlineExecutor (jobs run at submit)
- time.monotonic -> a fake clock advanced exactly one frame per step
- Panda3D global clock in non-real-time mode at 60 fps, offscreen window
"""
import concurrent.futures
import json
import os
import re
import sys
import tempfile
import time
from concurrent.futures import Future

FRAME_S = 1.0 / 60.0


class InlineExecutor:
    """Executor that runs every job synchronously at submit time."""

    def __init__(self, *args, **kwargs):
        pass

    def submit(self, fn, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # mirror a real executor: the error lives in the future
            future.set_exception(exc)
        return future

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        return iter([fn(*args) for args in zip(*iterables)])

    def shutdown(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.shutdown()
        return False


class FakeMonotonic:
    """Stand-in for time.monotonic: advances only when the harness steps a frame."""

    def __init__(self, start_s: float = 1000.0) -> None:
        self.now_s = start_s

    def __call__(self) -> float:
        return self.now_s

    def advance(self, dt_s: float) -> None:
        self.now_s += dt_s


_fake_monotonic = FakeMonotonic()
_installed = False


def install_determinism(win_size: str = "960 600") -> None:
    global _installed
    if _installed:
        return
    _installed = True
    concurrent.futures.ThreadPoolExecutor = InlineExecutor
    concurrent.futures.ProcessPoolExecutor = InlineExecutor
    time.monotonic = _fake_monotonic
    from panda3d.core import ClockObject, loadPrcFileData
    loadPrcFileData("", f"window-type offscreen\nwin-size {win_size}\naudio-library-name null")
    clock = ClockObject.get_global_clock()
    clock.set_mode(ClockObject.M_non_real_time)
    clock.set_frame_rate(60)


def press(app, label: str) -> None:
    """Click the live DirectButton whose visible text is ``label`` (as a player would)."""
    from direct.gui.DirectButton import DirectButton
    from direct.gui.DirectGuiBase import DirectGuiWidget
    for widget in list(DirectGuiWidget.guiDict.values()):
        if isinstance(widget, DirectButton) and str(widget["text"]).strip() == label:
            command = widget["command"]
            if command is not None:
                command(*widget["extraArgs"])
                return
    raise LookupError(f"no live button labelled {label!r}")


class Harness:
    """What a scenario script may do: step frames, send events, press buttons."""

    def __init__(self, app) -> None:
        self.app = app

    def step(self, frames: int) -> None:
        for _ in range(frames):
            _fake_monotonic.advance(FRAME_S)
            self.app.taskMgr.step()

    def send(self, event: str) -> None:
        self.app.messenger.send(event)

    def press(self, label: str) -> None:
        press(self.app, label)


def _floats(values):
    return [float(x) for x in values]


def snapshot(app) -> dict:
    from tests.render.app_harness import hooks
    vessel = None
    if app.world.vessels:
        v = app.world.vessels[0]
        vessel = {
            "r": _floats(v.state.r),
            "v": _floats(v.state.v),
            "fuel_mass_kg": float(v.fuel_mass_kg),
            "throttle": float(v.throttle),
            "sas_mode": v.sas_mode,
            "orientation": _floats(v.orientation),
            "nodes": [repr(n) for n in v.nodes],
        }
    texts = sorted(
        np_.node().get_text()
        for np_ in app.aspect2d.find_all_matches("**/+TextNode")
        if np_.node().get_text().strip()
    )
    return {
        "sim_time_s": float(app.clock.sim_time_s),
        "warp": float(app.clock.warp),
        "vessel": vessel,
        "target": hooks.current_target_name(app),
        "texts": texts,
        "rig": {
            "distance_m": float(app.rig.distance_m),
            "azimuth": float(app.rig.azimuth),
            "elevation": float(app.rig.elevation),
        },
        # DirectGUI per-widget events carry creation-order ids (click-mouse1-pg17);
        # keep how many there are, not which id each got.
        "events": sorted(re.sub(r"-pg\d+$", "-pg*", str(e)) for e in app.messenger.get_events()),
    }


def build_app(mode: str):
    """Construct OrbitApp for 'sandbox' or 'solar' exactly as orbitsim.__main__ does."""
    from orbitsim.__main__ import _default_world, _solar_world
    from orbitsim.render.app import OrbitApp
    from orbitsim.sim.clock import SimClock
    OrbitApp.QUICKSAVE_PATH = os.path.join(tempfile.mkdtemp(prefix="orbitsim-harness-"), "quick.json")
    if mode == "solar":
        clock = SimClock(sim_time_s=30.0 * 365.25 * 86400.0, warp=1_000_000.0)
        return OrbitApp(_solar_world(), clock, solar_system=True)
    return OrbitApp(_default_world(), SimClock(sim_time_s=0.0, warp=100.0))


def run_scenario(name: str) -> dict:
    install_determinism()
    from tests.render.app_harness.scenarios import SCENARIOS
    mode, script = SCENARIOS[name]
    app = build_app(mode)
    harness = Harness(app)
    script(harness)
    result = snapshot(app)
    app.destroy()
    return result


def main(argv) -> int:
    name, out_path = argv[1], argv[2]
    result = run_scenario(name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, sort_keys=True)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
