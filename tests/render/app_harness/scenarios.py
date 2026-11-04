"""Scripted player sessions. Each is (mode, script(harness)); never edit after goldens exist."""
from tests.render.app_harness import hooks


def _launch_sandbox(h):
    h.press("LAUNCH SANDBOX")


def boot_idle(h):
    _launch_sandbox(h)
    h.step(120)


def burn(h):
    _launch_sandbox(h)
    h.step(30)
    h.send("z")
    h.step(60)
    h.send("x")
    h.step(60)


def sas_and_views(h):
    _launch_sandbox(h)
    h.send("t")
    for key in "123456789":
        h.send(key)
        h.step(10)
    h.send("m")
    h.step(20)
    h.send("m")
    h.send("f1")
    h.send("f1")


def node_edit_execute(h):
    _launch_sandbox(h)
    hooks.set_jog(h.app, "pro", 0.6)
    h.step(30)
    h.send("mouse1-up")
    h.press("Next Ap")
    h.step(10)
    h.press("Execute Burn")
    h.step(900)


def warp_keys(h):
    _launch_sandbox(h)
    for _ in range(3):
        h.send("period")
        h.step(20)
    h.send("0")
    h.step(60)
    h.send("comma")
    h.send("comma")
    h.step(20)


def target_intercept(h):
    _launch_sandbox(h)
    hooks.select_target(h.app, "Moon")
    h.step(10)
    h.send("i")
    h.step(120)


def porkchop_escape(h):
    _launch_sandbox(h)
    hooks.select_target(h.app, "Moon")
    h.send("p")
    h.step(60)
    h.send("escape")
    h.step(10)
    h.send("p")
    h.step(30)
    h.press("CLOSE")
    h.step(10)


def quicksave_roundtrip(h):
    _launch_sandbox(h)
    h.step(30)
    h.send("f5")
    h.send("z")
    h.step(60)
    h.send("x")
    h.send("f9")
    h.step(30)


def zoom_with_target_node(h):
    # Regression: zooming rebuilds the planet orbit lines every frame; that loop used
    # to rebind `node` (the maneuver node) to a NodePath, crashing the closest-approach
    # block with AttributeError: 'NodePath' object has no attribute 'magnitude_mps'.
    _launch_sandbox(h)
    hooks.select_target(h.app, "Moon")
    h.step(10)
    h.send("i")
    h.step(10)
    for i in range(120):
        if i % 10 == 0:
            h.send("wheel_up")
        h.step(1)


def solar_boot_warp(h):
    h.press("OPEN SOLAR VIEWER")
    h.step(60)
    h.send("period")
    h.step(60)
    h.send("comma")
    h.step(30)


SCENARIOS = {
    "boot_idle": ("sandbox", boot_idle),
    "burn": ("sandbox", burn),
    "sas_and_views": ("sandbox", sas_and_views),
    "node_edit_execute": ("sandbox", node_edit_execute),
    "warp_keys": ("sandbox", warp_keys),
    "target_intercept": ("sandbox", target_intercept),
    "porkchop_escape": ("sandbox", porkchop_escape),
    "quicksave_roundtrip": ("sandbox", quicksave_roundtrip),
    "zoom_with_target_node": ("sandbox", zoom_with_target_node),
    "solar_boot_warp": ("solar", solar_boot_warp),
}
