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
    # Panda3D's real key names (",", "."), not "comma"/"period" (plan 3 warp-key fix).
    for _ in range(3):
        h.send(".")
        h.step(20)
    h.send("0")
    h.step(60)
    h.send(",")
    h.send(",")
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
    h.send(".")                     # Panda3D's real key names (plan 3 warp-key fix)
    h.step(60)
    h.send(",")
    h.step(30)


def solar_porkchop_key(h):
    # Regression: 'p' is bound in both modes, but the porkchop state only exists in
    # the sandbox, so pressing it in the solar viewer raised AttributeError
    # ('_porkchop_future') out of run().
    h.press("OPEN SOLAR VIEWER")
    h.step(10)
    h.send("p")
    h.step(10)


def nudge_keys(h):
    """Edit a burn with the keyboard: taps and holds on every nudge axis (plan 3)."""
    _launch_sandbox(h)
    h.step(10)
    h.send("]")                     # tap: node 10 s ahead
    h.send("]-up")
    h.send("]")                     # hold: node later, ramping
    h.step(90)
    h.send("]-up")
    h.send("j")                     # tap: +1 m/s prograde
    h.send("j-up")
    h.send("l")                     # hold: normal, ramping
    h.step(60)
    h.send("l-up")
    h.send(";")                     # tap: -1 m/s radial
    h.send(";-up")
    h.step(30)


def _click(h, px):
    """A left-click tap at bottom-up pixel ``px`` (headless mouse via hooks)."""
    hooks.set_mouse_px(h.app, px)
    h.step(1)
    h.send("mouse1")
    h.send("mouse1-up")
    h.step(1)


def orbit_menu_add_burn(h):
    """Click your orbit, choose "Add burn here" (plan 3)."""
    _launch_sandbox(h)
    h.step(30)
    _click(h, hooks.orbit_point_px(h.app, 40))
    h.press("Add burn here")
    h.step(30)


def orbit_menu_warp_here(h):
    """Open the orbit menu, dismiss it with Escape, reopen, "Warp to here" (plan 3)."""
    _launch_sandbox(h)
    h.step(30)
    _click(h, hooks.orbit_point_px(h.app, 40))
    h.send("escape")
    _click(h, hooks.orbit_point_px(h.app, 40))
    h.press("Warp to here")
    h.step(10)


def click_body_targets(h):
    """Clicking a body targets it; clicking empty space changes nothing (plan 3)."""
    _launch_sandbox(h)
    h.step(30)
    for _ in range(24):
        h.send("wheel_down")
    h.step(90)
    _click(h, hooks.body_px(h.app, "Moon"))
    _click(h, (5.0, 5.0))
    h.step(10)


def handle_drag_prograde(h):
    """Add a burn from the orbit menu, then pull its prograde handle (plan 3)."""
    _launch_sandbox(h)
    h.step(30)
    _click(h, hooks.orbit_point_px(h.app, 40))
    h.press("Add burn here")
    h.step(5)
    start = hooks.handle_px(h.app, "pro+")
    node = hooks.node_px(h.app)
    hooks.set_mouse_px(h.app, start)
    h.step(1)
    h.send("mouse1")
    hooks.set_mouse_px(h.app, (start[0] + 1.5 * (start[0] - node[0]),
                               start[1] + 1.5 * (start[1] - node[1])))
    h.step(60)
    h.send("mouse1-up")
    h.step(30)


def node_drag_slides_time(h):
    """Add a burn, nudge +1 m/s, then drag the node along the orbit (plan 3)."""
    _launch_sandbox(h)
    h.step(30)
    _click(h, hooks.orbit_point_px(h.app, 40))
    h.press("Add burn here")
    h.send("j")
    h.send("j-up")
    h.step(5)
    hooks.set_mouse_px(h.app, hooks.node_px(h.app))
    h.step(1)
    h.send("mouse1")
    hooks.set_mouse_px(h.app, hooks.orbit_point_px(h.app, 80))
    h.step(10)
    h.send("mouse1-up")
    h.step(5)


def moon_transfer_status(h):
    """Target the Moon (amber status), press Plan Moon transfer (green, ~200 km pass)."""
    _launch_sandbox(h)
    hooks.select_target(h.app, "Moon")
    h.step(150)
    h.press("Plan Moon transfer")
    h.step(30)          # applied, settled (PREVIEW_SETTLE_S = 0.25 s), exact preview in


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
    "nudge_keys": ("sandbox", nudge_keys),
    "orbit_menu_add_burn": ("sandbox", orbit_menu_add_burn),
    "orbit_menu_warp_here": ("sandbox", orbit_menu_warp_here),
    "click_body_targets": ("sandbox", click_body_targets),
    "handle_drag_prograde": ("sandbox", handle_drag_prograde),
    "node_drag_slides_time": ("sandbox", node_drag_slides_time),
    "moon_transfer_status": ("sandbox", moon_transfer_status),
    "solar_boot_warp": ("solar", solar_boot_warp),
    "solar_porkchop_key": ("solar", solar_porkchop_key),
}
