"""Keyboard/mouse bindings, throttle, SAS, camera orbit, ship-view toggle, escape."""

import numpy as np
from orbitsim.core.attitude import (
    quat_from_axis_angle,
    quat_multiply,
    quat_normalize,
    quat_rotate_vector,
)


class FlightInput:
    """Stateless controller: holds only ``self.app``; all state lives on OrbitApp."""

    MOUSE_ORBIT_SENS = 3.0     # radians per unit of normalized mouse travel
    ROTATE_RATE_RADPS = 0.8       # manual pitch/yaw/roll rate
    THROTTLE_STEP = 0.5           # throttle change per second for shift/ctrl
    SHIP_VIEW_DISTANCE_M = 80.0   # default close framing for ship view

    def __init__(self, app) -> None:
        self.app = app

    def setup_input(self) -> None:
        self.app.accept("wheel_up", lambda: self.app.rig.zoom(0.86))
        self.app.accept("wheel_down", lambda: self.app.rig.zoom(1.0 / 0.86))
        # Right-click + drag orbits the camera (both sandbox and solar modes).
        self.app._rmb_down = False
        self.app._last_mouse = None
        self.app.accept("mouse3", self._rmb, [True])
        self.app.accept("mouse3-up", self._rmb, [False])
        self.app.accept("arrow_left", lambda: self.app.rig.orbit(-0.1, 0.0))
        self.app.accept("arrow_right", lambda: self.app.rig.orbit(0.1, 0.0))
        self.app.accept("arrow_up", lambda: self.app.rig.orbit(0.0, 0.1))
        self.app.accept("arrow_down", lambda: self.app.rig.orbit(0.0, -0.1))
        # Panda3D names punctuation keys by the character (KeyboardButton.ascii_key(".")
        # is "."), not "period"/"comma" -- those never fired from a real keyboard.
        self.app.accept(".", self.app.warp_controls.warp_up_guarded)  # warp up (blocked while thrusting)
        self.app.accept(",", self.app.clock.warp_down)  # warp down
        self.app.accept("0", self.app.warp_controls.max_warp_guarded)
        self.app.accept("p", self.app.porkchop_modal.toggle_porkchop)  # porkchop delta-V plot
        self.app.accept("f1", self.app.keybind_overlay.toggle)  # keybind help overlay
        self.app.accept("escape", self._handle_escape)

        if not self.app.solar_system and self.app.world.vessels:
            self.app._keys = {k: False for k in ("w", "s", "a", "d", "q", "e", "shift", "control")}
            for k in list(self.app._keys):
                self.app.accept(k, self._set_key, [k, True])
                self.app.accept(f"{k}-up", self._set_key, [k, False])
            self.app._mouse1_down_px = None
            self.app.accept("mouse1", self.app.targeting.on_mouse1_down)   # tap (no drag) picks a target
            self.app.accept("z", self._throttle_full)
            self.app.accept("x", self._throttle_cut)
            self.app.accept("u", self.app.maneuver_editor.toggle_unlimited_dv)
            self.app.accept("t", self.toggle_sas)
            self.app.accept("m", self._toggle_ship_view)   # snap map <-> ship view
            sas_keys = ["PROGRADE", "RETROGRADE", "NORMAL", "ANTINORMAL",
                        "RADIAL_IN", "RADIAL_OUT", "TARGET", "ANTITARGET", "MANEUVER"]
            for i, mode in enumerate(sas_keys, start=1):
                self.app.accept(str(i), self.set_sas, [mode])
            self.app.accept("i", self.app.targeting.plan_intercept)
            # Burn nudges: tap for a step, hold to ramp (see render/node_edit.py).
            from orbitsim.render.node_edit import NUDGE_KEYS
            for key in NUDGE_KEYS:
                self.app.accept(key, self.app.maneuver_editor.on_nudge_key, [key, True])
                self.app.accept(f"{key}-up", self.app.maneuver_editor.on_nudge_key, [key, False])
            self.app.accept("f5", self.app.quicksave.save)
            self.app.accept("f9", self.app.quicksave.load)

    def key_held(self, key: str):
        """Whether the keyboard reports ``key`` down; None when there's no input
        device to ask (headless/offscreen), so callers keep their own state."""
        watcher = getattr(self.app, "mouseWatcherNode", None)
        if watcher is None:
            return None
        from panda3d.core import KeyboardButton
        return bool(watcher.is_button_down(KeyboardButton.ascii_key(key)))

    def _toggle_ship_view(self) -> None:
        """Snap the camera between remembered map and ship-view distances ('m')."""
        from orbitsim.render.ship_model import SHIP_VIEW_NEAR_M
        # "In ship view" means already in close framing; anything farther (incl. a
        # mid-fade zoom) counts as the map side and is remembered as such.
        in_ship_view = self.app.rig.target_distance_m <= SHIP_VIEW_NEAR_M
        if in_ship_view:
            self.app._ship_distance_m = self.app.rig.target_distance_m  # remember ship framing
            self.app.rig.move_to_distance(getattr(self.app, "_map_distance_m", 2.0e7))
        else:
            self.app._map_distance_m = self.app.rig.target_distance_m  # remember map framing
            self.app.rig.move_to_distance(
                getattr(self.app, "_ship_distance_m", self.SHIP_VIEW_DISTANCE_M)
            )

    def _rmb(self, down):
        self.app._rmb_down = down
        self.app._last_mouse = None

    def _handle_escape(self):
        """Close the topmost transient surface before opening settings."""
        if self.app.operation.status.running:
            self.app.trajectory_view.cancel_operation()
            return
        if getattr(self.app.keybind_overlay, "visible", False):
            self.app.keybind_overlay.hide()
            return
        if getattr(self.app, "_porkchop_card", None) is not None:
            self.app._porkchop_card.remove_node()
            self.app._porkchop_card = None
            if getattr(self.app, "_porkchop_modal", None) is not None:
                self.app._porkchop_modal.destroy()
                self.app._porkchop_modal = None
            return
        self.app.settings_panel.toggle()

    def apply_mouse_orbit(self):
        """Orbit the camera while the right mouse button is held and dragged."""
        mw = self.app.mouseWatcherNode
        if mw is None or not (self.app._rmb_down and mw.has_mouse()):
            self.app._last_mouse = None
            return
        x, y = mw.get_mouse_x(), mw.get_mouse_y()
        if self.app._last_mouse is not None:
            dx = x - self.app._last_mouse[0]
            dy = y - self.app._last_mouse[1]
            self.app.rig.orbit(dx * self.MOUSE_ORBIT_SENS, dy * self.MOUSE_ORBIT_SENS)
        self.app._last_mouse = (x, y)

    def _set_key(self, key, down):
        self.app._keys[key] = down

    def _throttle_full(self):
        self.app.world.vessels[0].throttle = 1.0

    def _throttle_cut(self):
        self.app.world.vessels[0].throttle = 0.0

    def toggle_sas(self):
        v = self.app.world.vessels[0]
        v.sas_mode = "STABILITY" if v.sas_mode == "OFF" else "OFF"

    def set_sas(self, mode):
        if mode == "MANEUVER" and self.app.maneuver_editor.current_node().magnitude_mps <= 0.0:
            self.app._flash_message("Plan a maneuver first")
            return
        self.app.world.vessels[0].sas_mode = mode

    def apply_flight_input(self, dt):
        """Manual throttle trim + rotation from held keys (sandbox flight)."""
        if self.app.solar_system or not self.app.world.vessels:
            return
        v = self.app.world.vessels[0]
        k = self.app._keys
        if k["shift"]:
            v.throttle = min(1.0, v.throttle + self.THROTTLE_STEP * dt)
        if k["control"]:
            v.throttle = max(0.0, v.throttle - self.THROTTLE_STEP * dt)
        ax = np.zeros(3)
        if k["w"]:
            ax = ax + np.array([1.0, 0.0, 0.0])    # pitch
        if k["s"]:
            ax = ax + np.array([-1.0, 0.0, 0.0])
        if k["a"]:
            ax = ax + np.array([0.0, -1.0, 0.0])   # yaw left
        if k["d"]:
            ax = ax + np.array([0.0, 1.0, 0.0])
        if k["q"]:
            ax = ax + np.array([0.0, 0.0, 1.0])    # roll
        if k["e"]:
            ax = ax + np.array([0.0, 0.0, -1.0])
        if np.linalg.norm(ax) > 0.0:
            v.sas_mode = "OFF"                      # taking manual control
            world_axis = quat_rotate_vector(v.orientation, ax)
            dq = quat_from_axis_angle(world_axis, self.ROTATE_RATE_RADPS * dt)
            v.orientation = quat_normalize(quat_multiply(dq, v.orientation))
