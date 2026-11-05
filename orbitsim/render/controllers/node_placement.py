"""Hover your orbit for a ghost node; click it for "Add burn here" / "Warp to here"."""

import numpy as np

from orbitsim.render.hud.format import fmt_lead
from orbitsim.render.node_edit import (
    MIN_NODE_LEAD_S, menu_clear_of, menu_pos, node_epoch_ok, node_epoch_window,
    screen_constant_scale,
)
from orbitsim.render.picking import pick_polyline_epoch, point_at

TAP_MOVE_PT = 6.0        # a click that moves further than this is a drag, not a tap
LINE_HIT_PT = 14.0       # how close to the orbit line a click/hover must be
GHOST_RADIUS_PT = 6.0
GHOST_COLOR = (0.22, 0.84, 1.0, 0.6)    # ORBIT_STYLE cyan, translucent


class NodePlacement:
    """Stateless controller: holds only ``self.app``; all state lives on OrbitApp."""

    def __init__(self, app) -> None:
        self.app = app

    def build_node_placement(self) -> None:
        self.app._hover_pick = None
        self.app._ghost_np = None
        self.app._ghost_label = None
        self.app._orbit_menu = None
        self.app._orbit_menu_pick = None      # (epoch_s, world point [m]) the menu is for

    # --- picking -------------------------------------------------------------

    def _pt(self, pt: float) -> float:
        return pt * self.app.line_px_scale()

    def orbit_pick_at(self, px, tol_px, window=None):
        """(epoch_s, world point [m]) of the live line under ``px``, or None."""
        cache = getattr(self.app, "_traj_world_cache", None)
        points = cache[0] if cache else None
        epochs = self.app._traj_epoch_cache[0] if cache else None
        if points is None or epochs is None or px is None:
            return None
        if window is None:
            window = (self.app.clock.sim_time_s + MIN_NODE_LEAD_S, float(epochs[-1]))
        pts_px = self.app.targeting.project_px(points)
        pick = pick_polyline_epoch(px, pts_px, epochs, window[0], window[1], tol_px)
        if pick is None:
            return None
        return pick.epoch_s, point_at(points, pick)

    def node_window(self):
        """Where a burn may go now: up to one orbit, or the first non-Earth SOI entry."""
        from orbitsim.core.elements import state_to_elements
        now = self.app.clock.sim_time_s
        period = None
        try:
            elem = state_to_elements(self.app.world.vessels[0].state)
            if elem.e < 1.0 and np.isfinite(elem.period_s):
                period = float(elem.period_s)
        except ValueError:
            period = None
        encs = self.app._traj_encounters[0] if getattr(self.app, "_traj_encounters", None) else None
        others = [e for e in (encs or []) if e.body_name not in ("Earth", "Sun")]
        entries = [e.entry_epoch_s for e in others]
        in_other_soi = any(e.start_index == 0 for e in others)   # line starts inside it
        epochs = self.app._traj_epoch_cache[0] if getattr(self.app, "_traj_epoch_cache", None) else None
        line_end = float(epochs[-1]) if epochs is not None else now + MIN_NODE_LEAD_S
        return node_epoch_window(now, period, entries, line_end, in_other_soi=in_other_soi)

    # --- mouse ---------------------------------------------------------------

    def on_mouse1_down(self) -> None:
        if self.app.operation.status.running:
            self.app._mouse1_down_px = None
            return
        if self.menu_open():
            self.close_menu()                     # a click elsewhere just closes the menu
            self.app._mouse1_down_px = None
            return
        px = self.app.flight_input.mouse_px()
        handle = self.app.node_handles.hit_test(px)
        if handle is not None:
            self.app.node_handles.begin_drag(handle, px)   # a drag, not a tap
            self.app._mouse1_down_px = None
            return
        self.app._mouse1_down_px = px

    def on_mouse1_up(self) -> None:
        if self.app.node_handles.dragging() is not None:
            self.app.node_handles.end_drag()
            return
        down = getattr(self.app, "_mouse1_down_px", None)
        self.app._mouse1_down_px = None
        if down is None:
            return
        click = self.app.flight_input.mouse_px()
        if click is None:
            return
        tap = self._pt(TAP_MOVE_PT)
        if abs(click[0] - down[0]) > tap or abs(click[1] - down[1]) > tap:
            return                                 # a drag, not a tap
        if self.app.targeting.pick_body_at(click):
            return
        hit = self.orbit_pick_at(click, self._pt(LINE_HIT_PT))
        if hit is not None:
            self._open_menu(*hit, click)

    # --- menu ----------------------------------------------------------------

    def menu_open(self) -> bool:
        return getattr(self.app, "_orbit_menu", None) is not None

    def close_menu(self) -> None:
        if getattr(self.app, "_orbit_menu", None) is not None:
            self.app._orbit_menu.destroy()
        self.app._orbit_menu = None
        self.app._orbit_menu_pick = None

    def _has_burn(self) -> bool:
        return (self.app._node_epoch_s is not None
                or any(v != 0.0 for v in self.app._dv.values()))

    def _open_menu(self, epoch_s, point_m, click_px) -> None:
        from direct.gui.DirectButton import DirectButton
        from direct.gui.DirectFrame import DirectFrame
        from orbitsim.render.ui.theme import THEME, button_options
        self.close_menu()
        self.app._orbit_menu_pick = (epoch_s, np.asarray(point_m, dtype=np.float64))
        items = []
        if epoch_s <= self.node_window()[1]:
            items.append(("Move burn here" if self._has_burn() else "Add burn here", self._add_here))
        items.append(("Warp to here", self._warp_here))
        height = 0.03 + 0.07 * len(items)
        frame = DirectFrame(frameColor=THEME.panel, frameSize=(0.0, 0.46, -height, 0.0),
                            parent=self.app.aspect2d)
        for i, (label, cmd) in enumerate(items):
            DirectButton(text=label, scale=0.045, pos=(0.23, 0.0, -0.06 - 0.07 * i),
                         command=cmd, parent=frame, **button_options())
        self.app._orbit_menu = frame
        self.app._orbit_menu_labels = [label for label, _ in items]
        self._place_menu(click_px)

    def _place_menu(self, px) -> None:
        w, h = self.app.win.get_x_size(), self.app.win.get_y_size()
        x, z = menu_pos(px, w, h)
        navball = getattr(self.app, "navball", None)
        if navball is not None:
            # The navball has its own display region drawn over all 2D GUI: keep clear.
            left, right, bottom, top = navball.dr.get_dimensions()
            aspect = w / h
            x, z = menu_clear_of((x, z), ((left * 2 - 1) * aspect, (right * 2 - 1) * aspect,
                                          bottom * 2 - 1, top * 2 - 1))
        self.app._orbit_menu.set_pos(x, 0.0, z)

    def _add_here(self) -> None:
        epoch_s = self.app._orbit_menu_pick[0]
        self.close_menu()
        if node_epoch_ok(epoch_s, self.app.clock.sim_time_s):
            self.app.maneuver_editor.place_node(epoch_s)
        else:
            self.app._flash_message("That point has passed")

    def _warp_here(self) -> None:
        epoch_s = self.app._orbit_menu_pick[0]
        self.close_menu()
        self.app.targeting.coast_to(epoch_s)

    # --- per frame -----------------------------------------------------------

    def _hide_ghost(self) -> None:
        if self.app._ghost_np is not None:
            self.app._ghost_np.hide()
        if self.app._ghost_label is not None:
            self.app._ghost_label.node().set_text("")
            self.app._ghost_label.hide()

    def frame_hover(self) -> None:
        """Ghost node under the mouse on your orbit; keep an open menu on its point."""
        if not hasattr(self.app, "_ghost_np"):
            return                                 # not built (solar viewer)
        if self.menu_open():
            self._hide_ghost()
            px = self.app.targeting.project_px(self.app._orbit_menu_pick[1][None, :])[0]
            if np.isfinite(px).all():
                self._place_menu(px)
            return
        fi = self.app.flight_input
        mouse = fi.mouse_px()
        handles = self.app.node_handles
        if (mouse is None or fi.mouse_over_gui() or getattr(self.app, "_rmb_down", False)
                or self.app.operation.status.running or handles.dragging() is not None
                or handles.hit_test(mouse) is not None):
            self._hide_ghost()
            return
        hit = self.orbit_pick_at(mouse, self._pt(LINE_HIT_PT))
        self.app._hover_pick = hit
        if hit is None:
            self._hide_ghost()
            return
        epoch_s, point = hit
        self._show_ghost(point, fmt_lead(epoch_s - self.app.clock.sim_time_s))

    def _show_ghost(self, point_m, text) -> None:
        from orbitsim.render.geometry import make_uv_sphere
        from orbitsim.render.world_labels import build_world_label
        app = self.app
        if app._ghost_np is None:
            app._ghost_np = make_uv_sphere(1.0, 8, 12)
            app._ghost_np.reparent_to(app.render)
            app._ghost_np.set_color(*GHOST_COLOR)
            app._ghost_np.set_transparency(True)
            app._ghost_np.set_light_off(1)
            app._ghost_label = build_world_label(app.render, "", color=GHOST_COLOR[:3] + (1.0,),
                                                 scale=11.0)
        from panda3d.core import Point3
        x, y, z = app.transform.to_render(point_m)
        dist = (app.camera.get_pos(app.render) - Point3(x, y, z)).length()
        r = screen_constant_scale(dist, self._pt(GHOST_RADIUS_PT), app.camLens.get_fov()[1],
                                  app.win.get_y_size())
        app._ghost_np.set_pos(x, y, z)
        app._ghost_np.set_scale(r)
        app._ghost_np.show()
        app._ghost_label.node().set_text(text)
        app._ghost_label.set_pos(x, y, z + 8.0)
        app._ghost_label.show()
