"""3D drag handles on the burn node: prograde/normal/radial delta-V (and, in Task 4,
dragging the node itself along the orbit)."""

import math

from orbitsim.render.node_edit import (
    DRAG_MAX_RATE_MPS, HANDLE_HIT_PX, HANDLE_OFFSET_PX, HANDLE_RADIUS_PX, HANDLES,
    MAX_DRAG_PX, MIN_AXIS_PX, NODE_HIT_PX, NODE_RADIUS_PX, HandleDrag, axis_unit_px,
    handle_drag_value, jog_rate, screen_constant_scale, snap_dv, snap_epoch,
)


class NodeHandles:
    """Stateless controller: holds only ``self.app``; all state lives on OrbitApp."""

    def __init__(self, app) -> None:
        self.app = app

    def build_node_handles(self) -> None:
        self.app._drag = None
        self.app._handle_nps = {}
        self.app._handle_px = {}
        self.app._node_px = None
        self.app._handle_hover = None

    def _pt(self, pt: float) -> float:
        return pt * self.app.line_px_scale()

    def _hide(self) -> None:
        for np_ in self.app._handle_nps.values():
            np_.hide()
        self.app._handle_px = {}
        self.app._node_px = None

    def _handle_np(self, kind, rgb, minus):
        np_ = self.app._handle_nps.get(kind)
        if np_ is None:
            from orbitsim.render.geometry import make_uv_sphere
            np_ = make_uv_sphere(1.0, 8, 12)
            np_.reparent_to(self.app.render)
            np_.set_light_off(1)
            np_.set_transparency(True)
            np_.set_color(*rgb, 0.55 if minus else 1.0)
            np_.set_bin("fixed", 40)
            np_.set_depth_test(False)
            self.app._handle_nps[kind] = np_
        return np_

    def frame_handles(self) -> None:
        """Place the node marker and its six handles at a constant on-screen size."""
        app = self.app
        if not hasattr(app, "_handle_nps"):
            return
        marker = getattr(app, "_node_marker_np", None)
        if marker is None or app._node_epoch_s is None:
            self._hide()
            return
        from panda3d.core import Point3
        from orbitsim.core.maneuvers import rtn_basis
        from orbitsim.core.propagate import propagate_kepler
        v0 = app.world.vessels[0]
        burn = propagate_kepler(v0.state, app._node_epoch_s - v0.state.epoch_s)
        try:
            axes = dict(zip(("pro", "nrm", "rad"), rtn_basis(burn.r, burn.v)))
        except ValueError:
            self._hide()
            return
        pos = marker.get_pos(app.render)
        d = (app.camera.get_pos(app.render) - Point3(pos)).length()
        fov_v, h = app.camLens.get_fov()[1], app.win.get_y_size()

        def size(px):
            return screen_constant_scale(d, self._pt(px), fov_v, h)
        marker.set_scale(size(NODE_RADIUS_PX))
        node_px = app.targeting.marker_px(burn.r)
        app._node_px = node_px
        offset = size(HANDLE_OFFSET_PX) * app.transform.scale_m_per_unit   # metres
        drag = app._drag
        app._handle_px = {}
        for kind, axis, sign, rgb in HANDLES:
            np_ = self._handle_np(kind, rgb, sign < 0)
            out = offset
            if drag is not None and drag.kind == kind:
                out *= 1.0 + max(0.0, self._drag_value())
            world = burn.r + sign * axes[axis] * out
            px = app.targeting.marker_px(world)
            unit = None if (px is None or node_px is None) else axis_unit_px(
                node_px, px, self._pt(MIN_AXIS_PX))
            if unit is None:
                np_.hide()
                app._handle_px[kind] = None
                continue
            np_.set_pos(*app.transform.to_render(world))
            r = size(HANDLE_RADIUS_PX) * (0.75 if sign < 0 else 1.0)
            if kind in (app._handle_hover, drag.kind if drag else None):
                r *= 1.3
            np_.set_scale(r)
            np_.show()
            app._handle_px[kind] = px

    def hit_test(self, px):
        """Which handle (or "node") is under ``px``, or None."""
        from orbitsim.render.picking import nearest_marker
        if px is None or getattr(self.app, "_node_px", None) is None:
            return None
        kinds = [k for k, p in self.app._handle_px.items() if p is not None]
        hit = nearest_marker(px, [self.app._handle_px[k] for k in kinds], self._pt(HANDLE_HIT_PX))
        if hit is not None:
            return kinds[hit]
        if nearest_marker(px, [self.app._node_px], self._pt(NODE_HIT_PX)) is not None:
            return "node"
        return None

    def begin_drag(self, kind, px) -> None:
        axis = None
        if kind != "node":
            axis = axis_unit_px(self.app._node_px, self.app._handle_px[kind], self._pt(MIN_AXIS_PX))
        self.app._drag = HandleDrag(kind, tuple(px), axis)

    def dragging(self):
        drag = getattr(self.app, "_drag", None)
        return None if drag is None else drag.kind

    def end_drag(self) -> None:
        drag = getattr(self.app, "_drag", None)
        self.app._drag = None
        if drag is None:
            return
        if drag.kind == "node":
            if self.app._node_epoch_s is not None:
                self.app.maneuver_editor.set_node_epoch(snap_epoch(self.app._node_epoch_s))
            return
        axis = dict((k, a) for k, a, _s, _c in HANDLES)[drag.kind]
        self.app.maneuver_editor.set_dv(axis, snap_dv(self.app._dv[axis]))

    def _drag_value(self) -> float:
        drag = self.app._drag
        mouse = self.app.flight_input.mouse_px()
        if drag is None or drag.axis_px is None or mouse is None:
            return 0.0
        return handle_drag_value(drag.start_px, mouse, drag.axis_px, self._pt(MAX_DRAG_PX))

    def apply_drag(self, real_dt: float) -> None:
        """Spring-loaded rate control: pulling along the handle changes delta-V."""
        drag = getattr(self.app, "_drag", None)
        if drag is None:
            return
        if self.app.flight_input.mouse1_held() is False:
            self.end_drag()                  # released over a panel / outside the window
            return
        mouse = self.app.flight_input.mouse_px()
        if mouse is None:
            return
        if drag.kind == "node":
            # Slide the burn along the orbit under the mouse, clamped to the current
            # orbit's window (never under MIN_NODE_LEAD_S, never past one orbit).
            placement = self.app.node_placement
            hit = placement.orbit_pick_at(mouse, math.inf, window=placement.node_window())
            if hit is not None:
                self.app.maneuver_editor.set_node_epoch(hit[0])
            return
        kind = dict((k, (a, s)) for k, a, s, _c in HANDLES)
        axis, sign = kind[drag.kind]
        rate = jog_rate(self._drag_value(), DRAG_MAX_RATE_MPS, 4.0, 0.02)
        if rate:
            self.app.maneuver_editor.set_dv(axis, self.app._dv[axis] + sign * rate * real_dt)
