"""The ONLY harness code allowed to touch OrbitApp private attributes.

During the app.py split, edit this file only to follow an attribute that moved;
never to change what a scenario does. Golden files are never edited.
"""


def select_target(app, name: str) -> None:
    """Select a navigation target by name, as a click on its marker would."""
    matches = [t for t in app._targets if t.name == name]
    if not matches:
        raise LookupError(f"no target named {name!r}")
    app.targeting.clear_target()
    app._target = matches[0]


def current_target_name(app):
    target = getattr(app, "_target", None)
    return None if target is None else target.name


def preview_lines(app):
    """(estimate_shown, exact_shown) for the maneuver preview line."""
    return (getattr(app, "_preview_estimate_np", None) is not None,
            getattr(app, "_preview_np", None) is not None)


def preview_at_current_scale(app) -> bool:
    """Whether the drawn exact preview was built at the current zoom scale."""
    return (getattr(app, "_preview_np", None) is not None
            and getattr(app, "_preview_built_scale", None) == app.transform.scale_m_per_unit)


def preview_start_error_m(app) -> float:
    """Distance [m] from the drawn exact preview's first point to where the burn would
    really start now (apply_maneuver on the live vessel state); inf if none is drawn."""
    import numpy as np
    from orbitsim.core.maneuvers import apply_maneuver
    points = getattr(app, "_preview_points", None)
    if getattr(app, "_preview_np", None) is None or points is None:
        return float("inf")
    start = apply_maneuver(app.world.vessels[0].state, app.maneuver_editor.current_node()).r
    return float(np.linalg.norm(np.asarray(points[0]) - start))


def set_key_held(app, key: str, held) -> None:
    """Override what the keyboard reports for ``key`` (headless has no device);
    held=None removes the override."""
    overrides = app.__dict__.setdefault("_test_key_held", {})
    if held is None:
        overrides.pop(key, None)
    else:
        overrides[key] = held
    real = type(app.flight_input).key_held
    fi = app.flight_input
    fi.key_held = lambda k, _o=overrides, _fi=fi: _o[k] if k in _o else real(_fi, k)


def node_state(app) -> tuple:
    """(node epoch or None, dV prograde, normal, radial) of the planned burn."""
    return (app._node_epoch_s, app._dv["pro"], app._dv["nrm"], app._dv["rad"])


def set_mouse_px(app, px) -> None:
    """Put the (headless) mouse at bottom-up pixel ``px``; None = off-window."""
    app.flight_input.mouse_px = lambda: px


def orbit_point_epoch(app, i: int) -> float:
    """Epoch of live-line sample ``i``."""
    return float(app._traj_epoch_cache[0][i])


def orbit_point_px(app, i: int):
    """Pixel position of live-line sample ``i``."""
    return app.targeting.marker_px(app._traj_world_cache[0][i])


def first_sample_after(app, lead_s: float) -> int:
    """First live-line sample at least ``lead_s`` ahead of now."""
    epochs = app._traj_epoch_cache[0]
    now = app.clock.sim_time_s
    return next(i for i, e in enumerate(epochs) if e >= now + lead_s)


def body_px(app, name: str):
    """Pixel position of the named navigation target's marker."""
    target = next(t for t in app._targets if t.name == name)
    return app.targeting.marker_px(target.state_at(app.clock.sim_time_s).r)


def orbit_menu_labels(app):
    """Item labels of the open orbit menu, or None when closed."""
    if getattr(app, "_orbit_menu", None) is None:
        return None
    return list(app._orbit_menu_labels)


def settings_visible(app) -> bool:
    return not app.settings_panel._frame.is_hidden()


def node_px(app):
    """Pixel position of the burn node marker, or None when there are no handles."""
    return getattr(app, "_node_px", None)


def handle_px(app, kind: str):
    return app._handle_px.get(kind)


def set_mouse1_held(app, held) -> None:
    """Override what the mouse device reports for the left button; None restores it."""
    if held is None:
        vars(app.flight_input).pop("mouse1_held", None)
    else:
        app.flight_input.mouse1_held = lambda: held


def dragging(app):
    return app.node_handles.dragging()


def orbit_line_ready(app) -> bool:
    """Whether the live orbit line has been sampled (real pools compute it async)."""
    cache = getattr(app, "_traj_world_cache", None)
    return bool(cache) and cache[0] is not None


def aim_status(app):
    """(kind, text) of the target status box, or None when it isn't shown."""
    status = getattr(app, "_aim_status", None)
    return None if status is None else (status.kind, status.text)


def encounter_marker_count(app) -> int:
    """How many predicted-encounter markers are shown on the map."""
    return sum(1 for m in app._encounter_marker_nps if not m.is_hidden())


def set_burn(app, pro: float = 0.0, nrm: float = 0.0, rad: float = 0.0) -> None:
    """Set the planned burn's delta-V [m/s], as the handles or nudge keys would."""
    for axis, value in (("pro", pro), ("nrm", nrm), ("rad", rad)):
        app.maneuver_editor.set_dv(axis, value)


def burn_panel(app):
    """(readout text or None, labels of the shown buttons in order) of the burn panel."""
    readout = app._burn_readout_text
    text = None if readout.is_hidden() else readout.getText()
    labels = [str(b["text"]).strip() for b in app._burn_buttons if not b.is_hidden()]
    return text, labels


def _frame_rect_px(app, frame, size=None):
    """Pixel rect (x0, y0, x1, y1), bottom-up, of a DirectFrame's drawn frameSize."""
    from panda3d.core import Point3
    l, r, b, t = frame["frameSize"] if size is None else size
    w, h = app.win.get_x_size(), app.win.get_y_size()
    p0 = app.render2d.get_relative_point(frame, Point3(l, 0, b))
    p1 = app.render2d.get_relative_point(frame, Point3(r, 0, t))
    return ((min(p0.x, p1.x) + 1) * 0.5 * w, (min(p0.z, p1.z) + 1) * 0.5 * h,
            (max(p0.x, p1.x) + 1) * 0.5 * w, (max(p0.z, p1.z) + 1) * 0.5 * h)


def shown_panel_rects_px(app):
    """{name: pixel rect} of every HUD panel drawn now, measured from the widgets
    themselves (frames, the speed chip, the navball's display region)."""
    rects = {}
    for name, root in app._hud_panels.items():
        if root.is_hidden():
            continue
        frame = app.hud._panel._bg if name == "orbit_fuel" else root
        if frame.is_hidden():
            continue
        if name == "speed_chip":
            button = app.vel_readout._button
            rects[name] = _frame_rect_px(app, button, button.getBounds())
            continue
        rects[name] = _frame_rect_px(app, frame)
    if getattr(app, "navball", None) is not None:
        l, r, b, t = app.navball.dr.get_dimensions()
        w, h = app.win.get_x_size(), app.win.get_y_size()
        rects["navball"] = (l * w, b * h, r * w, t * h)
    return rects


def labels_over_panels(app):
    """Texts of shown world (render) labels whose drawn box (card, or text frame)
    overlaps a shown panel. Measured from the widgets: the label's card corners are
    placed along the camera's right/up axes (labels are eye billboards) and projected
    through the lens, so a wide label anchored just off a panel still counts."""
    from panda3d.core import Point2, Point3, Vec3
    rects = list(shown_panel_rects_px(app).values())
    right = app.render.get_relative_vector(app.cam, Vec3(1, 0, 0))
    up = app.render.get_relative_vector(app.cam, Vec3(0, 0, 1))
    w, h = app.win.get_x_size(), app.win.get_y_size()

    def px(pt):
        q = Point2()
        if not app.camLens.project(app.cam.get_relative_point(app.render, pt), q):
            return None
        return ((q.x * 0.5 + 0.5) * w, (q.y * 0.5 + 0.5) * h)

    out = []
    for np_ in app.render.find_all_matches("**/+TextNode"):
        tn = np_.node()
        text = tn.get_text()
        if not text.strip() or np_.is_hidden():
            continue
        anchor = np_.get_pos(app.render)
        if px(anchor) is None:
            continue
        l, r, b, t = tn.get_card_actual() if tn.has_card() else tn.get_frame_actual()
        s = np_.get_scale(app.render)
        corners = [px(anchor + right * (x * s[0]) + up * (z * s[2]))
                   for x in (l, r) for z in (b, t)]
        corners = [c for c in corners if c is not None]
        if not corners:
            continue
        x0, x1 = min(c[0] for c in corners), max(c[0] for c in corners)
        y0, y1 = min(c[1] for c in corners), max(c[1] for c in corners)
        if any(x0 < a1 and a0 < x1 and y0 < b1 and b0 < y1 for a0, b0, a1, b1 in rects):
            out.append(text)
    return sorted(out)


def shown_world_labels(app):
    """Texts of every shown world (render) label."""
    return sorted(np_.node().get_text() for np_ in app.render.find_all_matches("**/+TextNode")
                  if np_.node().get_text().strip() and not np_.is_hidden())


def orbit_points_under_navball(app) -> int:
    """How many live-orbit-line samples project inside the navball's display region."""
    l, r, b, t = app.navball.dr.get_dimensions()
    w, h = app.win.get_x_size(), app.win.get_y_size()
    count = 0
    for point in app._traj_world_cache[0]:
        px = app.targeting.marker_px(point)
        if px is not None and l * w <= px[0] <= r * w and b * h <= px[1] <= t * h:
            count += 1
    return count


def world_label_texts(app):
    """Texts of every world (render) label, shown or hidden."""
    return sorted(np_.node().get_text() for np_ in app.render.find_all_matches("**/+TextNode")
                  if np_.node().get_text().strip())


def lagrange_state(app):
    """{L-point: (marker alpha scale, label shown)}."""
    return {name: (float(marker.get_color_scale()[3]), not label.is_hidden())
            for name, marker, label in zip(("L1", "L2", "L3", "L4", "L5"),
                                           app._lagrange_nps, app._lagrange_labels)}


def solar_orbit_nodes(app):
    """{planet: its solar-viewer orbit line NodePath} (a copy)."""
    return dict(app._solar_orbit_nps)


def solar_orbit_frame_scale(app) -> float:
    """The solar-viewer orbit frame's x scale (render units per baked unit)."""
    return float(app._solar_orbit_frame.get_scale()[0])


def _project_fb_px(app, point_render):
    """A render-space point to framebuffer pixels (y up), or None if it doesn't project."""
    from panda3d.core import Point2
    q = Point2()
    if not app.camLens.project(app.cam.get_relative_point(app.render, point_render), q):
        return None
    return ((q.x * 0.5 + 0.5) * app.win.get_fb_x_size(),
            (q.y * 0.5 + 0.5) * app.win.get_fb_y_size())


def _camera_axes(app):
    quat = app.camera.get_quat(app.render)
    return quat.get_right(), quat.get_up()


def screen_radius_px(app, node_name: str):
    """On-screen radius [px] of the render node ``node_name`` (a unit sphere scaled to
    its radius), measured from the scene: its centre and centre + camera-right x scale
    are projected. None if hidden or its centre doesn't project."""
    import math
    np_ = app.render.find(f"**/{node_name}")
    if np_.is_empty() or np_.is_hidden():
        return None
    centre = np_.get_pos(app.render)
    right, _up = _camera_axes(app)
    a = _project_fb_px(app, centre)
    b = _project_fb_px(app, centre + right * np_.get_scale(app.render)[0])
    if a is None or b is None:
        return None
    return math.hypot(b[0] - a[0], b[1] - a[1])


def label_screen_height_px(app, text: str):
    """On-screen height [px] of the solar-viewer label ``text``'s card (its top and
    bottom projected along camera-up). None if hidden or it doesn't project."""
    import math
    for label in app._planet_labels:
        if label.node().get_text() != text:
            continue
        if label.is_hidden():
            return None
        _l, _r, b, t = label.node().get_card_actual()
        pos = label.get_pos(app.render)
        s = label.get_scale(app.render)[2]
        _right, up = _camera_axes(app)
        if _project_fb_px(app, pos) is None:
            return None
        top = _project_fb_px(app, pos + up * (t * s))
        bottom = _project_fb_px(app, pos + up * (b * s))
        if top is None or bottom is None:
            return None
        return math.hypot(top[0] - bottom[0], top[1] - bottom[1])
    raise LookupError(f"no solar label {text!r}")


def solar_marker_px(app):
    """{body: (x, y, radius_px) | None}: last frame's solar-viewer dot anchors."""
    return dict(app._solar_marker_px)


def solar_label_rects(app):
    """{body: (x0, y0, x1, y1) | None}: the last solar-viewer name layout (fb px)."""
    return dict(app._solar_label_rects)


def hover_solar_body(app, name: str) -> None:
    """Lay the solar names out again as if the mouse were over ``name``'s dot."""
    app.solar_view.frame_markers_and_labels(mouse_px=app._solar_marker_px[name][:2])
