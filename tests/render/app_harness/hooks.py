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


def set_jog(app, axis: str, value: float) -> None:
    """Deflect a maneuver jog slider ('pro', 'nrm', 'rad') to ``value`` in [-1, 1]."""
    app._jog[axis]["value"] = value


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
