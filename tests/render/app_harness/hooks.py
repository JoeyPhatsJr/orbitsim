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
