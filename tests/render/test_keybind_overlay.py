"""Pure-data tests for keybind overlay content (no DirectGUI needed)."""
from orbitsim.render.keybind_overlay import SANDBOX_BINDINGS, SOLAR_BINDINGS


def test_sandbox_bindings_cover_key_controls():
    # Keys may be grouped per row (e.g. "F5 / F9"), so check the documented tokens
    # appear anywhere in the key column, not as exact whole-row matches.
    keytext = " ".join(k for k, _ in SANDBOX_BINDINGS)
    for expected in ("F5", "F9", "F1", "Esc", "Z", "X", "T"):
        assert expected in keytext, expected


def test_solar_bindings_minimal_but_present():
    keytext = " ".join(k for k, _ in SOLAR_BINDINGS)
    assert "F1" in keytext and "Esc" in keytext
    # No flight controls in the solar viewer (throttle Z, SAS-mode keys 1-7).
    assert "Z" not in keytext and "1-7" not in keytext


def test_every_binding_has_a_description():
    for k, desc in SANDBOX_BINDINGS + SOLAR_BINDINGS:
        assert isinstance(k, str) and k
        assert isinstance(desc, str) and desc


def _accepted_literals():
    import pathlib
    import re
    src = "".join(p.read_text() for p in pathlib.Path("orbitsim/render").rglob("*.py"))
    return set(re.findall(r"accept\(\s*\"([^\"]+)\"", src))


def test_nudge_keys_listed():
    from orbitsim.render.node_edit import NUDGE_KEYS
    keytext = " ".join(k for k, _ in SANDBOX_BINDINGS).lower()
    assert all(k in keytext for k in NUDGE_KEYS)


def test_nudge_keys_do_not_clash():
    from orbitsim.render.node_edit import NUDGE_KEYS
    assert not _accepted_literals() & (set(NUDGE_KEYS) | {f"{k}-up" for k in NUDGE_KEYS})


def test_warp_keys_bound_by_their_panda_names():
    # ',' / '.' were bound as "comma" / "period", which Panda3D never sends: the warp
    # keys did nothing on a real keyboard.
    from panda3d.core import KeyboardButton
    names = {KeyboardButton.ascii_key(c).get_name() for c in ",."}
    accepted = _accepted_literals()
    assert names <= accepted and not {"comma", "period"} & accepted
