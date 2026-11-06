"""Pure math for editing a maneuver node by hand: keyboard nudges, jog curves, snapping.

No Panda3D. Input and presentation (how a key hold or a drag maps to a change), not
physics: the burn itself is a ``core.maneuvers.ManeuverNode`` built from app state.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NudgeSpec:
    """One nudge axis: a tap step, then a hold that ramps (doubling) up to a cap.

    tap : change applied on key-down.
    base_rate : rate [unit/s] once the hold delay has passed.
    doubling_s : the rate doubles every this many seconds of holding.
    max_rate : rate cap [unit/s].
    hold_delay_s : how long a key must be held before the ramp starts.
    """

    tap: float
    base_rate: float
    doubling_s: float
    max_rate: float
    hold_delay_s: float


NUDGE_DV = NudgeSpec(1.0, 10.0, 0.5, 1000.0, 0.3)       # m/s, m/s per s
NUDGE_TIME = NudgeSpec(10.0, 30.0, 0.5, 21600.0, 0.3)   # s, s per s

# key -> (axis, sign). Right hand, clear of WASD/QE/Z/X (fly), arrows (camera),
# Shift/Ctrl (throttle; usable as modifiers, so no fine/coarse modifier here).
NUDGE_KEYS: dict = {
    "j": ("pro", 1), "h": ("pro", -1),
    "l": ("nrm", 1), "k": ("nrm", -1),
    # Radial: the burn's radial axis (apply_maneuver / rtn_basis) points TOWARD the
    # central body, so ";" (radial in) is +1 and "'" (radial out) is -1.
    ";": ("rad", 1), "'": ("rad", -1),
    "]": ("time", 1), "[": ("time", -1),
}


def nudge_rate(held_s: float, spec: NudgeSpec) -> float:
    """Ramp rate after ``held_s`` seconds of holding (0 before the hold delay)."""
    if held_s < spec.hold_delay_s:
        return 0.0
    return min(spec.max_rate, spec.base_rate * 2.0 ** ((held_s - spec.hold_delay_s) / spec.doubling_s))


def jog_rate(value: float, max_rate: float, curve: float, deadzone: float) -> float:
    """Map a jog displacement in [-1, 1] to a signed rate (the slider's exponential curve).

    Zero inside the deadzone, ``max_rate`` at full deflection; same expressions in the
    same order as the original slider code, so results are bit-identical.
    """
    mag = abs(value)
    if mag <= deadzone:
        return 0.0
    shaped = (np.exp(curve * mag) - 1.0) / (np.exp(curve) - 1.0)
    return float(np.copysign(max_rate * shaped, value))


def snap_dv(x: float) -> float:
    """Clean delta-V after an edit ends [m/s, 0.1 m/s]."""
    return round(x, 1)


def snap_epoch(t: float) -> float:
    """Clean node epoch after an edit ends [s, whole seconds]."""
    return float(round(t))


# --- Where a node may go, and screen-space sizing (plan 3, Task 2) ---
import math  # noqa: E402

MIN_NODE_LEAD_S = 10.0   # a new/moved burn must be at least this far in the future


def node_epoch_window(now_s, period_s, encounter_entry_epochs, line_end_s, in_other_soi=False):
    """(earliest, latest) epoch a burn may be placed: from now + MIN_NODE_LEAD_S to the
    end of the current orbit -- one period, the next non-Earth SOI entry, or the line's
    end, whichever comes first (None period = unbound).

    An encounter already in progress (entered at or before now: you're orbiting the
    Moon) doesn't close the window, and inside another body's SOI the geocentric
    period means nothing, so it's ignored (``in_other_soi``)."""
    end = line_end_s
    if period_s is not None and not in_other_soi:
        end = min(end, now_s + period_s)
    upcoming = [e for e in encounter_entry_epochs if e > now_s]
    if upcoming:
        end = min(end, min(upcoming))
    return (now_s + MIN_NODE_LEAD_S, end)


# Auto-plans search departures from this far ahead: planning takes real seconds (warp is
# dropped to 1x for it), and a departure in the past would be clamped to "now" -- a
# Moon flyby burned 10 s late impacts, 250 s late misses the Moon entirely.
PLAN_LEAD_S = 60.0


def departure_grid_s(span_s, n, lead_s=PLAN_LEAD_S):
    """Departure offsets [s] from now for an auto-plan: ``n`` evenly over ``span_s``,
    starting ``lead_s`` ahead."""
    return np.linspace(lead_s, span_s + lead_s, n)


def node_epoch_ok(epoch_s, now_s) -> bool:
    """Whether a burn may still be placed at ``epoch_s`` (enough lead time left)."""
    return epoch_s >= now_s + MIN_NODE_LEAD_S


def screen_constant_scale(distance_units, size_px, fov_v_deg, viewport_h_px) -> float:
    """World size that appears ``size_px`` tall at ``distance_units`` from the camera."""
    return size_px * 2.0 * distance_units * math.tan(math.radians(fov_v_deg) / 2.0) / viewport_h_px


def menu_pos(px, width, height, menu_w=0.46, menu_h=0.17):
    """aspect2d (x, z) for a popup anchored just right/below a bottom-up pixel point,
    clamped so the whole menu stays on screen."""
    aspect = width / height
    x = (px[0] / width * 2.0 - 1.0) * aspect + 0.03
    z = px[1] / height * 2.0 - 1.0 - 0.03
    x = min(max(x, -aspect), aspect - menu_w)
    z = min(max(z, -1.0 + menu_h), 1.0)
    return (x, z)


def menu_clear_of(menu_xz, rect, menu_w=0.46, menu_h=0.17, gap=0.02):
    """Move a popup (top-left anchor ``menu_xz``) above ``rect`` (l, r, b, t in aspect2d)
    if it would overlap it -- for overlays drawn over the 2D GUI (the navball's own
    display region), where a popup underneath would be invisible."""
    x, z = menu_xz
    left, right, bottom, top = rect
    overlaps = x < right and x + menu_w > left and z > bottom and z - menu_h < top
    if not overlaps:
        return (x, z)
    return (x, top + menu_h + gap)


# --- 3D drag handles (plan 3, Task 3) ---
# (kind, axis, sign, rgb). Prograde lime, normal violet, radial coral.
HANDLES = (
    ("pro+", "pro", 1, (0.75, 1.0, 0.30)), ("pro-", "pro", -1, (0.75, 1.0, 0.30)),
    ("nrm+", "nrm", 1, (0.80, 0.50, 1.0)), ("nrm-", "nrm", -1, (0.80, 0.50, 1.0)),
    ("rad+", "rad", 1, (1.0, 0.45, 0.45)), ("rad-", "rad", -1, (1.0, 0.45, 0.45)),
)
HANDLE_OFFSET_PX = 36.0      # handle distance from the node, on screen
HANDLE_RADIUS_PX = 5.0
NODE_RADIUS_PX = 7.0
HANDLE_HIT_PX = 12.0
NODE_HIT_PX = 14.0
MIN_AXIS_PX = 8.0            # a handle pair this foreshortened is hidden (axis ~ at camera)
MAX_DRAG_PX = 150.0          # pull this far along the handle for full rate
DRAG_MAX_RATE_MPS = 400.0    # delta-V rate at full pull (the slider's curve)


@dataclass(frozen=True)
class HandleDrag:
    """An in-progress drag: which handle (or "node"), where it started, its screen axis."""

    kind: str
    start_px: tuple
    axis_px: tuple = None


def axis_unit_px(node_px, handle_px, min_len_px):
    """Unit screen direction node -> handle, or None when shorter than ``min_len_px``
    (the axis points nearly at the camera: hide the pair, never divide by ~0)."""
    dx, dy = handle_px[0] - node_px[0], handle_px[1] - node_px[1]
    length = math.hypot(dx, dy)
    if length < min_len_px:
        return None
    return (dx / length, dy / length)


def handle_drag_value(start_px, mouse_px, axis_unit, max_drag_px) -> float:
    """Jog value in [-1, 1]: how far the mouse is pulled along the handle's screen axis."""
    along = (mouse_px[0] - start_px[0]) * axis_unit[0] + (mouse_px[1] - start_px[1]) * axis_unit[1]
    return max(-1.0, min(1.0, along / max_drag_px))
