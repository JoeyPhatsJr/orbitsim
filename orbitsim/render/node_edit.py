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
    "'": ("rad", 1), ";": ("rad", -1),
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


def node_epoch_window(now_s, period_s, encounter_entry_epochs, line_end_s):
    """(earliest, latest) epoch a burn may be placed: from now + MIN_NODE_LEAD_S to the
    end of the current orbit -- one period, the first non-Earth SOI entry, or the line's
    end, whichever comes first (None period = unbound)."""
    end = line_end_s
    if period_s is not None:
        end = min(end, now_s + period_s)
    if encounter_entry_epochs:
        end = min(end, min(encounter_entry_epochs))
    return (now_s + MIN_NODE_LEAD_S, end)


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
