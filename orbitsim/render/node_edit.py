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
