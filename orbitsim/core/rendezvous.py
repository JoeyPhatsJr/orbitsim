"""Closest approach between two Keplerian trajectories (coarse scan + refine)."""
from dataclasses import dataclass

import numpy as np

from orbitsim.core.propagate import propagate_kepler
from orbitsim.core.state import StateVector


@dataclass(frozen=True)
class ClosestApproach:
    """Result of a closest-approach search.

    Attributes
    ----------
    t_ca_s : float
        Time of closest approach, seconds from now.
    separation_m : float
        Distance between the two bodies at closest approach [m].
    rel_speed_mps : float
        Relative speed |v_a - v_b| at closest approach [m/s].
    """

    t_ca_s: float
    separation_m: float
    rel_speed_mps: float


def _sep(state_a: StateVector, state_b: StateVector, t: float) -> float:
    ra = propagate_kepler(state_a, t).r
    rb = propagate_kepler(state_b, t).r
    return float(np.linalg.norm(ra - rb))


def closest_approach(
    state_a: StateVector, state_b: StateVector, window_s: float, coarse_samples: int = 720
) -> ClosestApproach:
    """Minimum separation of two trajectories over ``[0, window_s]``.

    Coarse-scans ``coarse_samples+1`` uniform times, then refines the best one with a
    ternary search over its bracketing interval. Raises ValueError on bad inputs.
    """
    if window_s <= 0.0:
        raise ValueError(f"window_s must be positive, got {window_s}")
    if coarse_samples < 2:
        raise ValueError(f"coarse_samples must be >= 2, got {coarse_samples}")

    times = np.linspace(0.0, window_s, coarse_samples + 1)
    seps = np.array([_sep(state_a, state_b, float(t)) for t in times])
    k = int(np.argmin(seps))

    # Ternary-search refine within [t_{k-1}, t_{k+1}].
    lo = times[max(0, k - 1)]
    hi = times[min(len(times) - 1, k + 1)]
    for _ in range(60):
        if hi - lo < 1e-3:
            break
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if _sep(state_a, state_b, m1) < _sep(state_a, state_b, m2):
            hi = m2
        else:
            lo = m1
    t_ca = 0.5 * (lo + hi)

    # Guard: never return a separation worse than the coarse minimum (ternary refine
    # only guarantees improvement when the bracket is unimodal).
    if seps[k] <= _sep(state_a, state_b, t_ca):
        t_ca = float(times[k])

    sa = propagate_kepler(state_a, t_ca)
    sb = propagate_kepler(state_b, t_ca)
    sep = float(np.linalg.norm(sa.r - sb.r))
    rel = float(np.linalg.norm(sa.v - sb.v))
    return ClosestApproach(t_ca_s=t_ca, separation_m=sep, rel_speed_mps=rel)
