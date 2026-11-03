"""Cubic-Hermite interpolation of a state between two sampled endpoint states."""
from orbitsim.core.state import StateVector


def hermite_state(start: StateVector, end: StateVector, t0_s: float, step_s: float,
                  t_s: float, mu: float) -> StateVector:
    """Interpolate position and velocity at ``t_s`` in ``[t0_s, t0_s + step_s]``.

    Uses the endpoint positions *and* velocities, so the result is C1 across
    sample knots (no position/velocity jumps between consecutive intervals).
    """
    u = (float(t_s) - t0_s) / step_s
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    r = h00 * start.r + h10 * step_s * start.v + h01 * end.r + h11 * step_s * end.v
    dh00 = (6.0 * u2 - 6.0 * u) / step_s
    dh10 = 3.0 * u2 - 4.0 * u + 1.0
    dh01 = (-6.0 * u2 + 6.0 * u) / step_s
    dh11 = 3.0 * u2 - 2.0 * u
    v = dh00 * start.r + dh10 * start.v + dh01 * end.r + dh11 * end.v
    return StateVector(r=r, v=v, mu=mu, epoch_s=float(t_s))
