"""Surface impact timing on a two-body conic.

Used by the sim's surface-contact pass to catch a vessel whose path dips under a
surface and comes back out within one long (high-warp) tick: checking only the
end-of-tick position lets it tunnel straight through the body.
"""
import numpy as np

from orbitsim.core.nbody import _cross3, _norm3

_TWO_PI = 2.0 * np.pi


def surface_crossing_s(r_rel_m, v_rel_mps, mu: float, radius_m: float):
    """Time [s] until the conic through (r, v) first descends to radius_m.

    Returns None when it never does: the periapsis is at or above radius_m, the
    vessel is already at or below it, or it is outbound on an open (e >= 1) path.
    An outbound ellipse comes back down after apoapsis, so it gets that time.

    Uses the eccentric/hyperbolic anomaly (r = a(1 - e cos E), resp.
    a(1 - e cosh F)), which stays defined for a purely radial path (h = 0, e = 1)
    where the true anomaly is not.
    """
    r_vec = np.asarray(r_rel_m, dtype=np.float64)
    v_vec = np.asarray(v_rel_mps, dtype=np.float64)
    r = _norm3(r_vec)
    if r <= radius_m:
        return None
    h_vec = _cross3(r_vec, v_vec)
    h2 = float(np.dot(h_vec, h_vec))
    energy = 0.5 * float(np.dot(v_vec, v_vec)) - mu / r
    e = float(np.sqrt(max(0.0, 1.0 + 2.0 * energy * h2 / mu**2)))
    rp = h2 / mu / (1.0 + e)
    if rp >= radius_m or energy == 0.0:
        return None
    rdotv = float(np.dot(r_vec, v_vec))
    a = -mu / (2.0 * energy)
    if energy < 0.0:                                   # ellipse (incl. radial)
        n = np.sqrt(mu / a**3)
        # e cos E = 1 - r/a, e sin E = r.v / sqrt(mu a): atan2 stays well
        # conditioned at the apsides, where arccos alone loses ~1e-7 rad.
        E0 = np.arctan2(rdotv / np.sqrt(mu * a), 1.0 - r / a) % _TWO_PI
        Ec = _TWO_PI - np.arccos(np.clip((1.0 - radius_m / a) / e, -1.0, 1.0))
        M0 = E0 - e * np.sin(E0)
        Mc = Ec - e * np.sin(Ec)
        return float(((Mc - M0) % _TWO_PI) / n)
    if rdotv >= 0.0:                                   # open path, outbound
        return None
    n = np.sqrt(mu / (-a) ** 3)
    F0 = np.arcsinh(rdotv / (e * np.sqrt(-mu * a)))   # e sinh F = r.v / sqrt(-mu a)
    Fc = -np.arccosh(max(1.0, (1.0 - radius_m / a) / e))
    M0 = e * np.sinh(F0) - F0
    Mc = e * np.sinh(Fc) - Fc
    return float((Mc - M0) / n)
