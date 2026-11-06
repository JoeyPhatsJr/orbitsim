"""Gravity-assist (flyby) geometry for unpowered hyperbolic encounters.

All functions are pure math — no graphics imports. Inputs/outputs in SI (m, s, rad).
The flyby model treats the encounter as an instantaneous deflection of the
v-infinity vector at the planet (patched-conic approximation for planning;
the actual N-body trajectory handles the real physics during flight).
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from orbitsim.core.constants import MU_SUN


@dataclass(frozen=True)
class ConicPeriapsis:
    """Periapsis of the osculating conic through a body-relative state.

    radius_m : periapsis distance from the body's centre [m].
    time_to_periapsis_s : signed; negative once periapsis has passed. For an ellipse it is
        the nearest periapsis (mean anomaly taken in (-pi, pi]).
    point_rel_m : periapsis position relative to the body [m], shape (3,).
    """

    radius_m: float
    time_to_periapsis_s: float
    point_rel_m: np.ndarray


def _conic_frame(r_rel_m, v_rel_mps, mu):
    """(r, v, h_vec, e, P_hat, Q_hat, p) of the osculating conic; ValueError on h = 0."""
    r = np.asarray(r_rel_m, dtype=np.float64)
    v = np.asarray(v_rel_mps, dtype=np.float64)
    r_mag = float(np.linalg.norm(r))
    if r_mag == 0.0:
        raise ValueError("state at the body's centre has no conic")
    h_vec = np.cross(r, v)
    h_mag = float(np.linalg.norm(h_vec))
    if h_mag <= 1e-12 * r_mag * max(float(np.linalg.norm(v)), 1e-300):
        raise ValueError("zero angular momentum (radial line): no periapsis direction")
    # Curtis: e = (1/mu)[(v^2 - mu/r) r - r v_r v], v_r = (r.v)/r  =>  r v_r = r.v
    e_vec = ((float(v @ v) - mu / r_mag) * r - float(r @ v) * v) / mu
    e = float(np.linalg.norm(e_vec))
    h_hat = h_vec / h_mag
    if e < 1e-12:   # circular: periapsis direction undefined, take the current point
        p_hat = r / r_mag
    else:
        p_hat = e_vec / e
    q_hat = np.cross(h_hat, p_hat)
    p = h_mag * h_mag / mu
    return r, v, h_vec, e, p_hat, q_hat, p


def conic_periapsis(r_rel_m, v_rel_mps, mu: float) -> ConicPeriapsis:
    """Periapsis radius, signed time to it, and its body-relative point (any conic).

    Raises ValueError for zero angular momentum (radial line, no periapsis direction).
    """
    r, _v, _h, e, p_hat, q_hat, p = _conic_frame(r_rel_m, v_rel_mps, mu)
    rp = p / (1.0 + e)
    nu = float(np.arctan2(r @ q_hat, r @ p_hat))     # true anomaly, (-pi, pi]
    if abs(e - 1.0) < 1e-12:                          # parabola: Barker's equation
        d = np.tan(0.5 * nu)
        t_since = 0.5 * np.sqrt(p**3 / mu) * (d + d**3 / 3.0)
    elif e < 1.0:
        a = p / (1.0 - e * e)
        ecc_anom = np.arctan2(np.sqrt(1.0 - e * e) * np.sin(nu), e + np.cos(nu))
        mean_anom = ecc_anom - e * np.sin(ecc_anom)
        t_since = mean_anom / np.sqrt(mu / a**3)
    else:
        a = p / (1.0 - e * e)                         # negative
        sinh_f = np.sqrt(e * e - 1.0) * np.sin(nu) / (1.0 + e * np.cos(nu))
        hyp_anom = np.arcsinh(sinh_f)
        mean_anom = e * np.sinh(hyp_anom) - hyp_anom
        t_since = mean_anom / np.sqrt(mu / (-a) ** 3)
    return ConicPeriapsis(radius_m=float(rp), time_to_periapsis_s=float(-t_since),
                          point_rel_m=rp * p_hat)


def b_plane(r_rel_m, v_rel_mps, mu: float) -> tuple[np.ndarray, float]:
    """B-plane coordinates ``([B.T, B.R] [m], v_inf [m/s])`` of a hyperbolic flyby.

    S is the incoming asymptote, T = unit(S x z), R = S x T, B = (S x h) / v_inf
    (|B| is the impact parameter h / v_inf). Raises ValueError unless the state is
    hyperbolic relative to ``mu``.
    """
    r = np.asarray(r_rel_m, dtype=np.float64)
    v = np.asarray(v_rel_mps, dtype=np.float64)
    energy = 0.5 * float(v @ v) - mu / float(np.linalg.norm(r))
    if energy <= 0.0:
        raise ValueError("state is bound to the body: no B-plane")
    v_inf = float(np.sqrt(2.0 * energy))
    _r, _v, h_vec, e, p_hat, q_hat, _p = _conic_frame(r, v, mu)
    s_hat = p_hat / e + (np.sqrt(e * e - 1.0) / e) * q_hat
    s_hat = s_hat / np.linalg.norm(s_hat)
    t_vec = np.cross(s_hat, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(t_vec) < 1e-12:                # asymptote along z: any T works
        t_vec = np.cross(s_hat, np.array([1.0, 0.0, 0.0]))
    t_hat = t_vec / np.linalg.norm(t_vec)
    r_hat = np.cross(s_hat, t_hat)
    b_vec = np.cross(s_hat, h_vec) / v_inf
    return np.array([float(b_vec @ t_hat), float(b_vec @ r_hat)]), v_inf


def v_infinity(v_ship: np.ndarray, v_planet: np.ndarray) -> np.ndarray:
    """Hyperbolic excess velocity vector relative to the planet."""
    return v_ship - v_planet


def flyby_periapsis(v_inf_mag: float, mu: float, delta: float) -> float:
    """Periapsis radius for a given deflection angle [rad] and v_inf [m/s].

    delta is the total turn angle of the v-infinity vector.
    From the hyperbolic relation: sin(delta/2) = 1 / e, and r_p = a(e-1)
    where a = mu / v_inf^2.
    """
    if delta <= 0.0 or delta >= np.pi:
        raise ValueError(f"deflection must be in (0, pi), got {delta}")
    e = 1.0 / np.sin(delta / 2.0)
    a = mu / v_inf_mag**2
    return a * (e - 1.0)


def flyby_deflection(v_inf_mag: float, mu: float, r_p: float) -> float:
    """Total deflection angle [rad] for a hyperbolic flyby.

    Given |v_inf| and periapsis radius r_p, returns the turn angle delta.
    delta = 2 * arcsin(1/e) where e = 1 + r_p * v_inf^2 / mu.
    """
    if r_p <= 0.0:
        raise ValueError(f"periapsis must be positive, got {r_p}")
    e = 1.0 + r_p * v_inf_mag**2 / mu
    sin_half = 1.0 / e
    return 2.0 * np.arcsin(np.clip(sin_half, 0.0, 1.0))


def rotate_v_infinity(v_inf: np.ndarray, delta: float,
                       plane_normal: np.ndarray | None = None) -> np.ndarray:
    """Rotate v_inf by deflection angle delta in the flyby plane.

    If plane_normal is None, the flyby plane is the ecliptic (normal = +Z),
    giving a prograde deflection. The rotation is toward the planet (inward
    side of the hyperbola).
    """
    if plane_normal is None:
        plane_normal = np.array([0.0, 0.0, 1.0])
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    v_hat = v_inf / np.linalg.norm(v_inf)
    perp = np.cross(plane_normal, v_hat)
    perp_mag = np.linalg.norm(perp)
    if perp_mag < 1e-12:
        perp = np.array([0.0, 1.0, 0.0])
    else:
        perp = perp / perp_mag
    c, s = np.cos(delta), np.sin(delta)
    return np.linalg.norm(v_inf) * (c * v_hat + s * perp)


def flyby_dv_equivalent(v_inf_mag: float, delta: float) -> float:
    """Equivalent free delta-V gained from the flyby [m/s].

    This is the magnitude of the change in heliocentric velocity:
    |dV| = 2 * v_inf * sin(delta/2).
    """
    return 2.0 * v_inf_mag * np.sin(delta / 2.0)


def max_deflection(v_inf_mag: float, mu: float, r_min: float) -> float:
    """Maximum deflection angle [rad] for a flyby with periapsis >= r_min."""
    return flyby_deflection(v_inf_mag, mu, r_min)


def flyby_exit_velocity(v_ship_in: np.ndarray, v_planet: np.ndarray,
                         mu: float, r_p: float,
                         plane_normal: np.ndarray | None = None) -> np.ndarray:
    """Heliocentric exit velocity after an unpowered flyby.

    Parameters
    ----------
    v_ship_in : (3,) heliocentric velocity of the ship at encounter
    v_planet : (3,) heliocentric velocity of the planet
    mu : gravitational parameter of the flyby body [m^3/s^2]
    r_p : periapsis distance of the hyperbolic pass [m]
    plane_normal : optional flyby plane normal (default: ecliptic +Z)

    Returns
    -------
    v_out : (3,) heliocentric velocity after the flyby
    """
    v_inf_in = v_infinity(v_ship_in, v_planet)
    v_inf_mag = float(np.linalg.norm(v_inf_in))
    if v_inf_mag < 1e-6:
        return v_ship_in.copy()
    delta = flyby_deflection(v_inf_mag, mu, r_p)
    v_inf_out = rotate_v_infinity(v_inf_in, delta, plane_normal)
    return v_planet + v_inf_out


def encounter_parameters(r_ship: np.ndarray, v_ship: np.ndarray,
                          r_planet: np.ndarray, v_planet: np.ndarray,
                          mu: float) -> dict:
    """Compute flyby encounter parameters for HUD display.

    Returns a dict with:
      v_inf_mag: hyperbolic excess speed [m/s]
      periapsis_m: closest approach distance [m] (if on hyperbolic approach)
      deflection_rad: turn angle [rad]
      dv_equivalent: free dV from the flyby [m/s]
      e: hyperbolic eccentricity
    """
    v_inf = v_infinity(v_ship, v_planet)
    v_inf_mag = float(np.linalg.norm(v_inf))
    if v_inf_mag < 1e-6:
        return {"v_inf_mag": 0.0, "periapsis_m": float("inf"),
                "deflection_rad": 0.0, "dv_equivalent": 0.0, "e": 1.0}
    r_rel = r_ship - r_planet
    v_rel = v_inf
    r_mag = float(np.linalg.norm(r_rel))
    h_vec = np.cross(r_rel, v_rel)
    h_mag = float(np.linalg.norm(h_vec))
    a = mu / v_inf_mag**2
    if h_mag > 1e-6:
        p = h_mag**2 / mu
        e = np.sqrt(1.0 + p / a)
        r_p = a * (e - 1.0)
    else:
        e = 1.0 + r_mag * v_inf_mag**2 / mu
        r_p = r_mag
    delta = 2.0 * np.arcsin(np.clip(1.0 / e, 0.0, 1.0))
    dv_eq = flyby_dv_equivalent(v_inf_mag, delta)
    return {
        "v_inf_mag": v_inf_mag,
        "periapsis_m": float(r_p),
        "deflection_rad": float(delta),
        "dv_equivalent": float(dv_eq),
        "e": float(e),
    }
