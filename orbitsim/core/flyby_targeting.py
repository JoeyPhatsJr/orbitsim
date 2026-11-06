"""Flyby targeting: correct a Lambert intercept onto a low, safe Moon pass (pure, SI).

The Lambert intercept (``optimize.intercept_node``) aims a two-body arc at the Moon's
**centre**. Under the full N-body field that seed passes ~44,000 km from the Moon (and a
centre aim is an impact course anyway). ``moon_flyby_node`` keeps the seed's burn epoch
and corrects its delta-V so the predicted pass has a periapsis ``FLYBY_TARGET_ALT_M``
above the surface. It works on the B-plane (Curtis ch. 8 / standard flyby targeting):

* B = (B.T, B.R) is the flyby's aim point on the plane through the Moon normal to the
  incoming asymptote; |B| is the impact parameter, related to the periapsis radius by
  b = r_p sqrt(1 + 2 mu / (r_p v_inf^2)).
* The target each iteration is the same B direction at the |B| that gives the wanted
  r_p (the smallest correction). A finite-difference Jacobian dB/d(dv) at the seed,
  damped Newton steps (halved until the miss shrinks) and Broyden rank-1 updates
  converge in a handful of evaluations.

Every evaluation propagates with ``nbody.propagate_solar_prediction`` -- the stepper the
drawn preview uses -- so the planned pass is the one the player sees. It never calls the
flight propagators (``World.step``'s).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np

from orbitsim.core.constants import MU_EARTH, MU_MOON, R_MOON
from orbitsim.core.flyby import b_plane, conic_periapsis
from orbitsim.core.maneuvers import ManeuverNode, apply_maneuver
from orbitsim.core.moon import moon_state_at
from orbitsim.core import nbody
from orbitsim.core.optimize import _map_rows

FLYBY_TARGET_ALT_M = 200e3        # aimed periapsis altitude above the Moon's surface [m]
FLYBY_TOL_M = 25e3                # converged when the pass is this close to the aim [m]
PASS_SAMPLE_S = 3600.0            # sampling cadence when looking for the pass [s]
PASS_HORIZON_S = 20 * 86400.0     # the Lambert TOF grid tops out at 14 d
FD_STEP_MPS = 0.02                # finite-difference step for the Jacobian [m/s]
MAX_ITER = 12
MAX_HALVINGS = 4


# The Moon never comes closer to Earth than ~356,000 km. An Earth orbit whose apoapsis,
# stretched by this margin, stays more than 3 Moon SOI radii inside that can't produce a
# pass within the horizon (first_moon_pass would raise after ~10^6 gravity evaluations).
_MOON_PERIGEE_M = 3.56e8
_REACH_MARGIN = 1.25


def _cannot_reach_moon(state) -> bool:
    """Whether a geocentric state is on an Earth orbit far too small to reach the Moon."""
    r = float(np.linalg.norm(state.r))
    v2 = float(state.v @ state.v)
    energy = 0.5 * v2 - MU_EARTH / r
    if energy >= 0.0:
        return False
    a = -MU_EARTH / (2.0 * energy)
    e_vec = ((v2 - MU_EARTH / r) * state.r - float(state.r @ state.v) * state.v) / MU_EARTH
    apoapsis = a * (1.0 + float(np.linalg.norm(e_vec)))
    return _REACH_MARGIN * apoapsis < _MOON_PERIGEE_M - 3.0 * nbody.MOON_SOI_M


@dataclass(frozen=True)
class MoonPass:
    """The first Moon pass of a predicted path (Moon-relative conic at its nearest sample)."""

    periapsis_radius_m: float
    periapsis_epoch_s: float
    b_plane_m: np.ndarray          # [B.T, B.R] [m]
    v_inf_mps: float
    in_soi: bool


@dataclass(frozen=True)
class FlybyPlan:
    node: ManeuverNode
    moon_pass: MoonPass
    converged: bool


def first_moon_pass(post_burn, horizon_s=PASS_HORIZON_S, sample_s=PASS_SAMPLE_S) -> MoonPass:
    """The first Moon pass after ``post_burn`` under the prediction propagator.

    Steps ``sample_s`` at a time and stops at the first local minimum of the Moon
    distance inside its SOI (else uses the global minimum over ``horizon_s``). Raises
    ValueError when the closest sample is beyond 3 SOI radii (no meaningful B-plane),
    the Moon-relative state there is bound, or the start is an Earth orbit far too
    small to reach the Moon (checked up front).
    """
    if _cannot_reach_moon(post_burn):
        raise ValueError("no Moon pass")
    cur = post_burn
    t_end = post_burn.epoch_s + horizon_s
    best_d, best = np.inf, None
    prev, prev_d = None, None
    with nbody.stable_prediction_ephemeris():
        while True:
            d = float(np.linalg.norm(cur.r - moon_state_at(cur.epoch_s).r))
            if prev_d is not None and d > prev_d and prev_d < nbody.MOON_SOI_M:
                best_d, best = prev_d, prev            # first in-SOI local minimum
                break
            if d < best_d:
                best_d, best = d, cur
            if cur.epoch_s >= t_end:
                break
            prev, prev_d = cur, d
            cur = nbody.propagate_solar_prediction(cur, sample_s, max_step_s=86400.0,
                                                   min_substep_s=1.0)
    if best_d > 3.0 * nbody.MOON_SOI_M:
        raise ValueError("no Moon pass")
    moon = moon_state_at(best.epoch_s)
    r_rel, v_rel = best.r - moon.r, best.v - moon.v
    try:
        b, v_inf = b_plane(r_rel, v_rel, MU_MOON)
    except ValueError as exc:                          # bound to the Moon
        raise ValueError("no Moon pass") from exc
    pe = conic_periapsis(r_rel, v_rel, MU_MOON)
    return MoonPass(periapsis_radius_m=pe.radius_m,
                    periapsis_epoch_s=best.epoch_s + pe.time_to_periapsis_s,
                    b_plane_m=b, v_inf_mps=v_inf, in_soi=bool(best_d < nbody.MOON_SOI_M))


def _node(epoch_s, dv_rtn) -> ManeuverNode:
    return ManeuverNode(float(epoch_s), float(dv_rtn[0]), float(dv_rtn[1]), float(dv_rtn[2]))


def _pass_for_dv(ship_state, epoch_s, dv_rtn) -> MoonPass:
    """Burn ``dv_rtn`` (pro, nrm, rad) at ``epoch_s``; the first Moon pass after it."""
    return first_moon_pass(apply_maneuver(ship_state, _node(epoch_s, dv_rtn)))


def _pass_for_axis(ship_state, epoch_s, dv_rtn, step_mps, axis) -> MoonPass:
    """``_pass_for_dv`` with component ``axis`` nudged by ``step_mps`` (picklable)."""
    dv = list(dv_rtn)
    dv[int(axis)] += step_mps
    return _pass_for_dv(ship_state, epoch_s, tuple(dv))


def _b_target(p: MoonPass, rp_target_m: float) -> np.ndarray:
    """Same B direction, at the impact parameter that gives periapsis ``rp_target_m``."""
    b_mag = rp_target_m * np.sqrt(1.0 + 2.0 * MU_MOON / (rp_target_m * p.v_inf_mps**2))
    norm = float(np.linalg.norm(p.b_plane_m))
    direction = p.b_plane_m / norm if norm > 0.0 else np.array([1.0, 0.0])
    return b_mag * direction


def moon_flyby_node(ship_state, seed: ManeuverNode, target_alt_m=FLYBY_TARGET_ALT_M,
                    executor=None) -> FlybyPlan:
    """Correct ``seed``'s delta-V (epoch fixed) so the first Moon pass has periapsis
    ``target_alt_m`` above the surface, within ``FLYBY_TOL_M`` of the prediction model.

    Returns the best iterate seen; ``converged`` says whether it met the tolerance.
    ``executor`` fans the three finite-difference evaluations out (ordered map) and runs
    every other evaluation on it too; pooled and serial results are bit-identical.
    Raises ValueError when the seed never gets near the Moon.
    """
    epoch_s = float(seed.epoch_s)
    rp_target = R_MOON + float(target_alt_m)
    x = np.array([seed.dv_prograde_mps, seed.dv_normal_mps, seed.dv_radial_mps],
                 dtype=np.float64)

    def evaluate(dv_rtn):
        # With an executor, even the sequential evaluations run there: the app calls
        # this from its planning *thread*, where ~0.3 s of CPU per evaluation would hold
        # the render process's GIL (frame time +47% while planning).
        if executor is None:
            return _pass_for_dv(ship_state, epoch_s, dv_rtn)
        return executor.submit(_pass_for_dv, ship_state, epoch_s, dv_rtn).result()

    p = evaluate(tuple(x))

    # Jacobian dB/d(dv) at the seed, by forward differences.
    probes = _map_rows(partial(_pass_for_axis, ship_state, epoch_s, tuple(x), FD_STEP_MPS),
                       [0, 1, 2], executor)
    jac = np.column_stack([(q.b_plane_m - p.b_plane_m) / FD_STEP_MPS for q in probes])

    def miss(pass_):
        return abs(pass_.periapsis_radius_m - rp_target)

    best_x, best_p = x.copy(), p
    for _ in range(MAX_ITER):
        if miss(p) <= FLYBY_TOL_M:
            break
        b_star = _b_target(p, rp_target)
        err = float(np.linalg.norm(b_star - p.b_plane_m))
        step = np.linalg.pinv(jac) @ (b_star - p.b_plane_m)
        new_p = None
        for _half in range(MAX_HALVINGS + 1):
            try:
                cand = evaluate(tuple(x + step))
            except ValueError:
                cand = None
            if cand is not None and float(np.linalg.norm(b_star - cand.b_plane_m)) < err:
                new_p = cand
                break
            step = 0.5 * step
        if new_p is None:
            break                                      # stalled: keep the best so far
        db = new_p.b_plane_m - p.b_plane_m
        jac = jac + np.outer(db - jac @ step, step) / float(step @ step)
        x, p = x + step, new_p
        if miss(p) < miss(best_p):
            best_x, best_p = x.copy(), p
    return FlybyPlan(node=_node(epoch_s, best_x), moon_pass=best_p,
                     converged=bool(miss(best_p) <= FLYBY_TOL_M))
