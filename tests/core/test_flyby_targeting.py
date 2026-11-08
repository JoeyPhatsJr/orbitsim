"""Flyby targeting: correct a Lambert intercept seed onto a ~200 km Moon pass.

The Lambert planner aims at the Moon's centre, which in N-body misses by ~44,000 km.
moon_flyby_node corrects the seed's delta-V on the B-plane, using the same prediction
propagator the drawn preview uses (never the flight propagators).
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from orbitsim.core import nbody as nb
from orbitsim.core.constants import MU_EARTH, MU_MOON, R_MOON
from orbitsim.core.elements import state_to_elements
from orbitsim.core.flyby import conic_periapsis
from orbitsim.core.flyby_targeting import (
    FLYBY_TARGET_ALT_M, FLYBY_TOL_M, FlybyPlan, MoonPass, first_moon_pass, moon_flyby_node,
)
from orbitsim.core.maneuvers import ManeuverNode, apply_maneuver
from orbitsim.core.moon import moon_state_at
from orbitsim.core.optimize import intercept_node
from orbitsim.core.state import StateVector

# Measured online (DE440) value of the app's "I" seed for the default sandbox LEO.
# (Radial is + out since the radial-sign fix; it was measured as -38.14 on the old axis.)
SEED = ManeuverNode(3669.3721983198875, 3129.318767185485, -9.7657275283089, 38.144927341423305)

# Post-burn state of the centre-aimed Moon intercept (tests/render/test_preview_accuracy.py).
PINNED = StateVector(
    r=np.array([-8739999.637530766, -1247379.9275047905, -178197.13320472895]),
    v=np.array([1496.0148216363118, -9205.571968293121, -1324.9368000308782]),
    mu=398600400000000.0, epoch_s=3669.367180937791)


def _leo():
    from orbitsim.__main__ import _default_world      # pure sim/core imports only
    return _default_world().vessels[0].state


def _independent_pe_alt(post):
    """Reference: prediction propagator at a different cadence (600 s) + conic Pe."""
    cur, prev, prev_d = post, None, None
    with nb.stable_prediction_ephemeris():
        for _ in range(int(20 * 86400 / 600)):
            m = moon_state_at(cur.epoch_s)
            d = float(np.linalg.norm(cur.r - m.r))
            if prev_d is not None and d > prev_d and prev_d < nb.MOON_SOI_M:
                mp = moon_state_at(prev.epoch_s)
                return conic_periapsis(prev.r - mp.r, prev.v - mp.v, MU_MOON).radius_m - R_MOON
            prev, prev_d = cur, d
            cur = nb.propagate_solar_prediction(cur, 600.0, max_step_s=86400.0, min_substep_s=1.0)
    raise AssertionError("no Moon pass in 20 days")


@pytest.fixture
def de440():
    nb.refresh_ephemeris_cache(PINNED.epoch_s)
    if not nb.ephemeris_available():
        pytest.skip("DE440 ephemeris not available offline")


@pytest.fixture(scope="module")
def seed():
    """The app's 'I' Lambert seed: == SEED online, its circular-Moon twin offline."""
    nb.refresh_ephemeris_cache(0.0)          # DE440 if present, circular Moon otherwise
    st = _leo()
    return intercept_node(st, moon_state_at(0.0), MU_EARTH,
                          np.linspace(0, state_to_elements(st).period_s, 24),
                          np.linspace(3.0e3, 14 * 86400.0, 48))


@pytest.fixture(scope="module")
def plan(seed):
    return moon_flyby_node(_leo(), seed)


def test_first_moon_pass_matches_plan2_converged_truth(de440):
    p = first_moon_pass(PINNED)
    assert isinstance(p, MoonPass)
    assert abs(p.periapsis_radius_m - 45_670e3) <= 0.01 * (45_670e3 - R_MOON)
    assert abs(p.periapsis_epoch_s - 296.259 * 3600) <= 600.0 and p.in_soi
    assert p.b_plane_m.shape == (2,) and p.v_inf_mps > 0.0


def test_seed_is_the_measured_intercept(seed):
    nb.refresh_ephemeris_cache(0.0)          # the autouse fixture empties it per test
    if not nb.ephemeris_available():
        pytest.skip("SEED was measured with DE440")
    assert seed.epoch_s == pytest.approx(SEED.epoch_s, abs=1e-6)
    assert seed.dv_prograde_mps == pytest.approx(SEED.dv_prograde_mps, abs=1e-6)
    assert seed.dv_radial_mps == pytest.approx(SEED.dv_radial_mps, abs=1e-6)


def test_moon_flyby_node_reaches_200_km(seed, plan):
    st = _leo()
    assert isinstance(plan, FlybyPlan)
    assert plan.converged and plan.node.epoch_s == seed.epoch_s
    assert abs(plan.moon_pass.periapsis_radius_m - R_MOON - FLYBY_TARGET_ALT_M) <= FLYBY_TOL_M
    assert 150e3 <= _independent_pe_alt(apply_maneuver(st, plan.node)) <= 250e3
    assert abs(plan.node.magnitude_mps - seed.magnitude_mps) < 150.0


def test_moon_flyby_node_executor_matches_serial(seed, plan):
    with ThreadPoolExecutor(3) as ex:
        pooled = moon_flyby_node(_leo(), seed, executor=ex)
    assert pooled.node == plan.node
    assert np.array_equal(pooled.moon_pass.b_plane_m, plan.moon_pass.b_plane_m)
    assert pooled.converged == plan.converged


def test_moon_flyby_node_raises_when_no_pass(monkeypatch):
    calls = {"n": 0}
    real = nb.solar_system_accel

    def counted(r, t):
        calls["n"] += 1
        return real(r, t)
    monkeypatch.setattr(nb, "solar_system_accel", counted)
    with pytest.raises(ValueError):
        moon_flyby_node(_leo(), ManeuverNode(3669.37, 10.0, 0.0, 0.0))   # stays in LEO
    # An orbit that can't reach the Moon is rejected without 20 days of fine steps
    # (that took ~35 s and ~10^6 gravity evaluations).
    assert calls["n"] < 1_000, calls["n"]


def test_planner_never_uses_flight_propagators(monkeypatch, seed, plan):
    for name in ("propagate_solar_system", "propagate_earth_moon"):
        monkeypatch.setattr(nb, name, lambda *a, **k: pytest.fail("planner used flight physics"))
    again = moon_flyby_node(_leo(), seed)
    assert again.node == plan.node


def test_with_an_executor_no_evaluation_runs_on_the_calling_thread(monkeypatch, seed, plan):
    # The app calls the planner from its planning *thread*; any evaluation run there
    # holds the render process's GIL for ~0.3 s (node_planned frame time +47%). With an
    # executor (the app's process pool) every evaluation must go to it.
    import threading
    from orbitsim.core import flyby_targeting as ft
    caller = threading.get_ident()
    threads = []
    real = ft.first_moon_pass

    def recorded(*a, **k):
        threads.append(threading.get_ident())
        return real(*a, **k)
    monkeypatch.setattr(ft, "first_moon_pass", recorded)
    with ThreadPoolExecutor(3) as ex:
        pooled = moon_flyby_node(_leo(), seed, executor=ex)
    assert threads and caller not in threads
    assert pooled.node == plan.node


def _flown_pe_alt(post, chunk_s):
    """Fly like the app: flight propagator in frame-sized chunks, the ephemeris cache
    refreshed before each one; conic Moon Pe at the closest chunk sample."""
    cur, prev, prev_d = post, None, None
    for _ in range(int(20 * 86400 / chunk_s)):
        m = moon_state_at(cur.epoch_s)
        d = float(np.linalg.norm(cur.r - m.r))
        if prev_d is not None and d > prev_d and prev_d < nb.MOON_SOI_M:
            mp = moon_state_at(prev.epoch_s)
            return conic_periapsis(prev.r - mp.r, prev.v - mp.v, MU_MOON).radius_m - R_MOON
        prev, prev_d = cur, d
        nb.refresh_ephemeris_cache(cur.epoch_s)
        cur = nb.propagate_solar_system(cur, chunk_s)
    raise AssertionError("no Moon pass in 20 days")


def test_flown_plan_passes_where_the_plan_says(seed, plan):
    # The planner aims with the prediction propagator; flight must fly that plan to
    # the same pass. Measured (2026-11-07): plan 198.0 km; flown 204.5 / 199.3 /
    # 197.2 km in 3600 / 600 / 60 s chunks; converged truth 193.8 km.
    nb.refresh_ephemeris_cache(0.0)
    if not nb.ephemeris_available():
        pytest.skip("measured with DE440")
    planned = plan.moon_pass.periapsis_radius_m - R_MOON
    post = apply_maneuver(_leo(), plan.node)
    for chunk_s in (3600.0, 600.0):
        assert abs(_flown_pe_alt(post, chunk_s) - planned) <= 15e3, chunk_s
