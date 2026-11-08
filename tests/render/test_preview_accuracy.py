"""The drawn Moon-transfer preview must predict the converged path.

Planning (2026-11-04) measured today's preview at a 6,057 km Moon pass when the
converged path passes at ~45,670 km: periapsis-sized steps with a 120 s floor are far
too coarse near LEO, and a flyby amplifies the departure error. See
docs/superpowers/plans/2026-11-04-map-readability-2-live-preview.md.
"""

import numpy as np
import pytest

from orbitsim.core import nbody as nb
from orbitsim.core.constants import R_MOON
from orbitsim.core.state import StateVector
from orbitsim.render.trajectory_sampling import sample_preview

# Post-burn state the app previews for the "I" Moon intercept (captured 2026-11-04).
PINNED = StateVector(
    r=np.array([-8739999.637530766, -1247379.9275047905, -178197.13320472895]),
    v=np.array([1496.0148216363118, -9205.571968293121, -1324.9368000308782]),
    mu=398600400000000.0, epoch_s=3669.367180937791)
# Converged truth: two fine methods agree to 0.03% (current-distance steps at divisor
# 10,000: 45,677 km / 296.258 h; periapsis steps at divisor 3,200: 45,664 km / 296.26 h).
REF_PERIAPSIS_RADIUS_M = 45_670.0e3
REF_PERIAPSIS_EPOCH_S = 296.259 * 3600.0
PREVIEW_FLOOR_S = 1.0          # OrbitApp.PREDICTION_MIN_SUBSTEP_S


def _moon(encounters):
    return next(e for e in encounters if e.body_name == "Moon")


@pytest.fixture
def de440():
    nb.refresh_ephemeris_cache(PINNED.epoch_s)
    if not nb.ephemeris_available():
        pytest.skip("DE440 ephemeris not available offline")


def test_moon_transfer_preview_matches_converged_truth(de440):
    _pts, encounters, _ap = sample_preview(PINNED, True, PREVIEW_FLOOR_S)
    moon = _moon(encounters)
    ref_alt = REF_PERIAPSIS_RADIUS_M - R_MOON
    alt = moon.periapsis_radius_m - R_MOON
    assert abs(alt - ref_alt) <= 0.01 * ref_alt, f"Moon periapsis altitude {alt/1e3:,.0f} km vs {ref_alt/1e3:,.0f} km"
    assert abs(moon.periapsis_epoch_s - REF_PERIAPSIS_EPOCH_S) <= 600.0


def test_moon_transfer_preview_stays_cheap(de440, monkeypatch):
    # Deterministic speed check (gravity evaluations, not wall time): the spec wants the
    # exact preview in about a second. With the 4th-order step (flight-accuracy plan)
    # divisor 300 costs ~5.3k evals (~0.6 s) at 0.02% error; 1000 cost ~17k (~2 s).
    calls = {"n": 0}
    real = nb.solar_system_accel

    def counted(r, t):
        calls["n"] += 1
        return real(r, t)
    monkeypatch.setattr(nb, "solar_system_accel", counted)
    sample_preview(PINNED, True, PREVIEW_FLOOR_S)
    assert calls["n"] <= 6_000, calls["n"]


def test_escape_preview_is_finite_and_fast_enough(de440, monkeypatch):
    # Counted like the Moon preview above (a wall-clock bound failed under parallel
    # test load): ~3.0k gravity evaluations, ~1 s on the dev machine.
    calls = {"n": 0}
    real = nb.solar_system_accel

    def counted(r, t):
        calls["n"] += 1
        return real(r, t)
    monkeypatch.setattr(nb, "solar_system_accel", counted)
    esc = StateVector(r=PINNED.r, v=PINNED.v * 1.25, mu=PINNED.mu, epoch_s=PINNED.epoch_s)
    pts, _enc, _ap = sample_preview(esc, True, PREVIEW_FLOOR_S)
    assert np.isfinite(pts).all()
    assert calls["n"] <= 3_500, calls["n"]


# Plan 4: the B-plane planner prototype's node (planned against the pre-Yoshida preview,
# where it read a 190 km pass). Converged truth today (prediction propagator at divisor
# 3,000, 3600 s chunks, conic Pe; re-measured 2026-11-05): -30.4 km, a grazing impact.
# The preview's 256 samples are 2.8 h apart; the old parabola through them read 3,298 km.
LOW_PASS_NODE_DV = (3131.43288823, -27.139359, -87.11806646)
LOW_PASS_NODE_EPOCH_S = 3669.3721983198875
REF_LOW_PASS_ALT_M = -30.4e3


def test_low_moon_pass_reads_its_true_periapsis(de440):
    from orbitsim.__main__ import _default_world
    from orbitsim.core.maneuvers import ManeuverNode, apply_maneuver
    node = ManeuverNode(LOW_PASS_NODE_EPOCH_S, *LOW_PASS_NODE_DV)
    post = apply_maneuver(_default_world().vessels[0].state, node)
    _pts, encounters, _ap = sample_preview(post, True, PREVIEW_FLOOR_S)
    moon = _moon(encounters)
    assert abs((moon.periapsis_radius_m - R_MOON) - REF_LOW_PASS_ALT_M) < 25e3, \
        (moon.periapsis_radius_m - R_MOON)     # the parabola read 3,258 km


def test_preview_reports_target_approach(de440):
    _pts, _enc, ap = sample_preview(PINNED, True, PREVIEW_FLOOR_S, target_name="Moon")
    assert ap.body_name == "Moon" and ap.distance_m < 1.0e8


def test_preview_without_target_has_no_approach(de440):
    _pts, _enc, ap = sample_preview(PINNED, True, PREVIEW_FLOOR_S)
    assert ap is None
