"""Tests for encounter detection along a predicted trajectory.

Encounters are runs of samples whose dominant body differs from the frame's
primary body (Moon / planet SOI passes in the geocentric sandbox). The core
function is pure: it takes sampled positions + epochs and a classifier, and
returns closest-approach and hyperbolic-excess-speed records. These tests use a
synthetic classifier so they run offline with no ephemeris.
"""

import numpy as np
import pytest

from orbitsim.core.encounters import find_encounters, Encounter


def _straight_line(v_mps, b_m, t0, t1, n):
    """Sample a straight-line flyby of a body at the origin.

    The ship moves along +x at speed v with impact parameter b on +y, so the
    closest approach to the origin is exactly b at x = 0 (t at the midpoint).
    Returns (points, epochs).
    """
    epochs = np.linspace(t0, t1, n)
    x = v_mps * epochs  # x = 0 at t = 0
    pts = np.stack([x, np.full(n, b_m), np.zeros(n)], axis=1)
    return pts, epochs


def test_finds_single_encounter_with_correct_periapsis():
    # Body of radius R at the origin, SOI = 50,000 km. A straight-line pass with
    # impact parameter 8,000 km: closest approach is exactly 8,000 km at t=0.
    # A straight line is the zero-gravity (v_inf -> infinity) limit, so use a
    # near-massless body; vis-viva at the SOI edge then leaves v_inf ~ the speed.
    soi = 5.0e7
    b = 8.0e6
    mu = 1.0e11  # tiny body: 2mu/r at the SOI edge is negligible vs v^2
    radius = 1.7e6
    pts, epochs = _straight_line(v_mps=1000.0, b_m=b, t0=-1.0e5, t1=1.0e5, n=401)

    def dominant(r, t):
        if np.linalg.norm(r) < soi:
            return "Moon", np.zeros(3), mu, radius
        return "Earth", np.zeros(3), 0.0, 0.0

    encs = find_encounters(pts, epochs, dominant, primary_name="Earth")
    assert len(encs) == 1
    enc = encs[0]
    assert isinstance(enc, Encounter)
    assert enc.body_name == "Moon"
    # Periapsis equals the impact parameter to sub-sample accuracy.
    assert abs(enc.periapsis_radius_m - b) < 1.0e3
    # Periapsis is at the midpoint (t = 0).
    assert abs(enc.periapsis_epoch_s) < 1.0e3
    assert enc.closed  # trajectory enters and exits the SOI
    # A straight line is the v_inf -> infinity limit: measured v_inf ~ the speed.
    assert abs(enc.v_inf_mps - 1000.0) / 1000.0 < 0.05


def test_no_encounter_when_never_enters_soi():
    pts, epochs = _straight_line(v_mps=1000.0, b_m=8.0e6, t0=-1.0e5, t1=1.0e5, n=201)

    def dominant(r, t):
        return "Earth", np.zeros(3), 0.0, 0.0  # never dominated by another body

    assert find_encounters(pts, epochs, dominant, primary_name="Earth") == []


def test_open_encounter_when_trajectory_ends_inside_soi():
    # Sample only the approach half: the run never exits the SOI.
    soi = 5.0e7
    pts, epochs = _straight_line(v_mps=1000.0, b_m=8.0e6, t0=-1.0e5, t1=0.0, n=201)

    def dominant(r, t):
        if np.linalg.norm(r) < soi:
            return "Moon", np.zeros(3), 4.9e12, 1.7e6
        return "Earth", np.zeros(3), 0.0, 0.0

    encs = find_encounters(pts, epochs, dominant, primary_name="Earth")
    assert len(encs) == 1
    assert not encs[0].closed
    assert encs[0].exit_epoch_s is None


def test_periapsis_below_surface_is_an_impact():
    # Impact parameter smaller than the body radius -> flagged as an impact.
    soi = 5.0e7
    b = 1.0e6  # inside a 1.7e6 m radius
    pts, epochs = _straight_line(v_mps=1000.0, b_m=b, t0=-1.0e5, t1=1.0e5, n=401)

    def dominant(r, t):
        if np.linalg.norm(r) < soi:
            return "Moon", np.zeros(3), 4.9e12, 1.7e6
        return "Earth", np.zeros(3), 0.0, 0.0

    enc = find_encounters(pts, epochs, dominant, primary_name="Earth")[0]
    assert enc.impact
    assert enc.periapsis_radius_m < enc.body_radius_m


def test_two_separate_encounters_are_reported_in_order():
    # Two SOI dips separated by a stretch dominated by the primary.
    soi = 1.0e7
    epochs = np.linspace(0.0, 10.0, 1001)
    # A wobble in y that dips inside the SOI at t=2.5 and t=7.5 only (endpoints
    # sit well outside), giving exactly two interior encounters.
    y = 2.0e7 - 1.9e7 * np.cos(2.0 * np.pi * (epochs - 2.5) / 5.0)
    pts = np.stack([np.zeros_like(epochs), y, np.zeros_like(epochs)], axis=1)

    def dominant(r, t):
        if np.linalg.norm(r) < soi:
            return "Moon", np.zeros(3), 4.9e12, 1.7e6
        return "Earth", np.zeros(3), 0.0, 0.0

    encs = find_encounters(pts, epochs, dominant, primary_name="Earth")
    assert len(encs) == 2
    assert encs[0].periapsis_epoch_s < encs[1].periapsis_epoch_s
    assert all(e.body_name == "Moon" for e in encs)


def test_moving_body_periapsis_is_relative_to_body_center():
    # The body drifts in +x; classifier reports its moving center. Periapsis must
    # be measured against that moving center, not the frame origin.
    soi = 5.0e7
    b = 6.0e6
    vbody = 500.0
    epochs = np.linspace(-1.0e5, 1.0e5, 401)
    center_x = vbody * epochs
    # Ship tracks the body in x and passes it with impact parameter b in y.
    pts = np.stack([center_x, np.full_like(epochs, b), np.zeros_like(epochs)], axis=1)

    def dominant(r, t):
        center = np.array([vbody * t, 0.0, 0.0])
        if np.linalg.norm(r - center) < soi:
            return "Moon", center, 4.9e12, 1.7e6
        return "Earth", np.zeros(3), 0.0, 0.0

    enc = find_encounters(pts, epochs, dominant, primary_name="Earth")[0]
    assert abs(enc.periapsis_radius_m - b) < 1.0e3


def test_grazing_pass_between_samples_is_detected():
    # A fast, shallow flyby whose entire SOI pass falls *between* two coarse
    # samples: no sampled point lands inside the SOI, so the sample-only scan
    # misses it. Gap refinement (default) must recover it.
    #
    # Body at (0, y_off, 0) with SOI = R; ship flies straight along +x through
    # the origin, y = z = 0. Closest approach is y_off = 0.6 R < R (inside), but
    # the two samples that bracket x = 0 sit at |x| = R, i.e. distance
    # sqrt(R^2 + (0.6R)^2) = 1.17 R > R (outside). No sample is inside the SOI.
    soi = 5.0e7
    y_off = 0.6 * soi
    delta = soi  # bracketing samples at |x| = R
    mu = 1.0e11  # near-massless: v_inf ~ speed
    radius = 1.0e6
    center = np.array([0.0, y_off, 0.0])
    # Samples at x = -3d, -2d, -d, +d, +2d, +3d (deliberately skipping x = 0).
    xs = np.array([-3, -2, -1, 1, 2, 3], dtype=float) * delta
    v = 1000.0
    epochs = xs / v  # x = v * t  ->  t = x / v
    pts = np.stack([xs, np.zeros(6), np.zeros(6)], axis=1)

    def dominant(r, t):
        if np.linalg.norm(r - center) < soi:
            return "Mars", center, mu, radius
        return "Earth", np.zeros(3), 0.0, 0.0

    # Sample-only scan (refinement disabled) reproduces the miss.
    assert find_encounters(pts, epochs, dominant, primary_name="Earth", refine_steps=0) == []

    # With gap refinement on (the default), the graze is recovered.
    encs = find_encounters(pts, epochs, dominant, primary_name="Earth")
    mars = [e for e in encs if e.body_name == "Mars"]
    assert len(mars) == 1
    enc = mars[0]
    # Closest approach is the impact parameter y_off, resolved between samples.
    assert abs(enc.periapsis_radius_m - y_off) < 0.05 * y_off
    assert enc.periapsis_radius_m < soi
    assert not enc.impact  # y_off well above the surface
    # The drawn segment anchors to the two bracketing samples (valid indices).
    assert enc.start_index == 2 and enc.end_index == 3
    assert 0 <= enc.periapsis_index < len(pts)


def test_refinement_does_not_invent_encounters_on_a_clear_path():
    # A straight pass that stays comfortably outside the SOI everywhere: gap
    # refinement must not manufacture a spurious encounter.
    soi = 1.0e7
    center = np.array([0.0, 5.0e7, 0.0])  # far from the y = 0 track
    xs = np.linspace(-1.0e8, 1.0e8, 50)
    epochs = xs / 1000.0
    pts = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)

    def dominant(r, t):
        if np.linalg.norm(r - center) < soi:
            return "Mars", center, 1.0e11, 1.0e6
        return "Earth", np.zeros(3), 0.0, 0.0

    assert find_encounters(pts, epochs, dominant, primary_name="Earth") == []


# ---------------------------------------------------------------------------
# Integration: a real translunar arc under the offline (circular-Moon) N-body.
# ---------------------------------------------------------------------------
def test_translunar_arc_detects_a_moon_encounter():
    from orbitsim.core.state import StateVector
    from orbitsim.core.constants import MU_EARTH
    from orbitsim.core.moon import moon_state_at
    from orbitsim.core.nbody import propagate_earth_moon, MOON_SOI_M
    from orbitsim.core.encounters import find_encounters, earth_moon_dominant

    # Build a guaranteed SOI pass in the Moon frame — fly across the Moon at a
    # fixed offset with a modest relative speed — then shift into the Earth
    # frame and propagate under the real Earth+Moon N-body field. (A free-return
    # from LEO would need a lead angle the Moon doesn't have at t0; the detection
    # logic is what's under test here, not translunar targeting.)
    t0 = 0.0
    m = moon_state_at(t0)
    u = np.array([1.0, 0.0, 0.0])  # cross-track approach axis
    w = np.array([0.0, 0.0, 1.0])  # offset axis (orbit normal)
    b0, v_rel = 1.2e7, 800.0
    r0 = m.r + (-1.3 * MOON_SOI_M) * u + b0 * w  # just outside the SOI
    v0 = m.v + v_rel * u  # heading across the Moon
    state = StateVector(r=r0, v=v0, mu=MU_EARTH, epoch_s=t0)

    n = 512
    dt = (4.0 * 86400.0) / n
    pts = np.empty((n, 3))
    epochs = np.empty(n)
    cur = state
    pts[0], epochs[0] = cur.r, cur.epoch_s
    for i in range(1, n):
        cur = propagate_earth_moon(cur, dt)
        pts[i], epochs[i] = cur.r, cur.epoch_s

    encs = find_encounters(pts, epochs, earth_moon_dominant, primary_name="Earth")
    moon_encs = [e for e in encs if e.body_name == "Moon"]
    assert len(moon_encs) == 1
    enc = moon_encs[0]
    assert enc.closed and not enc.impact
    # Periapsis inside the SOI, above the surface, and near the aimed offset
    # (Moon gravity focuses it somewhat below the b0 = 12,000 km approach line).
    assert enc.body_radius_m < enc.periapsis_radius_m < MOON_SOI_M
    assert enc.periapsis_radius_m < b0
    # v_inf is positive and below the approach speed (vis-viva at the SOI edge).
    assert 0.0 < enc.v_inf_mps < v_rel
