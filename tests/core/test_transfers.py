"""Tests for closed-form transfers."""
import numpy as np
from orbitsim.core.transfers import TransferSolution, hohmann
from orbitsim.core.constants import MU_EARTH


def test_hohmann_leo_to_geo_known_answer():
    """Curtis-style LEO->GEO: r1=6678 km, r2=42164 km.

    Expect dv1 ~ 2.42 km/s, dv2 ~ 1.47 km/s, total ~ 3.89 km/s, t ~ 5.26 h.
    Tolerance 1%.
    """
    r1 = 6678e3
    r2 = 42164e3
    sol = hohmann(r1, r2, MU_EARTH)
    assert sol.kind == "hohmann"
    assert len(sol.burns) == 2
    # total
    np.testing.assert_allclose(sol.dv_total_mps, 3890.0, rtol=0.01)
    # individual burns
    np.testing.assert_allclose(abs(sol.burns[0].dv_prograde_mps), 2420.0, rtol=0.01)
    np.testing.assert_allclose(abs(sol.burns[1].dv_prograde_mps), 1470.0, rtol=0.01)
    # transfer time ~ 5.26 hours
    np.testing.assert_allclose(sol.time_of_flight_s, 5.26 * 3600.0, rtol=0.01)


def test_hohmann_burns_are_prograde():
    sol = hohmann(7.0e6, 1.0e7, MU_EARTH)
    assert sol.burns[0].dv_prograde_mps > 0
    assert sol.burns[1].dv_prograde_mps > 0


from orbitsim.core.transfers import bielliptic


def test_bielliptic_three_burns():
    sol = bielliptic(7000e3, 105000e3, 210000e3, MU_EARTH)
    assert sol.kind == "bielliptic"
    assert len(sol.burns) == 3
    assert sol.dv_total_mps > 0


def test_bielliptic_cheaper_when_ratio_large():
    """For r2/r1 well above 11.94, bi-elliptic (large rb) beats Hohmann."""
    r1 = 7000e3
    r2 = 16.0 * r1  # ratio 16 > 11.94
    rb = 60.0 * r1
    h = hohmann(r1, r2, MU_EARTH)
    be = bielliptic(r1, r2, rb, MU_EARTH)
    assert be.dv_total_mps < h.dv_total_mps


def test_hohmann_cheaper_when_ratio_small():
    """For r2/r1 below 11.94, Hohmann beats bi-elliptic."""
    r1 = 7000e3
    r2 = 3.0 * r1  # ratio 3 < 11.94
    rb = 60.0 * r1
    h = hohmann(r1, r2, MU_EARTH)
    be = bielliptic(r1, r2, rb, MU_EARTH)
    assert h.dv_total_mps < be.dv_total_mps


from orbitsim.core.transfers import plane_change


def test_plane_change_formula():
    v = 7700.0
    di = np.deg2rad(10.0)
    expected = 2.0 * v * np.sin(di / 2.0)
    assert abs(plane_change(v, di) - expected) < 1e-9


def test_plane_change_zero():
    assert plane_change(7700.0, 0.0) == 0.0


from orbitsim.core.transfers import lambert
from orbitsim.core.state import StateVector
from orbitsim.core.propagate import propagate_kepler


def test_lambert_reproduces_hohmann():
    """A Lambert arc over the Hohmann tof reproduces Hohmann within 2%."""
    r1 = 7000e3
    r2 = 14000e3
    h = hohmann(r1, r2, MU_EARTH)
    tof = h.time_of_flight_s
    r1_vec = np.array([r1, 0.0, 0.0])
    # Hohmann arrival is nearly opposite (perturbed slightly to avoid numerical degeneracy).
    r2_vec_raw = np.array([-r2 * 0.9999999, r2 * 0.0001, 0.0])
    r2_vec = r2_vec_raw / np.linalg.norm(r2_vec_raw) * r2
    v1, v2 = lambert(r1_vec, r2_vec, tof, MU_EARTH)
    v_circ1 = np.sqrt(MU_EARTH / r1)
    dv1 = np.linalg.norm(v1 - np.array([0.0, v_circ1, 0.0]))
    np.testing.assert_allclose(dv1, abs(h.burns[0].dv_prograde_mps), rtol=0.02)


def test_lambert_arc_lands_on_target():
    """Propagating r1 with the solved v1 for tof lands within 1 km of r2."""
    r1_vec = np.array([8000e3, 0.0, 0.0])
    r2_vec = np.array([0.0, 12000e3, 2000e3])
    tof = 3600.0
    v1, v2 = lambert(r1_vec, r2_vec, tof, MU_EARTH)
    state = StateVector(r=r1_vec, v=v1, mu=MU_EARTH)
    arrived = propagate_kepler(state, tof)
    assert np.linalg.norm(arrived.r - r2_vec) < 1000.0
