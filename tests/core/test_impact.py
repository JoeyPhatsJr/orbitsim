"""Known-answer tests for core.impact.surface_crossing_s (conic time to surface)."""
import numpy as np
import pytest

from orbitsim.core.constants import MU_EARTH, R_EARTH
from orbitsim.core.impact import surface_crossing_s
from orbitsim.core.propagate import propagate_kepler
from orbitsim.core.state import StateVector


def _radius_after(r, v, dt):
    st = propagate_kepler(StateVector(r=np.asarray(r, float), v=np.asarray(v, float),
                                      mu=MU_EARTH, epoch_s=0.0), dt)
    return float(np.linalg.norm(st.r))


def test_radial_drop_matches_free_fall_time():
    # Dropped from rest at 200 km: radial ellipse (h = 0, e = 1). Free-fall time to
    # radius R from rest at r0 (Curtis-style radial Kepler): with a = r0/2,
    # t = sqrt(a^3/mu) * (E - sin E)|... evaluated between E(r0)=pi and E(R).
    r0 = R_EARTH + 2.0e5
    a = r0 / 2.0
    E = np.arccos(1.0 - R_EARTH / a)              # e = 1: r = a(1 - cos E)
    t_expected = np.sqrt(a**3 / MU_EARTH) * (np.pi - (E - np.sin(E)))
    t = surface_crossing_s(np.array([r0, 0.0, 0.0]), np.zeros(3), MU_EARTH, R_EARTH)
    assert t == pytest.approx(t_expected, rel=1e-9)
    assert 190.0 < t < 210.0                       # ~202 s, sanity


def test_elliptic_deorbit_crossing_lands_exactly_on_the_surface():
    # Apoapsis 400 km, periapsis 100 km below the surface: starting at apoapsis.
    ra, rp = R_EARTH + 4.0e5, R_EARTH - 1.0e5
    a = 0.5 * (ra + rp)
    va = np.sqrt(MU_EARTH * (2.0 / ra - 1.0 / a))
    r, v = np.array([ra, 0.0, 0.0]), np.array([0.0, va, 0.0])
    t = surface_crossing_s(r, v, MU_EARTH, R_EARTH)
    assert t is not None
    assert _radius_after(r, v, t) == pytest.approx(R_EARTH, abs=1e-3)
    # It is the *first* (inbound) crossing: just before, still above the surface.
    assert _radius_after(r, v, t - 1.0) > R_EARTH


def test_outbound_ellipse_crosses_after_apoapsis():
    # Just after periapsis-side launch, climbing: the next crossing comes after
    # coming back over apoapsis, so the time exceeds half the time above ground.
    ra, rp = R_EARTH + 4.0e5, R_EARTH - 1.0e5
    a = 0.5 * (ra + rp)
    va = np.sqrt(MU_EARTH * (2.0 / ra - 1.0 / a))
    r_apo, v_apo = np.array([ra, 0.0, 0.0]), np.array([0.0, va, 0.0])
    t_down = surface_crossing_s(r_apo, v_apo, MU_EARTH, R_EARTH)
    # Run time backwards from apoapsis to get an outbound state 100 s before it.
    st = propagate_kepler(StateVector(r=r_apo, v=v_apo, mu=MU_EARTH, epoch_s=0.0), -100.0)
    assert float(np.dot(st.r, st.v)) > 0.0         # outbound
    t = surface_crossing_s(st.r, st.v, MU_EARTH, R_EARTH)
    assert t == pytest.approx(t_down + 100.0, rel=1e-9)


def test_hyperbolic_impactor_crossing():
    # Inbound hyperbola from 50,000 km aimed 3,000 km off-centre (periapsis inside).
    r = np.array([5.0e7, 3.0e6, 0.0])
    v = np.array([-8000.0, 0.0, 0.0])
    t = surface_crossing_s(r, v, MU_EARTH, R_EARTH)
    assert t is not None
    assert _radius_after(r, v, t) == pytest.approx(R_EARTH, abs=1e-2)
    assert _radius_after(r, v, t - 5.0) > R_EARTH


def test_no_crossing_when_periapsis_is_above_the_surface():
    r = 7.0e6
    v = np.sqrt(MU_EARTH / r)
    assert surface_crossing_s(np.array([r, 0, 0.0]), np.array([0, v, 0.0]),
                              MU_EARTH, R_EARTH) is None


def test_no_crossing_for_an_outbound_escape():
    r = np.array([R_EARTH + 3.0e5, 0.0, 0.0])
    v = np.array([12000.0, 1000.0, 0.0])           # radially out, hyperbolic
    assert surface_crossing_s(r, v, MU_EARTH, R_EARTH) is None


def test_no_crossing_when_already_at_or_below_the_surface():
    r = np.array([R_EARTH - 10.0, 0.0, 0.0])
    assert surface_crossing_s(r, np.array([-10.0, 0, 0]), MU_EARTH, R_EARTH) is None


def _near_parabolic(energy_jpkg, r0=4.0e8, rp=R_EARTH - 1.0e6):
    """Inbound at r0 with specific energy ~energy_jpkg and periapsis ~rp."""
    h = np.sqrt(2.0 * MU_EARTH * rp)                   # parabolic h for that periapsis
    v2 = 2.0 * (energy_jpkg + MU_EARTH / r0)
    vt = h / r0
    vr = -np.sqrt(v2 - vt * vt)
    return np.array([r0, 0.0, 0.0]), np.array([vr, vt, 0.0])


def test_near_parabolic_crossing_is_continuous_across_escape_energy():
    # Review of PR #19: at -1e-6 J/kg (400,000 km out) M = E - e sin E lost all its
    # digits and the crossing came out as t = 0 (instant landing from 400,000 km).
    rp = R_EARTH - 1.0e6
    p = 2.0 * rp
    nu0 = -np.arccos(p / 4.0e8 - 1.0)
    nuc = -np.arccos(p / R_EARTH - 1.0)
    barker = lambda nu: 0.5 * np.sqrt(p**3 / MU_EARTH) * (np.tan(nu / 2) + np.tan(nu / 2)**3 / 3)
    t_parabola = barker(nuc) - barker(nu0)             # Barker's equation, exact at e = 1
    for energy in (-1.0e-6, -1.0e-3, 0.0, 1.0e-3, 1.0e-6):
        t = surface_crossing_s(*_near_parabolic(energy), MU_EARTH, R_EARTH)
        assert t == pytest.approx(t_parabola, rel=1e-6), energy
    for energy in (-10.0, 10.0):                       # regular branches agree nearby
        t = surface_crossing_s(*_near_parabolic(energy), MU_EARTH, R_EARTH)
        assert t == pytest.approx(t_parabola, rel=1e-3), energy


def test_radial_parabolic_infall():
    r0 = 4.0e8
    v = -np.sqrt(2.0 * MU_EARTH / r0)                  # escape speed, straight down
    expected = (r0**1.5 - R_EARTH**1.5) / (1.5 * np.sqrt(2.0 * MU_EARTH))
    t = surface_crossing_s(np.array([r0, 0.0, 0.0]), np.array([v, 0.0, 0.0]),
                           MU_EARTH, R_EARTH)
    assert t == pytest.approx(expected, rel=1e-6)
