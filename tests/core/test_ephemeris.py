"""Ephemeris sanity anchors (Skyfield / DE440).

These tests download de440s.bsp on first run (~30 MB). They are slow once,
cached after. If offline (kernel missing and not downloadable), the whole
module is skipped via the availability guard.
"""
import numpy as np
import pytest

skyfield = pytest.importorskip("skyfield")
from orbitsim.core import ephemeris
from orbitsim.core.ephemeris import body_state
from orbitsim.core.constants import MU_SUN

pytestmark = pytest.mark.skipif(
    not ephemeris.available(), reason="DE440 kernel unavailable (offline)"
)


# 2030-01-01 00:00 TDB, seconds past J2000 (approx): 30 years * 365.25 d.
T_2030 = 30.0 * 365.25 * 86400.0


def test_earth_heliocentric_distance_about_1au():
    state = body_state("EARTH", T_2030, center="SUN")
    au = 1.495978707e11
    assert abs(state.r_mag - au) / au < 0.02  # within 2%


def test_earth_orbital_speed_about_29_8_kms():
    state = body_state("EARTH", T_2030, center="SUN")
    assert abs(state.v_mag - 29.8e3) / 29.8e3 < 0.02


def test_mars_heliocentric_distance_range():
    state = body_state("MARS", T_2030, center="SUN")
    # Mars heliocentric distance varies ~1.38–1.67 AU.
    au = 1.495978707e11
    assert 1.3 * au < state.r_mag < 1.7 * au
