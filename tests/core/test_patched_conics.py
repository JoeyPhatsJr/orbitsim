"""Tests for patched-conic SOI logic and frame shifts."""
import numpy as np
import pytest

pytest.importorskip("skyfield")
from orbitsim.core.patched_conics import dominant_body
from orbitsim.core.bodies import SUN, EARTH, PLANETS
from orbitsim.core.ephemeris import body_state

T = 30.0 * 365.25 * 86400.0  # ~2030


def test_point_near_earth_is_dominated_by_earth():
    earth = body_state("EARTH", T, center="SUN")
    # 100,000 km from Earth, well within Earth's SOI (~924,000 km).
    pos = earth.r + np.array([1.0e8, 0.0, 0.0])
    dom = dominant_body(pos, T, [SUN] + PLANETS)
    assert dom.name == "Earth"


def test_deep_space_point_is_dominated_by_sun():
    pos = np.array([0.7 * 1.496e11, 0.0, 0.0])  # 0.7 AU, far from any planet
    dom = dominant_body(pos, T, [SUN] + PLANETS)
    assert dom.name == "Sun"


from orbitsim.core.patched_conics import shift_frame
from orbitsim.core.state import StateVector
from orbitsim.core.constants import MU_EARTH, MU_SUN


def test_frame_shift_round_trip():
    """Earth-centered -> Sun-centered -> Earth-centered returns the original."""
    r = np.array([7.0e6, 1.0e6, -2.0e6])
    v = np.array([0.0, 7.5e3, 0.1e3])
    state_earth = StateVector(r=r, v=v, mu=MU_EARTH, epoch_s=T)
    state_helio = shift_frame(state_earth, "EARTH", "SUN", T, MU_SUN)
    state_back = shift_frame(state_helio, "SUN", "EARTH", T, MU_EARTH)
    assert np.linalg.norm(state_back.r - r) < 1.0          # 1 m
    assert np.linalg.norm(state_back.v - v) < 1e-3         # 1 mm/s
