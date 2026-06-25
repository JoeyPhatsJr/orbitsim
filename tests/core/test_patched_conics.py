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
