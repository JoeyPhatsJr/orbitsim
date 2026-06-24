"""Tests for CelestialBody dataclass and sphere-of-influence calculations."""
import pytest
from orbitsim.core.bodies import CelestialBody


def test_soi_radius_earth_about_sun():
    """SOI of Earth about Sun: a=1.496e11 m => r_SOI ≈ 9.24e8 m (within 5%)."""
    sun = CelestialBody(name="Sun", mu=1.32712e20, radius_m=6.96e8)
    earth = CelestialBody(
        name="Earth", mu=3.986e14, radius_m=6.378e6, parent=sun
    )

    a = 1.496e11  # meters (1 AU)
    soi = earth.soi_radius_m(a)

    expected = 9.24e8
    relative_error = abs(soi - expected) / expected
    assert relative_error < 0.05, (
        f"SOI {soi} not within 5% of {expected} (error: {relative_error*100:.2f}%)"
    )


def test_soi_no_parent():
    """SOI returns inf when body has no parent."""
    sun = CelestialBody(name="Sun", mu=1.32712e20, radius_m=6.96e8)
    soi = sun.soi_radius_m(1e12)
    assert soi == float("inf")


def test_body_frozen():
    """CelestialBody is frozen (immutable dataclass)."""
    body = CelestialBody(name="Earth", mu=3.986e14, radius_m=6.378e6)
    with pytest.raises(AttributeError):
        body.mu = 3.987e14


def test_prebuilt_bodies_exist():
    from orbitsim.core.bodies import SUN, EARTH, MOON
    from orbitsim.core.constants import MU_EARTH, R_EARTH
    assert EARTH.mu == MU_EARTH
    assert EARTH.radius_m == R_EARTH
    assert EARTH.parent is SUN
    assert MOON.parent is EARTH
    assert SUN.parent is None


def test_prebuilt_earth_j2():
    from orbitsim.core.bodies import EARTH
    from orbitsim.core.constants import J2_EARTH
    assert EARTH.j2 == J2_EARTH
