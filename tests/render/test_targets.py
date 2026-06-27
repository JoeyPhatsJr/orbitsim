import numpy as np
from orbitsim.render.targets import MoonTarget
from orbitsim.core.moon import moon_state_at


def test_name_and_delegation():
    t = MoonTarget()
    assert t.name == "Moon"
    for ts in (0.0, 1.0e5, 3.0e5):
        np.testing.assert_allclose(t.state_at(ts).r, moon_state_at(ts).r, rtol=0, atol=0)


def test_moon_target_supports_closest_approach():
    from orbitsim.render.targets import MoonTarget
    assert MoonTarget.supports_closest_approach is True


def test_lagrange_target_position_matches_core():
    import numpy as np
    from orbitsim.render.targets import LagrangePointTarget
    from orbitsim.core.nbody import earth_fixed_lagrange_points
    t = 1.0e5
    tgt = LagrangePointTarget("L1", "L1")
    assert tgt.name == "L1"
    assert tgt.supports_closest_approach is False
    st = tgt.state_at(t)
    assert np.allclose(st.r, earth_fixed_lagrange_points(t)["L1"])


def test_lagrange_target_velocity_matches_finite_difference():
    # The point's velocity must equal a central finite-difference of its position
    # (proves v is the true rigid-rotation velocity, not a z-axis approximation).
    import numpy as np
    from orbitsim.render.targets import LagrangePointTarget
    from orbitsim.core.nbody import earth_fixed_lagrange_points
    t, dt = 1.0e5, 1.0
    st = LagrangePointTarget("L4", "L4").state_at(t)
    r0 = earth_fixed_lagrange_points(t - dt)["L4"]
    r1 = earth_fixed_lagrange_points(t + dt)["L4"]
    v_fd = (r1 - r0) / (2.0 * dt)
    assert np.linalg.norm(st.v - v_fd) < 1.0   # within 1 m/s
