import numpy as np
from orbitsim.render.targets import MoonTarget
from orbitsim.core.moon import moon_state_at


def test_name_and_delegation():
    t = MoonTarget()
    assert t.name == "Moon"
    for ts in (0.0, 1.0e5, 3.0e5):
        np.testing.assert_allclose(t.state_at(ts).r, moon_state_at(ts).r, rtol=0, atol=0)
