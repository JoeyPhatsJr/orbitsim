"""Pure helpers for the live maneuver preview (no Panda3D)."""
import numpy as np

from orbitsim.core.constants import MU_EARTH
from orbitsim.core.elements import state_to_elements
from orbitsim.core.state import StateVector
from orbitsim.render.orbit_lines import sample_orbit_points
from orbitsim.render.trajectory_helpers import kepler_estimate_points, should_submit_exact


def test_estimate_points_match_the_two_body_orbit():
    st = StateVector(r=np.array([6.878e6, 0.0, 0.0]), v=np.array([0.0, 9000.0, 0.0]), mu=MU_EARTH)
    np.testing.assert_array_equal(kepler_estimate_points(st, 64),
                                  sample_orbit_points(state_to_elements(st), 64))


def test_estimate_points_hyperbolic():
    st = StateVector(r=np.array([6.878e6, 0.0, 0.0]), v=np.array([0.0, 12500.0, 0.0]), mu=MU_EARTH)
    pts = kepler_estimate_points(st, 64)
    assert pts.shape == (64, 3) and np.isfinite(pts).all()


def test_estimate_points_none_for_degenerate_state():
    st = StateVector(r=np.array([6.878e6, 0.0, 0.0]), v=np.zeros(3), mu=MU_EARTH)
    assert kepler_estimate_points(st, 64) is None


K1, K2 = (100.0, 0.0, 0.0, 50.0), (110.0, 0.0, 0.0, 50.0)


def test_no_submit_while_still_changing():
    assert not should_submit_exact(K2, K1, None, last_change_t=10.0, now=10.1, settle_s=0.25)


def test_submit_once_settled():
    assert should_submit_exact(K2, K1, None, last_change_t=10.0, now=10.3, settle_s=0.25)


def test_no_resubmit_when_exact_is_current():
    # Today the 6 s preview job was resubmitted every 0.2 s even when nothing changed.
    assert not should_submit_exact(K2, K2, None, last_change_t=0.0, now=99.0, settle_s=0.25)


def test_no_submit_while_a_job_is_in_flight():
    assert not should_submit_exact(K2, K1, K1, last_change_t=0.0, now=99.0, settle_s=0.25)
