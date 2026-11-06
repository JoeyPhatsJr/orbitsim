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


# The exact preview must follow the ship, not just the burn: a burn-now node's start
# moves with the vessel, and a manual burn / quickload changes a scheduled node's start.
from orbitsim.render.trajectory_helpers import (  # noqa: E402
    PREVIEW_DRIFT_TOL_M, PREVIEW_DRIFT_TOL_MPS, preview_start_drifted, should_refresh_exact,
)

S0 = StateVector(r=np.array([7.0e6, 0.0, 0.0]), v=np.array([0.0, 7.5e3, 0.0]), mu=MU_EARTH)


def test_start_not_drifted_within_tolerance():
    near = StateVector(r=S0.r + [0.5 * PREVIEW_DRIFT_TOL_M, 0, 0],
                       v=S0.v + [0, 0.5 * PREVIEW_DRIFT_TOL_MPS, 0], mu=MU_EARTH)
    assert not preview_start_drifted(S0, near)


def test_start_drifted_by_position_or_velocity():
    moved = StateVector(r=S0.r + [10_000.0, 0, 0], v=S0.v, mu=MU_EARTH)    # ship moved on
    kicked = StateVector(r=S0.r, v=S0.v + [0, 5.0, 0], mu=MU_EARTH)       # manual burn
    assert preview_start_drifted(S0, moved) and preview_start_drifted(S0, kicked)


def test_start_drifted_when_nothing_drawn_yet():
    assert preview_start_drifted(None, S0)


def test_refresh_only_for_the_drawn_burn_when_idle_and_paced():
    k = (100.0, 0.0, 0.0, None)
    assert should_refresh_exact(k, k, None, drifted=True, now=1.0, last_submit_t=0.7, refresh_s=0.2)
    assert not should_refresh_exact(k, k, None, drifted=False, now=1.0, last_submit_t=0.0, refresh_s=0.2)
    assert not should_refresh_exact(k, k, None, drifted=True, now=1.0, last_submit_t=0.9, refresh_s=0.2)
    assert not should_refresh_exact(k, k, k, drifted=True, now=1.0, last_submit_t=0.0, refresh_s=0.2)
    assert not should_refresh_exact(k, None, None, drifted=True, now=1.0, last_submit_t=0.0, refresh_s=0.2)


def test_preview_key_changes_with_target():
    from orbitsim.core.maneuvers import ManeuverNode
    from orbitsim.render.trajectory_helpers import maneuver_preview_key
    n = ManeuverNode(10.0, 1.0, 0.0, 0.0)
    assert maneuver_preview_key(n, 10.0, "Moon") != maneuver_preview_key(n, 10.0, None)
    assert maneuver_preview_key(n, 10.0, "Moon") != maneuver_preview_key(n, 10.0, "Mars")


def test_encounter_source_exact_preview():
    from orbitsim.render.trajectory_helpers import encounter_source
    assert encounter_source(True, True, ["p"], ["l"]) == ["p"]


def test_encounter_source_editing_shows_none():
    from orbitsim.render.trajectory_helpers import encounter_source
    assert encounter_source(True, False, ["p"], ["l"]) == []


def test_encounter_source_no_burn_uses_live():
    from orbitsim.render.trajectory_helpers import encounter_source
    assert encounter_source(False, False, [], ["l"]) == ["l"]
    assert encounter_source(False, False, [], None) == []
