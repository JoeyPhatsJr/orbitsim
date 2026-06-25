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
