"""Tests for rocket-equation flight physics."""
import numpy as np
import pytest
from orbitsim.core.flight import (
    tsiolkovsky_dv, mass_flow_rate, thrust_accel_mps2,
)


def test_tsiolkovsky_known_answer():
    # ve=3000, m0=2000, mf=1000 -> 3000*ln(2) = 2079.44 m/s.
    assert abs(tsiolkovsky_dv(3000.0, 2000.0, 1000.0) - 3000.0 * np.log(2.0)
               ) < 1e-9


def test_tsiolkovsky_zero_fuel_is_zero_dv():
    assert tsiolkovsky_dv(3000.0, 1000.0, 1000.0) == 0.0


def test_tsiolkovsky_rejects_bad_masses():
    with pytest.raises(ValueError):
        tsiolkovsky_dv(3000.0, 1000.0, 2000.0)   # mf > m0


def test_mass_flow_rate():
    # ṁ = throttle*thrust/ve = 1.0 * 30000 / 3000 = 10 kg/s.
    assert abs(mass_flow_rate(1.0, 30000.0, 3000.0) - 10.0) < 1e-12
    assert mass_flow_rate(0.0, 30000.0, 3000.0) == 0.0


def test_thrust_accel():
    # a = throttle*thrust/mass = 0.5 * 30000 / 1500 = 10 m/s^2.
    assert abs(thrust_accel_mps2(0.5, 30000.0, 1500.0) - 10.0) < 1e-12
