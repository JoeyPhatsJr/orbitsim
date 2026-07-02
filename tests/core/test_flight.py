"""Tests for rocket-equation flight physics."""
import numpy as np
import pytest
from orbitsim.core.flight import (
    tsiolkovsky_dv, fuel_burned_for_dv, mass_flow_rate, thrust_accel_mps2,
    integrate_powered,
)
from orbitsim.core.state import StateVector
from orbitsim.core.propagate import propagate_kepler
from orbitsim.core.constants import MU_EARTH


def test_tsiolkovsky_known_answer():
    # ve=3000, m0=2000, mf=1000 -> 3000*ln(2) = 2079.44 m/s.
    assert abs(tsiolkovsky_dv(3000.0, 2000.0, 1000.0) - 3000.0 * np.log(2.0)
               ) < 1e-9


def test_tsiolkovsky_zero_fuel_is_zero_dv():
    assert tsiolkovsky_dv(3000.0, 1000.0, 1000.0) == 0.0


def test_tsiolkovsky_rejects_bad_masses():
    with pytest.raises(ValueError):
        tsiolkovsky_dv(3000.0, 1000.0, 2000.0)   # mf > m0


def test_fuel_burned_known_answer():
    # dv = ve -> mf = m0/e, so burned = m0*(1 - 1/e).
    assert abs(fuel_burned_for_dv(3000.0, 1000.0, 3000.0)
               - 1000.0 * (1.0 - np.exp(-1.0))) < 1e-9
    assert fuel_burned_for_dv(3000.0, 1000.0, 0.0) == 0.0


def test_fuel_burned_inverts_tsiolkovsky():
    # Burning fuel_burned_for_dv(dv) leaves a mass whose tsiolkovsky dv equals dv.
    ve, m0, dv = 3200.0, 1800.0, 1450.0
    mf = m0 - fuel_burned_for_dv(ve, m0, dv)
    assert abs(tsiolkovsky_dv(ve, m0, mf) - dv) < 1e-9


def test_fuel_burned_rejects_bad_input():
    with pytest.raises(ValueError):
        fuel_burned_for_dv(0.0, 1000.0, 100.0)   # ve <= 0
    with pytest.raises(ValueError):
        fuel_burned_for_dv(3000.0, 1000.0, -1.0)  # dv < 0


def test_mass_flow_rate():
    # ṁ = throttle*thrust/ve = 1.0 * 30000 / 3000 = 10 kg/s.
    assert abs(mass_flow_rate(1.0, 30000.0, 3000.0) - 10.0) < 1e-12
    assert mass_flow_rate(0.0, 30000.0, 3000.0) == 0.0


def test_thrust_accel():
    # a = throttle*thrust/mass = 0.5 * 30000 / 1500 = 10 m/s^2.
    assert abs(thrust_accel_mps2(0.5, 30000.0, 1500.0) - 10.0) < 1e-12


def test_zero_throttle_matches_kepler():
    """With no thrust the integrator is pure two-body gravity -> must track
    Kepler."""
    r = np.array([7.0e6, 0.0, 0.0])
    v = np.array([0.0, np.sqrt(MU_EARTH / 7.0e6), 0.0])
    s = StateVector(r=r, v=v, mu=MU_EARTH)
    dt = 120.0
    out, fuel = integrate_powered(
        s, dry_mass_kg=1000.0, fuel_kg=500.0,
        thrust_dir_unit=np.array([0.0, 1.0, 0.0]),
        throttle=0.0, max_thrust_n=30000.0, ve_mps=3000.0, dt_s=dt, substeps=200,
    )
    kep = propagate_kepler(s, dt)
    assert fuel == 500.0                                   # no fuel burned
    assert np.linalg.norm(out.r - kep.r) < 1.0            # within 1 m of analytic
    # Energy conserved (no thrust): epsilon unchanged.
    assert (abs(out.specific_energy - s.specific_energy) /
            abs(s.specific_energy) < 1e-7)


def test_free_space_burn_matches_rocket_equation():
    """In free space (mu=0), a full burn to depletion reaches ve*ln(m0/mf)."""
    s = StateVector(r=np.array([0.0, 0.0, 0.0]), v=np.array([0.0, 0.0, 0.0]),
                    mu=0.0)
    dry, fuel0, thrust, ve = 1000.0, 1000.0, 30000.0, 3000.0
    mdot = thrust / ve                                     # 10 kg/s
    burn_time = fuel0 / mdot                               # 100 s
    out, fuel = integrate_powered(
        s, dry_mass_kg=dry, fuel_kg=fuel0,
        thrust_dir_unit=np.array([1.0, 0.0, 0.0]),
        throttle=1.0, max_thrust_n=thrust, ve_mps=ve, dt_s=burn_time,
        substeps=2000,
    )
    expected_dv = ve * np.log((dry + fuel0) / dry)         # 3000*ln(2)=2079.44
    assert abs(out.v[0] - expected_dv) / expected_dv < 1e-3
    assert abs(fuel) < 1e-6                                 # fuel depleted


def test_burn_stops_when_fuel_exhausted():
    """Asking for more burn than fuel allows: speed caps at the rocket-equation
    dv."""
    s = StateVector(r=np.array([0.0, 0.0, 0.0]), v=np.array([0.0, 0.0, 0.0]),
                    mu=0.0)
    out, fuel = integrate_powered(
        s, dry_mass_kg=1000.0, fuel_kg=100.0,
        thrust_dir_unit=np.array([1.0, 0.0, 0.0]),
        throttle=1.0, max_thrust_n=30000.0, ve_mps=3000.0, dt_s=1000.0,
        substeps=2000,
    )
    expected_dv = 3000.0 * np.log(1100.0 / 1000.0)
    assert fuel == 0.0
    assert abs(out.v[0] - expected_dv) / expected_dv < 2e-3


def test_powered_burn_second_order_accuracy():
    """A finite LEO burn at the default substep count lands within metres of a
    fine-substep reference.

    Strang (half-impulse / gravity drift / half-impulse) splitting is 2nd
    order; applying the whole substep impulse before the drift is 1st order
    and misses this reference by ~700 m at substeps=50.
    """
    r0 = 7.0e6
    s = StateVector(r=np.array([r0, 0.0, 0.0]),
                    v=np.array([0.0, np.sqrt(MU_EARTH / r0), 0.0]), mu=MU_EARTH)
    kwargs = dict(
        dry_mass_kg=1000.0, fuel_kg=800.0,
        thrust_dir_unit=np.array([0.0, 1.0, 0.0]),
        throttle=1.0, max_thrust_n=30000.0, ve_mps=3000.0, dt_s=60.0,
    )
    ref, ref_fuel = integrate_powered(s, substeps=20000, **kwargs)
    out, fuel = integrate_powered(s, substeps=50, **kwargs)
    assert np.linalg.norm(out.r - ref.r) < 5.0
    assert abs(fuel - ref_fuel) < 1e-9      # exact rocket-equation bookkeeping


from orbitsim.core.nbody import earth_moon_accel, propagate_earth_moon


def test_nbody_free_space_burn_matches_rocket_equation():
    """In free space (mu=0, no gravity), N-body burn dV telescopes to ve*ln(m0/mf)."""
    from orbitsim.core.flight import integrate_powered_nbody
    s = StateVector(r=np.array([0.0, 0.0, 0.0]), v=np.array([0.0, 0.0, 0.0]),
                    mu=0.0, epoch_s=0.0)
    dry, fuel0, thrust, ve = 1000.0, 1000.0, 30000.0, 3000.0
    mdot = thrust / ve          # 10 kg/s
    burn_time = fuel0 / mdot    # 100 s
    out, fuel = integrate_powered_nbody(
        s, dry_mass_kg=dry, fuel_kg=fuel0,
        thrust_dir_unit=np.array([1.0, 0.0, 0.0]),
        throttle=1.0, max_thrust_n=thrust, ve_mps=ve, dt_s=burn_time,
    )
    expected_dv = ve * np.log((dry + fuel0) / dry)   # 3000*ln(2) ≈ 2079.44 m/s
    assert abs(out.v[0] - expected_dv) / expected_dv < 1e-3
    assert abs(fuel) < 1e-6     # fuel depleted to exactly 0


def test_nbody_fuel_reaches_zero_exactly():
    """Asking for more burn time than fuel allows: fuel ends at 0, not negative."""
    from orbitsim.core.flight import integrate_powered_nbody
    s = StateVector(r=np.array([0.0, 0.0, 0.0]), v=np.array([0.0, 0.0, 0.0]),
                    mu=0.0, epoch_s=0.0)
    out, fuel = integrate_powered_nbody(
        s, dry_mass_kg=1000.0, fuel_kg=100.0,
        thrust_dir_unit=np.array([1.0, 0.0, 0.0]),
        throttle=1.0, max_thrust_n=30000.0, ve_mps=3000.0, dt_s=1000.0,
    )
    expected_dv = 3000.0 * np.log(1100.0 / 1000.0)
    assert fuel == 0.0
    assert abs(out.v[0] - expected_dv) / expected_dv < 2e-3


def test_nbody_moon_perturbation_diverges_from_twobody():
    """Near the Moon, N-body trajectory diverges measurably from two-body."""
    from orbitsim.core.flight import integrate_powered_nbody, integrate_powered
    from orbitsim.core.moon import moon_state_at
    # Place ship 5000 km from the Moon (deep in its gravity well).
    t0 = 0.0
    rM = moon_state_at(t0).r
    r_ship = rM + np.array([5.0e6, 0.0, 0.0])
    v_ship = np.array([0.0, 500.0, 0.0])
    s = StateVector(r=r_ship, v=v_ship, mu=MU_EARTH, epoch_s=t0)
    dt = 3600.0   # 1 hour burn
    # N-body burn
    out_nbody, _ = integrate_powered_nbody(
        s, dry_mass_kg=1000.0, fuel_kg=5000.0,
        thrust_dir_unit=np.array([0.0, 1.0, 0.0]),
        throttle=0.1, max_thrust_n=10000.0, ve_mps=3000.0, dt_s=dt,
    )
    # Two-body burn (same call, two-body gravity)
    out_2body, _ = integrate_powered(
        s, dry_mass_kg=1000.0, fuel_kg=5000.0,
        thrust_dir_unit=np.array([0.0, 1.0, 0.0]),
        throttle=0.1, max_thrust_n=10000.0, ve_mps=3000.0, dt_s=dt, substeps=50,
    )
    divergence = np.linalg.norm(out_nbody.r - out_2body.r)
    assert divergence > 1000.0, f"expected N-body divergence near Moon, got {divergence:.1f} m"
