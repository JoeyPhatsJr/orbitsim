"""Physical constants from astropy.constants (authoritative source, not hand-typed)."""
from astropy import constants as ac, units as u

# Gravitational constant [m^3/(kg*s^2)]
G = float(ac.G.to(u.m**3 / (u.kg * u.s**2)).value)

# Standard gravitational parameters μ = G·M  [m^3/s^2]
MU_SUN = float((ac.G * ac.M_sun).to(u.m**3 / u.s**2).value)
MU_EARTH = float((ac.G * ac.M_earth).to(u.m**3 / u.s**2).value)
# Moon mass from IAU: GM_moon = 4.90486959e12 m^3/s^2 (Curtis, Vallado)
MU_MOON = 4.90486959e12

# Equatorial radii [m]
R_EARTH = float(ac.R_earth.to(u.m).value)
R_SUN = float(ac.R_sun.to(u.m).value)
# Moon mean radius from IAU (no astropy constant available)
R_MOON = 1.7374e6

# Earth's oblateness coefficient J2 (dimensionless, source: IERS)
J2_EARTH = 1.08263e-3
