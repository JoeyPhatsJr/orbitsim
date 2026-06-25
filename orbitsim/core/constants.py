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

# Planetary standard gravitational parameters mu = GM [m^3/s^2].
# Source: IAU / JPL DE440 (NASA planetary fact sheet), documented values.
MU_MERCURY = 2.2032e13
MU_VENUS = 3.24859e14
MU_MARS = 4.282837e13
MU_JUPITER = 1.26686534e17
MU_SATURN = 3.7931187e16
MU_URANUS = 5.793939e15
MU_NEPTUNE = 6.836529e15

# Mean equatorial radii [m] (NASA planetary fact sheet).
R_MERCURY = 2.4397e6
R_VENUS = 6.0518e6
R_MARS = 3.3962e6
R_JUPITER = 7.1492e7
R_SATURN = 6.0268e7
R_URANUS = 2.5559e7
R_NEPTUNE = 2.4764e7

# Oblateness J2 (dimensionless) for the bodies where it matters (NASA fact sheet).
J2_MARS = 1.96045e-3
J2_JUPITER = 1.4736e-2
