# 01 — Phase 1: The Physics Core (test-driven)

This is the foundation. Everything else trusts it. Implement **strictly test-first** using the
known-answer values below. Work the tasks **in order** — each builds on the last.

All code here lives in `orbitsim/core/` and must obey the rules in `00-OVERVIEW.md` (SI units,
float64, frozen dataclasses, no rendering imports). Reference frame: J2000/ICRF, origin at the
central body.

> **Reference textbook for all formulas:** Curtis, *Orbital Mechanics for Engineering Students*
> (3rd ed.). Algorithm/Example numbers cited below come from it. Vallado is the deeper backup.

---

## Task 1.1 — `constants.py`

Pull constants from `astropy` so they are authoritative, then freeze them as module-level
`float64`. Do **not** hand-type digits.

```python
from astropy import constants as ac, units as u

G = float(ac.G.to(u.m**3 / (u.kg * u.s**2)).value)        # 6.674e-11

# Standard gravitational parameters μ = G·M  [m^3/s^2]
MU_SUN     = float((ac.G * ac.M_sun).to(u.m**3/u.s**2).value)   # 1.32712e20
MU_EARTH   = float((ac.G * ac.M_earth).to(u.m**3/u.s**2).value) # 3.986e14
MU_MOON    = ...   # use ac.GM_... where available, else G*M
# Equatorial radii [m]
R_EARTH = float(ac.R_earth.to(u.m).value)                 # 6.378137e6
```

For bodies astropy lacks a clean GM for, use IAU values (document the source in a comment).
J2 for Earth = `1.08263e-3` (dimensionless; cite IERS).

**Test (`tests/core/test_constants.py`):** assert `MU_EARTH` ≈ `3.986004418e14` within 1e6, and
`R_EARTH` ≈ `6.378137e6` within 1.0. These anchor the unit system.

---

## Task 1.2 — `bodies.py`

```python
@dataclass(frozen=True)
class CelestialBody:
    name: str
    mu: float           # standard gravitational parameter μ = GM  [m^3/s^2]
    radius_m: float     # mean equatorial radius [m]
    j2: float = 0.0     # oblateness coefficient (dimensionless)
    rotation_period_s: float = float("inf")  # sidereal rotation period [s]
    parent: "CelestialBody | None" = None    # body it orbits (None for the Sun)

    def soi_radius_m(self, semi_major_axis_m: float) -> float:
        """Sphere-of-influence radius. r_SOI = a · (m/M_parent)^(2/5).
        Approximate with (mu/mu_parent)^(2/5) since μ ∝ m. Returns inf if no parent."""
```

Provide pre-built instances `EARTH`, `SUN`, `MOON` (Phase 5 adds the rest). Hardcode `parent`
links (Earth.parent = SUN, Moon.parent = EARTH).

**Test:** Earth's SOI given `a = 1.496e11` m should be ≈ `9.24e8` m (924,000 km), tolerance 5%.

---

## Task 1.3 — `state.py`

```python
@dataclass(frozen=True)
class StateVector:
    """Position & velocity in an inertial frame centered on a body. SI, float64."""
    r: np.ndarray   # shape (3,), meters
    v: np.ndarray   # shape (3,), meters/second
    mu: float       # μ of the central body [m^3/s^2]
    epoch_s: float = 0.0   # seconds past J2000 (TDB)

    # validate in __post_init__: r.shape==(3,), dtype float64, finite. Make arrays read-only.

    @property
    def r_mag(self) -> float: ...      # |r|
    @property
    def v_mag(self) -> float: ...      # |v|
    @property
    def specific_energy(self) -> float:  # ε = v²/2 − μ/r
        ...
    @property
    def angular_momentum(self) -> np.ndarray:  # h = r × v
        ...
```

**Test:** for a circular LEO (`r=[7.0e6,0,0]`, `v=[0,√(μ/7e6),0]`): `specific_energy` ≈ `−μ/(2·7e6)`,
`|angular_momentum|` ≈ `r_mag·v_mag`. Tolerance 1e-6 relative.

---

## Task 1.4 — `elements.py` + conversions (the heart of Phase 1)

```python
@dataclass(frozen=True)
class KeplerianElements:
    a: float        # semi-major axis [m]  (negative for hyperbola)
    e: float        # eccentricity [-]
    i: float        # inclination [rad], 0..π
    raan: float     # right ascension of ascending node Ω [rad], 0..2π
    argp: float     # argument of periapsis ω [rad], 0..2π
    nu: float       # true anomaly ν [rad], 0..2π
    mu: float       # μ of central body [m^3/s^2]
    epoch_s: float = 0.0

    @property
    def period_s(self) -> float:    # 2π√(a³/μ); inf/ValueError if a<=0
    @property
    def semi_latus_rectum(self) -> float:  # p = a(1−e²)  (= h²/μ)
```

### 1.4a `state_to_elements(state) -> KeplerianElements`  (Curtis Algorithm 4.1)

Given `r`, `v`, `mu`:
```
r   = |r|;   v = |v|
vr  = dot(r, v) / r                      # radial velocity (sign gives ν quadrant)
h   = cross(r, v);   h_mag = |h|
i   = arccos(h_z / h_mag)
N   = cross([0,0,1], h);   N_mag = |N|   # node line
Ω   = arccos(N_x / N_mag);  if N_y < 0: Ω = 2π − Ω        # 0 if N_mag==0 (equatorial)
evec = (1/μ) * ((v² − μ/r)*r_vec − dot(r_vec,v_vec)*v_vec)
e   = |evec|
ω   = arccos(dot(N, evec)/(N_mag*e));  if evec_z < 0: ω = 2π − ω   # handle e≈0, N≈0
ν   = arccos(dot(evec, r_vec)/(e*r));  if vr < 0: ν = 2π − ν       # handle e≈0
a   = 1 / (2/r − v²/μ)                    # vis-viva inverse; works for all conics
```
Edge cases you MUST handle (write tests for each):
- **Circular** (`e≈0`): ω undefined → set ω=0, measure ν from the node line (or x-axis if also
  equatorial). Use a tolerance `1e-11`.
- **Equatorial** (`i≈0`): Ω undefined → set Ω=0, fold its angle into ω.
- Clamp `arccos` arguments to `[−1, 1]` before calling (floating error can give 1.0000000002).

### 1.4b `elements_to_state(elements) -> StateVector`  (Curtis Algorithm 4.2)

```
p = a*(1 − e²);   h = sqrt(μ * p)
# perifocal frame (PQW)
r_pqw = (p / (1 + e*cos ν)) * [cos ν, sin ν, 0]
v_pqw = (μ / h) * [−sin ν, e + cos ν, 0]
# rotation perifocal → inertial: Q = R3(−Ω) @ R1(−i) @ R3(−ω)
r_inertial = Q @ r_pqw
v_inertial = Q @ v_pqw
```
where `R3(θ)` rotates about z, `R1(θ)` about x (standard active rotation matrices — write them
as small helpers in `frames.py` and unit-test that `R3(0)=I`, `R3(θ)@R3(−θ)=I`).

### KNOWN-ANSWER TESTS (must pass)

**`state_to_elements`** — Curtis Example 4.3. Inputs are in km, km/s; **convert to SI in the test**
(`*1e3`, μ = `3.986004418e14`):
```
r = [-6045, -3490, 2500] km          v = [-3.457, 6.618, 2.533] km/s
Expected:
  a    ≈ 8788 km   (8.788e6 m)        e    ≈ 0.17121
  i    ≈ 153.249°  (2.6748 rad)       Ω    ≈ 255.279° (4.4552 rad)
  ω    ≈ 20.068°   (0.3503 rad)       ν    ≈ 28.446°  (0.4965 rad)
```
Tolerance: 0.5% on `a` and `e`; 0.2° on angles.

**`elements_to_state`** — Curtis Example 4.7 (a **hyperbolic** case, validates the all-conic path).
Given `h = 80000 km²/s, e = 1.4, i = 30°, Ω = 40°, ω = 60°, ν = 30°` (convert `h,p` to `a` via
`a = p/(1−e²) = (h²/μ)/(1−e²)`, which is negative — that's correct for a hyperbola):
```
Expected r ≈ [-4040, 4815, 3629] km     v ≈ [-10.39, -4.772, 1.744] km/s
```
Tolerance: 2 km on position, 0.02 km/s on velocity. (These are anchors; the round-trip test
below is the authoritative correctness check.)

**ROUND-TRIP (authoritative, use `hypothesis`):** for random valid elements
(`0 ≤ e < 0.99`, `a > R_earth`, all angles in range), `state_to_elements(elements_to_state(x))`
returns `x` within `1e-7` relative. This catches sign/quadrant bugs the textbook anchors miss.

---

## Task 1.5 — `kepler.py` (Kepler's equation)

Anomaly conversions and the time-of-flight solver. Implement the **elliptical** path first, then
**hyperbolic**. (A universal-variable solver is a nice later refactor; not required now.)

```python
def true_to_eccentric_anomaly(nu: float, e: float) -> float:
    # tan(E/2) = sqrt((1−e)/(1+e)) · tan(ν/2)      (elliptical, e<1)
def eccentric_to_mean_anomaly(E: float, e: float) -> float:
    # M = E − e·sin E                               (Kepler's equation)
def solve_kepler_elliptic(M: float, e: float, tol=1e-12, max_iter=50) -> float:
    # Newton–Raphson: E_{n+1} = E_n − (E_n − e·sin E_n − M)/(1 − e·cos E_n)
    # init: E0 = M + e·sin M (or M+e if e>0.8). Must converge in < ~10 iters.
def mean_to_true_anomaly(M: float, e: float) -> float: ...
```
Hyperbolic analogues (`e>1`): `M = e·sinh F − F`, Newton on `F`. Parabolic (`e≈1`): Barker's
equation — implement or raise `NotImplementedError` for now with a clear message.

### KNOWN-ANSWER TESTS
- `solve_kepler_elliptic(M=1.0, e=0.5)` → `E ≈ 1.498701` (tol 1e-6). Verify residual
  `E − e·sin E − M < 1e-12`.
- Round-trip: `mean_to_true_anomaly(eccentric_to_mean_anomaly(true_to_eccentric_anomaly(ν,e)... ))`
  chains return the input (tol 1e-10) for `e ∈ {0, 0.3, 0.7, 0.95}` and many ν.
- `e=0`: `E == M == ν` exactly.

---

## Task 1.6 — `propagate.py` (two-body propagation)

```python
def propagate_kepler(state: StateVector, dt: float) -> StateVector:
    """Analytic two-body propagation by Δt seconds (the 'on-rails' path).
    1. elements = state_to_elements(state)
    2. M0 = mean anomaly at elements.nu;   n = sqrt(μ / |a|³)
    3. M = M0 + n·dt   (use sinh form for hyperbola)
    4. nu = mean_to_true_anomaly(M, e)
    5. return elements_to_state(elements with new nu, epoch += dt)
    """

def propagate_numeric(state, dt, *, j2=False, third_bodies=()) -> StateVector:
    """High-fidelity path via scipy.integrate.solve_ivp (DOP853, rtol=1e-10, atol=1e-3 m).
    Acceleration = −μ r/|r|³  (+ J2 term if j2)  (+ Σ third-body if provided).
    Used when thrusting or when perturbations matter; analytic path used otherwise."""
```

J2 acceleration (add when `j2=True`), with `R` = body radius, `z` = r_z:
```
factor = 1.5 * J2 * μ * R² / r⁵
a_x = factor * x * (5 z²/r² − 1)
a_y = factor * y * (5 z²/r² − 1)
a_z = factor * z * (5 z²/r² − 3)
```

### TESTS (conservation — these are the real validators)
- **Period closure:** propagate a LEO orbit by exactly `period_s` → position returns to start
  within **1 mm** (analytic) / **1 m** (numeric). Half a period → apoapsis on the far side.
- **Energy & angular momentum** constant to 1e-9 (analytic) / 1e-7 (numeric, rtol-limited) over
  10 random Δt.
- **Analytic vs numeric agreement:** with `j2=False`, both must agree to < 1 m over one period
  (proves the integrator and the closed-form match).
- **Vis-viva** holds at every propagated point.
- **J2 sanity:** with `j2=True` on an inclined LEO over many orbits, the RAAN should regress
  (nodal precession ≈ `−1.5 n J2 (R/p)² cos i`) — assert sign and rough magnitude.

---

## Phase 1 exit criteria

`pytest tests/core -q` is **100% green**, including all known-answer tests, all round-trip
property tests, and all conservation tests. At that point — and only then — proceed to
`docs/02-rendering-scale.md`.

> Hand-off note for whoever runs this: implement one task, get its tests green, commit, then the
> next. If a conservation test drifts, the bug is in the most recently added function — do not
> paper over it by loosening tolerances.
