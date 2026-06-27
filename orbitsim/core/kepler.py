"""Kepler equation solvers and anomaly conversions."""
import numpy as np


def true_to_eccentric_anomaly(nu: float, e: float) -> float:
    """Convert true anomaly to eccentric anomaly (elliptical, e < 1).

    Parameters
    ----------
    nu : float
        True anomaly [rad].
    e : float
        Eccentricity, 0 <= e < 1.

    Returns
    -------
    float
        Eccentric anomaly E [rad], in [0, 2pi).
    """
    E = 2.0 * np.arctan2(
        np.sqrt(1.0 - e) * np.sin(nu / 2.0),
        np.sqrt(1.0 + e) * np.cos(nu / 2.0),
    )
    return E % (2.0 * np.pi)


def eccentric_to_mean_anomaly(E: float, e: float) -> float:
    """Convert eccentric anomaly to mean anomaly (Kepler's equation).

    M = E - e * sin(E)

    Parameters
    ----------
    E : float
        Eccentric anomaly [rad].
    e : float
        Eccentricity, 0 <= e < 1.

    Returns
    -------
    float
        Mean anomaly M [rad], in [0, 2pi).
    """
    M = E - e * np.sin(E)
    return M % (2.0 * np.pi)


def solve_kepler_elliptic(M: float, e: float, tol: float = 1e-12, max_iter: int = 50) -> float:
    """Solve Kepler's equation M = E - e*sin(E) for E via Newton-Raphson.

    Parameters
    ----------
    M : float
        Mean anomaly [rad].
    e : float
        Eccentricity, 0 <= e < 1.
    tol : float
        Convergence tolerance on |f(E)|.
    max_iter : int
        Maximum Newton iterations.

    Returns
    -------
    float
        Eccentric anomaly E [rad].

    Raises
    ------
    RuntimeError
        If Newton-Raphson fails to converge.
    """
    M_norm = M % (2.0 * np.pi)
    if M_norm < 1e-14:
        return 0.0
    if e < 0.8:
        E = M_norm + e * np.sin(M_norm)
    else:
        E = np.pi

    for _ in range(max_iter):
        f = E - e * np.sin(E) - M_norm
        fp = 1.0 - e * np.cos(E)
        dE = f / fp
        E -= dE
        if abs(f) < tol:
            return E

    raise RuntimeError(f"Kepler solve did not converge: M={M}, e={e}")


def mean_to_true_anomaly(M: float, e: float) -> float:
    """Convert mean anomaly to true anomaly (elliptical).

    Parameters
    ----------
    M : float
        Mean anomaly [rad].
    e : float
        Eccentricity, 0 <= e < 1.

    Returns
    -------
    float
        True anomaly [rad], in [0, 2pi).
    """
    E = solve_kepler_elliptic(M, e)
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(E / 2.0),
        np.sqrt(1.0 - e) * np.cos(E / 2.0),
    )
    return nu % (2.0 * np.pi)


def true_to_hyperbolic_anomaly(nu: float, e: float) -> float:
    """Convert true anomaly to hyperbolic anomaly F (e > 1).

    Parameters
    ----------
    nu : float
        True anomaly [rad].
    e : float
        Eccentricity, e > 1.

    Returns
    -------
    float
        Hyperbolic anomaly F [rad].
    """
    return 2.0 * np.arctanh(np.sqrt((e - 1.0) / (e + 1.0)) * np.tan(nu / 2.0))


def hyperbolic_to_mean_anomaly(F: float, e: float) -> float:
    """Convert hyperbolic anomaly to hyperbolic mean anomaly.

    M_h = e * sinh(F) - F

    Parameters
    ----------
    F : float
        Hyperbolic anomaly [rad].
    e : float
        Eccentricity, e > 1.

    Returns
    -------
    float
        Hyperbolic mean anomaly M_h.
    """
    return e * np.sinh(F) - F


def solve_kepler_hyperbolic(
    M: float, e: float, tol: float = 1e-12, max_iter: int = 50
) -> float:
    """Solve hyperbolic Kepler's equation M = e*sinh(F) - F for F via Newton-Raphson.

    Parameters
    ----------
    M : float
        Hyperbolic mean anomaly.
    e : float
        Eccentricity, e > 1.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum Newton iterations.

    Returns
    -------
    float
        Hyperbolic anomaly F.

    Raises
    ------
    RuntimeError
        If Newton-Raphson fails to converge.
    """
    F = M / (e - 1.0) if abs(M) < 1.0 else np.sign(M) * np.log(2.0 * abs(M) / e)

    for _ in range(max_iter):
        f = e * np.sinh(F) - F - M
        fp = e * np.cosh(F) - 1.0
        dF = f / fp
        F -= dF
        # Converge on the Newton step, not the absolute residual |f|. When M and e are
        # large (e.g. a ship deep in the Moon's well has e~73, M~2500 about Earth),
        # e*sinh(F) ~ M dwarfs 1.0, so |f| floors at the float64 ULP of that magnitude
        # (~1e-12) and never drops below an absolute tol — Newton has converged but the
        # |f|<tol test loops forever. |dF| reaches machine epsilon and is scale-free.
        if abs(dF) <= tol * (1.0 + abs(F)):
            return F

    raise RuntimeError(f"Hyperbolic Kepler solve did not converge: M={M}, e={e}")


def mean_to_true_anomaly_hyperbolic(M: float, e: float) -> float:
    """Convert hyperbolic mean anomaly to true anomaly (e > 1).

    Parameters
    ----------
    M : float
        Hyperbolic mean anomaly.
    e : float
        Eccentricity, e > 1.

    Returns
    -------
    float
        True anomaly [rad].
    """
    F = solve_kepler_hyperbolic(M, e)
    nu = 2.0 * np.arctan2(
        np.sqrt(e + 1.0) * np.sinh(F / 2.0),
        np.sqrt(e - 1.0) * np.cosh(F / 2.0),
    )
    return nu
