"""Pure quaternion attitude helpers (float64, SI). Convention: q = [w, x, y, z],
unit norm. The ship's nose is the body +Z axis rotated by the orientation quaternion."""
import math

import numpy as np

_NOSE_BODY = np.array([0.0, 0.0, 1.0])


def quat_identity() -> np.ndarray:
    """Identity rotation [1, 0, 0, 0]."""
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return q scaled to unit norm."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0.0:
        raise ValueError("cannot normalize a zero quaternion")
    return q / n


def quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Quaternion for a rotation of angle_rad about `axis` (axis need not be unit)."""
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n == 0.0:
        return quat_identity()
    axis = axis / n
    half = angle_rad / 2.0
    s = np.sin(half)
    return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a*b (apply b first, then a)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3-vector v by quaternion q."""
    w = q[0]
    u = np.asarray(q[1:], dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Angle [0, pi] between two non-zero vectors (arccos argument clamped)."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        raise ValueError("angle_between requires non-zero vectors")
    c = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return float(np.arccos(c))


def nose_direction(q: np.ndarray) -> np.ndarray:
    """Unit nose (thrust) direction = body +Z rotated by q."""
    return quat_rotate_vector(q, _NOSE_BODY)


def slew_toward(
    q: np.ndarray,
    target_dir: np.ndarray,
    max_rate_radps: float,
    dt_s: float,
) -> np.ndarray:
    """Rotate q so its nose turns toward target_dir, by at most max_rate*dt
    (no overshoot).

    Parameters
    ----------
    q : np.ndarray
        Current orientation quaternion [w, x, y, z].
    target_dir : np.ndarray
        Desired nose direction (need not be unit).
    max_rate_radps : float
        Maximum slew rate [rad/s].
    dt_s : float
        Time step [s].
    """
    nose = nose_direction(q)
    target = np.asarray(target_dir, dtype=np.float64)
    if np.linalg.norm(target) == 0.0:
        return q
    gap = angle_between(nose, target)
    if gap < 1e-12:
        return q
    step = min(gap, max_rate_radps * dt_s)
    if step <= 0.0:
        return q
    axis = np.cross(nose, target)
    if np.linalg.norm(axis) < 1e-12:
        # Parallel or antiparallel: pick any axis perpendicular to the nose.
        seed = (np.array([1.0, 0.0, 0.0]) if abs(nose[0]) < 0.9
                else np.array([0.0, 1.0, 0.0]))
        axis = np.cross(nose, seed)
    dq = quat_from_axis_angle(axis, step)
    return quat_normalize(quat_multiply(dq, q))


from orbitsim.core.state import StateVector

SAS_MODES = (
    "PROGRADE", "RETROGRADE", "NORMAL", "ANTINORMAL",
    "RADIAL_IN", "RADIAL_OUT", "TARGET", "ANTITARGET",
)


def heading_pitch(orientation_q: np.ndarray, state: StateVector) -> tuple[float, float]:
    """Return ship-nose heading and pitch [rad] relative to the local horizon."""
    r = np.asarray(state.r, dtype=np.float64)
    v = np.asarray(state.v, dtype=np.float64)
    prograde = v / np.linalg.norm(v)
    radial_out = np.cross(v, np.cross(r, v))
    radial_out = radial_out / np.linalg.norm(radial_out)
    east = np.cross(radial_out, prograde)
    nose = nose_direction(orientation_q)

    pitch = math.asin(float(np.clip(np.dot(nose, radial_out), -1.0, 1.0)))
    heading = math.atan2(float(np.dot(nose, east)), float(np.dot(nose, prograde)))
    heading %= 2.0 * math.pi
    if math.isclose(heading, 2.0 * math.pi, abs_tol=1e-12):
        heading = 0.0
    return heading, pitch


def sas_target_dir(mode, state: StateVector, target_pos=None) -> np.ndarray:
    """Unit nose direction for an SAS hold mode, from the vessel's orbital state.

    Radial-out uses the orthonormal RTN axis v × h (consistent with
    core.maneuvers), not r/|r|.
    """
    v = np.asarray(state.v, dtype=np.float64)
    v_hat = v / np.linalg.norm(v)
    h = np.cross(state.r, state.v)
    h_hat = h / np.linalg.norm(h)
    radial_out = np.cross(v, h)
    radial_out = radial_out / np.linalg.norm(radial_out)
    mode = mode.upper()
    if mode == "PROGRADE":
        return v_hat
    if mode == "RETROGRADE":
        return -v_hat
    if mode == "NORMAL":
        return h_hat
    if mode == "ANTINORMAL":
        return -h_hat
    if mode == "RADIAL_OUT":
        return radial_out
    if mode == "RADIAL_IN":
        return -radial_out
    if mode in ("TARGET", "ANTITARGET"):
        if target_pos is None:
            raise ValueError(f"{mode} requires target_pos")
        d = np.asarray(target_pos, dtype=np.float64) - state.r
        n = np.linalg.norm(d)
        if n == 0.0:
            raise ValueError("target coincides with vessel")
        d = d / n
        return d if mode == "TARGET" else -d
    raise ValueError(f"unknown SAS mode: {mode}")
