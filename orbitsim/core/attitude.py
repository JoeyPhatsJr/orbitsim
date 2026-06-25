"""Pure quaternion attitude helpers (float64, SI). Convention: q = [w, x, y, z],
unit norm. The ship's nose is the body +Z axis rotated by the orientation quaternion."""
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
