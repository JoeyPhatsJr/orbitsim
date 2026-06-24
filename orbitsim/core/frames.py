"""Frame rotations (perifocal ↔ inertial) and rotation matrix helpers."""
import numpy as np


def rotation_matrix_3(theta: float) -> np.ndarray:
    """Rotation matrix about the z-axis (active rotation).

    Parameters
    ----------
    theta : float
        Rotation angle [rad].

    Returns
    -------
    np.ndarray
        3x3 rotation matrix for rotation by θ about z-axis.

    Notes
    -----
    R3(θ) = [cos(θ)  -sin(θ)  0]
             [sin(θ)   cos(θ)  0]
             [0        0       1]
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rotation_matrix_1(theta: float) -> np.ndarray:
    """Rotation matrix about the x-axis (active rotation).

    Parameters
    ----------
    theta : float
        Rotation angle [rad].

    Returns
    -------
    np.ndarray
        3x3 rotation matrix for rotation by θ about x-axis.

    Notes
    -----
    R1(θ) = [1  0         0      ]
             [0  cos(θ)  -sin(θ)]
             [0  sin(θ)   cos(θ)]
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float64,
    )


def perifocal_to_inertial(
    r_pqw: np.ndarray,
    v_pqw: np.ndarray,
    raan: float,
    inclination: float,
    argp: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate position & velocity from perifocal (PQW) to inertial frame.

    Parameters
    ----------
    r_pqw : np.ndarray
        Position in perifocal frame [m], shape (3,).
    v_pqw : np.ndarray
        Velocity in perifocal frame [m/s], shape (3,).
    raan : float
        Right ascension of ascending node Ω [rad].
    inclination : float
        Inclination i [rad].
    argp : float
        Argument of periapsis ω [rad].

    Returns
    -------
    r_inertial : np.ndarray
        Position in inertial frame [m], shape (3,).
    v_inertial : np.ndarray
        Velocity in inertial frame [m/s], shape (3,).

    Notes
    -----
    Rotation matrix: Q = R3(Ω) @ R1(i) @ R3(ω) (active rotation)
    """
    # Q = R3(Ω) @ R1(i) @ R3(ω) (active rotation, perifocal → inertial)
    Q = rotation_matrix_3(raan) @ rotation_matrix_1(inclination) @ rotation_matrix_3(argp)

    r_inertial = Q @ r_pqw
    v_inertial = Q @ v_pqw

    return np.asarray(r_inertial, dtype=np.float64), np.asarray(
        v_inertial, dtype=np.float64
    )


def inertial_to_perifocal(
    r_inertial: np.ndarray,
    v_inertial: np.ndarray,
    raan: float,
    inclination: float,
    argp: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate position & velocity from inertial to perifocal (PQW) frame.

    Parameters
    ----------
    r_inertial : np.ndarray
        Position in inertial frame [m], shape (3,).
    v_inertial : np.ndarray
        Velocity in inertial frame [m/s], shape (3,).
    raan : float
        Right ascension of ascending node Ω [rad].
    inclination : float
        Inclination i [rad].
    argp : float
        Argument of periapsis ω [rad].

    Returns
    -------
    r_pqw : np.ndarray
        Position in perifocal frame [m], shape (3,).
    v_pqw : np.ndarray
        Velocity in perifocal frame [m/s], shape (3,).

    Notes
    -----
    Inverse of perifocal_to_inertial. Q^T = R3(-ω) @ R1(-i) @ R3(-Ω).
    """
    Q_inv = rotation_matrix_3(-argp) @ rotation_matrix_1(-inclination) @ rotation_matrix_3(-raan)

    r_pqw = Q_inv @ r_inertial
    v_pqw = Q_inv @ v_inertial

    return np.asarray(r_pqw, dtype=np.float64), np.asarray(
        v_pqw, dtype=np.float64
    )
