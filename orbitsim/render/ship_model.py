"""Procedural ship model + zoom->visibility helpers for the 3rd-person ship view.

Pure helpers (view_blend, model_node_scale) import only stdlib so they are
unit-testable without Panda3D, mirroring camera_rig.py. All Panda3D imports
live INSIDE build_ship_model().
"""

# Camera-distance window over which the map marker cross-fades to the ship model.
SHIP_VIEW_NEAR_M = 200.0    # at/below: ship model only
SHIP_VIEW_FAR_M = 5000.0    # at/above: map marker only


def view_blend(distance_m: float) -> tuple[float, float]:
    """Return (marker_alpha, model_alpha) for a camera-to-vessel distance [m].

    Beyond FAR the marker is fully shown; within NEAR the model is. In between
    they linearly cross-fade and sum to 1.0.
    """
    if distance_m >= SHIP_VIEW_FAR_M:
        return (1.0, 0.0)
    if distance_m <= SHIP_VIEW_NEAR_M:
        return (0.0, 1.0)
    # model_alpha = 1 at NEAR, 0 at FAR
    model_alpha = (SHIP_VIEW_FAR_M - distance_m) / (SHIP_VIEW_FAR_M - SHIP_VIEW_NEAR_M)
    return (1.0 - model_alpha, model_alpha)


def model_node_scale(scale_m_per_unit: float) -> float:
    """Node scale that renders a metres-built mesh true-size at the given zoom."""
    return 1.0 / scale_m_per_unit
