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


def build_ship_model():
    """Build a stylized ship NodePath (metres, nose +Z, lit by scene lights)."""
    import math
    from panda3d.core import (
        Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat,
        GeomVertexWriter, NodePath,
    )

    fmt = GeomVertexFormat.get_v3n3()
    vdata = GeomVertexData("ship", fmt, Geom.UHStatic)
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    tris = GeomTriangles(Geom.UHStatic)
    n_row = [0]  # next free vertex index (list for closure mutation)

    def add(px, py, pz, nx, ny, nz):
        vertex.add_data3(px, py, pz)
        normal.add_data3(nx, ny, nz)
        i = n_row[0]
        n_row[0] += 1
        return i

    seg = 16
    body_r = 1.0
    body_z0, body_z1 = -3.0, 3.0   # cylinder body
    nose_z = 6.0                   # cone apex (nose tip, +Z)

    # --- cylinder body (side wall) ---
    ring0, ring1 = [], []
    for j in range(seg + 1):
        a = 2.0 * math.pi * j / seg
        cx, cy = math.cos(a), math.sin(a)
        ring0.append(add(cx * body_r, cy * body_r, body_z0, cx, cy, 0.0))
        ring1.append(add(cx * body_r, cy * body_r, body_z1, cx, cy, 0.0))
    for j in range(seg):
        a0, b0 = ring0[j], ring0[j + 1]
        a1, b1 = ring1[j], ring1[j + 1]
        tris.add_vertices(a0, a1, b0)
        tris.add_vertices(b0, a1, b1)

    # --- nose cone (body_z1 -> nose_z apex at +Z) ---
    apex = add(0.0, 0.0, nose_z, 0.0, 0.0, 1.0)
    base = []
    for j in range(seg + 1):
        a = 2.0 * math.pi * j / seg
        cx, cy = math.cos(a), math.sin(a)
        # outward+up normal approximation for the cone wall
        base.append(add(cx * body_r, cy * body_r, body_z1, cx * 0.7, cy * 0.7, 0.7))
    for j in range(seg):
        tris.add_vertices(base[j], base[j + 1], apex)

    # --- 3 fins at the tail (z = body_z0), so roll is visible ---
    fin_h = 1.6   # how far the fin sticks out radially
    fin_z0, fin_z1 = body_z0, body_z0 + 2.0
    for k in range(3):
        a = 2.0 * math.pi * k / 3
        cx, cy = math.cos(a), math.sin(a)
        nx, ny = -math.sin(a), math.cos(a)  # fin face normal (tangent dir)
        root_lo = add(cx * body_r, cy * body_r, fin_z0, nx, ny, 0.0)
        root_hi = add(cx * body_r, cy * body_r, fin_z1, nx, ny, 0.0)
        tip = add(cx * (body_r + fin_h), cy * (body_r + fin_h), fin_z0, nx, ny, 0.0)
        tris.add_vertices(root_lo, root_hi, tip)
        tris.add_vertices(tip, root_hi, root_lo)  # back face (double-sided)

    geom = Geom(vdata)
    geom.add_primitive(tris)
    node = GeomNode("ship")
    node.add_geom(geom)
    np_ = NodePath(node)
    np_.set_color(0.85, 0.87, 0.9, 1.0)
    return np_
