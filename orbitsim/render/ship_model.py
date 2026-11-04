"""Procedural ship model + zoom->visibility helpers for the 3rd-person ship view.

Pure helpers (view_blend, model_node_scale) import only stdlib so they are
unit-testable without Panda3D, mirroring camera_rig.py. All Panda3D imports
live INSIDE build_ship_model().
"""

import math

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


SHIP_ICON_RENDER_UNITS = 22.0   # map icon nose-to-tail length in render units (camera sits
                                # 1000 units from the focus, so this is a constant on-screen size)
SHIP_ICON_OUTLINE_SCALE = 1.18  # outline hull size relative to the icon body
SHIP_ICON_BODY_COLOR = (0.92, 0.95, 1.0, 1.0)
# Dark rim: the icon always sits on its own cyan orbit line, so a cyan rim would merge
# with it. White body pops on Earth's night side, the dark rim on the day side.
SHIP_ICON_OUTLINE_COLOR = (0.02, 0.05, 0.10, 1.0)


def _norm(v):
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return None if n < 1e-9 else (v[0] / n, v[1] / n, v[2] / n)


def _reject(v, unit):
    """v minus its component along unit (projection onto the plane normal to unit)."""
    d = v[0] * unit[0] + v[1] * unit[1] + v[2] * unit[2]
    return (v[0] - d * unit[0], v[1] - d * unit[1], v[2] - d * unit[2])


def icon_axes(nose, view_dir, up_hint):
    """Model axes (x, y, z) that show the ship icon side-on to the camera.

    The icon is a 2.5D marker: its long axis (model +Z) is the nose direction as seen
    on screen (projected off the view direction), its side (model +Y) faces the
    camera, so it never collapses to a dot when the nose points at the camera; then
    it falls back to screen-up. Returns a right-handed orthonormal basis (x = y x z).
    """
    v = _norm(view_dir) or (0.0, 1.0, 0.0)
    y = (-v[0], -v[1], -v[2])
    z = (_norm(_reject(nose, v)) or _norm(_reject(up_hint, v))
         or _norm(_reject((0.0, 0.0, 1.0), v)) or _norm(_reject((1.0, 0.0, 0.0), v)))
    x = (y[1] * z[2] - y[2] * z[1], y[2] * z[0] - y[0] * z[2], y[0] * z[1] - y[1] * z[0])
    return x, y, z


def build_ship_icon():
    """The ship model as the constant-size map icon: unlit, light body, dark rim.

    The rim is the same model scaled up, drawn first, two-sided and without writing
    depth; the body is then drawn over it, so only the rim shows around the edges.
    (The model's faces aren't consistently wound, so a back-face inverted hull would
    draw over the body.) Orient it with the vessel's attitude quaternion each frame;
    place it at the vessel's render position.
    """
    from panda3d.core import NodePath

    root = NodePath("ship_icon")
    holder = root.attach_new_node("ship_icon_geometry")
    body = build_ship_model()
    lo, hi = body.get_tight_bounds()
    centre = (lo + hi) * 0.5
    length = float(hi.z - lo.z)

    outline = build_ship_model()
    outline.set_name("outline")
    outline.reparent_to(holder)
    s = SHIP_ICON_OUTLINE_SCALE
    outline.set_scale(s)
    outline.set_pos(centre * (1.0 - s))          # scale about the model's centre
    outline.set_two_sided(True)
    outline.set_depth_write(False)               # never hides the body drawn after it
    outline.set_color(*SHIP_ICON_OUTLINE_COLOR)
    outline.set_bin("fixed", 20)

    body.set_name("body")
    body.reparent_to(holder)
    body.set_two_sided(True)
    body.set_color(*SHIP_ICON_BODY_COLOR)
    body.set_bin("fixed", 21)

    holder.set_scale(SHIP_ICON_RENDER_UNITS / length)
    root.set_light_off(1)
    # A marker, not scenery: at map zoom the icon spans ~1,900 km, so depth-testing it
    # half-buried it in Earth. Flat unlit colours make self-overlap invisible.
    root.set_depth_test(False)
    return root


def build_plume():
    """Translucent additive exhaust cone along -Z (tail). Length 1 at unit scale."""
    import math
    from panda3d.core import (
        Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat,
        GeomVertexWriter, NodePath, ColorBlendAttrib, TransparencyAttrib,
    )
    fmt = GeomVertexFormat.get_v3()
    vdata = GeomVertexData("plume", fmt, Geom.UHStatic)
    vtx = GeomVertexWriter(vdata, "vertex")
    tris = GeomTriangles(Geom.UHStatic)
    seg, r, z0, z1 = 12, 0.8, -3.0, -7.0   # base at tail (-3), apex further -Z
    apex = 0
    vtx.add_data3(0, 0, z1)
    base = []
    for j in range(seg + 1):
        a = 2.0 * math.pi * j / seg
        vtx.add_data3(math.cos(a) * r, math.sin(a) * r, z0)
        base.append(j + 1)
    for j in range(seg):
        tris.add_vertices(apex, base[j], base[j + 1])
    geom = Geom(vdata); geom.add_primitive(tris)
    node = GeomNode("plume"); node.add_geom(geom)
    np_ = NodePath(node)
    np_.set_light_off()
    np_.set_transparency(TransparencyAttrib.M_alpha)
    np_.set_attrib(ColorBlendAttrib.make(ColorBlendAttrib.M_add))
    np_.set_color(1.0, 0.6, 0.2, 0.6)
    return np_
