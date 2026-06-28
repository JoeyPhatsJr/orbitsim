"""Procedural geometry helpers (no external model assets)."""
import math

from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
)


def make_uv_sphere(radius: float = 1.0, num_lat: int = 24, num_lon: int = 48,
                   with_uv: bool = False) -> NodePath:
    """Build a UV sphere NodePath of the given radius (render units).

    Parameters
    ----------
    radius : float
        Sphere radius in render units.
    num_lat, num_lon : int
        Latitude/longitude subdivisions.
    with_uv : bool
        If True, include equirectangular texture coordinates (v3n3t2 format).
        If False (default), use v3n3 format (no texcoords).

    Returns
    -------
    NodePath
        A NodePath wrapping the sphere geometry, centered at the origin.
    """
    fmt = GeomVertexFormat.get_v3n3t2() if with_uv else GeomVertexFormat.get_v3n3()
    vdata = GeomVertexData("sphere", fmt, Geom.UHStatic)
    vdata.set_num_rows((num_lat + 1) * (num_lon + 1))
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    texcoord = GeomVertexWriter(vdata, "texcoord") if with_uv else None

    for i in range(num_lat + 1):
        theta = math.pi * i / num_lat
        sin_t, cos_t = math.sin(theta), math.cos(theta)
        for j in range(num_lon + 1):
            phi = 2.0 * math.pi * j / num_lon
            x = sin_t * math.cos(phi)
            y = sin_t * math.sin(phi)
            z = cos_t
            vertex.add_data3(x * radius, y * radius, z * radius)
            normal.add_data3(x, y, z)
            if texcoord is not None:
                texcoord.add_data2(j / num_lon, 1.0 - i / num_lat)

    tris = GeomTriangles(Geom.UHStatic)
    row = num_lon + 1
    for i in range(num_lat):
        for j in range(num_lon):
            a = i * row + j
            b = a + 1
            c = a + row
            d = c + 1
            tris.add_vertices(a, c, b)
            tris.add_vertices(b, c, d)

    geom = Geom(vdata)
    geom.add_primitive(tris)
    node = GeomNode("sphere")
    node.add_geom(geom)
    return NodePath(node)


def make_wireframe_sphere(n_lat: int = 9, n_lon: int = 12,
                          color=(0.55, 0.75, 1.0, 1.0), segments: int = 48) -> NodePath:
    """Unit-radius wireframe sphere (latitude rings + longitude meridians).

    Light off, alpha-transparent, depth-tested with depth-write off so it reads as a
    boundary without punching through the body or trajectory behind it.
    """
    from panda3d.core import LineSegs, TransparencyAttrib

    ls = LineSegs("wireframe_sphere")
    ls.set_color(*color)
    ls.set_thickness(1.2)

    # Latitude rings: constant z = cos(theta), radius sin(theta).
    for i in range(1, n_lat):
        theta = math.pi * i / n_lat
        z = math.cos(theta)
        r = math.sin(theta)
        for j in range(segments + 1):
            phi = 2.0 * math.pi * j / segments
            x, y = r * math.cos(phi), r * math.sin(phi)
            (ls.move_to if j == 0 else ls.draw_to)(x, y, z)

    # Longitude meridians: constant phi, theta sweeps pole to pole.
    for k in range(n_lon):
        phi = 2.0 * math.pi * k / n_lon
        for i in range(segments + 1):
            theta = math.pi * i / segments
            x = math.sin(theta) * math.cos(phi)
            y = math.sin(theta) * math.sin(phi)
            z = math.cos(theta)
            (ls.move_to if i == 0 else ls.draw_to)(x, y, z)

    node = NodePath(ls.create())
    node.set_light_off()
    node.set_transparency(TransparencyAttrib.M_alpha)
    node.set_depth_test(True)
    node.set_depth_write(False)
    return node
