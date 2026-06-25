"""Star background: a textured inside-out sky sphere, or a procedural point field
when the star texture is unavailable."""
import numpy as np
from panda3d.core import (
    Filename, CullFaceAttrib, GeomVertexFormat, GeomVertexData, GeomVertexWriter,
    GeomPoints, Geom, GeomNode, NodePath,
)

from orbitsim.render.geometry import make_uv_sphere
from orbitsim.render.textures import texture_path

_SKY_RADIUS = 5000.0     # render units; depth-test is off, so this only clears the near plane
_STAR_COUNT = 3000


def random_star_dirs(n: int, seed: int = 0):
    """Return n unit direction vectors uniformly on the sphere (deterministic per seed)."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return [tuple(float(c) for c in row) for row in v]


def _background(node: NodePath) -> NodePath:
    node.set_bin("background", 0)
    node.set_depth_write(False)
    node.set_depth_test(False)
    node.set_light_off()
    return node


def _procedural_points() -> NodePath:
    rng_dirs = random_star_dirs(_STAR_COUNT, seed=42)
    bright = np.random.default_rng(42).uniform(0.4, 1.0, size=_STAR_COUNT)
    fmt = GeomVertexFormat.get_v3c4()
    vdata = GeomVertexData("stars", fmt, Geom.UHStatic)
    vdata.set_num_rows(_STAR_COUNT)
    vw = GeomVertexWriter(vdata, "vertex")
    cw = GeomVertexWriter(vdata, "color")
    for (x, y, z), b in zip(rng_dirs, bright):
        vw.add_data3(x * _SKY_RADIUS, y * _SKY_RADIUS, z * _SKY_RADIUS)
        cw.add_data4(b, b, b, 1.0)
    pts = GeomPoints(Geom.UHStatic)
    pts.add_consecutive_vertices(0, _STAR_COUNT)
    geom = Geom(vdata)
    geom.add_primitive(pts)
    gnode = GeomNode("stars")
    gnode.add_geom(geom)
    np_ = NodePath(gnode)
    np_.set_render_mode_thickness(2)
    return np_


def build_starfield(base) -> NodePath:
    """A textured inside-out sky sphere, or a procedural point field if offline."""
    path = texture_path("stars")
    if path is not None:
        sky = make_uv_sphere(1.0, 32, 64, with_uv=True)
        sky.set_scale(_SKY_RADIUS)
        sky.set_texture(base.loader.load_texture(Filename.from_os_specific(path)))
        sky.set_attrib(CullFaceAttrib.make_reverse())   # see it from the inside
        sky.set_color(1, 1, 1, 1)
        return _background(sky)
    return _background(_procedural_points())
