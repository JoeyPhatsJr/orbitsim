"""Tests for procedural sphere geometry."""
from panda3d.core import GeomVertexReader
from orbitsim.render.geometry import make_uv_sphere


def _vdata(np_):
    return np_.node().get_geom(0).get_vertex_data()


def test_plain_sphere_has_no_texcoord():
    vd = _vdata(make_uv_sphere(1.0, 4, 8))
    assert not vd.get_format().has_column("texcoord")


def test_uv_sphere_has_texcoord_column():
    vd = _vdata(make_uv_sphere(1.0, 4, 8, with_uv=True))
    assert vd.get_format().has_column("texcoord")


def test_uv_sphere_corner_texcoords():
    # First vertex (i=0, j=0) -> u=0, v=1 ; equirectangular convention.
    vd = _vdata(make_uv_sphere(1.0, 4, 8, with_uv=True))
    r = GeomVertexReader(vd, "texcoord")
    u, v = r.get_data2()
    assert abs(u - 0.0) < 1e-6 and abs(v - 1.0) < 1e-6
