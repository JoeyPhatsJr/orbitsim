"""Tests for the floating-origin RenderTransform precision guarantee."""
import numpy as np
from orbitsim.render.floating_origin import RenderTransform


def test_origin_maps_to_zero():
    origin = np.array([1.0e11, 2.0e11, -3.0e11])
    rt = RenderTransform(origin_m=origin, scale_m_per_unit=1.0e6)
    out = rt.to_render(origin)
    assert out == (0.0, 0.0, 0.0)


def test_precision_preserved_across_float32_cast():
    """A 1 mm offset from a point 1e11 m away must survive the float32 cast.

    This is the whole reason the renderer works: subtract the float64 origin
    BEFORE casting to float32, so 1e-3 m local detail is not lost in 1e11.
    """
    far = np.array([1.0e11, 0.0, 0.0])
    offset = np.array([1.0e-3, 0.0, 0.0])
    point = far + offset
    scale = 1.0e-3  # 1 render unit == 1 mm, so the offset maps to ~1.0 render units
    rt = RenderTransform(origin_m=far, scale_m_per_unit=scale)
    rx, ry, rz = rt.to_render(point)
    implied_distance_m = rx * scale

    # The floating-origin guarantee is that the float32 cast adds NOTHING beyond
    # float64's own limit. float64 alone can only resolve ~1.5e-5 m (its ULP) at
    # 1e11, so we compare the float32 render result against the pure-float64
    # subtraction: they must agree to far within that ULP.
    float64_ref = float((point - far)[0])
    assert abs(implied_distance_m - float64_ref) < 1e-9  # float32 cast added ~nothing
    # ...and the 1 mm offset clearly survived (did not collapse toward 0):
    assert abs(implied_distance_m - 1.0e-3) < 1e-4


def test_scale_divides():
    rt = RenderTransform(origin_m=np.zeros(3), scale_m_per_unit=1000.0)
    out = rt.to_render(np.array([2000.0, 0.0, 0.0]))
    assert abs(out[0] - 2.0) < 1e-9


def test_set_origin_updates_mapping():
    rt = RenderTransform(origin_m=np.zeros(3), scale_m_per_unit=1.0)
    rt.set_origin(np.array([5.0, 0.0, 0.0]))
    out = rt.to_render(np.array([5.0, 0.0, 0.0]))
    assert out == (0.0, 0.0, 0.0)
