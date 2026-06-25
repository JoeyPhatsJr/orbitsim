"""Earth builder: must always return a usable node (textured or flat fallback)."""
from panda3d.core import loadPrcFileData

loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "audio-library-name null")

from direct.showbase.ShowBase import ShowBase
from orbitsim.render.earth import build_earth, set_sun_dir


def test_build_earth_returns_a_node():
    base = ShowBase()
    earth, atmo = build_earth(base)
    assert earth is not None and not earth.is_empty()
    # set_sun_dir must not raise whether or not a shader is attached.
    set_sun_dir(earth, (1.0, 0.0, 0.0))
    base.destroy()
