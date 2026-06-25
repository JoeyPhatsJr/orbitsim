"""Build a textured, day/night-shaded Earth with a fresnel atmosphere shell.

Falls back to a flat-blue lit sphere if textures or shaders are unavailable, so the
sandbox always renders."""
import os

from panda3d.core import Shader, Vec3

from orbitsim.render.geometry import make_uv_sphere
from orbitsim.render.textures import texture_path

_SHADER_DIR = os.path.join(os.path.dirname(__file__), "shaders")


def _load_shader(vert, frag):
    try:
        return Shader.load(
            Shader.SL_GLSL,
            vertex=os.path.join(_SHADER_DIR, vert),
            fragment=os.path.join(_SHADER_DIR, frag),
        )
    except Exception:
        return None


def build_earth(base):
    """Return (earth_np, atmosphere_np|None).

    Textured + day/night-shadered when the maps and shaders are available, otherwise a
    flat-blue lit sphere fallback (earth_np, None) so the scene always renders.
    """
    day = texture_path("earth_day")
    night = texture_path("earth_night")
    earth_shader = _load_shader("earth.vert", "earth.frag")

    if day is None or night is None or earth_shader is None:
        earth = make_uv_sphere(1.0, 24, 48)
        earth.set_color(0.2, 0.4, 0.9, 1.0)
        return earth, None

    earth = make_uv_sphere(1.0, 48, 96, with_uv=True)
    earth.set_shader(earth_shader)
    earth.set_shader_input("dayTex", base.loader.load_texture(day))
    earth.set_shader_input("nightTex", base.loader.load_texture(night))
    earth.set_shader_input("sunDir", Vec3(1, 0, 0))
    earth.set_light_off()  # the shader does its own lighting

    atmo = None
    atmo_shader = _load_shader("atmosphere.vert", "atmosphere.frag")
    if atmo_shader is not None:
        atmo = make_uv_sphere(1.025, 32, 64)
        atmo.set_shader(atmo_shader)
        atmo.set_shader_input("wspos_view", Vec3(0, -1000, 0))
        atmo.set_transparency(True)
        atmo.set_two_sided(True)
        atmo.set_bin("fixed", 10)
        atmo.set_depth_write(False)
        atmo.set_light_off()
    return earth, atmo


def set_sun_dir(earth_np, sun_dir_render) -> None:
    """Update the Earth shader's sun direction (no-op for the flat fallback)."""
    try:
        earth_np.set_shader_input("sunDir", Vec3(*sun_dir_render))
    except Exception:
        pass
