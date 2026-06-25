# Realistic Earth (Textured, Lit, Day/Night, Atmosphere) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat-shaded central sphere with a realistic textured Earth — real surface map, physically-correct day/night terminator with city lights, and a blue atmosphere rim — with basic textures on the solar-mode planets.

**Architecture:** A UV-emitting sphere feeds an Earth builder that applies a GLSL day/night shader (terminator driven by the real ephemeris Sun direction) plus a fresnel atmosphere shell. Textures download+cache on first run with an offline flat-color fallback, so nothing crashes without a network. All in `render/` (may import `core/`); no `core/`/`sim/` changes.

**Tech Stack:** Python 3.10, Panda3D (GLSL shaders, Texture loading), numpy, urllib, pytest.

## Global Constraints

- All new code lives in `render/`. `render/` may import `core/` (e.g. ephemeris) but NOT `sim` internals beyond what `app.py` already uses. `core/`/`sim/` must NOT change.
- The app must NEVER crash for a missing/failed texture — always fall back to the current flat-color sphere.
- Texture cache dir: `data/textures/` (gitignored). Downloads use a `User-Agent` header, a timeout, atomic write (`.part` then rename), and image-magic validation (JPEG `FF D8`, PNG `89 50`).
- **Verified texture URLs (use exactly these):**
  - Earth day: `https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/earth_atmos_2048.jpg`
  - Earth night lights: `https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/earth_lights_2048.png`
- Equirectangular UV convention: `u = lon_index / num_lon`, `v = 1 - lat_index / num_lat`.
- `black` is NOT installed; clean code at line length ≤ 100, skip formatting.
- Always use `.venv/Scripts/python`. Full suite: `.venv/Scripts/python -m pytest tests/ -q`.
- Commit after each task with the exact message given. Use ONLY `git add <specific files>` — NEVER `git add -A`. Never stage: `data/`, `debug_curtis.py`, `kickbacks.vsix`, `.hypothesis/`, `CLAUDE.md`, `porkchop.png`.
- Render tasks end with a HUMAN VISUAL CHECKPOINT; verify headlessly first with `loadPrcFileData("", "window-type offscreen")` + `app.taskMgr.step()` + offscreen screenshots.

## Gate

Continuous-thrust flight model complete (123 tests green). No dependency on it; this is graphics-only.

## Existing API available

```python
from orbitsim.render.geometry import make_uv_sphere            # currently v3n3, no UVs
from orbitsim.core.ephemeris import body_state                 # body_state("SUN", t, center="EARTH").r
from panda3d.core import GeomVertexFormat, GeomVertexWriter, Texture, Shader, DirectionalLight, Vec3, Vec4
# app.py builds self.central_np = make_uv_sphere(1.0, 24, 48) in _start_sim (sandbox branch).
```

---

## Task 1: UV texture coordinates on the sphere

**Files:**
- Modify: `orbitsim/render/geometry.py`
- Test: `tests/render/test_geometry.py` (create if absent)

**Interfaces:**
- Produces: `make_uv_sphere(radius=1.0, num_lat=24, num_lon=48, with_uv=False) -> NodePath`.
  When `with_uv=True`, geometry uses format `v3n3t2` and writes equirectangular texcoords.

- [ ] **Step 1: Write the failing test**

Create/append `tests/render/test_geometry.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/render/test_geometry.py -q`
Expected: FAIL (`test_uv_sphere_has_texcoord_column` — plain format has no texcoord; the `with_uv` kwarg does not exist yet → TypeError).

- [ ] **Step 3: Add the with_uv path**

In `orbitsim/render/geometry.py`, change the signature and add texcoord writing:
```python
def make_uv_sphere(radius: float = 1.0, num_lat: int = 24, num_lon: int = 48,
                   with_uv: bool = False) -> NodePath:
```
Replace the format/data/writer setup:
```python
    fmt = GeomVertexFormat.get_v3n3t2() if with_uv else GeomVertexFormat.get_v3n3()
    vdata = GeomVertexData("sphere", fmt, Geom.UHStatic)
    vdata.set_num_rows((num_lat + 1) * (num_lon + 1))
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    texcoord = GeomVertexWriter(vdata, "texcoord") if with_uv else None
```
Inside the `for j` loop, after writing vertex+normal, add:
```python
            if texcoord is not None:
                texcoord.add_data2(j / num_lon, 1.0 - i / num_lat)
```
(Keep the rest of the function unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/render/test_geometry.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/render/geometry.py tests/render/test_geometry.py
git commit -m "Earth Task 1: UV texcoords on the procedural sphere"
```

---

## Task 2: Texture download + cache with offline fallback

**Files:**
- Create: `orbitsim/render/textures.py`
- Test: `tests/render/test_textures.py`

**Interfaces:**
- Produces:
  ```python
  TEXTURE_URLS: dict[str, str]                  # "earth_day", "earth_night" -> URL
  def texture_path(name: str, cache_dir: str | None = None) -> str | None
      # cached local path; downloads on first call; None if unavailable/invalid
  ```

- [ ] **Step 1: Write the failing tests**

Create `tests/render/test_textures.py`:
```python
"""Tests for the texture download/cache (offline-safe)."""
import os
from orbitsim.render import textures


def test_registry_has_earth_keys():
    assert "earth_day" in textures.TEXTURE_URLS
    assert "earth_night" in textures.TEXTURE_URLS


def test_cache_hit_returns_existing_valid_file(tmp_path):
    # Pre-place a file with a valid JPEG magic; no network should be needed.
    p = tmp_path / "earth_day.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
    got = textures.texture_path("earth_day", cache_dir=str(tmp_path))
    assert got == str(p)


def test_unknown_name_returns_none(tmp_path):
    assert textures.texture_path("not_a_planet", cache_dir=str(tmp_path)) is None


def test_bad_magic_is_rejected(tmp_path, monkeypatch):
    # Simulate a download that returns an HTML CAPTCHA page (no image magic).
    def fake_fetch(url, timeout=30):
        return b"<html>captcha</html>"
    monkeypatch.setattr(textures, "_fetch", fake_fetch)
    assert textures.texture_path("earth_day", cache_dir=str(tmp_path)) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/render/test_textures.py -q`
Expected: FAIL (ModuleNotFoundError: orbitsim.render.textures).

- [ ] **Step 3: Implement textures.py**

Create `orbitsim/render/textures.py`:
```python
"""Download + cache real surface texture maps; offline-safe (flat-color fallback
is the caller's job when this returns None)."""
import os
import urllib.request

_BASE = "https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/"
TEXTURE_URLS = {
    "earth_day": _BASE + "earth_atmos_2048.jpg",
    "earth_night": _BASE + "earth_lights_2048.png",
}

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "textures"
)
# Image magic numbers: JPEG starts FF D8, PNG starts 89 50 4E 47.
_MAGICS = (b"\xff\xd8", b"\x89PNG")


def _ext_for(url: str) -> str:
    return ".png" if url.lower().endswith(".png") else ".jpg"


def _looks_like_image(data: bytes) -> bool:
    return any(data.startswith(m) for m in _MAGICS)


def _fetch(url: str, timeout: int = 30) -> bytes:
    """Download raw bytes with a browser-like User-Agent."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=timeout).read()


def texture_path(name: str, cache_dir: str | None = None):
    """Return a cached local path for `name`, downloading on first use.

    Returns None if the name is unknown, the download fails, or the bytes are not
    a valid image (e.g. a CAPTCHA HTML page). The caller falls back to a flat color.
    """
    url = TEXTURE_URLS.get(name)
    if url is None:
        return None
    cache_dir = cache_dir or _DATA_DIR
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{name}{_ext_for(url)}")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    try:
        data = _fetch(url)
    except Exception:
        return None
    if not _looks_like_image(data):
        return None
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/render/test_textures.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Real-download smoke check (manual, not a unit test)**

Run:
```bash
.venv/Scripts/python -c "from orbitsim.render.textures import texture_path; print(texture_path('earth_day')); print(texture_path('earth_night'))"
```
Expected: two real paths under `data/textures/` (downloads ~512 KB + ~735 KB once). If offline, prints `None None` — acceptable (fallback path).

- [ ] **Step 6: Commit**

```bash
git add orbitsim/render/textures.py tests/render/test_textures.py
git commit -m "Earth Task 2: texture download+cache with offline fallback"
```

---

## Task 3: Earth GLSL shader (day/night + atmosphere)

**Files:**
- Create: `orbitsim/render/shaders/earth.vert`
- Create: `orbitsim/render/shaders/earth.frag`
- Create: `orbitsim/render/shaders/atmosphere.vert`
- Create: `orbitsim/render/shaders/atmosphere.frag`

**Interfaces:**
- Produces: GLSL shader files consumed by Task 4. Uniforms: `sunDir` (vec3, world space),
  `dayTex`/`nightTex` (sampler2D) for Earth; `sunDir` for atmosphere.

- [ ] **Step 1: Write the Earth shaders**

Create `orbitsim/render/shaders/earth.vert`:
```glsl
#version 120
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
attribute vec4 p3d_Vertex;
attribute vec3 p3d_Normal;
attribute vec2 p3d_MultiTexCoord0;
varying vec2 uv;
varying vec3 worldNormal;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    uv = p3d_MultiTexCoord0;
    worldNormal = normalize(mat3(p3d_ModelMatrix) * p3d_Normal);
}
```
Create `orbitsim/render/shaders/earth.frag`:
```glsl
#version 120
uniform sampler2D dayTex;
uniform sampler2D nightTex;
uniform vec3 sunDir;
varying vec2 uv;
varying vec3 worldNormal;
void main() {
    float lit = dot(normalize(worldNormal), normalize(sunDir));
    float f = smoothstep(-0.1, 0.1, lit);          // 0 = night, 1 = day
    vec3 day = texture2D(dayTex, uv).rgb * clamp(lit, 0.05, 1.0);
    vec3 night = texture2D(nightTex, uv).rgb;      // city lights
    gl_FragColor = vec4(mix(night, day, f), 1.0);
}
```

- [ ] **Step 2: Write the atmosphere shaders**

Create `orbitsim/render/shaders/atmosphere.vert`:
```glsl
#version 120
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
uniform vec3 wspos_view;        // camera world position (set from Python)
attribute vec4 p3d_Vertex;
attribute vec3 p3d_Normal;
varying vec3 worldNormal;
varying vec3 viewDir;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    vec3 wpos = (p3d_ModelMatrix * p3d_Vertex).xyz;
    worldNormal = normalize(mat3(p3d_ModelMatrix) * p3d_Normal);
    viewDir = normalize(wspos_view - wpos);
}
```
Create `orbitsim/render/shaders/atmosphere.frag`:
```glsl
#version 120
varying vec3 worldNormal;
varying vec3 viewDir;
void main() {
    float rim = 1.0 - max(dot(normalize(worldNormal), normalize(viewDir)), 0.0);
    float a = pow(rim, 3.0);                        // bright at the limb
    gl_FragColor = vec4(0.3, 0.6, 1.0, a * 0.9);    // sky-blue halo
}
```

- [ ] **Step 3: Verify the shaders compile**

Run:
```bash
.venv/Scripts/python -c "
from panda3d.core import loadPrcFileData
loadPrcFileData('', 'window-type offscreen')
from direct.showbase.ShowBase import ShowBase
from panda3d.core import Shader
b = ShowBase()
s = Shader.load(Shader.SL_GLSL, vertex='orbitsim/render/shaders/earth.vert', fragment='orbitsim/render/shaders/earth.frag')
a = Shader.load(Shader.SL_GLSL, vertex='orbitsim/render/shaders/atmosphere.vert', fragment='orbitsim/render/shaders/atmosphere.frag')
print('earth shader ok:', s is not None and not s.is_error() if hasattr(s,'is_error') else s is not None)
print('atmo shader ok:', a is not None)
b.destroy()
"
```
Expected: both print `ok: True` (no GLSL compile errors logged).

- [ ] **Step 4: Commit**

```bash
git add orbitsim/render/shaders/earth.vert orbitsim/render/shaders/earth.frag orbitsim/render/shaders/atmosphere.vert orbitsim/render/shaders/atmosphere.frag
git commit -m "Earth Task 3: GLSL day/night + atmosphere shaders"
```

---

## Task 4: Earth builder (`render/earth.py`)

**Files:**
- Create: `orbitsim/render/earth.py`
- Test: `tests/render/test_earth.py`

**Interfaces:**
- Consumes: `make_uv_sphere(with_uv=True)`, `texture_path`, the Task 3 shaders.
- Produces:
  ```python
  def build_earth(base) -> tuple[NodePath, NodePath | None]
      # returns (earth_np, atmosphere_np or None). Falls back to a flat-blue lit
      # sphere (earth_np, None) if textures/shaders are unavailable.
  def set_sun_dir(earth_np, sun_dir_render) -> None   # update the shader uniform
  ```

- [ ] **Step 1: Write the failing test**

Create `tests/render/test_earth.py`:
```python
"""Earth builder: must always return a usable node (textured or flat fallback)."""
from panda3d.core import loadPrcFileData

loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "audio-library-name null")

from direct.showbase.ShowBase import ShowBase
from orbitsim.render.earth import build_earth, set_sun_dir


def test_build_earth_returns_a_node(tmp_path):
    base = ShowBase()
    earth, atmo = build_earth(base)
    assert earth is not None and not earth.is_empty()
    # set_sun_dir must not raise whether or not a shader is attached.
    set_sun_dir(earth, (1.0, 0.0, 0.0))
    base.destroy()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/render/test_earth.py -q`
Expected: FAIL (ModuleNotFoundError: orbitsim.render.earth).

- [ ] **Step 3: Implement earth.py**

Create `orbitsim/render/earth.py`:
```python
"""Build a textured, day/night-shaded Earth with a fresnel atmosphere shell.

Falls back to a flat-blue lit sphere if textures or shaders are unavailable, so the
sandbox always renders."""
import os

from panda3d.core import Shader, Texture, Vec3

from orbitsim.render.geometry import make_uv_sphere
from orbitsim.render.textures import texture_path

_SHADER_DIR = os.path.join(os.path.dirname(__file__), "shaders")


def _load_shader(vert, frag):
    try:
        s = Shader.load(
            Shader.SL_GLSL,
            vertex=os.path.join(_SHADER_DIR, vert),
            fragment=os.path.join(_SHADER_DIR, frag),
        )
        return s
    except Exception:
        return None


def build_earth(base):
    """Return (earth_np, atmosphere_np|None). Textured+shadered when possible,
    else a flat-blue lit sphere fallback."""
    day = texture_path("earth_day")
    night = texture_path("earth_night")
    earth_shader = _load_shader("earth.vert", "earth.frag")

    if day is None or night is None or earth_shader is None:
        # Fallback: plain lit blue sphere (current look).
        earth = make_uv_sphere(1.0, 24, 48)
        earth.set_color(0.2, 0.4, 0.9, 1.0)
        return earth, None

    earth = make_uv_sphere(1.0, 48, 96, with_uv=True)
    day_tex = base.loader.load_texture(day)
    night_tex = base.loader.load_texture(night)
    earth.set_shader(earth_shader)
    earth.set_shader_input("dayTex", day_tex)
    earth.set_shader_input("nightTex", night_tex)
    earth.set_shader_input("sunDir", Vec3(1, 0, 0))
    earth.set_light_off()  # shader does its own lighting

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/render/test_earth.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add orbitsim/render/earth.py tests/render/test_earth.py
git commit -m "Earth Task 4: Earth builder with textured/flat fallback"
```

---

## Task 5: Use the Earth in the sandbox + sun direction — HUMAN VISUAL CHECKPOINT

**Files:**
- Modify: `orbitsim/render/app.py`

**Interfaces:**
- Consumes: `build_earth`, `set_sun_dir`, `ephemeris.body_state`.
- Produces: the sandbox central body is the textured Earth, lit/terminated by the real
  Sun direction each frame.

- [ ] **Step 1: Build the Earth instead of the flat sphere (sandbox)**

In `orbitsim/render/app.py`, add imports near the top:
```python
from orbitsim.render.earth import build_earth, set_sun_dir
```
In `_start_sim`, in the sandbox (`else`/non-solar) branch, replace the central-body
sphere creation
```python
        self.central_np = make_uv_sphere(1.0, 24, 48)
        self.central_np.reparent_to(self.render)
        self.central_np.set_color(0.2, 0.4, 0.9, 1.0)
```
with:
```python
        if not self.solar_system:
            self.central_np, self._atmo_np = build_earth(self)
            self.central_np.reparent_to(self.render)
            if self._atmo_np is not None:
                self._atmo_np.reparent_to(self.central_np)
        else:
            self.central_np = make_uv_sphere(1.0, 24, 48)
            self.central_np.reparent_to(self.render)
            self.central_np.set_color(1.0, 0.85, 0.2, 1.0)
            self.central_np.set_light_off()
            self.central_np.set_scale(10.0)
```
(Adjust to fit the existing solar/sandbox split; keep the solar Sun styling already present. Ensure `self._atmo_np = None` is set in the solar branch.)

- [ ] **Step 2: Drive the sun direction each frame (sandbox `_update`)**

In the sandbox `_update`, after the central body is positioned/scaled, add:
```python
        # Real Sun direction (Earth->Sun) for the day/night terminator + light.
        from orbitsim.core.ephemeris import body_state
        try:
            sun_r = body_state("SUN", self.clock.sim_time_s, center="EARTH").r
            sun_render = self.transform.to_render(sun_r) - self.transform.to_render(np.zeros(3))
            sun_dir = np.asarray(sun_render, dtype=float)
            n = np.linalg.norm(sun_dir)
            if n > 0:
                sun_dir = sun_dir / n
                set_sun_dir(self.central_np, tuple(sun_dir))
                self._sun_light_np.set_pos(tuple(sun_dir * 1000.0))
                self._sun_light_np.look_at(0, 0, 0)
        except Exception:
            pass
```

- [ ] **Step 3: Add a sun directional light (sandbox `_start_sim`)**

In the sandbox branch of `_start_sim`, after building Earth, add a directional light and
keep a low ambient:
```python
            from panda3d.core import DirectionalLight, AmbientLight, Vec4
            amb = AmbientLight("amb")
            amb.set_color(Vec4(0.08, 0.08, 0.1, 1))
            self.render.set_light(self.render.attach_new_node(amb))
            sun = DirectionalLight("sun")
            sun.set_color(Vec4(1.0, 1.0, 0.95, 1))
            self._sun_light_np = self.render.attach_new_node(sun)
            self.render.set_light(self._sun_light_np)
```
(If the existing lighting block already runs for both modes, leave the solar path as-is
and only add the sun-light node for the sandbox so `_update` can aim it.)

- [ ] **Step 4: Headless smoke + screenshot**

Create `tmp_earth_check.py`:
```python
from panda3d.core import loadPrcFileData, Filename
loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "audio-library-name null")
loadPrcFileData("", "win-size 900 700")
from orbitsim.__main__ import _default_world
from orbitsim.sim.clock import SimClock
from orbitsim.render.app import OrbitApp
app = OrbitApp(_default_world(), SimClock(0.0, 1.0))
app._on_play()
for _ in range(5):
    app.taskMgr.step()
app.win.save_screenshot(Filename.from_os_specific("tmp_earth.png"))
app.destroy()
print("OK")
```
Run: `PYTHONPATH=. .venv/Scripts/python tmp_earth_check.py`, open `tmp_earth.png`. Confirm a textured Earth with a day/night terminator and a blue rim (or, if offline, the flat-blue fallback — still no crash). Delete temp files.

- [ ] **Step 5: Full suite green**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 6: HUMAN VISUAL CHECKPOINT**

Run `.venv/Scripts/python -m orbitsim`, Play. Reviewer confirms: a realistic textured
Earth, a clear day/night terminator (city lights on the dark side), a blue atmosphere
limb; advancing time-warp sweeps the terminator across the surface.

- [ ] **Step 7: Commit**

```bash
git add orbitsim/render/app.py
git commit -m "Earth Task 5: textured Earth in sandbox with real-sun terminator"
```

---

## Task 6: Basic textures on solar-mode planets

**Files:**
- Modify: `orbitsim/render/app.py`
- Modify: `orbitsim/render/textures.py` (add Earth-in-solar reuse; no new URLs required)

**Interfaces:**
- Consumes: `make_uv_sphere(with_uv=True)`, `texture_path`.
- Produces: solar-mode planet markers use a textured sphere with the Earth day map for
  Earth; others keep their flat colors (best-effort).

- [ ] **Step 1: Texture the Earth marker in solar mode**

In `orbitsim/render/app.py::_build_planets`, when building each marker, use a UV sphere
and apply the day map for Earth:
```python
            marker = make_uv_sphere(1.0, 12, 16, with_uv=(body.name == "Earth"))
            ...
            if body.name == "Earth":
                from orbitsim.render.textures import texture_path
                p = texture_path("earth_day")
                if p is not None:
                    marker.set_texture(self.loader.load_texture(p))
                    marker.set_color(1, 1, 1, 1)
```
(Leave the other planets on their existing flat colors — best-effort per scope.)

- [ ] **Step 2: Smoke-check imports**

Run: `.venv/Scripts/python -c "import orbitsim.render.app; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Headless solar screenshot**

Reuse the solar headless check (offscreen, `--solar` world, `_on_play`, step, screenshot);
confirm Earth's marker shows surface texture (or stays flat if offline). No crash.

- [ ] **Step 4: Full suite green**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 5: HUMAN VISUAL CHECKPOINT**

Run `.venv/Scripts/python -m orbitsim --solar`; Earth's dot shows a surface texture.

- [ ] **Step 6: Commit**

```bash
git add orbitsim/render/app.py orbitsim/render/textures.py
git commit -m "Earth Task 6: textured Earth marker in solar mode"
```

---

## Exit Criteria

- UV-textured sphere with correct equirectangular texcoords (unit-tested).
- Texture download/cache works with image-magic validation and returns `None` (flat
  fallback) when offline/invalid — never crashes (unit-tested).
- Sandbox Earth is textured with a real-sun day/night terminator, city lights, and an
  atmosphere rim; terminator sweeps under time-warp (visual checkpoint).
- Solar-mode Earth marker is textured.
- `pytest tests/ -q` fully green; no `core/`/`sim/` changes.

## Self-Review Notes

- Spec coverage: UV geometry (Task 1), download/cache+fallback (Task 2), shaders (Task 3),
  Earth builder+fallback (Task 4), sandbox integration + ephemeris sun + light (Task 5),
  solar planets (Task 6). All mapped.
- Offline-safety is enforced at every layer: `texture_path` returns `None`, `build_earth`
  falls back to the flat sphere, `set_sun_dir` is a no-op without a shader, and the
  `_update` sun block is wrapped in try/except.
- Unit-testable pieces (geometry UVs, texture cache) are real pytest tests and stay
  offline (monkeypatched `_fetch`); shaders/visuals are checkpoint-verified per project
  convention.
- Type consistency: `make_uv_sphere(..., with_uv)`, `texture_path(name, cache_dir)`,
  `build_earth(base) -> (np, np|None)`, `set_sun_dir(np, dir)` used identically across tasks.
```
