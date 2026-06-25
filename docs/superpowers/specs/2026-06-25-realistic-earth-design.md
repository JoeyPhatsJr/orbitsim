# Realistic Celestial Bodies (Earth-focused) — Design

**Status:** approved (design phase). Next: implementation plan via writing-plans.
**Date:** 2026-06-25

## Goal

Replace the flat-shaded sphere look with a realistic, textured, lit Earth: a real
surface map, a physically-correct day/night terminator with city lights on the night
side, and a soft blue atmosphere rim. Solar-mode planets get basic surface textures.
This is the first graphics sub-project; vessel/trajectory visuals and full HUD/navball
polish are separate later cycles.

## Decisions (locked during brainstorming)

1. **Focus:** Earth gets the full treatment (it fills the sandbox view); solar-mode
   planets get basic flat textures.
2. **Texture source:** download public-domain maps on first run, cache in
   `data/textures/` (same pattern as the DE440 kernel); graceful flat-color fallback if
   offline. **Verified source:** the three.js example textures on GitHub raw
   (NASA-derived, public domain), which download cleanly in this environment:
   - Day: `https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/earth_atmos_2048.jpg` (≈512 KB JPEG)
   - Night lights: `https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/earth_lights_2048.png` (≈735 KB PNG)
   (solarsystemscope.com is CAPTCHA-walled and unusable for direct download.)
3. **Day/night + city lights:** a custom GLSL shader on the Earth sphere.
4. **Atmosphere rim:** a back-rendered transparent fresnel shell.
5. **Sun direction:** the real Earth→Sun unit vector from the ephemeris at the current
   sim time, so the terminator is physically correct and sweeps under time-warp.

## Architecture (respects the project's layering)

```
render/geometry.py   MODIFIED. Add a UV-emitting textured sphere (format v3n3t2).
render/textures.py   NEW. Download+cache texture maps; offline flat-color fallback.
render/earth.py      NEW. Build the textured, shadered Earth + atmosphere shell;
                     expose update(sun_dir) to drive the terminator each frame.
render/shaders/      NEW. earth.vert / earth.frag (GLSL): day/night blend + rim.
render/app.py        MODIFIED. Use the Earth builder for the sandbox central body;
                     compute sun_dir from ephemeris each frame; sun light; texture
                     the solar-mode planet markers.
data/textures/       gitignored cache (add to .gitignore).
```

`render/` may import `core/` (ephemeris for the sun vector) — that's allowed by the
layering rule. No graphics code leaks into `core/`/`sim/`.

## 1. Textured sphere geometry (`render/geometry.py`)

Add `make_uv_sphere(..., with_uv: bool = False)` (or a sibling `make_textured_sphere`).
When `with_uv`, use `GeomVertexFormat.get_v3n3t2()` and write per-vertex texcoords from
the existing lat/lon parametrization: `u = j / num_lon`, `v = 1.0 - i / num_lat`
(equirectangular, matching the NASA map projection). Existing flat-color callers keep
the `v3n3` path unchanged (default `with_uv=False`).

**Test (real unit test):** build a textured sphere; assert the vertex data has a
`texcoord` column and that a known vertex's `(u, v)` equals the expected lat/lon
fraction.

## 2. Texture download + cache (`render/textures.py`)

```python
def texture_path(name: str) -> str | None
    # returns a cached local path, downloading on first call; None if unavailable
```

- `name` keys a small registry mapping to the verified URLs (§Decisions).
- Cache dir: `data/textures/` (created on demand; gitignored).
- Download with `urllib` + a `User-Agent` header, a timeout, and atomic write
  (download to `.part`, then rename) so a partial download never poisons the cache.
- **Validate** the downloaded bytes look like an image (JPEG `FF D8` / PNG `89 50`
  magic); if not (e.g. a CAPTCHA HTML page), discard and return `None`.
- On any failure (offline, bad bytes, HTTP error) return `None` — callers fall back to
  a flat color. The app must never crash for lack of a texture.

**Tests:** (a) a cache *hit* (pre-place a fake valid-magic file) returns it without
network; (b) a bad-magic payload is rejected and returns `None`; (c) the registry
contains the Earth day/night keys. Network downloads are not asserted in unit tests
(they are exercised once at runtime / a manually-run check), keeping the suite offline.

## 3. Earth rendering (`render/earth.py` + `render/shaders/earth.*`)

`build_earth(radius_render_units) -> (NodePath, atmosphere NodePath)` and an
`update_sun(node, sun_dir_render)` helper.

- **Body:** textured UV sphere with the day map; the GLSL fragment shader samples both
  the day and night textures and blends them by the terminator:
  `f = smoothstep(-0.1, 0.1, dot(N, sunDir)); color = mix(nightTex, dayTex, f)`.
  The night texture is shown only where unlit, giving city lights on the dark side.
  Uniform `sunDir` is updated each frame.
- **Atmosphere shell:** a sphere ~2.5% larger, rendered with reverse culling
  (inside-out) and additive/alpha blending; fragment alpha is a fresnel term
  `pow(1 - max(dot(N, viewDir), 0), p)` tinted sky-blue → a halo bright at the limb.
- If a texture is unavailable (`texture_path` returned `None`), fall back to the
  current flat-blue lit sphere (no shader) so the sandbox still renders.

**Verification:** headless offscreen render + screenshot showing a textured Earth with a
visible day/night terminator and blue rim. (Shaders/visuals are checkpoint-verified, not
unit-tested, per project convention.)

## 4. Sun light + sun direction (`render/app.py`)

- Sandbox: each frame compute the Earth→Sun unit vector via
  `ephemeris.body_state("SUN", sim_time_s, center="EARTH").r` (normalized), map its
  direction into render space, and (a) feed it to the Earth shader uniform and (b) aim a
  `DirectionalLight` along it. This replaces the current fixed ambient/directional rig
  for the sandbox central body; a low ambient remains so the dark side isn't pure black
  beyond the city lights.
- Ephemeris is imported lazily (as in solar mode) so a texture-less/offline run still
  works and non-Earth central bodies degrade gracefully.

## 5. Solar-mode planets (`render/app.py::_build_planets`)

Build planet markers as textured spheres and apply each planet's surface map when
`texture_path` provides one; otherwise keep the current flat color. (Earth day map is
reused for Earth.) This is best-effort — missing planet maps simply stay flat-colored.

## 6. Testing strategy

- Real unit tests: textured-sphere UVs (§1), texture cache hit/reject/registry (§2).
  These stay offline and deterministic.
- Headless render checks + screenshots for the Earth shader, terminator, and atmosphere
  (§3) and the lit sandbox scene (§4), as done for the navball and solar viewer.
- Full suite stays green; no `core/`/`sim/` changes.

## 7. Scope boundaries (YAGNI)

**In:** textured UV spheres; download/cache with offline fallback; Earth day map + city-
lights night side + real-sun terminator + atmosphere rim; sun-directional lighting;
basic textures on solar-mode planets.

**Out (deferred):** cloud layer, specular ocean highlights, normal/bump maps, ring
systems (Saturn), star-field skybox, and any vessel/trajectory or HUD/navball changes.
No gameplay changes.

## Open implementation notes (not blockers)

- Panda3D loads GLSL via `Shader.load(Shader.SL_GLSL, vertex=..., fragment=...)`; keep
  the `.vert`/`.frag` files under `render/shaders/` and reference them by path.
- The night texture is a PNG; load with alpha. The day texture is a JPEG.
- Texture cache lives in `data/` which is already gitignored at `data/*.bsp`; add
  `data/textures/` explicitly.
