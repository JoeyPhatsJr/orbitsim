# Starfield / Skybox — Design

**Status:** approved (design phase). Next: implementation plan (in
`docs/superpowers/plans/2026-06-25-starfield.md`).
**Date:** 2026-06-25

## Goal

Replace the black void behind the scene with a real star background, so the sandbox
and solar views feel like space. The sky is fixed in the inertial (J2000) frame and sits
"at infinity" — orbiting the camera pans across a fixed star field with no parallax, and
the sky always renders behind Earth, the vessel, and orbit lines.

This is a graphics sub-project; it does not touch physics or gameplay.

## Decisions (locked during brainstorming)

1. **Look:** a real photographic star map (Milky Way / galaxy star field), not flat
   procedural-only.
2. **Source:** download on first run + cache (same pattern as the Earth maps), with a
   **procedural point-star fallback** if offline. **Verified URL** (downloads cleanly in
   this environment, MIT-licensed, ~673 KB PNG):
   `https://raw.githubusercontent.com/jeromeetienne/threex.planets/master/images/galaxy_starfield.png`
   (ESO panoramas fail SSL verification here; solarsystemscope is CAPTCHA-walled; the
   other GitHub candidates 404'd or were tiny — this is the reliable one.)
3. **Frame:** stars are inertial (world-fixed orientation); the sky sphere is re-centered
   on the camera each frame so it behaves as infinitely far.

## Architecture (render-only; respects layering)

```
render/textures.py   MODIFIED. Add a "stars" entry to TEXTURE_URLS (no logic change).
render/skybox.py      NEW. build_starfield(base) -> NodePath; random_star_dirs(n, seed)
                      pure helper for the procedural fallback.
render/app.py         MODIFIED. Build the starfield in _start_sim (both modes); recenter
                      it on the camera each frame in _update / _update_solar_system.
```

`render/` only. No `core/`/`sim/` changes.

## 1. Sky texture registry (`render/textures.py`)

Add one entry to `TEXTURE_URLS`:
```python
"stars": "https://raw.githubusercontent.com/jeromeetienne/threex.planets/master/images/galaxy_starfield.png",
```
`texture_path("stars")` then works exactly like the Earth maps (download/cache/validate,
`None` when offline/invalid). The file is a PNG, so the existing `_ext_for` (`.png`) and
PNG magic check already cover it. No other changes to `textures.py`.

## 2. Starfield builder (`render/skybox.py`)

```python
def random_star_dirs(n: int, seed: int = 0) -> list[tuple[float, float, float]]
    # n unit direction vectors, deterministic for a given seed (for the fallback)
def build_starfield(base) -> NodePath
    # textured inside-out sky sphere, or procedural points if the texture is unavailable
```

- **Textured path:** a UV sphere (`make_uv_sphere(1.0, 32, 64, with_uv=True)`) scaled
  large, with the star texture; reverse culling (`set_two_sided(True)` /
  `set_attrib(CullFaceAttrib.make_reverse())`) so it's visible from inside; `set_light_off`.
- **Procedural fallback:** build a `GeomPoints` node from `random_star_dirs(n)` placed on
  a large-radius sphere, white with per-point brightness jitter; `set_light_off`. Used
  when `texture_path("stars")` is `None`.
- **Background treatment (both paths):** `set_bin("background", 0)`, `set_depth_write(False)`,
  `set_depth_test(False)`, `set_light_off` — always drawn first and behind everything.

`random_star_dirs` must be a pure function (numpy with a seeded `default_rng`) returning
unit vectors, so it is unit-tested without graphics.

## 3. Integration (`render/app.py`)

- In `_start_sim` (run for both modes, before the task starts): `self.starfield =
  build_starfield(self)` and `self.starfield.reparent_to(self.render)`.
- Each frame (both `_update` and `_update_solar_system`), before `self.rig.apply()`, set
  the starfield's position to the camera's world position so the camera stays at its
  center:
  `self.starfield.set_pos(self.camera.get_pos(self.render))`.
  Its *orientation* stays fixed (identity in world space) → stars are inertial and pan as
  the camera rotates. Scale it well beyond the camera distance (e.g. radius such that it
  never clips the far plane; depth-test-off means size only needs to clear the near plane).

## 4. Testing

- **Unit (offline):** `"stars"` key present in `TEXTURE_URLS`; `random_star_dirs(n, seed)`
  returns `n` unit-length vectors, deterministic for a seed (same seed → same list,
  different seed → different).
- **Headless + screenshot:** `build_starfield` returns a non-empty node (textured or
  fallback); a sandbox screenshot shows stars behind a lit Earth (Earth and orbit still
  draw on top — verifies the background bin / depth settings).

## 5. Scope (YAGNI)

**In:** downloaded star map with procedural fallback; inertial, camera-centered sky;
correct back-layering; both modes.

**Out:** constellation/label overlays, multiple nebula layers, twinkling/animation,
HDR/bloom on stars (that belongs to a later lighting/bloom pass), and accurate real-star
catalog positions (the panorama is ambiance, not an astrometric sky).

## Open implementation notes (not blockers)

- Reverse culling: `from panda3d.core import CullFaceAttrib; node.set_attrib(
  CullFaceAttrib.make_reverse())` is the robust way to render the inside of the sphere.
- `GeomPoints` needs a render thickness: `node.set_render_mode_thickness(2)` (via
  RenderModeAttrib) or rely on default 1px; keep points small.
- Use `Filename.from_os_specific(path)` when loading the star texture (Panda cannot
  resolve raw Windows backslash paths — the same gotcha that bit the Earth textures).
