"""Central body, lighting, Sun direction, Moon (ring, body, SOI), Lagrange points, planets, starfield."""

import numpy as np
from orbitsim.core.constants import MU_SUN
from orbitsim.core.elements import state_to_elements
from orbitsim.core.moon import moon_state_at
from orbitsim.core.nbody import MOON_SOI_M
from orbitsim.core.planets import (
    EARTH_SOI_M,
    JUPITER_SOI_M,
    MARS_SOI_M,
    MERCURY_SOI_M,
    NEPTUNE_SOI_M,
    SATURN_SOI_M,
    URANUS_SOI_M,
    VENUS_SOI_M,
)
from orbitsim.core.state import StateVector
from orbitsim.render.earth import build_earth, set_sun_dir
from orbitsim.render.geometry import make_ring, make_uv_sphere
from orbitsim.render.line_style import planet_orbit_style, planet_orbits_visible, reference_style
from orbitsim.render.orbit_lines import (
    build_orbit_node,
    sample_orbit_points,
    sample_relative_orbit_points,
)
from orbitsim.render.skybox import build_starfield
from orbitsim.render.world_markers import distance_fade
from panda3d.core import AmbientLight, DirectionalLight, Filename, Vec4


class BodiesView:
    """Stateless controller: holds only ``self.app``; all state lives on OrbitApp."""

    def _target_name(self):
        """Name of the current target body (None if nothing / solar viewer)."""
        target = getattr(self.app, "_target", None)
        return getattr(target, "name", None) if target is not None else None

    MOON_RING_REFRESH_S = 6 * 3600.0  # sim-seconds between osculating Moon-ring rebuilds
    LAGRANGE_REFRESH_S = 10.0     # sim-seconds between L-point recomputes
    LAGRANGE_IDLE_ALPHA = 0.25    # an L-point that isn't the target fades to this
    SOI_COLOR = (0.45, 0.65, 1.0, 1.0)         # cool blue tint (outside)
    SOI_INSIDE_COLOR = (0.45, 1.0, 0.65, 1.0)  # greenish "captured by the Moon"
    SOI_BASE_ALPHA = 0.10                       # faint translucent shell
    SOI_INSIDE_ALPHA = 0.18
    SOI_FADE_NEAR_M = 1.5e9   # camera distance: full alpha when closer than this
    SOI_FADE_FAR_M = 1.5e10   # ... fading to zero past this (tune by screenshot)

    def __init__(self, app) -> None:
        self.app = app

    def build_central_body(self):
        """Sun marker (solar) or shaded Earth + atmosphere (sandbox)."""
        # Central body. Solar mode: fullbright Sun marker. Sandbox: the textured,
        # day/night-shaded Earth (with an atmosphere shell), or a flat-blue fallback.
        self.app._atmo_np = None
        if self.app.solar_system:
            self.app.central_np = make_uv_sphere(1.0, 24, 48)
            self.app.central_np.reparent_to(self.app.render)
            self.app.central_np.set_color(1.0, 0.85, 0.2, 1.0)
            self.app.central_np.set_light_off()
            self.app.central_np.set_scale(10.0)
        else:
            self.app.central_np, self.app._atmo_np = build_earth(self.app)
            self.app.central_np.reparent_to(self.app.render)
            if self.app._atmo_np is not None:
                self.app._atmo_np.reparent_to(self.app.central_np)

    def build_lighting(self):
        """Ambient + directional sun light."""
        # Lighting: low ambient + a sun directional light (aimed at the real Sun each
        # frame in the sandbox). The shadered Earth lights itself; the light is for the
        # flat fallback and any other lit geometry.
        amb = AmbientLight("amb")
        amb.set_color(Vec4(0.3, 0.3, 0.3, 1) if self.app.solar_system else Vec4(0.12, 0.12, 0.15, 1))
        self.app.render.set_light(self.app.render.attach_new_node(amb))
        sun = DirectionalLight("sun")
        sun.set_color(Vec4(1.0, 1.0, 0.95, 1))
        self.app._sun_light_np = self.app.render.attach_new_node(sun)
        self.app._sun_light_np.set_hpr(45, -45, 0)
        self.app.render.set_light(self.app._sun_light_np)

    def build_orbit_frame(self):
        """Orbit frame node + per-vessel trajectory caches."""
        world = self.app.world
        # Orbit frame: holds all Earth-centered orbit lines in world meters; repositioned +
        # rescaled once per frame so they track the floating origin without per-vertex rebuilds.
        self.app._orbit_frame = self.app.render.attach_new_node("orbit_frame")
        self.app._traj_world_cache = [None for _ in world.vessels]
        self.app._traj_epoch_cache = [None for _ in world.vessels]
        self.app._traj_future = [None for _ in world.vessels]
        self.app._traj_last_submit_t = [float("-inf") for _ in world.vessels]
        self.app._traj_last_complete_t = [float("-inf") for _ in world.vessels]
        self.app._traj_render_scale = [None for _ in world.vessels]
        self.app._traj_render_origin = [None for _ in world.vessels]
        self.app._traj_encounters = [None for _ in world.vessels]
        self.app._traj_approach = [None for _ in world.vessels]   # TargetApproach or None

    def build_moon_and_lagrange(self):
        """Moon marker, SOI shell, Lagrange markers."""
        from orbitsim.render.world_labels import build_labeled_marker, build_world_label
        self.app._moon_np = make_uv_sphere(1.0, 12, 16)
        self.app._moon_np.reparent_to(self.app.render)
        self.app._moon_np.set_color(0.7, 0.7, 0.72, 1.0)
        self.app._moon_np.set_light_off()
        self.app._moon_np.set_scale(7.0)
        self.app._moon_orbit_np = None      # built lazily in the update loop (scale-dependent)
        self.app._moon_orbit_scale = None
        self.app._moon_orbit_target = None  # target name the ring was styled for
        self.app._moon_orbit_epoch_s = None  # sim time of the osculating ellipse drawn
        # Moon sphere-of-influence: faint true-scale translucent tinted shell.
        from panda3d.core import TransparencyAttrib
        self.app._soi_np = make_uv_sphere(1.0, 24, 32)
        self.app._soi_np.reparent_to(self.app.render)
        self.app._soi_np.set_light_off()                       # flat, unlit tint
        self.app._soi_np.set_transparency(TransparencyAttrib.M_alpha)
        self.app._soi_np.set_depth_write(False)                # don't occlude the scene behind it
        self.app._soi_np.set_two_sided(True)                   # visible from inside too
        self.app._soi_np.hide()  # shown + placed each frame in _update
        # Lagrange-point markers (constant on-screen size) + billboard labels.
        self.app._lagrange_nps = []
        self.app._lagrange_labels = []
        self.app._lagrange_positions = {}
        for name in ("L1", "L2", "L3", "L4", "L5"):
            mk = make_uv_sphere(1.0, 8, 12)
            mk.reparent_to(self.app.render)
            mk.set_color(0.3, 0.9, 0.8, 1.0)   # teal — distinct from Moon/CA/node
            mk.set_light_off()
            mk.set_scale(4.0)
            self.app._lagrange_nps.append(mk)
            lbl = build_world_label(
                self.app.render, name, color=(0.5, 1.0, 0.9, 1.0), scale=12.0
            )
            self.app._lagrange_labels.append(lbl)

    def build_sandbox_planets(self):
        """Sandbox planets/Sun, Saturn ring, Moon body, Earth SOI, orbit lines."""
        from orbitsim.render.world_labels import build_labeled_marker, build_world_label
        from panda3d.core import TransparencyAttrib
        # Inner planets + Sun: markers, true-scale bodies, SOI spheres, orbit lines.
        from orbitsim.core.constants import (
            R_SUN, R_MERCURY, R_VENUS, R_MARS, R_MOON,
            R_JUPITER, R_SATURN, R_URANUS, R_NEPTUNE,
        )
        from orbitsim.render.textures import texture_path
        self.app._planet_sandbox_nps = {}    # name -> marker NodePath (constant on-screen size)
        self.app._planet_sandbox_labels = {}
        self.app._planet_body_nps = {}       # name -> true-scale textured body NodePath
        self.app._planet_body_radii = {}     # name -> physical radius [m]
        self.app._planet_soi_nps = {}        # name -> SOI sphere NodePath
        self.app._planet_soi_radii = {}      # name -> SOI radius [m]
        _planet_defs = [
            ("Sun", (1.0, 0.85, 0.2, 1.0), 10.0, float("inf"), R_SUN, "sun"),
            ("Mercury", (0.6, 0.6, 0.6, 1.0), 4.0, MERCURY_SOI_M, R_MERCURY, "mercury"),
            ("Venus", (0.9, 0.8, 0.5, 1.0), 5.0, VENUS_SOI_M, R_VENUS, "venus_surface"),
            ("Mars", (0.9, 0.4, 0.2, 1.0), 5.0, MARS_SOI_M, R_MARS, "mars"),
            ("Jupiter", (0.8, 0.7, 0.5, 1.0), 7.0, JUPITER_SOI_M, R_JUPITER, "jupiter"),
            ("Saturn", (0.9, 0.8, 0.6, 1.0), 6.5, SATURN_SOI_M, R_SATURN, "saturn"),
            ("Uranus", (0.6, 0.85, 0.9, 1.0), 5.5, URANUS_SOI_M, R_URANUS, "uranus"),
            ("Neptune", (0.3, 0.4, 0.9, 1.0), 5.5, NEPTUNE_SOI_M, R_NEPTUNE, "neptune"),
        ]
        _soi_colors = {
            "Mercury": (0.6, 0.6, 0.6, 1.0),
            "Venus": (0.9, 0.8, 0.5, 1.0),
            "Mars": (0.9, 0.5, 0.3, 1.0),
            "Jupiter": (0.8, 0.7, 0.5, 1.0),
            "Saturn": (0.9, 0.8, 0.6, 1.0),
            "Uranus": (0.6, 0.85, 0.9, 1.0),
            "Neptune": (0.3, 0.4, 0.9, 1.0),
        }
        for pname, color, sz, soi_r, radius_m, tex_name in _planet_defs:
            mk = make_uv_sphere(1.0, 12, 16)
            mk.reparent_to(self.app.render)
            mk.set_color(*color)
            mk.set_light_off()
            mk.set_scale(sz)
            self.app._planet_sandbox_nps[pname] = mk
            lbl = build_world_label(
                self.app.render, pname, color=(0.8, 0.85, 1.0, 1.0), scale=12.0
            )
            self.app._planet_sandbox_labels[pname] = lbl
            self.app._planet_soi_radii[pname] = soi_r
            self.app._planet_body_radii[pname] = radius_m
            body = make_uv_sphere(1.0, 32, 64, with_uv=True)
            body.reparent_to(self.app.render)
            body.set_color(*color)
            tp = texture_path(tex_name)
            if tp is not None:
                body.set_texture(
                    self.app.loader.load_texture(Filename.from_os_specific(tp)))
                body.set_color(1, 1, 1, 1)
            if pname == "Sun":
                body.set_light_off()
            body.hide()
            self.app._planet_body_nps[pname] = body
            if pname != "Sun" and np.isfinite(soi_r):
                soi = make_uv_sphere(1.0, 24, 32)
                soi.reparent_to(self.app.render)
                soi.set_light_off()
                soi.set_transparency(TransparencyAttrib.M_alpha)
                soi.set_depth_write(False)
                soi.set_two_sided(True)
                sc = _soi_colors.get(pname, (0.5, 0.5, 0.5, 1.0))
                soi.set_color(sc[0], sc[1], sc[2], self.SOI_BASE_ALPHA)
                soi.hide()
                self.app._planet_soi_nps[pname] = soi
        # Saturn ring (true-scale annular disk, inner ~1.11× R, outer ~2.33× R).
        self.app._saturn_ring_np = make_ring(inner_radius=1.11, outer_radius=2.33, num_segments=64)
        self.app._saturn_ring_np.reparent_to(self.app.render)
        self.app._saturn_ring_np.set_two_sided(True)
        self.app._saturn_ring_np.set_transparency(TransparencyAttrib.M_alpha)
        self.app._saturn_ring_np.set_color(0.85, 0.75, 0.6, 0.8)
        ring_tp = texture_path("saturn_ring")
        if ring_tp is not None:
            self.app._saturn_ring_np.set_texture(
                self.app.loader.load_texture(Filename.from_os_specific(ring_tp)))
            self.app._saturn_ring_np.set_color(1, 1, 1, 0.8)
        self.app._saturn_ring_np.hide()
        # True-scale Moon body (textured).
        self.app._moon_body_np = make_uv_sphere(1.0, 32, 64, with_uv=True)
        self.app._moon_body_np.reparent_to(self.app.render)
        moon_tp = texture_path("moon")
        if moon_tp is not None:
            self.app._moon_body_np.set_texture(
                self.app.loader.load_texture(Filename.from_os_specific(moon_tp)))
            self.app._moon_body_np.set_color(1, 1, 1, 1)
        else:
            self.app._moon_body_np.set_color(0.7, 0.7, 0.72, 1.0)
        self.app._moon_body_np.hide()
        # Earth SOI sphere (visible once the vessel gets far from Earth).
        earth_soi = make_uv_sphere(1.0, 24, 32)
        earth_soi.reparent_to(self.app.render)
        earth_soi.set_light_off()
        earth_soi.set_transparency(TransparencyAttrib.M_alpha)
        earth_soi.set_depth_write(False)
        earth_soi.set_two_sided(True)
        earth_soi.set_color(0.3, 0.5, 1.0, self.SOI_BASE_ALPHA)
        earth_soi.hide()
        self.app._earth_soi_np = earth_soi
        # Planet orbit reference lines (heliocentric circles, positioned at the Sun's
        # geocentric position each frame). Built lazily on first zoom like the Moon line.
        self.app._planet_orbit_nps = {}
        self.app._planet_orbit_scale = None
        self.app._planet_orbit_target = None  # target name the orbits were styled for
        self.app._planet_orbit_epoch_bucket = None
        self.app._sun_orbit_frame = self.app.render.attach_new_node("sun_orbit_frame")

    def build_starfield(self):
        """Inertial camera-centered star background (both modes)."""
        # Star background (both modes): inertial, camera-centered, behind everything.
        self.app.starfield = build_starfield(self.app)
        self.app.starfield.reparent_to(self.app.render)

    def update_starfield(self) -> None:
        """Keep the sky centered on the camera so stars sit at infinity (no parallax)."""
        if getattr(self.app, "starfield", None) is not None:
            self.app.starfield.set_pos(self.app.camera.get_pos(self.app.render))

    def frame_moon_ring(self, scale):
        """Rebuild the Moon's osculating orbit ring on zoom / every few sim-hours."""
        # The real Moon's orbit is eccentric and perturbed, so the ring is its *osculating*
        # ellipse (always passes through the Moon); rebuilt on zoom or every 6 sim-hours.
        t_now = self.app.clock.sim_time_s
        target = self._target_name()
        if (self.app._moon_orbit_scale != scale or self.app._moon_orbit_epoch_s is None
                or self.app._moon_orbit_target != target
                or abs(t_now - self.app._moon_orbit_epoch_s) > self.MOON_RING_REFRESH_S):
            self.app._moon_orbit_scale = scale
            self.app._moon_orbit_target = target
            self.app._moon_orbit_epoch_s = t_now
            if self.app._moon_orbit_np is not None:
                self.app._moon_orbit_np.remove_node()
            moon_elements = state_to_elements(moon_state_at(t_now))
            moon_pts = [tuple(p / scale) for p in sample_orbit_points(moon_elements, n=256)]
            self.app._moon_orbit_np = build_orbit_node(
                moon_pts, reference_style(target == "Moon"), px_scale=self.app.line_px_scale()
            )
            self.app._moon_orbit_np.reparent_to(self.app._orbit_frame)

    def frame_sun(self):
        """Aim the terminator and sun light at the real Sun."""
        # Real Sun direction (Earth->Sun) drives the day/night terminator + sun
        # light. The cached wrapper serves DE440 positions when available and
        # falls back to the circular approximation offline — never raises.
        from orbitsim.core.nbody import _csun as _cached_sun
        sun_dir = np.asarray(_cached_sun(self.app.clock.sim_time_s).r, dtype=float)
        n = np.linalg.norm(sun_dir)
        if n > 0:
            sun_dir = sun_dir / n
            set_sun_dir(self.app.central_np, tuple(sun_dir))
            self.app._sun_light_np.set_pos(float(sun_dir[0]) * 1000.0,
                                       float(sun_dir[1]) * 1000.0,
                                       float(sun_dir[2]) * 1000.0)
            self.app._sun_light_np.look_at(0, 0, 0)

    def frame_moon(self):
        """Moon marker, true-scale body and SOI shell."""
        # Moon position this frame.
        moon_now = moon_state_at(self.app.clock.sim_time_s)
        self.app._moon_np.set_pos(*self.app.transform.to_render(moon_now.r))
        # True-scale Moon body.
        from orbitsim.core.constants import R_MOON
        moon_render_pos = self.app.transform.to_render(moon_now.r)
        moon_body_scale = R_MOON / self.app.transform.scale_m_per_unit
        self.app._moon_body_np.set_pos(*moon_render_pos)
        self.app._moon_body_np.set_scale(max(moon_body_scale, 1e-3))
        self.app._moon_body_np.show()
        # Moon SOI wireframe: true-scale, brighter when the vessel is inside, camera-distance fade.
        soi_scale = MOON_SOI_M / self.app.transform.scale_m_per_unit
        self.app._soi_np.set_pos(*self.app.transform.to_render(moon_now.r))
        self.app._soi_np.set_scale(soi_scale)
        inside = float(np.linalg.norm(self.app.world.vessels[0].state.r - moon_now.r)) < MOON_SOI_M
        color = self.SOI_INSIDE_COLOR if inside else self.SOI_COLOR
        base_alpha = self.SOI_INSIDE_ALPHA if inside else self.SOI_BASE_ALPHA
        self.app._soi_np.set_color(color[0], color[1], color[2], base_alpha)
        fade = distance_fade(self.app.rig.distance_m, self.SOI_FADE_NEAR_M, self.SOI_FADE_FAR_M, minimum=0.0)
        self.app._soi_np.set_alpha_scale(fade)
        self.app._soi_np.show()

    def frame_lagrange(self):
        """Lagrange-point markers (bucketed recompute)."""
        # Lagrange points (rotate with the Moon). Each recompute runs three
        # brentq root-solves, so refresh by sim-time bucket instead of every
        # frame: the L-points move ~1 km/s, so a 10 s bucket is at most ~10 km
        # stale (invisible at map zoom) while cutting the per-frame cost to
        # zero at low warp. At high warp the bucket changes every frame and
        # the recompute cadence follows automatically.
        from orbitsim.core.nbody import earth_fixed_lagrange_points
        lag_bucket = int(self.app.clock.sim_time_s // self.LAGRANGE_REFRESH_S)
        if getattr(self.app, "_lagrange_bucket", None) != lag_bucket:
            self.app._lagrange_bucket = lag_bucket
            self.app._lagrange_cache = earth_fixed_lagrange_points(self.app.clock.sim_time_s)
        lps = self.app._lagrange_cache
        for name, mk, lbl in zip(("L1", "L2", "L3", "L4", "L5"),
                                 self.app._lagrange_nps, self.app._lagrange_labels):
            rx, ry, rz = self.app.transform.to_render(lps[name])
            mk.set_pos(rx, ry, rz)
            lbl.set_pos(rx, ry, rz + 6.0)
            self.app._lagrange_positions[name] = np.asarray(lps[name]).copy()

    def frame_planets(self, scale):
        """Planet/Sun markers, bodies, SOI shells, Earth SOI, orbit lines."""
        # Planets + Sun: position markers, SOI spheres, and orbit reference lines.
        from orbitsim.core.nbody import (
            _csun, _cmercury, _cvenus, _cmars,
            _cjupiter, _csaturn, _curanus, _cneptune,
        )
        _planet_state_fns = {
            "Sun": _csun,
            "Mercury": _cmercury,
            "Venus": _cvenus,
            "Mars": _cmars,
            "Jupiter": _cjupiter,
            "Saturn": _csaturn,
            "Uranus": _curanus,
            "Neptune": _cneptune,
        }
        t_now = self.app.clock.sim_time_s
        vessel_r = self.app.world.vessels[0].state.r if self.app.world.vessels else np.zeros(3)
        for pname, state_fn in _planet_state_fns.items():
            pstate = state_fn(t_now)
            px, py, pz = self.app.transform.to_render(pstate.r)
            self.app._planet_sandbox_nps[pname].set_pos(px, py, pz)
            self.app._planet_sandbox_labels[pname].set_pos(px, py, pz + 8.0)
            if pname in self.app._planet_body_nps:
                body_np = self.app._planet_body_nps[pname]
                body_r = self.app._planet_body_radii[pname]
                body_scale = body_r / self.app.transform.scale_m_per_unit
                body_np.set_pos(px, py, pz)
                body_np.set_scale(max(body_scale, 1e-3))
                body_np.show()
            if pname in self.app._planet_soi_nps:
                soi_np = self.app._planet_soi_nps[pname]
                soi_r = self.app._planet_soi_radii[pname]
                soi_scale = soi_r / self.app.transform.scale_m_per_unit
                soi_np.set_pos(px, py, pz)
                soi_np.set_scale(soi_scale)
                dist_to = float(np.linalg.norm(vessel_r - pstate.r))
                inside = dist_to < soi_r
                alpha = self.SOI_INSIDE_ALPHA if inside else self.SOI_BASE_ALPHA
                soi_np.set_alpha_scale(
                    distance_fade(self.app.rig.distance_m, soi_r * 1.5, soi_r * 15, minimum=0.0)
                )
                soi_np.set_color_scale(1, 1, 1, alpha)
                soi_np.show()
            if pname == "Saturn" and hasattr(self.app, "_saturn_ring_np"):
                from orbitsim.core.constants import R_SATURN
                ring_scale = R_SATURN / self.app.transform.scale_m_per_unit
                self.app._saturn_ring_np.set_pos(px, py, pz)
                self.app._saturn_ring_np.set_scale(max(ring_scale, 1e-3))
                self.app._saturn_ring_np.show()
        # Earth SOI sphere (centered at origin, visible when far from Earth).
        earth_soi_scale = EARTH_SOI_M / self.app.transform.scale_m_per_unit
        ex, ey, ez = self.app.transform.to_render(np.zeros(3))
        self.app._earth_soi_np.set_pos(ex, ey, ez)
        self.app._earth_soi_np.set_scale(earth_soi_scale)
        earth_dist = float(np.linalg.norm(vessel_r))
        earth_inside = earth_dist < EARTH_SOI_M
        earth_alpha = self.SOI_INSIDE_ALPHA if earth_inside else self.SOI_BASE_ALPHA
        self.app._earth_soi_np.set_alpha_scale(
            distance_fade(self.app.rig.distance_m, EARTH_SOI_M * 1.5, EARTH_SOI_M * 15, minimum=0.0)
        )
        self.app._earth_soi_np.set_color_scale(1, 1, 1, earth_alpha)
        self.app._earth_soi_np.show()
        # Planet orbit reference lines (heliocentric circles, offset by -Earth(t)).
        from orbitsim.core.nbody import _csun
        sun_geo = _csun(t_now).r
        sx, sy, sz = self.app.transform.to_render(sun_geo)
        self.app._sun_orbit_frame.set_pos(sx, sy, sz)
        if planet_orbits_visible(self.app.rig.distance_m):
            self.app._sun_orbit_frame.show()
        else:
            self.app._sun_orbit_frame.hide()
        # Refresh monthly in simulation time.  DE440 positions are in tilted ICRF
        # axes and the real orbits are eccentric, so flat mean-radius circles do
        # not describe the paths followed by the rendered planet markers.
        orbit_epoch_bucket = int(np.floor(t_now / (30.0 * 86400.0)))
        target = self._target_name()
        if (self.app._planet_orbit_scale != scale
                or self.app._planet_orbit_target != target
                or self.app._planet_orbit_epoch_bucket != orbit_epoch_bucket):
            self.app._planet_orbit_scale = scale
            self.app._planet_orbit_target = target
            self.app._planet_orbit_epoch_bucket = orbit_epoch_bucket
            earth_state = StateVector(
                r=np.zeros(3), v=np.zeros(3), mu=0.0, epoch_s=t_now
            )
            orbit_states = {
                "Mercury": _cmercury(t_now),
                "Venus": _cvenus(t_now),
                "Earth": earth_state,
                "Mars": _cmars(t_now),
                "Jupiter": _cjupiter(t_now),
                "Saturn": _csaturn(t_now),
                "Uranus": _curanus(t_now),
                "Neptune": _cneptune(t_now),
            }
            sun_state = _csun(t_now)
            for oname in ("Mercury", "Venus", "Earth", "Mars",
                          "Jupiter", "Saturn", "Uranus", "Neptune"):
                if oname in self.app._planet_orbit_nps:
                    self.app._planet_orbit_nps[oname].remove_node()
                orbit_points = sample_relative_orbit_points(
                    orbit_states[oname], sun_state, MU_SUN, n=256
                )
                pts = [tuple(point / scale) for point in orbit_points]
                style = planet_orbit_style(oname, target == oname)
                orbit_np = build_orbit_node(pts, style, px_scale=self.app.line_px_scale())
                orbit_np.reparent_to(self.app._sun_orbit_frame)
                self.app._planet_orbit_nps[oname] = orbit_np
