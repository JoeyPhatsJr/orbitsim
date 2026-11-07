"""Solar-system viewer mode: Sun-centered planets from the ephemeris, no vessel."""

import numpy as np
from orbitsim.core.planets import heliocentric_state
from orbitsim.render.geometry import make_uv_sphere
from panda3d.core import Filename


class SolarView:
    """Stateless controller: holds only ``self.app``; all state lives on OrbitApp."""

    # Distinct colours for the Sun + 8 planets (constant on-screen marker size).
    _PLANET_COLORS = {
        "Sun": (1.0, 0.85, 0.2, 1.0),
        "Mercury": (0.6, 0.6, 0.6, 1.0),
        "Venus": (0.9, 0.8, 0.5, 1.0),
        "Earth": (0.3, 0.5, 1.0, 1.0),
        "Mars": (0.9, 0.4, 0.2, 1.0),
        "Jupiter": (0.8, 0.7, 0.5, 1.0),
        "Saturn": (0.9, 0.8, 0.6, 1.0),
        "Uranus": (0.6, 0.85, 0.9, 1.0),
        "Neptune": (0.3, 0.4, 0.9, 1.0),
    }

    def __init__(self, app) -> None:
        self.app = app

    def build_planets(self) -> None:
        """Create constant-size markers + labels for the Sun and 8 planets."""
        from orbitsim.core.bodies import PLANETS, SUN

        self.app._planet_bodies = [SUN] + list(PLANETS)
        self.app._planet_nps = []
        self.app._planet_labels = []
        for body in self.app._planet_bodies:
            is_earth = body.name == "Earth"
            marker = make_uv_sphere(1.0, 12, 16, with_uv=is_earth)
            marker.reparent_to(self.app.render)
            marker.set_color(*self._PLANET_COLORS.get(body.name, (0.8, 0.8, 0.8, 1.0)))
            marker.set_light_off()
            marker.set_scale(8.0 if body.name == "Sun" else 4.0)
            if is_earth:
                from panda3d.core import Filename
                from orbitsim.render.textures import texture_path

                p = texture_path("earth_day")
                if p is not None:
                    marker.set_texture(self.app.loader.load_texture(Filename.from_os_specific(p)))
                    marker.set_color(1, 1, 1, 1)
            self.app._planet_nps.append(marker)
            # 3D billboard label on render (tracks the marker, faces the camera).
            from orbitsim.render.world_labels import build_world_label
            label = build_world_label(
                self.app.render, body.name, color=(0.8, 0.85, 1.0, 1.0), scale=15.0
            )
            self.app._planet_labels.append(label)

    def update_solar_system(self) -> None:
        """Place the Sun + planets for the current sim time (DE440 or fallback)."""
        from datetime import datetime, timedelta
        from orbitsim.core.ephemeris import available as ephemeris_available

        t = self.app.clock.sim_time_s
        self.app.transform.set_origin(np.zeros(3))  # heliocentric: Sun fixed at origin

        for body, marker, label in zip(self.app._planet_bodies, self.app._planet_nps, self.app._planet_labels):
            pos_m = np.zeros(3) if body.name == "Sun" else heliocentric_state(body.name, t).r
            rx, ry, rz = self.app.transform.to_render(pos_m)
            marker.set_pos(rx, ry, rz)
            label.set_pos(rx, ry, rz + 6.0)

        self.app.central_np.set_pos(*self.app.transform.to_render(np.zeros(3)))
        self.app.flight_input.apply_mouse_orbit()
        self.app.bodies_view.update_starfield()
        self.app.rig.apply()

        self.app.warp_controls.update_warp_readout()
        source = "JPL DE440" if ephemeris_available() else "circular approx (offline)"
        date = datetime(2000, 1, 1, 12, 0, 0) + timedelta(seconds=t)
        self.app.hud.text.setText(
            f"Solar system ({source})\n"
            f"Date: {date:%Y-%m-%d}\n"
            f"',' / '.' or the buttons change warp"
        )
