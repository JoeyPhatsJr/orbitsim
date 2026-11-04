"""Sandbox HUD wiring: navball, SAS panel, velocity readout, panel dock, apsis/encounter markers, per-frame readouts."""

import numpy as np
from orbitsim.core.moon import moon_state_at
from orbitsim.render.world_markers import distance_fade


class Readouts:
    """Stateless controller: holds only ``self.app``; all state lives on OrbitApp."""

    MAX_ENCOUNTER_MARKERS = 4   # rarely more than 1-2 flybys visible on a path

    def __init__(self, app) -> None:
        self.app = app

    def build_sandbox_controls(self):
        """Maneuver editor, navball, SAS panel, velocity readout, dock."""
        # Maneuver editor (operates on vessel 0). The burn is applied at vessel 0's
        # *current* state (dt=0), so the executed orbit matches the live preview.
        self.app.maneuver_editor.build_maneuver_ui()
        self.app._fuel_capacity = (
            self.app.world.vessels[0].fuel_mass_kg if self.app.world.vessels else 0.0
        )
        from orbitsim.render.navball import Navball
        self.app.navball = Navball(self.app)
        from orbitsim.render.sas_panel import SasPanel, VelocityReadout
        self.app.sas_panel = SasPanel(
            self.app, on_set_mode=self.app.flight_input.set_sas, on_toggle=self.app.flight_input.toggle_sas
        )
        self.app.vel_readout = VelocityReadout(self.app, lambda: self.app.hud.units)
        from orbitsim.render.ui.widgets import PanelDock
        self.app.panel_dock = PanelDock(self.app, [
            ("ORBIT", self.app.hud.toggle_orbit),
            ("VESSEL", self.app.hud.toggle_vessel),
            ("MANEUVER", self.app.maneuver_editor.toggle_maneuver_drawer),
        ])
        self.app._maneuver_frame.hide()  # clean-flight default; MNV dock button opens it

    def build_apsis_and_encounter_markers(self):
        """PE/AP markers + encounter periapsis cues."""
        from orbitsim.render.world_labels import build_labeled_marker, build_world_label
        self.app._apsis_positions = {"PE": None, "AP": None}
        self.app._apsis_nps = {}
        self.app._apsis_labels = {}
        for name, color in (
            ("PE", (0.30, 1.0, 0.58, 1.0)),
            ("AP", (0.38, 0.68, 1.0, 1.0)),
        ):
            marker, label = build_labeled_marker(
                self.app.render, name, color=color, marker_scale=5.5, label_scale=11.0
            )
            marker.hide()
            label.hide()
            self.app._apsis_nps[name] = marker
            self.app._apsis_labels[name] = label
        # Predicted-encounter periapsis cues: a small pool of gold markers +
        # labels, placed each frame at each upcoming SOI closest approach.
        self.app._encounter_marker_nps = []
        self.app._encounter_labels = []
        for _ in range(self.MAX_ENCOUNTER_MARKERS):
            mk, lbl = build_labeled_marker(
                self.app.render, "", color=self.app.ENCOUNTER_COLOR,
                marker_scale=5.0, label_scale=10.5,
            )
            mk.hide()
            lbl.hide()
            self.app._encounter_marker_nps.append(mk)
            self.app._encounter_labels.append(lbl)

    def place_apsis_markers(self) -> None:
        """Place cached Pe/Ap cues in the current floating-origin frame."""
        for name in ("PE", "AP"):
            world_pos = self.app._apsis_positions[name]
            marker = self.app._apsis_nps[name]
            label = self.app._apsis_labels[name]
            if world_pos is None:
                marker.hide()
                label.hide()
                continue
            rx, ry, rz = self.app.transform.to_render(world_pos)
            marker.set_pos(rx, ry, rz)
            label.set_pos(rx, ry, rz + 7.0)
            marker.show()

    def update_marker_readability(self, focus_m) -> None:
        """Fade distant cues and declutter their labels by navigation priority."""
        from panda3d.core import TransparencyAttrib
        from orbitsim.render.world_markers import declutter_indices, distance_fade

        candidates = []

        def add(label, marker, world_pos, priority, minimum=0.22):
            if world_pos is not None:
                candidates.append((label, marker, np.asarray(world_pos), priority, minimum))

        add(self.app._target_label, None, self.app._target_label_position, 100, 0.55)
        add(self.app._ca_labels["ship"], self.app._ca_marker_ship,
            self.app._ca_label_positions["ship"], 90, 0.40)
        add(self.app._ca_labels["target"], self.app._ca_marker_moon,
            self.app._ca_label_positions["target"], 85, 0.40)
        add(self.app._apsis_labels["PE"], self.app._apsis_nps["PE"],
            self.app._apsis_positions["PE"], 70)
        add(self.app._apsis_labels["AP"], self.app._apsis_nps["AP"],
            self.app._apsis_positions["AP"], 65)
        for name, marker, label in zip(
            ("L1", "L2", "L3", "L4", "L5"), self.app._lagrange_nps, self.app._lagrange_labels
        ):
            add(label, marker, self.app._lagrange_positions.get(name), 20)

        points_px = [self.app.targeting.marker_px(item[2]) for item in candidates]
        visible = declutter_indices(
            points_px, [item[3] for item in candidates], min_separation_px=52.0
        )
        near = max(self.app.rig.distance_m * 2.0, 1.0)
        far = max(self.app.rig.distance_m * 18.0, near + 1.0)
        for index, (label, marker, world_pos, _priority, minimum) in enumerate(candidates):
            alpha = distance_fade(
                float(np.linalg.norm(world_pos - focus_m)), near, far, minimum=minimum
            )
            label.set_alpha_scale(alpha)
            if index in visible:
                label.show()
            else:
                label.hide()
            if marker is not None and not marker.is_empty():
                marker.set_transparency(TransparencyAttrib.M_alpha)
                marker.set_alpha_scale(alpha)

    def _update_encounter_info(self, vessel, dom_body, dom_pos) -> None:
        """Compute and display flyby encounter parameters when approaching a planet."""
        if not self.app.world.solar_system:
            self.app._encounter_line = ""
            return
        if dom_body.name in ("Earth", "Moon"):
            self.app._encounter_line = ""
            return
        if dom_body.name == "Sun":
            self.app._encounter_line = ""
            return
        from orbitsim.core.flyby import encounter_parameters
        r_rel = vessel.state.r - dom_pos
        v_planet_state = None
        t_now = self.app.clock.sim_time_s
        from orbitsim.core.nbody import (
            _cmercury, _cvenus, _cmars,
            _cjupiter, _csaturn, _curanus, _cneptune,
        )
        _planet_vel_fns = {
            "Mercury": _cmercury, "Venus": _cvenus,
            "Mars": _cmars, "Jupiter": _cjupiter,
            "Saturn": _csaturn, "Uranus": _curanus,
            "Neptune": _cneptune,
        }
        fn = _planet_vel_fns.get(dom_body.name)
        if fn is None:
            self.app._encounter_line = ""
            return
        pstate = fn(t_now)
        params = encounter_parameters(
            vessel.state.r, vessel.state.v, dom_pos, pstate.v, dom_body.mu)
        v_inf_kms = params["v_inf_mag"] / 1000.0
        defl_deg = np.degrees(params["deflection_rad"])
        dv_eq = params["dv_equivalent"]
        pe_km = params["periapsis_m"] / 1000.0
        self.app._encounter_line = (
            f"Flyby {dom_body.name}: v-inf {v_inf_kms:,.1f} km/s  "
            f"δ {defl_deg:,.1f}°  Pe {pe_km:,.0f} km  "
            f"free dV {dv_eq:,.0f} m/s"
        )
        self.app.maneuver_editor.sync_maneuver_hud()

    def frame_readouts(self, target_pos, maneuver_dir):
        """HUD, navball, SAS panel, velocity, encounters, warp."""
        v0 = self.app.world.vessels[0]
        # Osculating elements about the dominant body.
        from orbitsim.core.constants import MU_MOON
        from orbitsim.core.bodies import MOON as MOON_BODY
        if self.app.world.solar_system:
            from orbitsim.core.nbody import osculating_elements_solar, dominant_body_solar
            elem = osculating_elements_solar(v0.state, self.app.clock.sim_time_s)
            dom_body, dom_pos = dominant_body_solar(v0.state.r, self.app.clock.sim_time_s)
            ref_radius = dom_body.radius_m
        else:
            from orbitsim.core.nbody import osculating_elements
            elem = osculating_elements(v0.state, self.app.clock.sim_time_s)
            moon_dominant = elem.mu == MU_MOON
            dom_body = MOON_BODY if moon_dominant else self.app.world.central
            dom_pos = np.zeros(3) if dom_body == self.app.world.central else moon_state_at(self.app.clock.sim_time_s).r
            ref_radius = dom_body.radius_m
        rp = elem.periapsis_radius
        ra = elem.apoapsis_radius
        try:
            period = elem.period_s
        except ValueError:
            period = float("nan")
        dist_from_dom = float(np.linalg.norm(v0.state.r - dom_pos))
        self.app.hud.update(
            sim_time_s=self.app.clock.sim_time_s,
            altitude_m=dist_from_dom - ref_radius,
            speed_mps=v0.state.v_mag,
            periapsis_m=rp - ref_radius,
            apoapsis_m=ra - ref_radius,
            period_s=period,
            inclination_rad=elem.i,
            dominant_body=dom_body.name,
        )
        g_local = dom_body.mu / max(dist_from_dom, 1.0) ** 2
        twr = (v0.max_thrust_n / (v0.mass_kg * g_local)) if v0.mass_kg > 0 else 0.0
        cap = getattr(self.app, "_fuel_capacity", 0.0)
        fuel_frac = v0.fuel_mass_kg / cap if cap > 0 else 0.0
        self.app.hud.update_flight(
            throttle=v0.throttle,
            fuel_kg=v0.fuel_mass_kg,
            fuel_frac=fuel_frac,
            mass_kg=v0.mass_kg,
            thrust_n=v0.max_thrust_n,
            twr=twr,
            dv_remaining=v0.delta_v_remaining,
            warp_locked=self.app.world.any_thrusting(),
        )
        self.app.navball.update(
            orientation_q=v0.orientation,
            state=v0.state,
            target_pos=target_pos,
            maneuver_dir=maneuver_dir,
        )
        from orbitsim.core.attitude import heading_pitch
        heading, pitch = heading_pitch(v0.orientation, v0.state)
        self.app.sas_panel.update(v0.sas_mode, heading, pitch)
        target_rel_speed = None
        if self.app._target is not None:
            target_velocity = self.app._target.state_at(self.app.clock.sim_time_s).v
            target_rel_speed = float(np.linalg.norm(v0.state.v - target_velocity))
        self.app.vel_readout.update(v0.state.v_mag, target_rel_speed)
        self._update_encounter_info(v0, dom_body, dom_pos)
        self.app.trajectory_view.update_encounter_cues()
        self.app.warp_controls.update_warp_readout()
