"""Next Pe / Next Ap place the burn like every other placement (review: Part 4b)."""
from types import SimpleNamespace

import numpy as np
import pytest

from orbitsim.core.constants import MU_EARTH
from orbitsim.core.maneuvers import time_to_apoapsis, time_to_periapsis
from orbitsim.core.state import StateVector
from orbitsim.render.controllers.maneuver_editor import ManeuverEditor


def _app(landed=False):
    rp, a = 7.0e6, 8.0e6
    v = np.sqrt(MU_EARTH * (2.0 / rp - 1.0 / a))
    st = StateVector(r=np.array([0.0, rp, 0.0]), v=np.array([-v, 0.0, 0.0]),
                     mu=MU_EARTH, epoch_s=100.0)
    vessel = SimpleNamespace(state=st, landed_on="Earth" if landed else None)
    toasts = []
    app = SimpleNamespace(world=SimpleNamespace(vessels=[vessel]),
                          clock=SimpleNamespace(sim_time_s=100.0),
                          _node_epoch_s=None, _auto_execute_node=True,
                          _flash_message=toasts.append)
    return app, toasts, st


@pytest.mark.parametrize("button, when", [("_node_to_pe", time_to_periapsis),
                                          ("_node_to_ap", time_to_apoapsis)])
def test_apsis_buttons_place_the_burn_exactly_and_disarm_auto_execute(button, when):
    app, _toasts, st = _app()
    getattr(ManeuverEditor(app), button)()
    assert app._node_epoch_s == 100.0 + when(st)     # exact apsis time, not snapped
    assert app._auto_execute_node is False


def test_apsis_buttons_refuse_while_landed():
    app, toasts, _st = _app(landed=True)
    ManeuverEditor(app)._node_to_ap()
    assert app._node_epoch_s is None
    assert toasts == ["Can't plan a burn while landed"]


def test_passed_node_executes_now_and_the_button_says_so():
    from orbitsim.render.controllers.maneuver_editor import execute_label
    assert execute_label(auto_armed=False, ttn_s=-30.0, tolerance_s=2.0) == "Burn Now"
    assert execute_label(auto_armed=False, ttn_s=600.0, tolerance_s=2.0) == "Jump & Execute"
    assert execute_label(auto_armed=False, ttn_s=1.0, tolerance_s=2.0) == "Execute Burn"
    assert execute_label(auto_armed=False, ttn_s=None, tolerance_s=2.0) == "Execute Burn"
    assert execute_label(auto_armed=True, ttn_s=600.0, tolerance_s=2.0) == "Cancel Auto Burn"
