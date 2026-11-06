"""The target status box: green encounter / amber miss / red impact / grey checking.

Spec ("Where to aim"): the box takes a colour only from an exact N-body result for the
current burn and target; "Pe" is the altitude above the target's surface, "closest" the
distance from its centre.
"""
import dataclasses

import numpy as np

from orbitsim.core.constants import MU_MOON, R_MOON
from orbitsim.core.encounters import Encounter, TargetApproach
from orbitsim.render.aim_status import AimStatus, aim_status
from orbitsim.render.ui.theme import THEME


def ENC(pe_alt, impact=False):
    return Encounter("Moon", 0, 1, 0.0, 1.0, 0, 0.0, R_MOON + pe_alt, np.zeros(3), 1000.0,
                     MU_MOON, R_MOON, True, impact)


AP = TargetApproach("Moon", 1.2e8, 0.0, np.zeros(3), np.ones(3))


def test_green_encounter_text():
    s = aim_status("Moon", [ENC(4_317e3)], AP, exact=True, estimate_closest_m=None)
    assert isinstance(s, AimStatus)
    assert (s.kind, s.text, s.icon) == ("encounter", "Moon encounter · Pe 4,317 km", "check")
    assert s.color == THEME.green


def test_amber_miss_uses_centre_distance():
    s = aim_status("Moon", [], AP, exact=True, estimate_closest_m=None)
    assert (s.kind, s.text, s.icon) == ("miss", "No encounter · closest 120,000 km", None)
    assert s.color == THEME.amber


def test_miss_without_approach():
    s = aim_status("Moon", [], None, exact=True, estimate_closest_m=None)
    assert (s.kind, s.text) == ("miss", "No encounter")


def test_red_impact():
    s = aim_status("Moon", [ENC(-500e3, impact=True)], AP, exact=True, estimate_closest_m=None)
    assert (s.kind, s.text, s.icon) == ("impact", "Impact", "cross")
    assert s.color == THEME.danger


def test_status_checking_when_key_stale():
    s = aim_status("Moon", [ENC(4_317e3)], AP, exact=False, estimate_closest_m=229e3)
    assert s.kind == "checking" and s.text == "checking… · closest ~229 km" and s.icon is None
    assert s.color == THEME.text_muted


def test_checking_without_estimate():
    s = aim_status("Moon", [], None, exact=False, estimate_closest_m=None)
    assert (s.kind, s.text) == ("checking", "checking…")


def test_encounter_for_another_body_is_not_the_targets():
    other = dataclasses.replace(ENC(1e6), body_name="Mars")
    assert aim_status("Moon", [other], AP, exact=True, estimate_closest_m=None).kind == "miss"
