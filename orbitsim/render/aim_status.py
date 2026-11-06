"""The target status box's content (pure: no scene objects, no Panda3D calls).

Spec ("Where to aim"): green "Moon encounter · Pe 4,317 km" when the exact N-body path
enters the target's SOI, red "Impact" when it hits the surface, amber "No encounter ·
closest 120,000 km" otherwise. "Pe" is the altitude above the target's surface; "closest"
is the distance from its centre. Until the exact result for the current burn and target is
in, the box is grey "checking…" with the Earth-only estimate, so its colour never flips
wrongly. The bundled font has no tick/cross glyphs, so the icon is drawn separately.
"""
from __future__ import annotations

from dataclasses import dataclass

from orbitsim.render.hud.format import fmt_dist
from orbitsim.render.ui.theme import THEME


@dataclass(frozen=True)
class AimStatus:
    kind: str                     # "encounter" | "miss" | "impact" | "checking"
    text: str
    icon: str | None              # "check" | "cross" | None
    color: tuple[float, float, float, float]


def aim_status(target_name: str, encounters, approach, exact: bool,
               estimate_closest_m: float | None) -> AimStatus:
    """Status for ``target_name`` from the exact path's encounters and approach."""
    if not exact:
        text = "checking…"
        if estimate_closest_m is not None:
            text += f" · closest ~{fmt_dist(estimate_closest_m)}"
        return AimStatus("checking", text, None, THEME.text_muted)
    enc = next((e for e in encounters or () if e.body_name == target_name), None)
    if enc is not None and enc.impact:
        return AimStatus("impact", "Impact", "cross", THEME.danger)
    if enc is not None:
        alt_km = (enc.periapsis_radius_m - enc.body_radius_m) / 1000.0
        return AimStatus("encounter", f"{target_name} encounter · Pe {alt_km:,.0f} km",
                         "check", THEME.green)
    if approach is None:
        return AimStatus("miss", "No encounter", None, THEME.amber)
    return AimStatus("miss", f"No encounter · closest {fmt_dist(approach.distance_m)}",
                     None, THEME.amber)
