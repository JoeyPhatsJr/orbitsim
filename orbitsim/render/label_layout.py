"""Screen-space label layout and pixel sizing (pure: no Panda3D).

Names are placed in one of four slots around their dot, by priority, so they never
stack on each other, on other dots, or on HUD rects; a name that fits nowhere is
hidden unless it is pinned (hovered). All rects are ``(x0, y0, x1, y1)`` framebuffer
pixels with y up.
"""
import math
from dataclasses import dataclass
from typing import Collection, Mapping, Optional, Sequence, Tuple

Rect = Tuple[float, float, float, float]

SLOTS = ("above", "right", "left", "below")


@dataclass(frozen=True)
class LabelRequest:
    """One name to place.

    key : unique id (the body name).
    anchor_px : the dot's centre in pixels, or None when off-screen / behind the camera.
    size_px : (width, height) of the label's card in pixels.
    marker_radius_px : the dot's on-screen radius in pixels.
    priority : higher is placed first.
    """

    key: str
    anchor_px: Optional[Tuple[float, float]]
    size_px: Tuple[float, float]
    marker_radius_px: float
    priority: int


def units_per_pixel(depth_units: float, vfov_deg: float, viewport_h_px: float) -> float:
    """World units one framebuffer pixel spans at camera depth ``depth_units``."""
    if not depth_units > 0.0:
        raise ValueError(f"depth must be > 0, got {depth_units}")
    if not 0.0 < vfov_deg < 180.0:
        raise ValueError(f"vertical fov must be in (0, 180) deg, got {vfov_deg}")
    if not viewport_h_px > 0:
        raise ValueError(f"viewport height must be > 0, got {viewport_h_px}")
    return 2.0 * depth_units * math.tan(math.radians(vfov_deg) / 2.0) / viewport_h_px


def slot_rect(anchor_px, size_px, clearance_px: float, slot: str) -> Rect:
    """The card's rect in ``slot`` around the anchor, its near edge ``clearance_px`` away."""
    x, y = anchor_px
    w, h = size_px
    if slot == "above":
        return (x - w / 2, y + clearance_px, x + w / 2, y + clearance_px + h)
    if slot == "right":
        return (x + clearance_px, y - h / 2, x + clearance_px + w, y + h / 2)
    if slot == "left":
        return (x - clearance_px - w, y - h / 2, x - clearance_px, y + h / 2)
    if slot == "below":
        return (x - w / 2, y - clearance_px - h, x + w / 2, y - clearance_px)
    raise ValueError(f"unknown label slot {slot!r}")


def rects_overlap(a: Rect, b: Rect) -> bool:
    """Whether two rects overlap (open intervals: touching edges don't count)."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _inside(rect: Rect, viewport_px) -> bool:
    return 0.0 <= rect[0] and rect[2] <= viewport_px[0] and 0.0 <= rect[1] and rect[3] <= viewport_px[1]


def layout_labels(
    requests: Sequence[LabelRequest],
    viewport_px: Tuple[float, float],
    *,
    gap_px: float = 3.0,
    obstacles: Sequence[Rect] = (),
    pinned: Collection[str] = (),
    preferred_slots: Optional[Mapping[str, str]] = None,
) -> dict:
    """Place each request's card, or hide it: ``{key: (slot, rect) | None}``.

    Pinned keys go first (input order), then the rest by descending priority (ties in
    input order). Each takes its preferred slot (last frame's, for stability) if free,
    else the first free one in ``SLOTS`` order: fully inside the viewport, clear of every
    placed card, every other dot and every obstacle. Nothing free: a pinned card takes
    its first candidate anyway; any other is hidden. Off-screen anchors are hidden and
    block nothing.
    """
    preferred_slots = preferred_slots or {}
    pinned = set(pinned)
    indexed = list(enumerate(requests))
    pinned_first = [ir for ir in indexed if ir[1].key in pinned]
    rest = sorted((ir for ir in indexed if ir[1].key not in pinned),
                  key=lambda ir: (-ir[1].priority, ir[0]))
    markers = {}
    for _, r in indexed:
        if r.anchor_px is not None:
            x, y = r.anchor_px
            m = r.marker_radius_px
            markers[r.key] = (x - m, y - m, x + m, y + m)

    out = {r.key: None for r in requests}
    placed = []
    for _, r in pinned_first + rest:
        if r.anchor_px is None:
            continue
        first = preferred_slots.get(r.key)
        candidates = ([first] if first in SLOTS else []) + [s for s in SLOTS if s != first]
        blockers = [m for k, m in markers.items() if k != r.key] + list(obstacles)
        clearance = r.marker_radius_px + gap_px
        choice = None
        for slot in candidates:
            rect = slot_rect(r.anchor_px, r.size_px, clearance, slot)
            if not _inside(rect, viewport_px):
                continue
            if any(rects_overlap(rect, b) for b in placed) or any(
                    rects_overlap(rect, b) for b in blockers):
                continue
            choice = (slot, rect)
            break
        if choice is None and r.key in pinned:
            slot = candidates[0]
            choice = (slot, slot_rect(r.anchor_px, r.size_px, clearance, slot))
        if choice is not None:
            out[r.key] = choice
            placed.append(choice[1])
    return out


def label_origin_offset_px(rect: Rect, card, font_px: float, anchor_px) -> Tuple[float, float]:
    """Pixel offset from the anchor to the text origin that centres the card on ``rect``.

    ``card`` is the label's ``(l, r, b, t)`` in text units; ``font_px`` pixels per unit.
    """
    l, r, b, t = card
    cx = (rect[0] + rect[2]) / 2.0 - (l + r) / 2.0 * font_px
    cy = (rect[1] + rect[3]) / 2.0 - (b + t) / 2.0 * font_px
    return (cx - anchor_px[0], cy - anchor_px[1])


def render2d_rect_px(lo, hi, viewport_px) -> Rect:
    """Map a render2d box (x, z in [-1, 1]) to y-up framebuffer pixels."""
    w, h = viewport_px
    return ((lo[0] + 1.0) * 0.5 * w, (lo[1] + 1.0) * 0.5 * h,
            (hi[0] + 1.0) * 0.5 * w, (hi[1] + 1.0) * 0.5 * h)
