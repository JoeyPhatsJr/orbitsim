# Repository Guidelines

## Project Structure & Module Organization

`orbitsim/` is split into strict one-way layers:

- `core/`: pure orbital physics, NumPy `float64`, SI units; never import Panda3D, `sim`, or `render`.
- `sim/`: mutable world state (`World`, `Vessel`, `SimClock`); imports `core` only.
- `render/`: Panda3D application, HUD, cameras, geometry, and visual assets; may import `sim` and `core`.

Tests mirror these layers under `tests/core`, `tests/sim`, and `tests/render`. Design notes and implementation plans live in `docs/`. Runtime downloads, screenshots, saves, and generated files belong in gitignored locations such as `data/` and `saves/`.

## Build, Test, and Development Commands

This is a Windows-first Python project. Always use the repository virtual environment:

```powershell
.venv/Scripts/python -m pytest tests/ -q        # full suite
.venv/Scripts/python -m pytest tests/core -q   # physics-only tests
.venv/Scripts/python -m orbitsim               # sandbox
.venv/Scripts/python -m orbitsim --solar       # solar-system viewer
```

Bare `python` may lack required rendering and astronomy dependencies. Packaging metadata and optional dependency groups are defined in `pyproject.toml`.

## Coding Style & Naming Conventions

Use four-space indentation, type hints for public interfaces, and a 100-character line limit. Black is configured but may not be installed; hand-format when necessary. Physics variables use explicit unit suffixes (`_m`, `_s`, `_rad`, `_mps`). Keep core arrays shape `(3,)`, `float64`, and immutable value objects frozen. Raise `ValueError` for invalid physics input rather than silently clamping.

## Testing Guidelines

Pytest and Hypothesis are the test frameworks. Name files `test_<module>.py` and tests `test_<behavior>`. Write known-answer tests before changing physics, then cover invariants such as energy, angular momentum, vis-viva, and state/element round trips. Never loosen tolerances merely to make a failure pass. Render changes should include pure helper tests where possible and offscreen Panda3D screenshot checks for layout-sensitive behavior.

## Commit & Pull Request Guidelines

Use short imperative or scoped subjects, for example `Fix maneuver preview render-thread stalls` or `HUD 2a: grouped panels`. Stage explicit paths; never use `git add .` or include caches, downloads, screenshots, saves, `CLAUDE.md`, or local settings. Keep commits focused and tests green. Pull requests should explain behavior changes, list verification commands, link the relevant plan or issue, and include before/after screenshots for visual changes.
