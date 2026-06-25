"""Tests for the procedural star-direction helper (offline, no graphics)."""
import math
from orbitsim.render.skybox import random_star_dirs


def test_returns_n_unit_vectors():
    dirs = random_star_dirs(100, seed=1)
    assert len(dirs) == 100
    for x, y, z in dirs:
        assert abs(math.sqrt(x * x + y * y + z * z) - 1.0) < 1e-9


def test_deterministic_for_seed():
    assert random_star_dirs(50, seed=7) == random_star_dirs(50, seed=7)


def test_different_seeds_differ():
    assert random_star_dirs(50, seed=1) != random_star_dirs(50, seed=2)
