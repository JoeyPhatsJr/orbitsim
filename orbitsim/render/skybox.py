"""Star background: a textured inside-out sky sphere, or a procedural point field
when the star texture is unavailable."""
import numpy as np


def random_star_dirs(n: int, seed: int = 0):
    """Return n unit direction vectors uniformly on the sphere (deterministic per seed)."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return [tuple(float(c) for c in row) for row in v]
