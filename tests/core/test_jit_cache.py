"""lamberthub's numba kernels must compile into an on-disk cache.

Without it every process (the app and each pool worker) re-JITs the Lambert solver on
its first call (~2.4 s of GIL-holding compile), which stutters the render thread on the
first intercept/porkchop plan of every session.
"""
import os
import subprocess
import sys
import types

from orbitsim.core.jit_cache import enable_jit_disk_cache


def test_lambert_solve_writes_numba_disk_cache(tmp_path):
    # Fresh interpreter: the cache must be switched on by importing our transfers module,
    # before anything compiles. NUMBA_CACHE_DIR redirects numba's cache files here.
    env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path))
    code = (
        "import numpy as np\n"
        "from orbitsim.core.transfers import lambert\n"
        "lambert(np.array([7e6, 0.0, 0.0]), np.array([0.0, 3.8e8, 1e7]),\n"
        "        3 * 86400.0, 3.986004418e14)\n"
    )
    subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=300)
    index_files = list(tmp_path.rglob("*.nbi"))
    assert any("izzo" in p.name for p in index_files), sorted(p.name for p in index_files)


def test_enable_jit_disk_cache_counts_dispatchers_and_never_raises():
    from numba import njit

    @njit
    def _square(x):
        return x * x

    @njit
    def _cube(x):
        return x * x * x

    def _refuse():
        raise RuntimeError("cannot cache here (e.g. unwritable cache dir)")

    _cube.enable_caching = _refuse   # a real Dispatcher whose caching fails

    good = types.ModuleType("fake_pkg_good")
    good._square = _square
    good.not_jitted = len            # non-dispatchers are ignored
    assert enable_jit_disk_cache([good]) == 1

    bad = types.ModuleType("fake_pkg_bad")
    bad._cube = _cube
    assert enable_jit_disk_cache([bad]) == 0   # degrades, doesn't crash
    assert _cube(2) == 8                       # and the function still works uncached
