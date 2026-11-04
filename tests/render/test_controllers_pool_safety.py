"""The harness's process-pool stand-in enforces the real pool's pickling rule.

Every golden scenario runs with it, so a controller that submits a bound method
(which would drag the controller -> OrbitApp -> Panda3D across the pickle) fails
the goldens instead of only failing on a real multi-process run.
"""
import pathlib
import subprocess
import sys
import threading

import pytest

from tests.render.app_harness.driver import InlineProcessExecutor

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _square(x):
    return x * x


class _HoldsLock:
    """Like a controller: its bound methods carry an unpicklable object graph."""

    def __init__(self) -> None:
        self.lock = threading.Lock()

    def work(self):
        return 1


def test_module_level_jobs_run_inline():
    pool = InlineProcessExecutor(max_workers=4)
    assert pool.submit(_square, 7).result() == 49
    assert list(pool.map(_square, [1, 2, 3])) == [1, 4, 9]


def test_pool_stand_in_rejects_bound_methods():
    pool = InlineProcessExecutor()
    with pytest.raises(TypeError):
        pool.submit(_HoldsLock().work)
    with pytest.raises(TypeError):
        list(pool.map(_HoldsLock().work, []))


def test_driver_installs_the_strict_pool_stand_in():
    # The goldens only enforce the pickling rule if the driver actually swaps it in.
    # Separate process: install_determinism() patches process-wide globals.
    code = (
        "import concurrent.futures as cf\n"
        "from tests.render.app_harness.driver import InlineProcessExecutor, install_determinism\n"
        "install_determinism()\n"
        "assert cf.ProcessPoolExecutor is InlineProcessExecutor, cf.ProcessPoolExecutor\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                          timeout=60)
    assert proc.returncode == 0, proc.stderr[-2000:]
