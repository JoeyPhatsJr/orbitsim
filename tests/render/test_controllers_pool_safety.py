"""The harness's process-pool stand-in enforces the real pool's pickling rule.

Every golden scenario runs with it, so a controller that submits a bound method
(which would drag the controller -> OrbitApp -> Panda3D across the pickle) fails
the goldens instead of only failing on a real multi-process run.
"""
import threading

import pytest

from tests.render.app_harness.driver import InlineProcessExecutor


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


def test_no_controller_method_reaches_process_pool():
    pool = InlineProcessExecutor()
    with pytest.raises(TypeError):
        pool.submit(_HoldsLock().work)
    with pytest.raises(TypeError):
        list(pool.map(_HoldsLock().work, []))
