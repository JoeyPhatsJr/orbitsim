import pytest

from orbitsim.core import nbody


@pytest.fixture(autouse=True)
def _isolate_ephemeris_cache(monkeypatch):
    """Give every test an empty per-frame JPL ephemeris cache.

    ``refresh_ephemeris_cache`` rebinds a module global in ``core.nbody``. Without
    this reset, any test that populates it makes every later test see real DE440
    planet positions instead of the circular fallbacks it expects, which produces
    failures that depend on test order. monkeypatch restores the original (empty)
    object on teardown, even if the test rebinds the name.
    """
    monkeypatch.setattr(nbody, "_ephemeris_cache", {})
