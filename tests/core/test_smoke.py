"""Smoke test: verify core imports."""


def test_imports():
    """Verify orbitsim.core package imports cleanly."""
    import orbitsim.core
    assert orbitsim.core is not None
