"""The process pool's entry points must be picklable module-level functions."""
import pickle


def test_pool_warmup_is_picklable_module_function():
    from orbitsim.render import process_pool

    assert process_pool.pool_warmup.__module__ == "orbitsim.render.process_pool"
    # Round-trips our own function in-process -- exactly what ProcessPoolExecutor
    # does when dispatching it; no untrusted data is unpickled.
    restored = pickle.loads(pickle.dumps(process_pool.pool_warmup))
    assert restored() is True
