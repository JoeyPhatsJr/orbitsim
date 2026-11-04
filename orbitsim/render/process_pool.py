"""The app's one persistent spawn-based process pool (see CLAUDE.md multi-core rules).

Everything submitted to this pool must be a module-level, picklable function --
never a bound OrbitApp or controller method (that would drag Panda3D across the
pickle).
"""
import os


def pool_warmup():
    """Trivial picklable task used to prove a spawned worker actually starts."""
    return True


def build_process_pool():
    """Create a spawn-based ``ProcessPoolExecutor`` and confirm a worker starts.

    All heavy CPU work (the planning grid search and the flight-time trajectory
    sampling) is submitted here so it runs across cores instead of GIL-serialising
    on one Python thread. Uses the ``spawn`` context explicitly so a worker never
    inherits the parent's OpenGL/Panda3D state (a fork-after-GL hazard on Linux).

    Returns the pool, or ``None`` if the environment cannot spawn workers (spawn
    blocked, no picklable entry point, sandboxed CI) — the caller then falls back
    to threads. Never raises: honours the project's degrade-don't-crash rule.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    pool = None
    try:
        workers = max(1, min((os.cpu_count() or 2) - 1, 16))
        ctx = multiprocessing.get_context("spawn")
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
        # Warm one worker now so a spawn failure surfaces deterministically here
        # (clean fallback) rather than on the first preview, and the first real
        # sample doesn't pay the whole cold-start cost.
        pool.submit(pool_warmup).result(timeout=30)
        return pool
    except Exception:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        return None
