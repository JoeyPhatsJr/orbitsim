"""On-disk caching for third-party numba kernels (lamberthub).

lamberthub decorates its solvers with plain ``@jit`` (no ``cache=True``), so every
process re-compiles them on its first call: ~2.4 s of GIL-holding LLVM work in the app
process (stuttering the render thread on the first intercept/porkchop plan of a session)
plus the same again in every pool worker. Switching each dispatcher's disk cache on
*before its first call* makes that a one-time cost per install: later processes load the
compiled code in ~0.1 s. Results are identical -- the cache stores the same machine code.

numba picks the cache location (``__pycache__`` beside the package if writable, else a
per-user cache dir; ``NUMBA_CACHE_DIR`` overrides). Pure helper with no orbitsim
imports; it never raises -- if caching can't be enabled the solver simply compiles
in-process as before.
"""
import importlib
import pkgutil


def enable_jit_disk_cache(modules) -> int:
    """Turn on numba's on-disk cache for every jitted function found in ``modules``.

    Returns how many distinct dispatchers were switched on. Non-dispatcher attributes
    are ignored; a dispatcher whose ``enable_caching`` fails is skipped (degrade, don't
    crash).
    """
    try:
        from numba.core.dispatcher import Dispatcher
    except Exception:
        return 0
    seen = set()
    enabled = 0
    for module in modules:
        for obj in list(vars(module).values()):
            if not isinstance(obj, Dispatcher) or id(obj) in seen:
                continue
            seen.add(id(obj))
            try:
                obj.enable_caching()
            except Exception:
                continue
            enabled += 1
    return enabled


def package_modules(package_name: str) -> list:
    """The package and all its importable submodules (unimportable ones are skipped)."""
    try:
        package = importlib.import_module(package_name)
    except Exception:
        return []
    modules = [package]
    for info in pkgutil.walk_packages(getattr(package, "__path__", []), package_name + "."):
        try:
            modules.append(importlib.import_module(info.name))
        except Exception:
            continue
    return modules
