"""Record goldens with a determinism gate: every scenario runs twice, outputs must match.

    python -m tests.render.app_harness.record [--check-only] [scenario ...]

Each run is a separate process (one ShowBase per process). Writes golden/<name>.json
only when both runs are byte-identical; never run this during the app.py split.
"""
import os
import subprocess
import sys
import tempfile

from tests.render.app_harness.scenarios import SCENARIOS

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
GOLDEN_DIR = os.path.join(HERE, "golden")


def run_driver(name: str, out_path: str, timeout_s: float = 300.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "tests.render.app_harness.driver", name, out_path],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout_s,
    )


def main(argv) -> int:
    check_only = "--check-only" in argv
    names = [a for a in argv[1:] if not a.startswith("--")] or list(SCENARIOS)
    tmp = tempfile.mkdtemp(prefix="orbitsim-golden-")
    failed = []
    for name in names:
        outs = []
        for run in ("a", "b"):
            path = os.path.join(tmp, f"{name}.{run}.json")
            proc = run_driver(name, path)
            if proc.returncode != 0:
                print(f"{name}: run {run} FAILED (exit {proc.returncode})\n{proc.stderr[-2000:]}")
                failed.append(name)
                break
            with open(path, encoding="utf-8") as f:
                outs.append(f.read())
        else:
            if outs[0] != outs[1]:
                print(f"{name}: NOT DETERMINISTIC (runs differ)")
                failed.append(name)
                continue
            print(f"{name}: deterministic")
            if not check_only:
                os.makedirs(GOLDEN_DIR, exist_ok=True)
                with open(os.path.join(GOLDEN_DIR, f"{name}.json"), "w", encoding="utf-8") as f:
                    f.write(outs[0])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
