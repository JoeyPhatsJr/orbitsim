"""Headless profiler: frame-time stats + hotspots for a fixed OrbitApp workload.

    python -m tests.render.app_harness.profile <workload> [--frames N] [--warmup N]
        [--inline] [--top K] [--json out.json]

The sim clock is fixed-step (60 fps), so every run simulates exactly the same thing;
frame times are real wall time (time.perf_counter) around each taskMgr.step().

By default the real thread/process pools are used, so off-thread work (trajectory
sampling, planning) costs what it costs in the game. ``--inline`` runs that work on the
render thread instead, which inflates frame times but puts it in the hotspot list.
"""
import argparse
import cProfile
import json
import os
import pstats
import sys
import time

from tests.render.app_harness import hooks


def _launch(h):
    h.press("LAUNCH SANDBOX")


def _idle_setup(h):
    _launch(h)


def _idle_frame(h, i):
    pass


def _high_warp_setup(h):
    _launch(h)
    h.send("0")            # max safe warp


def _burn_setup(h):
    _launch(h)
    h.send("z")            # full throttle (warp locks to 1x)


def _zoom_frame(h, i):
    if i % 30 == 0:
        h.send("wheel_up" if (i // 30) % 2 == 0 else "wheel_down")


def _node_setup(h):
    _launch(h)
    hooks.select_target(h.app, "Moon")
    h.step(5)
    h.send("i")            # plan an intercept: node + closest approach + preview


def _node_drag_setup(h):
    """A burn added from the orbit menu, its prograde handle grabbed (plan 3)."""
    _launch(h)
    h.step(30)
    for _ in range(600):                      # real pools sample the line asynchronously
        if hooks.orbit_line_ready(h.app):
            break
        h.step(1)
    hooks.set_mouse_px(h.app, hooks.orbit_point_px(h.app, 40))
    h.step(1)
    h.send("mouse1")
    h.send("mouse1-up")
    h.step(1)
    h.press("Add burn here")
    h.step(5)
    start, node = hooks.handle_px(h.app, "pro+"), hooks.node_px(h.app)
    length = ((start[0] - node[0]) ** 2 + (start[1] - node[1]) ** 2) ** 0.5
    h._drag = (start, ((start[0] - node[0]) / length, (start[1] - node[1]) / length))
    hooks.set_mouse_px(h.app, start)
    h.step(1)
    h.send("mouse1")


def _node_drag_frame(h, i):
    """Keep pulling the handle back and forth: the estimate redraws every frame."""
    (sx, sy), (ux, uy) = h._drag
    d = (i % 60) * 2.0
    hooks.set_mouse_px(h.app, (sx + ux * d, sy + uy * d))


def _solar_setup(h):
    h.press("OPEN SOLAR VIEWER")


# name -> (mode, setup(h), per_frame(h, i))
WORKLOADS = {
    "idle": ("sandbox", _idle_setup, _idle_frame),
    "high_warp": ("sandbox", _high_warp_setup, _idle_frame),
    "burn": ("sandbox", _burn_setup, _idle_frame),
    "zoom": ("sandbox", _idle_setup, _zoom_frame),
    "node_planned": ("sandbox", _node_setup, _idle_frame),
    "zoom_node_planned": ("sandbox", _node_setup, _zoom_frame),
    "solar": ("solar", _solar_setup, _idle_frame),
    "node_drag": ("sandbox", _node_drag_setup, _node_drag_frame),
}


def _percentile(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    k = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def _hotspots(profiler, top, sort_key="tottime_ms", only_project=False):
    stats = pstats.Stats(profiler)
    rows = []
    for (filename, line, func), (cc, nc, tt, ct, _callers) in stats.stats.items():
        name = os.path.relpath(filename) if filename.startswith(os.sep) else filename
        if only_project and not name.startswith("orbitsim"):
            continue
        rows.append({
            "function": f"{name}:{line}({func})",
            "calls": nc,
            "tottime_ms": tt * 1000.0,
            "cumtime_ms": ct * 1000.0,
        })
    rows.sort(key=lambda r: r[sort_key], reverse=True)
    return rows[:top]


def run(workload, frames, warmup, inline, top, sort_key="tottime_ms", only_project=False):
    from tests.render.app_harness.driver import Harness, build_app, install_determinism
    install_determinism(inline_executors=inline)
    mode, setup, per_frame = WORKLOADS[workload]
    app = build_app(mode)
    h = Harness(app)
    setup(h)
    h.step(warmup)
    profiler = cProfile.Profile()
    times = []
    for i in range(frames):
        per_frame(h, i)
        t0 = time.perf_counter()
        profiler.enable()
        h.step(1)
        profiler.disable()
        times.append((time.perf_counter() - t0) * 1000.0)
    app.destroy()
    s = sorted(times)
    return {
        "workload": workload,
        "inline_executors": inline,
        "frames": frames,
        "frame_ms": {
            "mean_ms": sum(times) / len(times),
            "p50_ms": _percentile(s, 0.50),
            "p95_ms": _percentile(s, 0.95),
            "max_ms": s[-1],
        },
        "hotspots": _hotspots(profiler, top, sort_key, only_project),
    }


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("workload")
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--warmup", type=int, default=60)
    p.add_argument("--inline", action="store_true", help="run background jobs on the render thread")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--json", dest="json_path")
    p.add_argument("--sort", choices=("tottime", "cumtime"), default="tottime")
    p.add_argument("--project", action="store_true", help="only list orbitsim/ functions")
    args = p.parse_args(argv[1:])
    if args.workload not in WORKLOADS:
        print(f"unknown workload {args.workload!r}; choose from {', '.join(WORKLOADS)}", file=sys.stderr)
        return 2
    report = run(args.workload, args.frames, args.warmup, args.inline, args.top,
                 f"{args.sort}_ms", args.project)
    fm = report["frame_ms"]
    print(f"{report['workload']}{' (inline)' if args.inline else ''}: {report['frames']} frames, "
          f"frame ms mean {fm['mean_ms']:.2f}  p50 {fm['p50_ms']:.2f}  p95 {fm['p95_ms']:.2f}  "
          f"max {fm['max_ms']:.2f}  (~{1000.0 / fm['mean_ms']:.0f} fps)")
    print(f"{'tottime ms':>11} {'cumtime ms':>11} {'calls':>8}  function")
    for r in report["hotspots"]:
        print(f"{r['tottime_ms']:11.1f} {r['cumtime_ms']:11.1f} {r['calls']:8d}  {r['function']}")
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
