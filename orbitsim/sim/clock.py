"""Simulation time and time-warp."""
import bisect


class SimClock:
    """Owns simulation time (seconds past J2000, TDB) and the time-warp rate.

    Parameters
    ----------
    sim_time_s : float
        Initial simulation time [s past J2000 TDB].
    warp : float
        Sim seconds advanced per real second. Must be one of WARP_STEPS.
    """

    WARP_STEPS = [1, 5, 10, 50, 100, 1_000, 10_000, 100_000, 1_000_000]

    def __init__(self, sim_time_s: float = 0.0, warp: float = 1.0) -> None:
        self.sim_time_s = float(sim_time_s)
        self.warp = float(warp)

    def advance(self, real_dt_s: float) -> float:
        """Advance sim time by real_dt_s * warp; return the sim_dt applied [s]."""
        sim_dt = real_dt_s * self.warp
        self.sim_time_s += sim_dt
        return sim_dt

    def _current_index(self) -> int:
        # Snap an arbitrary warp to the nearest table index.
        steps = self.WARP_STEPS
        pos = bisect.bisect_left(steps, self.warp)
        if pos >= len(steps):
            return len(steps) - 1
        if pos > 0 and abs(steps[pos - 1] - self.warp) <= abs(steps[pos] - self.warp):
            return pos - 1
        return pos

    def warp_up(self) -> None:
        """Increase warp to the next allowed step (clamped at max)."""
        idx = min(self._current_index() + 1, len(self.WARP_STEPS) - 1)
        self.warp = float(self.WARP_STEPS[idx])

    def warp_down(self) -> None:
        """Decrease warp to the previous allowed step (clamped at min)."""
        idx = max(self._current_index() - 1, 0)
        self.warp = float(self.WARP_STEPS[idx])
