"""Tiny rolling frame-time meter used to guard UI regressions."""
from collections import deque
import numpy as np


class FrameMeter:
    def __init__(self, capacity: int = 240):
        self.samples_ms = deque(maxlen=capacity)

    def add(self, dt_s: float) -> None:
        self.samples_ms.append(max(0.0, float(dt_s) * 1000.0))

    @property
    def p95_ms(self) -> float:
        return float(np.percentile(self.samples_ms, 95)) if self.samples_ms else 0.0

    @property
    def fps(self) -> float:
        if not self.samples_ms:
            return 0.0
        mean = float(np.mean(self.samples_ms))
        return 1000.0 / mean if mean > 0.0 else float("inf")
