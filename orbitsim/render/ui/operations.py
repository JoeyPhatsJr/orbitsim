"""Frame-budgeted operation state for long gameplay work."""
from dataclasses import dataclass


@dataclass(frozen=True)
class OperationStatus:
    label: str
    progress: float
    running: bool
    cancelled: bool = False


class OperationController:
    """Advance a generator incrementally without blocking Panda's render task."""

    def __init__(self):
        self._iterator = None
        self._label = ""
        self._progress = 0.0
        self._cancelled = False
        self.result = None

    @property
    def status(self):
        return OperationStatus(self._label, self._progress, self._iterator is not None,
                               self._cancelled)

    def start(self, label, iterator):
        self.cancel()
        self._label = label
        self._iterator = iter(iterator)
        self._progress = 0.0
        self._cancelled = False
        self.result = None

    def tick(self, steps=1):
        for _ in range(steps):
            if self._iterator is None:
                break
            try:
                value = next(self._iterator)
                if value is not None:
                    self._progress = max(0.0, min(1.0, float(value)))
            except StopIteration as done:
                self.result = done.value
                self._progress = 1.0
                self._iterator = None
        return self.status

    def cancel(self):
        if self._iterator is not None:
            close = getattr(self._iterator, "close", None)
            if close is not None:
                close()
            self._cancelled = True
        self._iterator = None
