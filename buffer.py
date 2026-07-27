from __future__ import annotations

import bisect
import threading
from collections import deque
from dataclasses import dataclass
from typing import Generic, TypeVar

from .model import TimedSample

T = TypeVar("T")


@dataclass(frozen=True)
class Selection(Generic[T]):
    sample: TimedSample | None
    lag_ns: int
    valid: bool


class TimeBuffer:
    """Bounded, thread-safe timestamp buffer with causal selection."""

    def __init__(self, capacity: int = 512) -> None:
        self._items: deque[TimedSample] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self.dropped = 0

    def append(self, sample: TimedSample) -> None:
        with self._lock:
            if len(self._items) == self._items.maxlen:
                self.dropped += 1
            if self._items and sample.local_mono_ns < self._items[-1].local_mono_ns:
                values = list(self._items)
                pos = bisect.bisect_right([x.local_mono_ns for x in values], sample.local_mono_ns)
                values.insert(pos, sample)
                self._items = deque(values[-self._items.maxlen :], maxlen=self._items.maxlen)
            else:
                self._items.append(sample)

    def latest(self) -> TimedSample | None:
        with self._lock:
            return self._items[-1] if self._items else None

    def select_before(self, target_ns: int, max_lag_ns: int) -> Selection:
        with self._lock:
            selected = next((x for x in reversed(self._items) if x.local_mono_ns <= target_ns), None)
        if selected is None:
            return Selection(None, 2**63 - 1, False)
        lag = target_ns - selected.local_mono_ns
        return Selection(selected, lag, selected.valid and lag <= max_lag_ns)

