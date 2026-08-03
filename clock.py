from __future__ import annotations

import statistics
import threading
from collections import deque


class ClockMapper:
    """Maps device-A monotonic time to device-B monotonic time.

    Offset samples use the NTP formula. The lowest-RTT half is retained and its
    median is used, limiting queueing-delay bias. Drift can later be added
    without changing packet or buffer APIs.
    """

    def __init__(self, window: int = 31) -> None:
        self._samples: deque[tuple[int, int]] = deque(maxlen=window)
        self._offset_ns = 0
        self._lock = threading.Lock()

    def observe_round_trip(
        self, a_send: int, b_recv: int, b_send: int, a_recv: int
    ) -> bool:
        """接收一组 NTP 四时间戳；只有偏移样本实际生效时返回 True。"""
        rtt = (a_recv - a_send) - (b_send - b_recv)
        offset_b_minus_a = ((b_recv - a_send) + (b_send - a_recv)) // 2
        if rtt < 0:
            return False
        with self._lock:
            self._samples.append((rtt, offset_b_minus_a))
            best = sorted(self._samples)[: max(1, len(self._samples) // 2)]
            self._offset_ns = int(statistics.median(x[1] for x in best))
        return True

    def set_offset_for_test(self, offset_ns: int) -> None:
        with self._lock:
            self._offset_ns = offset_ns

    def to_local(self, remote_ns: int) -> int:
        with self._lock:
            return remote_ns + self._offset_ns

    @property
    def offset_ns(self) -> int:
        with self._lock:
            return self._offset_ns

    @property
    def sample_count(self) -> int:
        with self._lock:
            return len(self._samples)
