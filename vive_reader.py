from __future__ import annotations

"""独立的双 VIVE Tracker 读取器，不依赖 MANUS 或 Aerohand。"""

import threading
import time
from typing import Any

import pysurvive


class DualViveReader:
    """通过 pysurvive 在后台持续读取左右 VIVE Tracker。"""

    def __init__(
        self,
        tracker_names: dict[str, str],
        survive_args: list[str] | None = None,
    ) -> None:
        self._context = pysurvive.SimpleContext(
            survive_args or ["teleop_collect"]
        )
        self._tracker_to_side = {
            name: side for side, name in tracker_names.items()
        }
        self._latest: dict[str, tuple[list[float], int]] = {}
        self._latest_lighthouses: dict[str, tuple[list[float], int]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="dual-vive", daemon=True
        )
        self._thread.start()

    @staticmethod
    def _name(obj: Any) -> str:
        value = obj.Name()
        return (
            value.decode(errors="replace")
            if isinstance(value, bytes)
            else str(value)
        )

    def _run(self) -> None:
        while self._context.Running() and not self._stop.is_set():
            obj = self._context.NextUpdated()
            if obj is None:
                continue
            name = self._name(obj)
            side = self._tracker_to_side.get(name)
            is_lighthouse = name.startswith("LH")
            if side is None and not is_lighthouse:
                continue
            pose, _ = obj.Pose()
            value = [
                float(pose.Pos[0]),
                float(pose.Pos[1]),
                float(pose.Pos[2]),
                float(pose.Rot[0]),
                float(pose.Rot[1]),
                float(pose.Rot[2]),
                float(pose.Rot[3]),
            ]
            timestamp_ns = time.perf_counter_ns()
            with self._lock:
                if side is not None:
                    self._latest[side] = (value, timestamp_ns)
                else:
                    self._latest_lighthouses[name] = (value, timestamp_ns)

    def snapshot(self) -> dict[str, tuple[list[float], int]]:
        with self._lock:
            return dict(self._latest)

    def lighthouse_snapshot(self) -> dict[str, tuple[list[float], int]]:
        """返回 Lighthouse 位姿快照，仅用于诊断或可视化。"""
        with self._lock:
            return dict(self._latest_lighthouses)

    def close(self) -> None:
        self._stop.set()

