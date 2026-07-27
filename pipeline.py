from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .adapters import CameraIO, RobotIO
from .buffer import TimeBuffer
from .dataset import FrameWriter
from .model import (
    BimanualControlCommand,
    BimanualRobotState,
    ControlCommand,
    TeleopCommand,
    TimedSample,
)
from .protocol import validate_bimanual_payload
from .safety import SafetyGate

LOG = logging.getLogger(__name__)


@dataclass
class Metrics:
    control_ticks: int = 0
    frame_ticks: int = 0
    written_frames: int = 0
    stale_teleop: int = 0
    writer_drops: int = 0
    control_overruns: int = 0


class RobotPipeline:
    def __init__(
        self,
        robot: RobotIO,
        cameras: dict[str, CameraIO],
        writer: FrameWriter,
        safety: dict[str, SafetyGate],
        rates: dict[str, Any],
        alignment: dict[str, Any],
        queue_capacity: int = 64,
    ) -> None:
        if set(cameras) != {"scene", "wrist_left", "wrist_right"}:
            raise ValueError("cameras 必须同时包含 scene、wrist_left、wrist_right")
        if set(safety) != {"left", "right"}:
            raise ValueError("safety 必须同时包含 left、right")
        self.robot = robot
        self.cameras = cameras
        self.writer = writer
        self.safety = safety
        capacity = int(alignment["buffer_capacity"])
        # 每路图像拥有独立缓冲，慢相机不会阻塞另一只腕部相机或控制线程。
        buffer_names = (
            "teleop", "robot_state", "control_action",
            "scene", "wrist_left", "wrist_right",
        )
        self.buffers = {name: TimeBuffer(capacity) for name in buffer_names}
        self.rates, self.alignment = rates, alignment
        self.write_queue: queue.Queue[dict[str, Any] | None] = queue.Queue(queue_capacity)
        self.stop_event = threading.Event()
        self.metrics = Metrics()
        self.threads: list[threading.Thread] = []

    def ingest_teleop(self, sample: TimedSample) -> None:
        self.buffers["teleop"].append(sample)

    def start(self) -> None:
        self.robot.connect()
        targets = [
            ("control", self._control_loop), ("state", self._state_loop),
            ("frame", self._frame_loop), ("writer", self._writer_loop),
        ]
        # 三个相机各占一个采集线程，不在控制线程中解码图像。
        targets.extend(
            (name, lambda camera_name=name: self._camera_loop(
                camera_name, self.cameras[camera_name]
            ))
            for name in ("scene", "wrist_left", "wrist_right")
        )
        self.threads = [threading.Thread(name=n, target=f, daemon=True) for n, f in targets]
        for thread in self.threads:
            thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        for thread in self.threads:
            if thread.name != "writer":
                thread.join(timeout=3)
        try:
            self.write_queue.put(None, timeout=1)
        except queue.Full:
            pass
        for thread in self.threads:
            if thread.name == "writer":
                thread.join(timeout=10)
        self.robot.stop()
        self.robot.disconnect()

    def _periodic(self, hz: float, callback: Any) -> None:
        period = int(1e9 / hz)
        deadline = time.perf_counter_ns()
        while not self.stop_event.is_set():
            deadline += period
            callback(deadline)
            remaining = deadline - time.perf_counter_ns()
            if remaining > 0:
                time.sleep(remaining / 1e9)
            elif callback == self._control_tick:
                self.metrics.control_overruns += 1

    def _control_loop(self) -> None:
        self._periodic(float(self.rates["control_hz"]), self._control_tick)

    def _control_tick(self, now_ns: int) -> None:
        selected = self.buffers["teleop"].latest()
        if selected is None:
            self.metrics.stale_teleop += 1
            self.metrics.control_ticks += 1
            return
        value = selected.value
        hand_dof = len(self.safety["left"].cfg.hand_min)
        try:
            validate_bimanual_payload(value, hand_dof)
        except ValueError:
            # 坏包只计作遥操作失效，不允许异常杀死实时控制线程。
            LOG.warning("丢弃非法双侧遥操作包 seq=%s", selected.seq)
            self.metrics.stale_teleop += 1
            self.metrics.control_ticks += 1
            return
        safe_by_side: dict[str, ControlCommand] = {}
        for side in ("left", "right"):
            side_value = value[side]
            command = TeleopCommand(
                np.asarray(side_value["arm_pose"], np.float32),
                np.asarray(side_value["hand_joints"], np.float32),
                selected.seq,
                bool(side_value.get("valid", True)),
            )
            safe_by_side[side] = self.safety[side].apply(
                command, now_ns - selected.local_mono_ns
            )
        # 左右命令源于同一个网络序列号，作为一份原子命令提交给硬件层。
        safe = BimanualControlCommand(
            safe_by_side["left"], safe_by_side["right"], selected.seq
        )
        self.robot.command(safe)
        self.buffers["control_action"].append(TimedSample(
            "control_action", selected.seq, selected.source_mono_ns, time.perf_counter_ns(), safe
        ))
        if safe.left.safety_flags & 1 or safe.right.safety_flags & 1:
            self.metrics.stale_teleop += 1
        self.metrics.control_ticks += 1

    def _state_loop(self) -> None:
        def tick(_: int) -> None:
            stamp = time.perf_counter_ns()
            self.buffers["robot_state"].append(TimedSample("robot_state", self.metrics.control_ticks, stamp, stamp, self.robot.read_state()))
        self._periodic(float(self.rates["state_hz"]), tick)

    def _camera_loop(self, name: str, camera: CameraIO) -> None:
        seq = 0
        def tick(_: int) -> None:
            nonlocal seq
            image = camera.read()
            stamp = time.perf_counter_ns()
            self.buffers[name].append(TimedSample(name, seq, stamp, stamp, image))
            seq += 1
        self._periodic(float(self.rates["camera_hz"]), tick)

    def _frame_loop(self) -> None:
        self._periodic(float(self.rates["frame_hz"]), self._frame_tick)

    def _frame_tick(self, target_ns: int) -> None:
        max_lag = int(float(self.alignment["max_lag_ms"]) * 1e6)
        names = (
            "scene", "wrist_left", "wrist_right",
            "robot_state", "control_action", "teleop",
        )
        selected = {name: self.buffers[name].select_before(target_ns, max_lag) for name in names}
        if any(x.sample is None for x in selected.values()):
            return
        state: BimanualRobotState = selected["robot_state"].sample.value
        action: BimanualControlCommand = selected["control_action"].sample.value

        def state_vector(side: str) -> np.ndarray:
            value = getattr(state, side)
            return np.concatenate(
                (value.arm_pose, value.arm_joints, value.hand_joints)
            )

        def action_vector(side: str) -> np.ndarray:
            value = getattr(action, side)
            return np.concatenate(
                (value.arm_pose, value.arm_joints, value.hand_joints)
            )

        frame = {
            # 数据向量固定按 left → right 排列，feature names 中也使用相同顺序。
            "observation.state": np.concatenate(
                (state_vector("left"), state_vector("right"))
            ).astype(np.float32),
            "action": np.concatenate(
                (action_vector("left"), action_vector("right"))
            ).astype(np.float32),
            "observation.images.scene": selected["scene"].sample.value,
            "observation.images.wrist_left": selected["wrist_left"].sample.value,
            "observation.images.wrist_right": selected["wrist_right"].sample.value,
            "alignment.lag_s": np.array([selected[n].lag_ns / 1e9 for n in names], np.float32),
            "alignment.valid": np.array([selected[n].valid for n in names], np.float32),
            "diagnostics.source_seq": np.array([action.source_seq], np.int64),
            "diagnostics.safety_flags": np.array(
                [action.left.safety_flags, action.right.safety_flags], np.int64
            ),
        }
        try:
            self.write_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.write_queue.get_nowait()
                self.write_queue.put_nowait(frame)
                self.metrics.writer_drops += 1
            except queue.Empty:
                pass
        self.metrics.frame_ticks += 1

    def _writer_loop(self) -> None:
        while True:
            item = self.write_queue.get()
            if item is None:
                break
            try:
                self.writer.add_frame(item)
                self.metrics.written_frames += 1
            except Exception:
                LOG.exception("dataset writer failed")
                self.stop_event.set()
                break
        self.writer.close()
