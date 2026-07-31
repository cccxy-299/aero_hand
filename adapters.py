from __future__ import annotations

import math
import time
from typing import Protocol

import numpy as np

from model import (
    BimanualControlCommand,
    BimanualRobotState,
    BimanualTeleopCommand,
    ControlCommand,
    RobotState,
    TeleopCommand,
)


class OperatorSource(Protocol):
    def read(self) -> BimanualTeleopCommand: ...


class RobotIO(Protocol):
    def initialize(self) -> None: ...
    def home(self) -> None: ...
    def prepare_start(self) -> None: ...
    def activate(self) -> None: ...
    def command(self, value: BimanualControlCommand) -> None: ...
    def read_state(self) -> BimanualRobotState: ...
    def deactivate(self) -> None: ...
    def close(self) -> None: ...


class CameraIO(Protocol):
    def connect(self) -> None: ...
    def read(self) -> tuple[np.ndarray, int]: ...
    def disconnect(self) -> None: ...


class SimOperator:
    """双侧遥操作仿真源，左右轨迹刻意设置为镜像，便于检查接线。"""

    def __init__(self, hand_dof: int = 7) -> None:
        self.seq = 0
        self.hand_dof = hand_dof
        self.start = time.perf_counter()

    def read(self) -> BimanualTeleopCommand:
        t = time.perf_counter() - self.start
        left_pose = np.array(
            [0.25 + 0.04 * math.sin(t), 0.16 + 0.03 * math.sin(t * 0.7), 0.25, 0, 0, 0],
            np.float32,
        )
        right_pose = np.array(
            [0.25 + 0.04 * math.sin(t), -0.16 - 0.03 * math.sin(t * 0.7), 0.25, 0, 0, 0],
            np.float32,
        )
        left_hand = np.full(self.hand_dof, 30 + 20 * math.sin(t), np.float32)
        right_hand = np.full(self.hand_dof, 30 + 20 * math.cos(t), np.float32)
        result = BimanualTeleopCommand(
            TeleopCommand(left_pose, left_hand, self.seq),
            TeleopCommand(right_pose, right_hand, self.seq),
            self.seq,
        )
        self.seq += 1
        return result


class SimRobot:
    """双 Piper + 双 Aerohand 的内存仿真适配器。"""

    def __init__(self, hand_dof: int = 7) -> None:
        self.state = "new"
        self.homed = False
        self._command = BimanualControlCommand(
            ControlCommand(
                np.array([0.25, 0.16, 0.25, 0, 0, 0], np.float32),
                np.zeros(6, np.float32), np.zeros(hand_dof, np.float32), 0,
            ),
            ControlCommand(
                np.array([0.25, -0.16, 0.25, 0, 0, 0], np.float32),
                np.zeros(6, np.float32), np.zeros(hand_dof, np.float32), 0,
            ),
            0,
        )

    @property
    def active(self) -> bool:
        return self.state == "active"

    def status_snapshot(self) -> dict[str, object]:
        return {
            "state": self.state,
            "initialized": self.state not in {"new", "closed"},
            "active": self.active,
            "homed": self.homed,
        }

    def initialize(self) -> None:
        if self.state == "idle_disabled":
            return
        if self.state != "new":
            raise RuntimeError(f"当前状态 {self.state} 不允许 initialize")
        self.state = "idle_disabled"

    def home(self) -> None:
        if self.state != "idle_disabled":
            raise RuntimeError(f"当前状态 {self.state} 不允许 home")
        self.homed = True

    def prepare_start(self) -> None:
        """仿真模式模拟 start 前回到已知机械臂起始姿态。"""
        if self.state != "idle_disabled":
            raise RuntimeError(f"当前状态 {self.state} 不允许 prepare_start")
        self.homed = True
        self.state = "prepared_enabled"

    def activate(self) -> None:
        if self.state not in {"idle_disabled", "prepared_enabled"}:
            raise RuntimeError(f"当前状态 {self.state} 不允许 activate")
        self.state = "active"

    def command(self, value: BimanualControlCommand) -> None:
        if not self.active:
            raise RuntimeError("仿真机器人尚未 activate")
        self._command = value

    def read_state(self) -> BimanualRobotState:
        def one_side(value: ControlCommand) -> RobotState:
            return RobotState(
                value.arm_pose.copy(), value.arm_joints.copy(), value.hand_joints.copy()
            )

        return BimanualRobotState(
            one_side(self._command.left), one_side(self._command.right)
        )

    def deactivate(self) -> None:
        if self.state in {"active", "prepared_enabled"}:
            self.state = "idle_disabled"

    def close(self) -> None:
        self.deactivate()
        self.state = "closed"

    # 兼容旧单进程仿真入口。
    def connect(self) -> None:
        self.initialize()
        self.activate()

    def stop(self) -> None:
        self.deactivate()

    def disconnect(self) -> None:
        self.close()


class SimCamera:
    def __init__(self, width: int, height: int, channel: int) -> None:
        self.width, self.height, self.channel = width, height, channel
        self.counter = 0

    def connect(self) -> None:
        pass

    def read(self) -> tuple[np.ndarray, int]:
        image = np.zeros((self.height, self.width, 3), np.uint8)
        image[:, :, self.channel] = self.counter % 255
        self.counter += 1
        return image, time.perf_counter_ns()

    def disconnect(self) -> None:
        pass
