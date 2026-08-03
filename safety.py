from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model import ControlCommand, TeleopCommand

FLAG_STALE = 1
FLAG_WORKSPACE = 2
FLAG_RATE = 4
FLAG_ESTOP = 8


def effective_control_step_m(
    configured_hardware_step_m: float,
    control_hz: float,
    arm_command_hz: float,
) -> float:
    """把每次硬件 move_p 步长换算为控制循环每 tick 的步长。"""
    if configured_hardware_step_m <= 0 or control_hz <= 0 or arm_command_hz <= 0:
        raise ValueError("步长、control_hz 和 arm_command_hz 必须为正数")
    return configured_hardware_step_m * min(1.0, arm_command_hz / control_hz)


@dataclass
class SafetyConfig:
    workspace_min: np.ndarray
    workspace_max: np.ndarray
    max_linear_step_m: float
    hand_min: np.ndarray
    hand_max: np.ndarray
    stale_timeout_ns: int


class SafetyGate:
    """单侧安全门。

    双侧系统为左右臂各创建一个实例，避免某一侧的历史位姿或急停状态污染另一侧。
    上层仍可同时触发两个实例的 emergency_stop() 实现整机急停。
    """

    def __init__(self, cfg: SafetyConfig, initial_pose: np.ndarray) -> None:
        self.cfg = cfg
        self._last_pose = initial_pose.astype(np.float32).copy()
        if self._last_pose.shape != (6,) or not np.all(np.isfinite(self._last_pose)):
            raise ValueError("安全门初始法兰位姿必须是有效6维数组")
        if np.any(self._last_pose[:3] < cfg.workspace_min) or np.any(
            self._last_pose[:3] > cfg.workspace_max
        ):
            raise ValueError(
                "安全门初始法兰位置不在工作空间内："
                f"position={self._last_pose[:3].tolist()}, "
                f"workspace=[{cfg.workspace_min.tolist()}, "
                f"{cfg.workspace_max.tolist()}]"
            )
        self._estop = False

    def emergency_stop(self) -> None:
        self._estop = True

    def reset(self) -> None:
        self._estop = False

    def apply(self, command: TeleopCommand, age_ns: int) -> ControlCommand:
        flags = 0
        pose = np.asarray(command.arm_pose, dtype=np.float32).copy()
        hand = np.asarray(command.hand_joints, dtype=np.float32).copy()
        if self._estop or not command.valid:
            flags |= FLAG_ESTOP
        if age_ns > self.cfg.stale_timeout_ns:
            flags |= FLAG_STALE
        if flags & (FLAG_ESTOP | FLAG_STALE):
            pose = self._last_pose.copy()
        else:
            clipped = np.clip(pose[:3], self.cfg.workspace_min, self.cfg.workspace_max)
            if not np.allclose(clipped, pose[:3]):
                flags |= FLAG_WORKSPACE
            pose[:3] = clipped
            delta = pose[:3] - self._last_pose[:3]
            norm = float(np.linalg.norm(delta))
            if norm > self.cfg.max_linear_step_m:
                pose[:3] = self._last_pose[:3] + delta * self.cfg.max_linear_step_m / norm
                flags |= FLAG_RATE
            self._last_pose = pose.copy()
        hand = np.clip(hand, self.cfg.hand_min, self.cfg.hand_max)
        return ControlCommand(pose, np.zeros(6, np.float32), hand, command.source_seq, flags)
