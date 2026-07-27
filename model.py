from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TimedSample:
    source: str
    seq: int
    source_mono_ns: int
    local_mono_ns: int
    value: Any
    valid: bool = True
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TeleopCommand:
    """单侧遥操作目标；左右侧各持有一个实例。"""

    arm_pose: np.ndarray
    hand_joints: np.ndarray
    source_seq: int
    valid: bool = True


@dataclass(frozen=True)
class RobotState:
    """单侧机器人反馈状态。"""

    arm_pose: np.ndarray
    arm_joints: np.ndarray
    hand_joints: np.ndarray


@dataclass(frozen=True)
class ControlCommand:
    """经过单侧安全门后的最终执行命令。"""

    arm_pose: np.ndarray
    arm_joints: np.ndarray
    hand_joints: np.ndarray
    source_seq: int
    safety_flags: int = 0


@dataclass(frozen=True)
class BimanualTeleopCommand:
    """设备 A 同一数据包内的左右侧遥操作目标。"""

    left: TeleopCommand
    right: TeleopCommand
    source_seq: int


@dataclass(frozen=True)
class BimanualRobotState:
    """设备 B 同一采样时刻读取到的左右机器人状态。"""

    left: RobotState
    right: RobotState


@dataclass(frozen=True)
class BimanualControlCommand:
    """左右安全命令的原子快照，避免两侧使用不同遥操作帧。"""

    left: ControlCommand
    right: ControlCommand
    source_seq: int
