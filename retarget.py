from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from model import BimanualTeleopCommand, TeleopCommand


class BimanualRetargeter(Protocol):
    def retarget(self, payload: dict[str, Any], source_seq: int) -> BimanualTeleopCommand: ...


class PassthroughRetargeter:
    """仿真数据已经是机器人目标，可直接转换为内部命令。"""

    def retarget(self, payload: dict[str, Any], source_seq: int) -> BimanualTeleopCommand:
        commands = {}
        for side in ("left", "right"):
            commands[side] = TeleopCommand(
                np.asarray(payload[side]["arm_pose"], np.float32),
                np.asarray(payload[side]["hand_joints"], np.float32),
                source_seq,
                bool(payload[side].get("valid", True)),
            )
        return BimanualTeleopCommand(commands["left"], commands["right"], source_seq)


@dataclass
class SideRetargetConfig:
    initial_pose: np.ndarray
    scale: float
    fixed_orientation: np.ndarray


class HardwareBimanualRetargeter:
    """设备 B 只映射 VIVE；7维灵巧手指令由设备 A 直接提供。"""

    def __init__(self, sides: dict[str, SideRetargetConfig]) -> None:
        self.sides = sides
        self._vive_reference: dict[str, np.ndarray] = {}

    def _vive_to_arm(self, side: str, vive_pose: list[float]) -> np.ndarray:
        pose = np.asarray(vive_pose, dtype=np.float32)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise ValueError(f"{side} VIVE pose 必须是 position(3)+quaternion_wxyz(4)")
        cfg = self.sides[side]
        if side not in self._vive_reference:
            self._vive_reference[side] = pose[:3].copy()
        delta = pose[:3] - self._vive_reference[side]
        # SteamVR: x右/y上/z后；Piper: x前/y左/z上。
        robot_delta = np.array(
            [-delta[2], -delta[0], delta[1]], dtype=np.float32
        ) * cfg.scale
        result = cfg.initial_pose.copy()
        result[:3] += robot_delta
        result[3:] = cfg.fixed_orientation
        return result

    def retarget(self, payload: dict[str, Any], source_seq: int) -> BimanualTeleopCommand:
        commands: dict[str, TeleopCommand] = {}
        for side in ("left", "right"):
            side_value = payload[side]
            try:
                arm = self._vive_to_arm(side, side_value["vive_pose"])
                # 设备 A 已完成 MANUS 重定向，设备 B 不再重复计算。
                hand = np.asarray(side_value["hand_joints"], dtype=np.float32)
                if hand.shape != (7,) or not np.all(np.isfinite(hand)):
                    raise ValueError(f"{side} hand_joints 必须是有效7维控制指令")
                valid = bool(side_value.get("valid", True))
            except (KeyError, TypeError, ValueError):
                arm = self.sides[side].initial_pose.copy()
                hand = np.zeros(7, np.float32)
                valid = False
            commands[side] = TeleopCommand(arm, hand, source_seq, valid)
        return BimanualTeleopCommand(commands["left"], commands["right"], source_seq)
