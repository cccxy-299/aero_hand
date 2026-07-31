from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Protocol

import numpy as np

from model import BimanualTeleopCommand, TeleopCommand

LOG = logging.getLogger(__name__)
SIDES = ("left", "right")


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
    position_map: np.ndarray


class HardwareBimanualRetargeter:
    """设备 B 映射 VIVE 位置；7维灵巧手指令由设备 A 直接提供。

    ``fixed_orientation`` 由 episode 创建阶段确定。安全默认值是 start 时读取的
    真实法兰姿态，因此首帧不会把当前位置与另一套姿态强行组合。
    """

    def __init__(self, sides: dict[str, SideRetargetConfig]) -> None:
        self.sides = sides
        self._vive_reference: dict[str, np.ndarray] = {}
        self._reference_source_seq: int | None = None
        self._last_mapping: dict[str, dict[str, list[float]]] = {}

    @property
    def references_ready(self) -> bool:
        return all(side in self._vive_reference for side in SIDES)

    def reference_snapshot(self) -> dict[str, list[float]]:
        return {
            side: value.tolist()
            for side, value in self._vive_reference.items()
        }

    @property
    def reference_source_seq(self) -> int | None:
        return self._reference_source_seq

    def mapping_snapshot(self) -> dict[str, dict[str, list[float]]]:
        return {
            side: {
                name: list(values)
                for name, values in snapshot.items()
            }
            for side, snapshot in self._last_mapping.items()
        }

    def _vive_to_arm(self, side: str, vive_pose: list[float]) -> np.ndarray:
        pose = np.asarray(vive_pose, dtype=np.float32)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise ValueError(f"{side} VIVE pose 必须是 position(3)+quaternion_wxyz(4)")
        cfg = self.sides[side]
        if side not in self._vive_reference:
            raise RuntimeError(f"{side} VIVE 相对位置参考尚未建立")
        delta = pose[:3] - self._vive_reference[side]
        # pysurvive 与旧 OpenVR demo 的世界坐标定义/标定方向并不保证一致，
        # 因此轴交换和符号必须由真机标定矩阵显式给出，不能继续硬编码。
        robot_delta = (cfg.position_map @ delta) * cfg.scale
        self._last_mapping[side] = {
            "vive_delta_xyz": delta.tolist(),
            "robot_delta_xyz": robot_delta.tolist(),
        }
        result = cfg.initial_pose.copy()
        result[:3] += robot_delta
        result[3:] = cfg.fixed_orientation
        return result

    def _invalid_result(self, source_seq: int) -> BimanualTeleopCommand:
        """双侧联锁无效：不允许只用同一数据包中的单侧数据建立参考。"""
        commands = {
            side: TeleopCommand(
                self.sides[side].initial_pose.copy(),
                np.zeros(7, np.float32),
                source_seq,
                False,
            )
            for side in SIDES
        }
        return BimanualTeleopCommand(
            commands["left"], commands["right"], source_seq
        )

    def retarget(self, payload: dict[str, Any], source_seq: int) -> BimanualTeleopCommand:
        parsed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        try:
            for side in SIDES:
                side_value = payload[side]
                vive = np.asarray(side_value["vive_pose"], dtype=np.float32)
                hand = np.asarray(side_value["hand_joints"], dtype=np.float32)
                if vive.shape != (7,) or not np.all(np.isfinite(vive)):
                    raise ValueError(
                        f"{side} VIVE pose 必须是有效 position(3)+quaternion_wxyz(4)"
                    )
                if hand.shape != (7,) or not np.all(np.isfinite(hand)):
                    raise ValueError(f"{side} hand_joints 必须是有效7维控制指令")
                if not bool(side_value.get("valid", True)):
                    raise ValueError(f"{side} 遥操作数据标记为无效")
                parsed[side] = (vive, hand)
        except (KeyError, TypeError, ValueError):
            # 旧方案在 Tracker 丢失时不发送新目标；双侧系统进一步做左右联锁。
            return self._invalid_result(source_seq)

        if not self.references_ready:
            # 左右参考必须来自同一个网络序列号。第一帧 delta=0，因此生成的机械臂
            # 目标严格等于 episode start 时读到的真实法兰参考位姿。
            self._vive_reference = {
                side: parsed[side][0][:3].copy()
                for side in SIDES
            }
            self._reference_source_seq = source_seq
            LOG.info(
                "双侧 VIVE 相对位置参考已建立 seq=%s left=%s right=%s",
                source_seq,
                self._vive_reference["left"].tolist(),
                self._vive_reference["right"].tolist(),
            )

        commands: dict[str, TeleopCommand] = {}
        for side in SIDES:
            vive, hand = parsed[side]
            arm = self._vive_to_arm(side, vive.tolist())
            # 设备 A 已完成 MANUS 重定向，设备 B 直接使用7维灵巧手命令。
            commands[side] = TeleopCommand(arm, hand, source_seq, True)
        return BimanualTeleopCommand(commands["left"], commands["right"], source_seq)
