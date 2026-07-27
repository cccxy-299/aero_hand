from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any

PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class Packet:
    kind: str
    seq: int
    source_mono_ns: int
    payload: dict[str, Any]

    def encode(self) -> bytes:
        obj = {
            "v": PROTOCOL_VERSION,
            "kind": self.kind,
            "seq": self.seq,
            "source_mono_ns": self.source_mono_ns,
            "payload": self.payload,
        }
        return json.dumps(obj, separators=(",", ":"), allow_nan=False).encode()

    @classmethod
    def decode(cls, data: bytes) -> "Packet":
        obj = json.loads(data)
        if obj.get("v") != PROTOCOL_VERSION:
            raise ValueError("protocol version mismatch")
        if not isinstance(obj.get("seq"), int) or obj["seq"] < 0:
            raise ValueError("invalid sequence")
        if not isinstance(obj.get("source_mono_ns"), int):
            raise ValueError("invalid source timestamp")
        if not isinstance(obj.get("payload"), dict):
            raise ValueError("invalid payload")
        return cls(obj["kind"], obj["seq"], obj["source_mono_ns"], obj["payload"])


class Sequencer:
    def __init__(self) -> None:
        self._seq = 0

    def packet(self, kind: str, payload: dict[str, Any]) -> Packet:
        packet = Packet(kind, self._seq, time.perf_counter_ns(), payload)
        self._seq += 1
        return packet


def validate_bimanual_payload(payload: dict[str, Any], hand_dof: int) -> None:
    """检查双侧包的结构、维度和有限值，防止坏数据进入控制线程。"""

    for side in ("left", "right"):
        if side not in payload or not isinstance(payload[side], dict):
            raise ValueError(f"teleop payload missing side: {side}")
        arm_pose = payload[side].get("arm_pose")
        hand_joints = payload[side].get("hand_joints")
        if not isinstance(arm_pose, list) or len(arm_pose) != 6:
            raise ValueError(f"{side}.arm_pose must contain 6 values")
        if not isinstance(hand_joints, list) or len(hand_joints) != hand_dof:
            raise ValueError(f"{side}.hand_joints must contain {hand_dof} values")
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in arm_pose + hand_joints):
            raise ValueError(f"{side} contains non-finite values")
