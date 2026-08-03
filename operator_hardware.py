from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from common_cls import ManusZmqReader, RetargetingConfig
from manus_aero_retargeting_two_hands import (
    ManusToAeroRetargeter,
)
from thumb_cmc_calibrator import ThumbCMCCalibrator
from vive_reader import DualViveReader


class HardwareOperatorSource:
    """设备 A 采集双侧数据，并将 MANUS 重定向为可直接执行的7维手指令。"""

    def __init__(self, cfg: dict[str, Any]) -> None:
        manus_cfg = cfg["manus"]
        self._manus = ManusZmqReader(manus_cfg["address"], manus_cfg["topic"])
        calibration_file = Path(manus_cfg["calibration_file"])
        if not calibration_file.is_absolute():
            calibration_file = Path(__file__).resolve().parents[1] / calibration_file
        left_calibrator = ThumbCMCCalibrator(
            operator_id=manus_cfg["operator_id"],
            side="left",
            calibration_file=str(calibration_file),
        )
        right_calibrator = ThumbCMCCalibrator(
            operator_id=manus_cfg["operator_id"],
            side="right",
            calibration_file=str(calibration_file),
        )
        # 启动阶段强制加载左右拇指标定；缺失时直接拒绝启动，避免错误动作。
        left_calibrator.load_profile()
        right_calibrator.load_profile()
        self._hand_retargeter = ManusToAeroRetargeter(
            RetargetingConfig(
                compact_7dof=True,
                output_degrees=True,
                enable_filter=bool(manus_cfg.get("enable_filter", True)),
                filter_alpha=float(manus_cfg.get("filter_alpha", 0.8)),
            ),
            left_calibrator=left_calibrator,
            right_calibrator=right_calibrator,
        )
        vive_cfg = cfg["vive"]
        self._vive = DualViveReader(
            {"left": vive_cfg["left_tracker_name"], "right": vive_cfg["right_tracker_name"]},
            vive_cfg.get("survive_args"),
        )
        self._max_age_ns = int(float(cfg.get("max_source_age_ms", 100)) * 1e6)
        self._last_hand = {side: np.zeros(7, np.float32) for side in ("left", "right")}
        self._last_hand_ns = 0

    def read_payload(self) -> dict[str, Any]:
        frame = self._manus.read_latest(timeout_ms=5)
        if frame is not None:
            try:
                # 参考实现内部依次调用 _retarget_from_ergonomics_two_hands、
                # 关节限位/滤波以及 to_compact_7dof，输出左右各7维。
                values = self._hand_retargeter.retarget_two_hands(frame)
                for side in ("left", "right"):
                    command = np.asarray(values[side], dtype=np.float32)
                    if command.shape != (7,) or not np.all(np.isfinite(command)):
                        raise ValueError(f"{side} MANUS重定向结果不是有效7维指令")
                    self._last_hand[side] = command
                self._last_hand_ns = time.perf_counter_ns()
            except (KeyError, TypeError, ValueError, RuntimeError, IndexError):
                # 保留上一条有效手指令，并通过 valid=false 阻止设备B执行新动作。
                pass
        vive = self._vive.snapshot()
        now = time.perf_counter_ns()
        payload: dict[str, Any] = {}
        for side in ("left", "right"):
            vive_value = vive.get(side)
            hand_age_ns = (
                now - self._last_hand_ns
                if self._last_hand_ns > 0
                else None
            )
            vive_age_ns = (
                now - vive_value[1]
                if vive_value is not None
                else None
            )
            valid = (
                vive_value is not None
                and self._last_hand_ns > 0
                and hand_age_ns is not None
                and hand_age_ns <= self._max_age_ns
                and vive_age_ns is not None
                and vive_age_ns <= self._max_age_ns
            )
            invalid_reasons: list[str] = []
            if self._last_hand_ns <= 0:
                invalid_reasons.append("manus_not_ready")
            elif hand_age_ns is not None and hand_age_ns > self._max_age_ns:
                invalid_reasons.append(
                    f"manus_stale({hand_age_ns / 1e6:.1f}ms)"
                )
            if vive_value is None:
                invalid_reasons.append("vive_missing")
            elif vive_age_ns is not None and vive_age_ns > self._max_age_ns:
                invalid_reasons.append(
                    f"vive_stale({vive_age_ns / 1e6:.1f}ms)"
                )
            payload[side] = {
                # 设备 B 可直接将该7维命令交给 Aerohand 安全门和驱动层。
                "hand_joints": self._last_hand[side].tolist(),
                "vive_pose": vive_value[0] if vive_value else [0.0] * 7,
                "valid": valid,
                # 额外字段不会进入控制指令，仅用于设备 B 启动门控和日志诊断。
                "invalid_reason": (
                    "ready" if valid else "+".join(invalid_reasons)
                ),
                "source_age_ms": {
                    "manus": (
                        round(max(0, hand_age_ns) / 1e6, 3)
                        if hand_age_ns is not None
                        else None
                    ),
                    "vive": (
                        round(max(0, vive_age_ns) / 1e6, 3)
                        if vive_age_ns is not None
                        else None
                    ),
                },
            }
        return payload

    def close(self) -> None:
        self._manus.close()
        self._vive.close()

    @property
    def vive_reader(self) -> DualViveReader:
        """暴露只读 VIVE 快照接口给可选 GUI。"""
        return self._vive
