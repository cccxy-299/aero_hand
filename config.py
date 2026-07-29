from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    required = ["network", "rates", "dataset", "robot", "cameras", "alignment"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"missing config sections: {missing}")
    for name in ("operator_hz", "control_hz", "state_hz", "camera_hz", "frame_hz"):
        if float(cfg["rates"][name]) <= 0:
            raise ValueError(f"rates.{name} must be positive")
    return cfg


def load_operator_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    required = ["network", "rates", "operator"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"missing operator config sections: {missing}")
    if float(cfg["rates"]["operator_hz"]) <= 0:
        raise ValueError("rates.operator_hz must be positive")
    if int(cfg["operator"]["hand_dof"]) != 7:
        raise ValueError("operator.hand_dof 必须为 7")
    manus = cfg["operator"].get("manus", {})
    for name in ("address", "topic", "operator_id", "calibration_file"):
        if name not in manus:
            raise ValueError(f"operator.manus.{name} is required")
    visual = cfg.get("visualization", {})
    if "update_hz" in visual and float(visual["update_hz"]) <= 0:
        raise ValueError("visualization.update_hz must be positive")
    return cfg


def load_robot_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    required = ["network", "rates", "dataset", "robot", "cameras", "alignment"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"missing robot config sections: {missing}")
    for name in ("control_hz", "state_hz", "camera_hz", "frame_hz"):
        if float(cfg["rates"][name]) <= 0:
            raise ValueError(f"rates.{name} must be positive")
    for side in ("left", "right"):
        if side not in cfg["robot"]:
            raise ValueError(f"robot.{side} is required for bimanual control")
    for camera in ("scene", "wrist_left", "wrist_right"):
        if camera not in cfg["cameras"]:
            raise ValueError(f"cameras.{camera} is required")
    scene = cfg["cameras"]["scene"]
    if scene.get("driver") != "realsense":
        raise ValueError("cameras.scene.driver 必须为 realsense")
    if int(scene.get("fps", cfg["rates"]["camera_hz"])) <= 0:
        raise ValueError("cameras.scene.fps must be positive")
    if int(scene.get("timeout_ms", 1000)) <= 0:
        raise ValueError("cameras.scene.timeout_ms must be positive")
    if int(cfg["robot"]["hand_dof"]) != 7:
        raise ValueError("robot.hand_dof 必须为 7")
    if len(cfg["robot"]["hand_min"]) != 7 or len(cfg["robot"]["hand_max"]) != 7:
        raise ValueError("robot.hand_min/hand_max 必须各包含 7 个值")
    runtime = cfg.get("runtime", {})
    if runtime.get("process_model", "spawn") != "spawn":
        raise ValueError("runtime.process_model 必须为 spawn")
    for name in ("ipc_queue_capacity", "status_queue_capacity"):
        if int(runtime.get(name, 1)) <= 0:
            raise ValueError(f"runtime.{name} must be positive")
    for name in ("shutdown_timeout_s", "writer_shutdown_timeout_s"):
        if float(runtime.get(name, 1)) <= 0:
            raise ValueError(f"runtime.{name} must be positive")
    return cfg
