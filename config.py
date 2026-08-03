from __future__ import annotations

import math
import logging
from pathlib import Path
from typing import Any

import yaml


LOG = logging.getLogger(__name__)


def _validate_zmq_network(cfg: dict[str, Any], require_robot_host: bool) -> None:
    network = cfg.get("network")
    if not isinstance(network, dict):
        raise ValueError("network 配置必须是对象")
    if str(network.get("transport", "zmq")).lower() != "zmq":
        raise ValueError("network.transport 必须为 zmq")
    data_port = int(network.get("data_port", 0))
    sync_port = int(network.get("sync_port", data_port + 1))
    if not 1 <= data_port <= 65535 or not 1 <= sync_port <= 65535:
        raise ValueError("network.data_port/sync_port 必须在 [1, 65535]")
    if data_port == sync_port:
        raise ValueError("network.data_port 与 sync_port 不能相同")
    network["sync_port"] = sync_port
    if int(network.get("max_packet_bytes", 65507)) <= 0:
        raise ValueError("network.max_packet_bytes 必须大于0")
    if float(network.get("startup_timeout_s", 5.0)) <= 0:
        raise ValueError("network.startup_timeout_s 必须大于0")
    if require_robot_host and not str(network.get("robot_host", "")).strip():
        raise ValueError("network.robot_host 不能为空")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    required = ["network", "rates", "dataset", "robot", "cameras", "alignment"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"missing config sections: {missing}")
    _validate_zmq_network(cfg, require_robot_host=False)
    for name in ("operator_hz", "control_hz", "state_hz", "camera_hz", "frame_hz"):
        if float(cfg["rates"][name]) <= 0:
            raise ValueError(f"rates.{name} must be positive")
    dataset_root = Path(cfg["dataset"]["root"])
    if not dataset_root.is_absolute():
        dataset_root = Path(__file__).resolve().parent / dataset_root
    cfg["dataset"]["root"] = str(dataset_root.resolve())
    return cfg


def load_operator_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    required = ["network", "rates", "operator"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"missing operator config sections: {missing}")
    _validate_zmq_network(cfg, require_robot_host=True)
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
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    required = ["network", "rates", "dataset", "robot", "cameras", "alignment"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"missing robot config sections: {missing}")
    _validate_zmq_network(cfg, require_robot_host=False)
    for name in ("control_hz", "state_hz", "camera_hz", "frame_hz"):
        if float(cfg["rates"][name]) <= 0:
            raise ValueError(f"rates.{name} must be positive")
    for side in ("left", "right"):
        if side not in cfg["robot"]:
            raise ValueError(f"robot.{side} is required for bimanual control")
        side_cfg = cfg["robot"][side]
        if "command_output_enabled" in side_cfg and not isinstance(
            side_cfg["command_output_enabled"], bool
        ):
            raise ValueError(
                f"robot.{side}.command_output_enabled 必须为布尔值"
            )
        for name in ("initial_pose",):
            value = side_cfg.get(name)
            if not isinstance(value, list) or len(value) != 6:
                raise ValueError(f"robot.{side}.{name} 必须包含 6 个值")
            if not all(math.isfinite(float(item)) for item in value):
                raise ValueError(f"robot.{side}.{name} 必须全部为有限值")
        for name in ("workspace_min", "workspace_max", "fixed_orientation"):
            value = side_cfg.get(name)
            if not isinstance(value, list) or len(value) != 3:
                raise ValueError(f"robot.{side}.{name} 必须包含 3 个值")
            if not all(math.isfinite(float(item)) for item in value):
                raise ValueError(f"robot.{side}.{name} 必须全部为有限值")
        position_map = side_cfg.get("vive_to_robot_matrix")
        if (
            not isinstance(position_map, list)
            or len(position_map) != 3
            or any(not isinstance(row, list) or len(row) != 3 for row in position_map)
        ):
            raise ValueError(
                f"robot.{side}.vive_to_robot_matrix 必须是3x3矩阵"
            )
        matrix = [[float(item) for item in row] for row in position_map]
        if not all(math.isfinite(item) for row in matrix for item in row):
            raise ValueError(
                f"robot.{side}.vive_to_robot_matrix 必须全部为有限值"
            )
        # 标定结果可以是任意正交坐标映射，而不只是0/±1轴交换矩阵。
        # det=-1 表示坐标约定含镜像，对当前位置增量仍然有效。
        gram = [
            [
                sum(matrix[k][i] * matrix[k][j] for k in range(3))
                for j in range(3)
            ]
            for i in range(3)
        ]
        orthogonality_error = math.sqrt(
            sum(
                (gram[i][j] - (1.0 if i == j else 0.0)) ** 2
                for i in range(3)
                for j in range(3)
            )
        )
        determinant = (
            matrix[0][0]
            * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
            - matrix[0][1]
            * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
            + matrix[0][2]
            * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
        )
        if orthogonality_error > 1e-3 or abs(abs(determinant) - 1.0) > 1e-3:
            raise ValueError(
                f"robot.{side}.vive_to_robot_matrix 必须是正交坐标映射；"
                f"orthogonality_error={orthogonality_error:.3e}, "
                f"det={determinant:.6f}"
            )
        if float(side_cfg.get("vive_scale", 0)) <= 0:
            raise ValueError(f"robot.{side}.vive_scale 必须大于0")
        orientation_mode = str(
            side_cfg.get("orientation_mode", "current_on_start")
        ).lower()
        if orientation_mode not in {"current_on_start", "configured_fixed"}:
            raise ValueError(
                f"robot.{side}.orientation_mode 必须为 "
                "current_on_start 或 configured_fixed"
            )
        workspace_min = [float(value) for value in side_cfg["workspace_min"]]
        workspace_max = [float(value) for value in side_cfg["workspace_max"]]
        if any(lower >= upper for lower, upper in zip(workspace_min, workspace_max)):
            raise ValueError(f"robot.{side} 工作空间上下界非法")
        if "home_pose" in side_cfg:
            home_pose = side_cfg["home_pose"]
            if not isinstance(home_pose, list) or len(home_pose) != 6:
                raise ValueError(f"robot.{side}.home_pose 必须包含 6 个值")
            home_values = [float(value) for value in home_pose]
            if not all(math.isfinite(value) for value in home_values):
                raise ValueError(f"robot.{side}.home_pose 必须全部为有限值")
        for name, default in (
            ("enable_timeout_s", 5),
            ("health_timeout_s", 3),
            ("arm_status_poll_interval_s", 1),
            ("home_timeout_s", 10),
            ("home_joint_tolerance_rad", 0.10),
        ):
            if float(side_cfg.get(name, default)) <= 0:
                raise ValueError(f"robot.{side}.{name} must be positive")
        if float(side_cfg.get("hand_home_wait_s", 8)) < 0:
            raise ValueError(
                f"robot.{side}.hand_home_wait_s must be non-negative"
            )
        for name, default in (
            ("speed_percent", 20),
            ("home_speed_percent", 10),
        ):
            value = int(side_cfg.get(name, default))
            if not 1 <= value <= 100:
                raise ValueError(f"robot.{side}.{name} must be in [1, 100]")
    for camera in ("scene", "wrist_left", "wrist_right"):
        if camera not in cfg["cameras"]:
            raise ValueError(f"cameras.{camera} is required")
        if "hardware_enabled" in cfg["cameras"][camera] and not isinstance(
            cfg["cameras"][camera]["hardware_enabled"], bool
        ):
            raise ValueError(
                f"cameras.{camera}.hardware_enabled 必须为布尔值"
            )
    if not isinstance(
        cfg["cameras"].get(
            "hardware_enabled", cfg["robot"].get("enabled", False)
        ),
        bool,
    ):
        raise ValueError("cameras.hardware_enabled 必须为布尔值")
    scene = cfg["cameras"]["scene"]
    if scene.get("driver") != "realsense":
        raise ValueError("cameras.scene.driver 必须为 realsense")
    if int(scene.get("fps", cfg["rates"]["camera_hz"])) <= 0:
        raise ValueError("cameras.scene.fps must be positive")
    if int(scene.get("timeout_ms", 1000)) <= 0:
        raise ValueError("cameras.scene.timeout_ms must be positive")
    wrist_devices: list[str] = []
    for camera in ("wrist_left", "wrist_right"):
        wrist = cfg["cameras"][camera]
        if wrist.get("driver") != "opencv":
            raise ValueError(f"cameras.{camera}.driver 必须为 opencv")
        device = wrist.get("device")
        if device is None or (
            isinstance(device, str) and not device.strip()
        ):
            raise ValueError(f"cameras.{camera}.device 不能为空")
        wrist_devices.append(str(device))
        camera_hardware_enabled = bool(
            wrist.get(
                "hardware_enabled",
                cfg["cameras"].get(
                    "hardware_enabled", cfg["robot"].get("enabled", False)
                ),
            )
        )
        device_text = str(device)
        if (
            camera_hardware_enabled
            and device_text.startswith("/dev/video")
            and device_text.removeprefix("/dev/video").isdigit()
        ):
            LOG.warning(
                "cameras.%s.device=%s 是不稳定的枚举路径；正式部署请改为 "
                "/dev/v4l/by-id/...",
                camera,
                device_text,
            )
        if str(wrist.get("backend", "v4l2")).lower() not in {
            "v4l2",
            "any",
        }:
            raise ValueError(
                f"cameras.{camera}.backend 必须为 v4l2 或 any"
            )
        fourcc = str(wrist.get("fourcc", "MJPG"))
        if len(fourcc) != 4:
            raise ValueError(
                f"cameras.{camera}.fourcc 必须为4个字符"
            )
        for name, default in (
            ("buffer_size", 1),
            ("open_timeout_ms", 5000),
            ("read_timeout_ms", 2000),
            ("opencv_threads", 1),
        ):
            if int(wrist.get(name, default)) <= 0:
                raise ValueError(
                    f"cameras.{camera}.{name} must be positive"
                )
        if float(wrist.get("fps", cfg["rates"]["camera_hz"])) <= 0:
            raise ValueError(f"cameras.{camera}.fps must be positive")
        if float(wrist.get("fps_tolerance", 2.0)) < 0:
            raise ValueError(
                f"cameras.{camera}.fps_tolerance must be non-negative"
            )
    if wrist_devices[0] == wrist_devices[1]:
        raise ValueError(
            "cameras.wrist_left.device 与 wrist_right.device 不能相同"
        )
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
    episode = cfg.setdefault("episode", {})
    if (
        bool(cfg["robot"].get("enabled", False))
        and bool(cfg["robot"].get("require_home_before_start", True))
        and bool(episode.get("auto_start", False))
    ):
        raise ValueError(
            "真机 require_home_before_start=true 时不能启用 episode.auto_start；"
            "请启动后显式执行 home"
        )
    for name in ("worker_stop_timeout_s", "state_stop_timeout_s"):
        if float(cfg["robot"].get(name, 3)) <= 0:
            raise ValueError(f"robot.{name} must be positive")
    for name, default in (
        ("arm_command_hz", 30),
        ("arm_feedback_hz", 30),
        ("hand_command_hz", 60),
        ("hand_feedback_hz", 30),
    ):
        if float(cfg["robot"].get(name, default)) <= 0:
            raise ValueError(f"robot.{name} must be positive")
    for name, default in (
        ("hardware_start_timeout_s", 20),
        ("hardware_request_timeout_s", 30),
    ):
        if float(cfg["robot"].get(name, default)) <= 0:
            raise ValueError(f"robot.{name} must be positive")
    if int(cfg["robot"].get("hardware_command_queue_capacity", 16)) <= 0:
        raise ValueError("robot.hardware_command_queue_capacity must be positive")
    for name, default in (("min_frames", 10), ("command_queue_capacity", 32)):
        if int(episode.get(name, default)) <= 0:
            raise ValueError(f"episode.{name} must be positive")
    for name, default in (
        ("camera_start_timeout_s", 15),
        ("camera_shutdown_timeout_s", 5),
        ("camera_failure_timeout_ms", 1500),
    ):
        if float(episode.get(name, default)) <= 0:
            raise ValueError(f"episode.{name} must be positive")
    if float(episode.get("camera_retry_delay_ms", 10)) < 0:
        raise ValueError("episode.camera_retry_delay_ms must be non-negative")
    if int(episode.get("camera_error_report_interval", 30)) <= 0:
        raise ValueError("episode.camera_error_report_interval must be positive")
    for camera in ("scene", "wrist_left", "wrist_right"):
        camera_cfg = cfg["cameras"][camera]
        if float(camera_cfg.get("startup_delay_ms", 0)) < 0:
            raise ValueError(
                f"cameras.{camera}.startup_delay_ms must be non-negative"
            )
        for name in ("timeout_ms", "failure_timeout_ms"):
            if name in camera_cfg and float(camera_cfg[name]) <= 0:
                raise ValueError(f"cameras.{camera}.{name} must be positive")
        if "retry_delay_ms" in camera_cfg and float(camera_cfg["retry_delay_ms"]) < 0:
            raise ValueError(
                f"cameras.{camera}.retry_delay_ms must be non-negative"
            )
        if (
            "error_report_interval" in camera_cfg
            and int(camera_cfg["error_report_interval"]) <= 0
        ):
            raise ValueError(
                f"cameras.{camera}.error_report_interval must be positive"
            )
    if float(episode.get("min_camera_fps", 1)) <= 0:
        raise ValueError("episode.min_camera_fps must be positive")
    unique_ratio = float(episode.get("min_camera_unique_ratio", 0.7))
    if not 0 < unique_ratio <= 1:
        raise ValueError("episode.min_camera_unique_ratio must be in (0, 1]")
    error_ratio = float(episode.get("max_camera_error_ratio", 0.10))
    if not 0 <= error_ratio < 1:
        raise ValueError("episode.max_camera_error_ratio must be in [0, 1)")
    dataset = cfg["dataset"]
    for name in ("queue_capacity", "encoder_queue_maxsize"):
        if name in dataset and int(dataset[name]) <= 0:
            raise ValueError(f"dataset.{name} must be positive")
    for name in ("image_writer_processes", "image_writer_threads"):
        if name in dataset and int(dataset[name]) < 0:
            raise ValueError(f"dataset.{name} must be non-negative")
    if dataset.get("encoder_threads") is not None:
        if int(dataset["encoder_threads"]) <= 0:
            raise ValueError("dataset.encoder_threads must be positive")
    dataset_root = Path(cfg["dataset"]["root"])
    if not dataset_root.is_absolute():
        # 固定相对于代码目录解析，避免两次启动时因 cwd 不同写到两个数据集。
        dataset_root = Path(__file__).resolve().parent / dataset_root
    cfg["dataset"]["root"] = str(dataset_root.resolve())
    return cfg
