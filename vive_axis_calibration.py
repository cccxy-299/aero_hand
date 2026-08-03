from __future__ import annotations

"""VIVE Tracker 到 Piper 基坐标系的交互式轴标定。

该模块只接收设备 A 发来的 ZMQ 位姿，不导入或创建任何机械臂 SDK 实例。
标定结果是把 VIVE 世界坐标增量映射到 Piper 基坐标增量的正交矩阵。
位置分量映射允许行列式为 -1（坐标约定包含镜像），但该矩阵不能直接作为
SO(3) 旋转左乘 Tracker 姿态。
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable

import numpy as np
import yaml
import zmq


LOG = logging.getLogger("vive-axis-calibration")
SIDES = ("left", "right")
DIRECTION_SPECS = (
    ("+X", np.array([1.0, 0.0, 0.0])),
    ("-X", np.array([-1.0, 0.0, 0.0])),
    ("+Y", np.array([0.0, 1.0, 0.0])),
    ("-Y", np.array([0.0, -1.0, 0.0])),
    ("+Z", np.array([0.0, 0.0, 1.0])),
    ("-Z", np.array([0.0, 0.0, -1.0])),
)


@dataclass(frozen=True)
class StationaryCapture:
    mean: np.ndarray
    std_xyz: np.ndarray
    samples: int


@dataclass(frozen=True)
class DirectionMeasurement:
    label: str
    desired_robot_direction: np.ndarray
    vive_delta: np.ndarray
    displacement_m: float
    start: StationaryCapture
    end: StationaryCapture


@dataclass(frozen=True)
class AxisFitResult:
    matrix: np.ndarray
    rms_error_deg: float
    max_error_deg: float
    direction_errors_deg: dict[str, float]
    opposite_pair_errors_deg: dict[str, float]
    orthogonality_error: float
    determinant: float
    singular_values: np.ndarray
    estimated_unit_scale: float
    measurements: tuple[DirectionMeasurement, ...]


class PoseReceiverFailed(RuntimeError):
    """ZMQ 接收线程已经终止，继续交互重试也无法恢复。"""


class ZmqPoseReceiver:
    """在独立线程中独占 ZMQ socket，并保存最近的有效 Tracker 位置。"""

    def __init__(
        self,
        bind: str,
        validator: Callable[[Any], dict[str, Any]],
        max_source_age_ms: float,
    ) -> None:
        self._bind = bind
        self._validator = validator
        self._max_source_age_ms = float(max_source_age_ms)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._condition = threading.Condition()
        self._samples: dict[str, deque[tuple[int, np.ndarray]]] = {
            side: deque(maxlen=4096) for side in SIDES
        }
        self._error: BaseException | None = None
        self._received = 0
        self._invalid = 0
        self._thread = threading.Thread(
            target=self._run,
            name="vive-axis-calibration-zmq",
            daemon=True,
        )

    def start(self, timeout_s: float = 5.0) -> None:
        self._thread.start()
        if not self._ready.wait(timeout_s):
            raise TimeoutError(f"等待标定 ZMQ 端口绑定超时 {timeout_s:.1f}s")
        self._raise_if_failed()

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            LOG.warning("标定 ZMQ 接收线程未能在2秒内停止")

    def status(self) -> dict[str, int]:
        with self._condition:
            return {
                "received": self._received,
                "invalid": self._invalid,
                **{
                    f"samples_{side}": len(self._samples[side])
                    for side in SIDES
                },
            }

    def capture_stationary(
        self,
        side: str,
        window_s: float,
        min_samples: int,
        max_std_m: float,
        timeout_s: float,
    ) -> StationaryCapture:
        """只使用调用之后到达的样本，避免把移动过程中的旧数据混入端点。"""
        if side not in SIDES:
            raise ValueError(f"未知侧别: {side}")
        start_ns = time.perf_counter_ns()
        window_ns = int(window_s * 1e9)
        deadline = time.monotonic() + timeout_s
        selected: list[np.ndarray] = []
        while time.monotonic() < deadline:
            self._raise_if_failed()
            with self._condition:
                selected = [
                    position
                    for timestamp_ns, position in self._samples[side]
                    if timestamp_ns >= start_ns
                ]
                elapsed_ns = time.perf_counter_ns() - start_ns
                if len(selected) >= min_samples and elapsed_ns >= window_ns:
                    break
                self._condition.wait(timeout=0.05)
        if len(selected) < min_samples:
            raise RuntimeError(
                f"{side} 在 {timeout_s:.1f}s 内仅收到 {len(selected)} 个有效样本，"
                f"要求至少 {min_samples} 个；receiver={self.status()}"
            )
        points = np.asarray(selected, dtype=np.float64)
        mean = np.mean(points, axis=0)
        std_xyz = np.std(points, axis=0)
        if float(np.max(std_xyz)) > max_std_m:
            raise ValueError(
                f"{side} 端点不稳定：std_xyz={std_xyz.round(6).tolist()}m，"
                f"阈值={max_std_m:.6f}m"
            )
        return StationaryCapture(mean, std_xyz, len(points))

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise PoseReceiverFailed("标定 ZMQ 接收线程失败") from self._error

    def _run(self) -> None:
        socket: Any = None
        try:
            context = zmq.Context.instance()
            socket = context.socket(zmq.PULL)
            socket.setsockopt(zmq.RCVHWM, 1)
            socket.setsockopt(zmq.CONFLATE, 1)
            socket.setsockopt(zmq.LINGER, 0)
            socket.bind(self._bind)
            self._ready.set()
            while not self._stop.is_set():
                if not socket.poll(timeout=100, flags=zmq.POLLIN):
                    continue
                try:
                    message = self._validator(socket.recv_json())
                    receive_ns = time.perf_counter_ns()
                    with self._condition:
                        self._received += 1
                        for side in SIDES:
                            item = message["sides"][side]
                            age_ms = item.get("age_ms")
                            if (
                                not bool(item.get("valid", False))
                                or age_ms is None
                                or float(age_ms) > self._max_source_age_ms
                            ):
                                continue
                            position = np.asarray(
                                item["vive_pose"][:3], dtype=np.float64
                            )
                            self._samples[side].append(
                                (receive_ns, position.copy())
                            )
                        self._condition.notify_all()
                except (KeyError, TypeError, ValueError):
                    with self._condition:
                        self._invalid += 1
                    LOG.warning("标定模式丢弃非法 VIVE 数据包", exc_info=True)
        except BaseException as exc:
            self._error = exc
            self._ready.set()
            with self._condition:
                self._condition.notify_all()
        finally:
            if socket is not None:
                socket.close(linger=0)


def fit_axis_map(
    measurements: list[DirectionMeasurement],
    expected_distance_m: float,
) -> AxisFitResult:
    """通过 Kabsch/正交 Procrustes 拟合 ``robot_delta = R @ vive_delta``。"""
    if len(measurements) != len(DIRECTION_SPECS):
        raise ValueError("轴标定必须包含 +X/-X/+Y/-Y/+Z/-Z 六个方向")
    sources = np.stack(
        [item.vive_delta / item.displacement_m for item in measurements]
    )
    targets = np.stack(
        [item.desired_robot_direction for item in measurements]
    )

    # 对行向量求正交 Procrustes，再转成供列向量使用的正交矩阵。
    # 不强制 det=+1：如果 VIVE 与 Piper 使用不同手性坐标约定，位置分量映射
    # 合理地会包含一次镜像；强制 SO(3) 反而会牺牲一个轴的正确方向。
    covariance = sources.T @ targets
    u, singular_values, vt = np.linalg.svd(covariance)
    matrix = vt.T @ u.T

    predicted = (matrix @ sources.T).T
    dots = np.clip(np.sum(predicted * targets, axis=1), -1.0, 1.0)
    errors_deg = np.degrees(np.arccos(dots))
    direction_errors = {
        item.label: float(error)
        for item, error in zip(measurements, errors_deg, strict=True)
    }
    normalized = {
        item.label: item.vive_delta / item.displacement_m
        for item in measurements
    }
    pair_errors = {}
    for axis in "XYZ":
        dot = float(
            np.clip(
                np.dot(normalized[f"+{axis}"], -normalized[f"-{axis}"]),
                -1.0,
                1.0,
            )
        )
        pair_errors[axis] = float(np.degrees(np.arccos(dot)))

    lengths = np.asarray(
        [item.displacement_m for item in measurements], dtype=np.float64
    )
    return AxisFitResult(
        matrix=matrix,
        rms_error_deg=float(np.sqrt(np.mean(errors_deg**2))),
        max_error_deg=float(np.max(errors_deg)),
        direction_errors_deg=direction_errors,
        opposite_pair_errors_deg=pair_errors,
        orthogonality_error=float(
            np.linalg.norm(matrix.T @ matrix - np.eye(3), ord="fro")
        ),
        determinant=float(np.linalg.det(matrix)),
        singular_values=singular_values,
        estimated_unit_scale=float(expected_distance_m / np.median(lengths)),
        measurements=tuple(measurements),
    )


def _prompt(message: str) -> None:
    try:
        value = input(message).strip().lower()
    except EOFError as exc:
        raise RuntimeError("标定需要交互式终端输入") from exc
    if value in {"q", "quit", "exit"}:
        raise KeyboardInterrupt


def _capture_with_retry(
    receiver: ZmqPoseReceiver,
    side: str,
    description: str,
    window_s: float,
    min_samples: int,
    max_std_m: float,
) -> StationaryCapture:
    while True:
        _prompt(f"{description}，保持静止后按 Enter 开始采样（输入 q 退出）：")
        try:
            capture = receiver.capture_stationary(
                side=side,
                window_s=window_s,
                min_samples=min_samples,
                max_std_m=max_std_m,
                timeout_s=max(5.0, window_s * 4.0),
            )
            LOG.info(
                "%s 采样成功：mean=%s std=%s samples=%d",
                side,
                capture.mean.round(6).tolist(),
                capture.std_xyz.round(6).tolist(),
                capture.samples,
            )
            return capture
        except PoseReceiverFailed:
            raise
        except (RuntimeError, ValueError) as exc:
            LOG.warning("%s；请保持 Tracker 静止并重试", exc)


def collect_side_measurements(
    receiver: ZmqPoseReceiver,
    side: str,
    expected_distance_m: float,
    min_displacement_m: float,
    window_s: float,
    min_samples: int,
    max_std_m: float,
) -> list[DirectionMeasurement]:
    LOG.info(
        "开始 %s Tracker 标定。只移动该侧 Tracker；机械臂不会连接或运动。",
        side,
    )
    measurements: list[DirectionMeasurement] = []
    for label, desired in DIRECTION_SPECS:
        while True:
            start = _capture_with_retry(
                receiver,
                side,
                f"[{side} {label}] 将 Tracker 放在舒适起点",
                window_s,
                min_samples,
                max_std_m,
            )
            end = _capture_with_retry(
                receiver,
                side,
                f"[{side} {label}] 沿 Piper 基坐标 {label} 方向移动约 "
                f"{expected_distance_m * 100:.1f}cm",
                window_s,
                min_samples,
                max_std_m,
            )
            delta = end.mean - start.mean
            displacement = float(np.linalg.norm(delta))
            if displacement < min_displacement_m:
                LOG.warning(
                    "%s %s 有效位移仅 %.4fm，小于阈值 %.4fm；该方向重新采集",
                    side,
                    label,
                    displacement,
                    min_displacement_m,
                )
                continue
            measurement = DirectionMeasurement(
                label=label,
                desired_robot_direction=desired.copy(),
                vive_delta=delta,
                displacement_m=displacement,
                start=start,
                end=end,
            )
            measurements.append(measurement)
            LOG.info(
                "%s %s 完成：vive_delta=%s length=%.4fm",
                side,
                label,
                delta.round(6).tolist(),
                displacement,
            )
            break
    return measurements


def _round_matrix(matrix: np.ndarray) -> list[list[float]]:
    result = np.round(matrix.astype(float), 8)
    result[np.abs(result) < 5e-9] = 0.0
    return result.tolist()


def _result_diagnostics(result: AxisFitResult) -> dict[str, Any]:
    return {
        "rms_error_deg": round(result.rms_error_deg, 4),
        "max_error_deg": round(result.max_error_deg, 4),
        "direction_errors_deg": {
            key: round(value, 4)
            for key, value in result.direction_errors_deg.items()
        },
        "opposite_pair_errors_deg": {
            key: round(value, 4)
            for key, value in result.opposite_pair_errors_deg.items()
        },
        "orthogonality_error": round(result.orthogonality_error, 10),
        "determinant": round(result.determinant, 10),
        "contains_reflection": bool(result.determinant < 0),
        "singular_values": np.round(result.singular_values, 8).tolist(),
        "estimated_unit_scale": round(result.estimated_unit_scale, 6),
        "measurements": {
            item.label: {
                "vive_delta": np.round(item.vive_delta, 8).tolist(),
                "displacement_m": round(item.displacement_m, 6),
                "start_std_xyz_m": np.round(item.start.std_xyz, 8).tolist(),
                "end_std_xyz_m": np.round(item.end.std_xyz, 8).tolist(),
                "start_samples": item.start.samples,
                "end_samples": item.end.samples,
            }
            for item in result.measurements
        },
    }


def _non_overwriting_path(path: Path) -> Path:
    if not path.exists():
        return path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")


def run_axis_calibration(
    *,
    bind: str,
    validator: Callable[[Any], dict[str, Any]],
    sides: tuple[str, ...],
    output_path: Path,
    configured_scales: dict[str, float],
    max_source_age_ms: float,
    expected_distance_m: float,
    min_displacement_m: float,
    window_s: float,
    min_samples: int,
    max_std_m: float,
    max_fit_error_deg: float,
    max_pair_error_deg: float,
) -> Path:
    """执行完整交互式标定，质量合格后写出独立 YAML 片段。"""
    LOG.warning(
        "轴标定安全模式：不会创建 Piper 进程，不会 enable/home/move_p。"
        "请确保设备 A 的 VIVE ZMQ 发送端已经启动。"
    )
    receiver = ZmqPoseReceiver(bind, validator, max_source_age_ms)
    results: dict[str, AxisFitResult] = {}
    try:
        receiver.start()
        for side in sides:
            measurements = collect_side_measurements(
                receiver=receiver,
                side=side,
                expected_distance_m=expected_distance_m,
                min_displacement_m=min_displacement_m,
                window_s=window_s,
                min_samples=min_samples,
                max_std_m=max_std_m,
            )
            result = fit_axis_map(measurements, expected_distance_m)
            LOG.info(
                "%s FIT matrix=%s rms=%.3fdeg max=%.3fdeg directions=%s "
                "pair=%s det=%.6f estimated_unit_scale=%.4f",
                side,
                _round_matrix(result.matrix),
                result.rms_error_deg,
                result.max_error_deg,
                {
                    key: round(value, 3)
                    for key, value in result.direction_errors_deg.items()
                },
                {
                    key: round(value, 3)
                    for key, value in result.opposite_pair_errors_deg.items()
                },
                result.determinant,
                result.estimated_unit_scale,
            )
            failures = []
            if result.max_error_deg > max_fit_error_deg:
                failures.append(
                    f"max_error={result.max_error_deg:.2f}deg>"
                    f"{max_fit_error_deg:.2f}deg"
                )
            worst_pair = max(result.opposite_pair_errors_deg.values())
            if worst_pair > max_pair_error_deg:
                failures.append(
                    f"opposite_pair_error={worst_pair:.2f}deg>"
                    f"{max_pair_error_deg:.2f}deg"
                )
            if abs(abs(result.determinant) - 1.0) > 1e-6:
                failures.append(f"det={result.determinant:.8f} 非正交坐标映射")
            if result.orthogonality_error > 1e-6:
                failures.append(
                    f"orthogonality_error={result.orthogonality_error:.3e}"
                )
            if failures:
                raise RuntimeError(
                    f"{side} 轴标定质量不合格，不生成配置：{failures}"
                )
            if result.determinant < 0:
                LOG.warning(
                    "%s 标定矩阵 det=-1，包含坐标镜像。该结果可用于位置增量；"
                    "后续姿态映射必须使用矩阵共轭，不能直接把它当旋转矩阵。",
                    side,
                )
            results[side] = result
    finally:
        receiver.close()

    output = {
        "robot": {
            side: {
                # 轴标定不擅自改变操作增益；保留原 robot.yaml 的 vive_scale。
                "vive_scale": float(configured_scales[side]),
                "vive_to_robot_matrix": _round_matrix(results[side].matrix),
            }
            for side in sides
        },
        "calibration": {
            "kind": "vive_to_piper_axis_map",
            "created_at": datetime.now().astimezone().isoformat(),
            "bind": bind,
            "expected_distance_m": float(expected_distance_m),
            "max_source_age_ms": float(max_source_age_ms),
            "configured_scale_preserved": True,
            "sides": {
                side: _result_diagnostics(result)
                for side, result in results.items()
            },
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    actual_path = _non_overwriting_path(output_path)
    actual_path.write_text(
        yaml.safe_dump(output, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    LOG.info("轴标定配置已保存：%s", actual_path)
    LOG.info(
        "可合并的 YAML 片段：\n%s",
        yaml.safe_dump(
            {"robot": output["robot"]},
            sort_keys=False,
            allow_unicode=True,
        ).rstrip(),
    )
    LOG.info("请人工检查后，把 robot.left/right 下的两个字段合并进 robot.yaml")
    return actual_path
