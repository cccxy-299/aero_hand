"""仅使用 OpenCV VideoCapture 的相机接口。

该模块不依赖 pyvizionsdk，也不接入机器人主 pipeline，主要用于先验证 Linux
V4L2 下双腕相机能否稳定并发取流。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import threading
import time
from typing import Any

import cv2
import numpy as np


class OpenCVCameraError(RuntimeError):
    """OpenCV 相机基础异常。"""


class OpenCVCameraReadError(OpenCVCameraError):
    """单次取帧失败；测试程序可统计后继续重试。"""

    reason = "read_failed"


@dataclass(frozen=True)
class OpenCVCameraConfig:
    """OpenCV/V4L2 相机配置。"""

    device: str | int
    width: int = 640
    height: int = 480
    fps: float = 30.0
    fourcc: str = "MJPG"
    backend: str = "v4l2"
    buffer_size: int = 1
    open_timeout_ms: int = 5000
    read_timeout_ms: int = 2000
    strict_fourcc: bool = True
    strict_resolution: bool = True
    fps_tolerance: float = 2.0
    name: str = "opencv-camera"


def _decode_fourcc(value: float) -> str:
    number = int(value)
    return "".join(chr((number >> (8 * index)) & 0xFF) for index in range(4))


def _backend_id(name: str) -> int:
    normalized = name.strip().lower()
    if normalized == "v4l2":
        if not hasattr(cv2, "CAP_V4L2"):
            raise OpenCVCameraError("当前 OpenCV 不包含 CAP_V4L2")
        return int(cv2.CAP_V4L2)
    if normalized == "any":
        return int(cv2.CAP_ANY)
    raise ValueError(f"不支持的 OpenCV backend: {name!r}")


class OpenCVCamera:
    """同步、单读取者的 OpenCV 相机接口。

    类内部不创建线程。需要双相机并发时，由上层给每只相机分配一个线程或进程；
    同一个实例的 ``read_*`` 会通过锁保证不会被两个线程同时调用。
    """

    def __init__(self, config: OpenCVCameraConfig) -> None:
        self.config = config
        self._capture: cv2.VideoCapture | None = None
        self._read_lock = threading.Lock()
        self._property_results: dict[str, bool] = {}
        self._actual_properties: dict[str, Any] = {}

    @property
    def connected(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    @property
    def actual_properties(self) -> dict[str, Any]:
        """返回驱动最终接受的参数，不假设 ``set`` 一定成功。"""
        return dict(self._actual_properties)

    def connect(self) -> None:
        if self.connected:
            return
        self._property_results = {}
        self._actual_properties = {}
        cfg = self.config
        if cfg.width <= 0 or cfg.height <= 0 or cfg.fps <= 0:
            raise ValueError("width、height 和 fps 必须大于 0")
        if len(cfg.fourcc) != 4:
            raise ValueError("fourcc 必须是四个字符，例如 MJPG 或 YUYV")

        backend = _backend_id(cfg.backend)
        capture = cv2.VideoCapture()
        try:
            # 这些超时属性在部分 V4L2/OpenCV 版本中可能不受支持，因此仅记录
            # set 返回值；并发测试仍由外层 watchdog 负责发现永久阻塞。
            self._set_if_available(
                capture,
                "open_timeout_ms",
                "CAP_PROP_OPEN_TIMEOUT_MSEC",
                float(cfg.open_timeout_ms),
            )
            self._set_if_available(
                capture,
                "read_timeout_ms",
                "CAP_PROP_READ_TIMEOUT_MSEC",
                float(cfg.read_timeout_ms),
            )
            if not capture.open(cfg.device, backend):
                raise OpenCVCameraError(
                    f"{cfg.name} 无法打开设备 {cfg.device!r}，"
                    f"backend={cfg.backend}"
                )

            self._property_results.update(
                {
                    "fourcc": bool(
                        capture.set(
                            cv2.CAP_PROP_FOURCC,
                            cv2.VideoWriter_fourcc(*cfg.fourcc),
                        )
                    ),
                    "width": bool(
                        capture.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
                    ),
                    "height": bool(
                        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
                    ),
                    "fps": bool(capture.set(cv2.CAP_PROP_FPS, cfg.fps)),
                    "buffer_size": bool(
                        capture.set(
                            cv2.CAP_PROP_BUFFERSIZE, cfg.buffer_size
                        )
                    ),
                }
            )

            actual = {
                "device": cfg.device,
                "backend_requested": cfg.backend,
                "backend_actual": (
                    capture.getBackendName()
                    if hasattr(capture, "getBackendName")
                    else "unknown"
                ),
                "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
                "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                "fps": float(capture.get(cv2.CAP_PROP_FPS)),
                "fourcc": _decode_fourcc(
                    capture.get(cv2.CAP_PROP_FOURCC)
                ),
                "buffer_size": float(
                    capture.get(cv2.CAP_PROP_BUFFERSIZE)
                ),
                "property_set_results": dict(self._property_results),
            }
            if (
                cfg.strict_fourcc
                and str(actual["fourcc"]).upper() != cfg.fourcc.upper()
            ):
                raise OpenCVCameraError(
                    f"{cfg.name} FourCC设置未生效：requested="
                    f"{cfg.fourcc}, actual={actual['fourcc']!r}"
                )
            if cfg.strict_resolution and (
                actual["width"] != cfg.width
                or actual["height"] != cfg.height
            ):
                raise OpenCVCameraError(
                    f"{cfg.name} 分辨率设置未生效：requested="
                    f"{cfg.width}x{cfg.height}, actual="
                    f"{actual['width']}x{actual['height']}"
                )
            actual_fps = float(actual["fps"])
            if (
                actual_fps > 0
                and abs(actual_fps - cfg.fps) > cfg.fps_tolerance
            ):
                raise OpenCVCameraError(
                    f"{cfg.name} 帧率设置未生效：requested={cfg.fps}, "
                    f"actual={actual_fps}"
                )

            self._capture = capture
            self._actual_properties = actual
        except BaseException:
            capture.release()
            raise

    def _set_if_available(
        self,
        capture: cv2.VideoCapture,
        result_name: str,
        property_name: str,
        value: float,
    ) -> None:
        property_id = getattr(cv2, property_name, None)
        if property_id is None:
            self._property_results[result_name] = False
            return
        self._property_results[result_name] = bool(
            capture.set(property_id, value)
        )

    def read_bgr(self) -> tuple[np.ndarray, int]:
        """阻塞读取一帧，返回 BGR 图像和设备 B 单调时钟时间戳（ns）。"""
        capture = self._capture
        if capture is None or not capture.isOpened():
            raise OpenCVCameraError(f"{self.config.name} 尚未连接")

        with self._read_lock:
            ok, frame = capture.read()
            stamp_ns = time.perf_counter_ns()
        if not ok or frame is None or frame.size == 0:
            raise OpenCVCameraReadError(
                f"{self.config.name} VideoCapture.read() 返回空帧"
            )
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise OpenCVCameraReadError(
                f"{self.config.name} 图像 shape 异常: {frame.shape}"
            )
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        return frame, stamp_ns

    def read_rgb(self) -> tuple[np.ndarray, int]:
        frame_bgr, stamp_ns = self.read_bgr()
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), stamp_ns

    def disconnect(self) -> None:
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.release()

    def describe(self) -> dict[str, Any]:
        return {
            "requested": asdict(self.config),
            "actual": self.actual_properties,
        }

    def __enter__(self) -> "OpenCVCamera":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()
