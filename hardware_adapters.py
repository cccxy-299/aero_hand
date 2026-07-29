from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from model import (
    BimanualControlCommand,
    BimanualRobotState,
    ControlCommand,
    RobotState,
)

LOG = logging.getLogger(__name__)


class LatestCommandWorker:
    """只执行最新命令的非阻塞硬件工作线程。

    控制线程只做 put_nowait，不会被 CAN/串口调用阻塞。队列满时覆盖旧命令。
    """

    def __init__(self, name: str, callback: Callable[[np.ndarray], None]) -> None:
        self.name = name
        self.callback = callback
        self.queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.last_error: str | None = None
        self.dropped = 0
        self.thread.start()

    def submit(self, value: np.ndarray) -> None:
        try:
            self.queue.put_nowait(value.copy())
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            self.queue.put_nowait(value.copy())
            self.dropped += 1

    def _run(self) -> None:
        while True:
            value = self.queue.get()
            if value is None:
                return
            try:
                self.callback(value)
                self.last_error = None
            except Exception as exc:  # 硬件异常留在工作线程并进入诊断
                self.last_error = repr(exc)
                LOG.exception("%s 下发失败", self.name)

    def close(self) -> None:
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.queue.put(None)
        self.thread.join(timeout=3)


class DualPiperAerohand:
    """设备 B 的双 Piper + 双 Aerohand 真实硬件适配器。"""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.arms: dict[str, Any] = {}
        self.hands: dict[str, Any] = {}
        self.arm_locks = {side: threading.Lock() for side in ("left", "right")}
        self.workers: dict[str, LatestCommandWorker] = {}
        self.last_hand = {side: np.zeros(7, np.float32) for side in ("left", "right")}
        self.last_command: BimanualControlCommand | None = None

    def connect(self) -> None:
        # 同一适配器允许跨 episode 重新连接；旧会话容器必须已经由 disconnect 清空。
        if self.arms or self.hands or self.workers:
            raise RuntimeError("机器人适配器仍持有上一会话资源，拒绝重复 connect")
        try:
            from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config
            from aero_open_sdk.aero_hand import AeroHand
        except ImportError as exc:
            raise RuntimeError(
                "真实机器人模式需要安装 pyAgxArm 和 aero_open_sdk"
            ) from exc

        for side in ("left", "right"):
            print(f"{side} 侧设备连接中！")
            side_cfg = self.cfg[side]

            arm_config = create_agx_arm_config(
                robot=ArmModel.PIPER,
                firmeware_version=PiperFW.DEFAULT,
                interface=side_cfg["interface"],
                channel=str(side_cfg.get("channel", "0")),
                bitrate=int(side_cfg.get("bitrate", 1_000_000)),
            )
            print(f"{side} 侧 机械臂 连接中~~~~~")
            arm = AgxArmFactory.create_arm(arm_config)
            arm.connect()
            deadline = time.monotonic() + float(side_cfg.get("enable_timeout_s", 5))
            # while not arm.enable():
            #     if time.monotonic() >= deadline:
            #         raise RuntimeError(f"{side} Piper 使能超时")
            #     time.sleep(0.05)
            arm.set_speed_percent(int(side_cfg.get("speed_percent", 20)))
            self.arms[side] = arm
            hand = AeroHand(port=side_cfg["hand_port"])
            self.hands[side] = hand
            print(f"{side} 侧 灵巧手 连接中~~~~~")
            if bool(side_cfg.get("home_on_connect", False)):
                arm.move_j(side_cfg["initial_pose"])
                hand.send_homing()
            print(f"{side} 侧 设备连接成功~~~~")

        for side in ("left", "right"):
            self.workers[f"arm_{side}"] = LatestCommandWorker(
                f"piper-{side}", lambda value, s=side: self._send_arm(s, value)
            )
            self.workers[f"hand_{side}"] = LatestCommandWorker(
                f"aerohand-{side}", lambda value, s=side: self._send_hand(s, value)
            )

    def _send_arm(self, side: str, value: np.ndarray) -> None:
        with self.arm_locks[side]:
            # TODO: 打印并未执行
            pass
            # print(f"动作执行，hardware_adapter, _send_arm() side: {side}, value: {value}")
            # self.arms[side].move_p(value.tolist())

    def _send_hand(self, side: str, value: np.ndarray) -> None:
        if value.shape != (7,):
            raise ValueError(f"{side} Aerohand 控制指令必须为7维，实际为 {value.shape}")
        # TODO: 打印并未执行
        # print(f"动作执行，hardware_adapter, _send_hand() side: {side}, value: {value}" )
        # self.hands[side].set_joint_positions(value.tolist())

        state = self.hands[side].get_joint_positions_compact() # 获取7位
        state = np.asarray(state, np.float32)
        self.last_hand[side] = state if state.shape == (7,) else value.copy()

    def command(self, value: BimanualControlCommand) -> None:
        self.last_command = value
        for side in ("left", "right"):
            side_value: ControlCommand = getattr(value, side)
            self.workers[f"arm_{side}"].submit(side_value.arm_pose)
            self.workers[f"hand_{side}"].submit(side_value.hand_joints)

    def read_state(self) -> BimanualRobotState:
        states: dict[str, RobotState] = {}
        for side in ("left", "right"):
            with self.arm_locks[side]:
                flange = None
                joints = None
                fp = self.arms[side].get_flange_pose()
                ja = self.arms[side].get_joint_angles()
                if fp is not None:
                    flange = np.asarray(fp.msg, np.float32)
                if ja is not None:
                    joints = np.asarray(ja.msg, np.float32)

            states[side] = RobotState(flange, joints, self.last_hand[side].copy())
        return BimanualRobotState(states["left"], states["right"])

    def stop(self) -> None:
        for worker in list(self.workers.values()):
            worker.close()
        self.workers.clear()

    def disconnect(self) -> None:
        for side in ("left", "right"):
            arm = self.arms.get(side)
            if arm is not None:
                try:
                    arm.disable()
                    time.sleep(3)
                    arm.disconnect()
                except Exception:
                    LOG.exception("%s Piper 断开失败", side)
        self.arms.clear()
        for side, hand in list(self.hands.items()):
            try:
                if hasattr(hand, "close"):
                    hand.close()
                elif hasattr(hand, "disconnect"):
                    hand.disconnect()
            except Exception:
                LOG.exception("%s Aerohand 断开失败", side)
        self.hands.clear()


class TechNexionCamera:
    """复用参考 Demo 的 CameraInterface，并保留真实采集时间戳。"""

    def __init__(self, cfg: dict[str, Any], name: str, width: int, height: int, fps: int) -> None:
        demo_dir = Path(__file__).resolve().parents[1] / "lerobot_demo"
        if str(demo_dir) not in sys.path:
            sys.path.insert(0, str(demo_dir))
        try:
            from camera import CameraInterface
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("TechNexion 相机模式需要 pyvizionsdk 和 opencv") from exc
        self.camera = CameraInterface(
            cam_num=int(cfg["cam_num"]),
            fps=fps,
            format_idx=(
                int(cfg["format_idx"])
                if cfg.get("format_idx") is not None
                else None
            ),
            name=name,
            target_width=width,
            target_height=height,
            timeout_ms=int(cfg.get("timeout_ms", 1000)),
            warmup_timeout_s=float(cfg.get("warmup_timeout_s", 5.0)),
            strict_fps=bool(cfg.get("strict_fps", True)),
            fps_tolerance=float(cfg.get("fps_tolerance", 1.0)),
            # 每路相机已由独立 OS 进程承载，不再在进程内套采集线程。
            background_capture=bool(cfg.get("background_capture", False)),
        )

    def connect(self) -> None:
        self.camera.connect()

    def read(self) -> tuple[np.ndarray, int]:
        image, timestamp_s = self.camera.get_rgb_with_timestamp()
        return image, int(timestamp_s * 1e9)

    def disconnect(self) -> None:
        self.camera.disconnect()


class IntelRealSenseColorCamera:
    """Intel RealSense 全景相机适配器，目前只启用彩色流。

    深度流没有被配置，因此不会产生不需要的深度帧和额外 USB 带宽。相机 SDK
    自带时间戳不一定与设备 B 的 ``perf_counter`` 同源，所以在彩色帧到达进程后
    立即记录设备 B 单调时钟，供统一时间缓冲和 LeRobot frame 对齐使用。
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        name: str,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        self.cfg = cfg
        self.name = name
        self.width = int(width)
        self.height = int(height)
        self.fps = int(cfg.get("fps", fps))
        self.timeout_ms = int(cfg.get("timeout_ms", 1000))
        self.warmup_frames = int(cfg.get("warmup_frames", 10))
        self.pipeline: Any | None = None
        self._rs: Any | None = None

    def connect(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "Intel RealSense scene相机需要安装 pyrealsense2"
            ) from exc

        pipeline = rs.pipeline()
        stream_config = rs.config()
        serial = str(self.cfg.get("serial", "")).strip()
        if serial:
            stream_config.enable_device(serial)

        # 明确只启用 RGB8 彩色流，不调用任何 depth stream。
        stream_config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.rgb8,
            self.fps,
        )
        pipeline.start(stream_config)
        self._rs = rs
        self.pipeline = pipeline

        # 丢弃自动曝光尚未稳定的启动帧。
        for _ in range(max(0, self.warmup_frames)):
            pipeline.wait_for_frames(self.timeout_ms)

    def read(self) -> tuple[np.ndarray, int]:
        if self.pipeline is None:
            raise RuntimeError(f"{self.name} RealSense尚未连接")
        frames = self.pipeline.wait_for_frames(self.timeout_ms)
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError(f"{self.name} 未收到彩色帧")

        image = np.asanyarray(color_frame.get_data())
        timestamp_ns = time.perf_counter_ns()
        expected_shape = (self.height, self.width, 3)
        if image.shape != expected_shape:
            raise ValueError(
                f"{self.name} 彩色帧尺寸错误: {image.shape}, 期望 {expected_shape}"
            )
        if image.dtype != np.uint8:
            image = image.astype(np.uint8, copy=False)
        # rs.format.rgb8 已是 LeRobot 需要的 RGB 顺序，不执行 BGR 转换。
        return image.copy(), timestamp_ns

    def disconnect(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            finally:
                self.pipeline = None
