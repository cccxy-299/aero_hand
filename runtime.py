from __future__ import annotations

import json
import logging
import signal
import threading
import time
from typing import Any

import numpy as np

from adapters import SimCamera, SimOperator, SimRobot
from clock import ClockMapper
from dataset import feature_schema, make_writer
from model import TimedSample
from network import ZmqClockSynchronizer, ZmqReceiver, ZmqSender
from pipeline import RobotPipeline
from protocol import (
    Sequencer,
    validate_bimanual_payload,
    validate_hardware_command_payload,
)
from retarget import (
    HardwareBimanualRetargeter,
    PassthroughRetargeter,
    SideRetargetConfig,
)
from safety import SafetyConfig, SafetyGate


LOG = logging.getLogger(__name__)


def make_pipeline(cfg: dict, ingest_network: bool) -> tuple[RobotPipeline, ZmqReceiver | None]:
    hand_dof = int(cfg["robot"]["hand_dof"])
    cameras = cfg["cameras"]
    schema = feature_schema(cameras["height"], cameras["width"], hand_dof)
    writer = make_writer(cfg["dataset"], schema, int(cfg["rates"]["frame_hz"]))
    safety: dict[str, SafetyGate] = {}
    for side in ("left", "right"):
        side_cfg = cfg["robot"][side]
        safety[side] = SafetyGate(
            SafetyConfig(
                np.asarray(side_cfg["workspace_min"], np.float32),
                np.asarray(side_cfg["workspace_max"], np.float32),
                float(side_cfg["max_linear_step_m"]),
                np.asarray(cfg["robot"]["hand_min"], np.float32),
                np.asarray(cfg["robot"]["hand_max"], np.float32),
                int(float(cfg["alignment"]["teleop_timeout_ms"]) * 1e6),
            ),
            np.asarray(side_cfg["initial_pose"], np.float32),
        )
    if bool(cfg["robot"]["enabled"]):
        # 旧单进程构造路径按需加载；正式 robot_main 不会在父进程加载硬件 SDK。
        from hardware_adapters import (
            DualPiperAerohand,
            IntelRealSenseColorCamera,
            OpenCVWristCamera,
        )

        robot_adapter = DualPiperAerohand(cfg["robot"])
        camera_adapters = {
            "scene": IntelRealSenseColorCamera(
                cameras["scene"],
                "scene",
                cameras["width"],
                cameras["height"],
                int(cfg["rates"]["camera_hz"]),
            ),
            "wrist_left": OpenCVWristCamera(
                cameras["wrist_left"],
                "wrist_left",
                cameras["width"],
                cameras["height"],
                int(cfg["rates"]["camera_hz"]),
            ),
            "wrist_right": OpenCVWristCamera(
                cameras["wrist_right"],
                "wrist_right",
                cameras["width"],
                cameras["height"],
                int(cfg["rates"]["camera_hz"]),
            ),
        }
        retargeter = HardwareBimanualRetargeter({
            side: SideRetargetConfig(
                np.asarray(cfg["robot"][side]["initial_pose"], np.float32),
                float(cfg["robot"][side].get("vive_scale", 0.6)),
                np.asarray(cfg["robot"][side]["fixed_orientation"], np.float32),
                np.asarray(
                    cfg["robot"][side]["vive_to_robot_matrix"],
                    np.float32,
                ),
            )
            for side in ("left", "right")
        })
    else:
        robot_adapter = SimRobot(hand_dof)
        camera_adapters = {
            "scene": SimCamera(cameras["width"], cameras["height"], 0),
            "wrist_left": SimCamera(cameras["width"], cameras["height"], 1),
            "wrist_right": SimCamera(cameras["width"], cameras["height"], 2),
        }
        retargeter = PassthroughRetargeter()
        # retargeter = HardwareBimanualRetargeter({
        #     side: SideRetargetConfig(
        #         np.asarray(cfg["robot"][side]["initial_pose"], np.float32),
        #         float(cfg["robot"][side].get("vive_scale", 0.6)),
        #         np.asarray(cfg["robot"][side]["fixed_orientation"], np.float32),
        #     )
        #     for side in ("left", "right")
        # })
    pipeline = RobotPipeline(
        robot_adapter, camera_adapters, writer, safety, retargeter,
        cfg["rates"], cfg["alignment"], int(cfg["dataset"]["queue_capacity"]),
    )
    receiver = None
    if ingest_network:
        data_port = int(cfg["network"]["data_port"])
        receiver = ZmqReceiver(
            data_port,
            int(cfg["network"].get("sync_port", data_port + 1)),
            ClockMapper(),
            pipeline.ingest_teleop,
            int(cfg["network"]["max_packet_bytes"]),
        )
    return pipeline, receiver


def operator(
    cfg: dict,
    stop_event: threading.Event | None = None,
    source: Any | None = None,
) -> None:
    """运行设备 A 发送循环。

    ``source`` 和 ``stop_event`` 可由 GUI 入口传入，使 Qt 主线程与网络发送线程
    共用同一个硬件数据源和退出信号。
    """
    hand_dof = int(cfg.get("operator", {}).get("hand_dof", cfg.get("robot", {}).get("hand_dof", 7)))
    hardware_enabled = bool(cfg.get("operator", {}).get("hardware_enabled", False))
    stop_event = stop_event or threading.Event()
    if source is None and hardware_enabled:
        from operator_hardware import HardwareOperatorSource
        source = HardwareOperatorSource(cfg["operator"])
    elif source is None:
        source = SimOperator(hand_dof)
    data_port = int(cfg["network"]["data_port"])
    sender = ZmqSender(cfg["network"]["robot_host"], data_port)
    synchronizer = ZmqClockSynchronizer(
        cfg["network"]["robot_host"],
        int(cfg["network"].get("sync_port", data_port + 1)),
    )
    seq = Sequencer()
    period_ns = int(1e9 / float(cfg["rates"]["operator_hz"]))
    deadline_ns = time.perf_counter_ns()
    sync_thread = threading.Thread(
        target=synchronizer.run,
        args=(stop_event,),
        name="operator-zmq-clock-sync",
        daemon=True,
    )
    sync_thread.start()
    try:
        while not stop_event.is_set():
            if synchronizer.error is not None:
                raise RuntimeError("设备A ZMQ 时钟同步线程失败") from synchronizer.error
            if hardware_enabled:
                payload = source.read_payload()
                validate_hardware_command_payload(payload)
            else:
                value = source.read()
                payload = {
                    side: {
                        "arm_pose": getattr(value, side).arm_pose.tolist(),
                        "hand_joints": getattr(value, side).hand_joints.tolist(),
                        "valid": getattr(value, side).valid,
                    }
                    for side in ("left", "right")
                }
                validate_bimanual_payload(payload, hand_dof)
            sender.send(seq.packet("teleop", payload))
            deadline_ns += period_ns
            remaining_ns = deadline_ns - time.perf_counter_ns()
            if remaining_ns > 0:
                stop_event.wait(remaining_ns / 1e9)
            else:
                deadline_ns = time.perf_counter_ns()
        if synchronizer.error is not None:
            raise RuntimeError(
                "设备A ZMQ 时钟同步线程失败"
            ) from synchronizer.error
    finally:
        stop_event.set()
        sync_thread.join(timeout=1.0)
        if hasattr(source, "close"):
            source.close()
        sender.close()
        LOG.info(
            "设备A ZMQ已停止：sent=%d dropped=%d sync_success=%d "
            "sync_failures=%d",
            sender.sent,
            sender.dropped,
            synchronizer.success,
            synchronizer.failures,
        )


def run_robot(cfg: dict) -> None:
    # 真机入口固定使用 spawn 多进程。控制进程不会与图像编码、磁盘写入争用 GIL，
    # 也不会继承父进程中预先打开的 ZMQ、CAN、串口或 USB 句柄。
    from process_pipeline import run_robot_multiprocess

    run_robot_multiprocess(cfg)


def simulate(cfg: dict, seconds: float) -> None:
    pipeline, _ = make_pipeline(cfg, False)
    source = SimOperator(int(cfg["robot"]["hand_dof"]))
    pipeline.start()
    end = time.perf_counter() + seconds
    period = 1.0 / float(cfg["rates"]["operator_hz"])
    try:
        while time.perf_counter() < end:
            value = source.read()
            stamp = time.perf_counter_ns()
            pipeline.ingest_teleop(TimedSample("teleop", value.source_seq, stamp, stamp, {
                "left": {
                    "arm_pose": value.left.arm_pose.tolist(),
                    "hand_joints": value.left.hand_joints.tolist(),
                    "valid": value.left.valid,
                },
                "right": {
                    "arm_pose": value.right.arm_pose.tolist(),
                    "hand_joints": value.right.hand_joints.tolist(),
                    "valid": value.right.valid,
                },
            }))
            time.sleep(period)
    finally:
        pipeline.stop()
    print(json.dumps(pipeline.metrics.__dict__, indent=2))
