from __future__ import annotations

import json
import signal
import threading
import time

import numpy as np

from .adapters import SimCamera, SimOperator, SimRobot
from .clock import ClockMapper
from .dataset import feature_schema, make_writer
from .model import TimedSample
from .network import UdpReceiver, UdpSender
from .pipeline import RobotPipeline
from .protocol import Sequencer, validate_bimanual_payload
from .safety import SafetyConfig, SafetyGate


def make_pipeline(cfg: dict, ingest_network: bool) -> tuple[RobotPipeline, UdpReceiver | None]:
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
    # 全景、左腕、右腕相机为三个完全独立的数据源和采集线程。
    camera_adapters = {
        "scene": SimCamera(cameras["width"], cameras["height"], 0),
        "wrist_left": SimCamera(cameras["width"], cameras["height"], 1),
        "wrist_right": SimCamera(cameras["width"], cameras["height"], 2),
    }
    pipeline = RobotPipeline(
        SimRobot(hand_dof), camera_adapters, writer, safety,
        cfg["rates"], cfg["alignment"], int(cfg["dataset"]["queue_capacity"]),
    )
    receiver = None
    if ingest_network:
        receiver = UdpReceiver(
            int(cfg["network"]["data_port"]), ClockMapper(), pipeline.ingest_teleop,
            int(cfg["network"]["max_packet_bytes"]),
        )
    return pipeline, receiver


def operator(cfg: dict) -> None:
    hand_dof = int(cfg.get("operator", {}).get("hand_dof", cfg.get("robot", {}).get("hand_dof", 16)))
    source = SimOperator(hand_dof)
    sender = UdpSender(cfg["network"]["robot_host"], int(cfg["network"]["data_port"]))
    seq = Sequencer()
    period = 1.0 / float(cfg["rates"]["operator_hz"])
    next_sync = 0.0
    try:
        while True:
            now = time.perf_counter()
            if now >= next_sync:
                sender.synchronize(seq.packet("sync_request", {}))
                next_sync = now + 1.0
            value = source.read()
            payload = {
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
            }
            # 设备 A 在发送前完成最基础的维度和 NaN/Inf 检查。
            validate_bimanual_payload(payload, hand_dof)
            sender.send(seq.packet("teleop", payload))
            time.sleep(period)
    finally:
        sender.close()


def run_robot(cfg: dict) -> None:
    pipeline, receiver = make_pipeline(cfg, True)
    assert receiver
    network_thread = threading.Thread(target=receiver.run, name="udp-receiver", daemon=True)
    pipeline.start()
    network_thread.start()
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        while not stop.wait(1):
            print(json.dumps(pipeline.metrics.__dict__))
    finally:
        receiver.close()
        pipeline.stop()


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
