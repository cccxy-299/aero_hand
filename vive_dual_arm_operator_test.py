from __future__ import annotations

"""设备 A：采集双 VIVE Tracker，通过 ZMQ PUSH 发送，并可选实时可视化。"""

import argparse
import logging
import math
from pathlib import Path
import signal
import threading
import time
from typing import Any

import yaml
import zmq

from vive_reader import DualViveReader


LOG = logging.getLogger("vive-dual-arm-operator-test")
SIDES = ("left", "right")


def _load(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    vive = cfg.get("operator", {}).get("vive", {})
    for name in ("left_tracker_name", "right_tracker_name"):
        if not str(vive.get(name, "")).strip():
            raise ValueError(f"operator.vive.{name} 未配置")
    return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="设备 A：双 VIVE Tracker -> ZMQ 测试发送端"
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "cfg" / "operator.yaml"),
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="默认 tcp://<operator.yaml 中的 robot_host>:17861",
    )
    parser.add_argument("--hz", type=float, default=None)
    parser.add_argument("--max-age-ms", type=float, default=100.0)
    visual_group = parser.add_mutually_exclusive_group()
    visual_group.add_argument(
        "--visualize",
        dest="visualize",
        action="store_true",
        help="开启双 VIVE 三维可视化窗口",
    )
    visual_group.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        help="关闭可视化，适用于无桌面环境",
    )
    parser.set_defaults(visualize=None)
    return parser


def _send_loop(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    reader: DualViveReader,
    stop_event: threading.Event,
) -> None:
    """发送最新 VIVE 快照；该循环可安全地运行在 Qt 主线程之外。"""
    vive_cfg = cfg["operator"]["vive"]
    endpoint = args.endpoint or f"tcp://{cfg['network']['robot_host']}:17861"
    hz = float(
        args.hz
        if args.hz is not None
        else cfg.get("rates", {}).get("operator_hz", 60.0)
    )
    period_ns = int(1e9 / hz)
    max_age_ns = int(args.max_age_ms * 1e6)

    context = zmq.Context.instance()
    socket = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(endpoint)

    seq = 0
    sent = 0
    dropped = 0
    last_report_ns = time.perf_counter_ns()
    deadline_ns = last_report_ns
    LOG.info(
        "ZMQ连接=%s，left=%s，right=%s，发送频率=%.1fHz",
        endpoint,
        vive_cfg["left_tracker_name"],
        vive_cfg["right_tracker_name"],
        hz,
    )
    try:
        while not stop_event.is_set():
            now_ns = time.perf_counter_ns()
            snapshot = reader.snapshot()
            sides: dict[str, Any] = {}
            for side in SIDES:
                value = snapshot.get(side)
                age_ns = now_ns - value[1] if value is not None else 2**63 - 1
                pose = value[0] if value is not None else [0.0] * 7
                valid_pose = len(pose) == 7 and all(
                    math.isfinite(float(item)) for item in pose
                )
                sides[side] = {
                    "vive_pose": pose,
                    "capture_mono_ns": int(value[1]) if value else 0,
                    "age_ms": round(age_ns / 1e6, 3) if value else None,
                    "valid": bool(
                        value is not None and age_ns <= max_age_ns and valid_pose
                    ),
                }
            packet = {
                "version": 1,
                "kind": "vive_dual_arm_test",
                "seq": seq,
                "source_mono_ns": now_ns,
                "sides": sides,
            }
            try:
                socket.send_json(packet, flags=zmq.DONTWAIT)
                sent += 1
            except zmq.Again:
                dropped += 1
            seq += 1

            if now_ns - last_report_ns >= 1_000_000_000:
                LOG.info(
                    "seq=%d sent=%d dropped=%d left_age_ms=%s "
                    "right_age_ms=%s valid=%s",
                    seq,
                    sent,
                    dropped,
                    sides["left"]["age_ms"],
                    sides["right"]["age_ms"],
                    sides["left"]["valid"] and sides["right"]["valid"],
                )
                last_report_ns = now_ns

            deadline_ns += period_ns
            remaining_ns = deadline_ns - time.perf_counter_ns()
            if remaining_ns > 0:
                stop_event.wait(remaining_ns / 1e9)
            else:
                # 落后一个周期后从当前时间重新计时，避免持续追赶并占满 CPU。
                deadline_ns = time.perf_counter_ns()
    finally:
        socket.close(linger=0)
        LOG.info("设备A测试发送端已停止：sent=%d dropped=%d", sent, dropped)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [vive-test/device-a] %(message)s",
    )
    cfg = _load(args.config)
    hz = float(
        args.hz
        if args.hz is not None
        else cfg.get("rates", {}).get("operator_hz", 60.0)
    )
    if hz <= 0 or args.max_age_ms <= 0:
        parser.error("--hz 和 --max-age-ms 必须大于0")
    visualize = (
        bool(cfg.get("visualization", {}).get("enabled", False))
        if args.visualize is None
        else bool(args.visualize)
    )

    vive_cfg = cfg["operator"]["vive"]
    reader = DualViveReader(
        {
            "left": str(vive_cfg["left_tracker_name"]),
            "right": str(vive_cfg["right_tracker_name"]),
        },
        vive_cfg.get("survive_args"),
    )
    stop_event = threading.Event()
    old_sigint = signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    old_sigterm = signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
    sender_thread: threading.Thread | None = None
    worker_errors: list[BaseException] = []

    try:
        if not visualize:
            _send_loop(cfg, args, reader, stop_event)
            return

        # Qt 事件循环必须位于主线程；ZMQ 发送只读取同一个 reader 的快照，
        # 不会重复连接 libsurvive，也不会阻塞三维窗口刷新。
        from visualization import ViveViewer

        def run_sender() -> None:
            try:
                _send_loop(cfg, args, reader, stop_event)
            except BaseException as exc:
                worker_errors.append(exc)
                stop_event.set()

        sender_thread = threading.Thread(
            target=run_sender,
            name="vive-zmq-sender",
            daemon=True,
        )
        sender_thread.start()
        ViveViewer(cfg, reader, stop_event).run()
    finally:
        stop_event.set()
        if sender_thread is not None:
            sender_thread.join(timeout=3.0)
        # 先让发送线程退出，再释放 VIVE reader，避免关闭期间仍在读取快照。
        reader.close()
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)

    if sender_thread is not None and sender_thread.is_alive():
        raise RuntimeError("ZMQ 发送线程未能在 3 秒内停止")
    if worker_errors:
        raise worker_errors[0]


if __name__ == "__main__":
    main()
