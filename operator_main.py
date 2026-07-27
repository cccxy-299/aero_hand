"""设备 A：MANUS/VIVE 操作者侧启动入口。"""

from __future__ import annotations

import argparse
import logging
import threading
from pathlib import Path

from config import load_operator_config
from runtime import operator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="设备 A：采集 MANUS/VIVE 数据并发送到机器人侧")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "cfg" / "operator.yaml"),
    )
    visual_group = parser.add_mutually_exclusive_group()
    visual_group.add_argument(
        "--visualize",
        dest="visualize",
        action="store_true",
        help="开启 VIVE 三维可视化窗口",
    )
    visual_group.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        help="关闭 VIVE 三维可视化窗口",
    )
    parser.set_defaults(visualize=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [device-a] %(message)s",
    )
    cfg = load_operator_config(args.config)
    visualize = (
        bool(cfg.get("visualization", {}).get("enabled", False))
        if args.visualize is None
        else bool(args.visualize)
    )
    if not visualize:
        operator(cfg)
        return
    if not bool(cfg["operator"].get("hardware_enabled", False)):
        parser.error("VIVE 可视化需要 operator.hardware_enabled=true")

    # Qt 事件循环必须运行在主线程，网络发送循环放到独立线程。
    from operator_hardware import HardwareOperatorSource
    from visualization import ViveViewer

    source = HardwareOperatorSource(cfg["operator"])
    stop_event = threading.Event()
    worker_errors: list[BaseException] = []

    def run_sender() -> None:
        try:
            operator(cfg, stop_event=stop_event, source=source)
        except BaseException as exc:
            worker_errors.append(exc)
            stop_event.set()

    sender_thread = threading.Thread(
        target=run_sender,
        name="operator-network-sender",
        daemon=True,
    )
    sender_thread.start()
    try:
        ViveViewer(cfg, source.vive_reader, stop_event).run()
    finally:
        stop_event.set()
        sender_thread.join(timeout=3)
    if worker_errors:
        raise worker_errors[0]


if __name__ == "__main__":
    main()
