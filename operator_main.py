"""设备 A：MANUS/VIVE 操作者侧启动入口。"""

from __future__ import annotations

import argparse
import logging

from .runtime import operator
from .config import load_operator_config


def main() -> None:
    parser = argparse.ArgumentParser(description="设备 A：采集 MANUS/VIVE 数据并发送到机器人侧")
    parser.add_argument("--config", default="teleop_collect/config/operator.yaml")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [device-a] %(message)s",
    )
    operator(load_operator_config(args.config))


if __name__ == "__main__":
    main()
