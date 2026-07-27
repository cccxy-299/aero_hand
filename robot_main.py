"""设备 B：机器人控制与 LeRobot 数据采集启动入口。"""

from __future__ import annotations

import argparse
import logging

from .runtime import run_robot
from .config import load_robot_config


def main() -> None:
    parser = argparse.ArgumentParser(description="设备 B：控制机器人并采集 LeRobot v3 数据集")
    parser.add_argument("--config", default="teleop_collect/config/robot.yaml")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [device-b] %(message)s",
    )
    run_robot(load_robot_config(args.config))


if __name__ == "__main__":
    main()
