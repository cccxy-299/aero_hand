"""无硬件端到端仿真启动入口。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import load_config
from runtime import simulate


def main() -> None:
    parser = argparse.ArgumentParser(description="本机端到端仿真")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "cfg" / "demo.yaml"),
    )
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    simulate(load_config(args.config), args.seconds)


if __name__ == "__main__":
    main()
