from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol
import numpy as np


def feature_schema(height: int, width: int, hand_dof: int) -> dict[str, dict[str, Any]]:
    """构造双臂、双手、双腕相机的 LeRobot v3 feature schema。"""

    per_side_size = 12 + hand_dof  # 末端 6 + 关节 6 + 灵巧手关节
    state_size = action_size = 2 * per_side_size
    modalities = [
        "scene", "wrist_left", "wrist_right",
        "robot_state", "control_action", "teleop",
    ]

    def vector_names(prefix: str) -> list[str]:
        names: list[str] = []
        for side in ("left", "right"):
            names.extend(f"{prefix}.{side}.arm_pose.{i}" for i in range(6))
            names.extend(f"{prefix}.{side}.arm_joint.{i}" for i in range(6))
            names.extend(f"{prefix}.{side}.hand_joint.{i}" for i in range(hand_dof))
        return names

    return {
        "observation.state": {
            "dtype": "float32", "shape": (state_size,),
            "names": vector_names("observation"),
        },
        "action": {
            "dtype": "float32", "shape": (action_size,),
            "names": vector_names("action"),
        },
        "observation.images.scene": {"dtype": "video", "shape": (height, width, 3)},
        "observation.images.wrist_left": {
            "dtype": "video", "shape": (height, width, 3)
        },
        "observation.images.wrist_right": {
            "dtype": "video", "shape": (height, width, 3)
        },
        "alignment.lag_s": {"dtype": "float32", "shape": (len(modalities),), "names": modalities},
        "alignment.valid": {"dtype": "float32", "shape": (len(modalities),), "names": modalities},
        "diagnostics.source_seq": {"dtype": "int64", "shape": (1,)},
        "diagnostics.safety_flags": {
            "dtype": "int64", "shape": (2,), "names": ["left", "right"]
        },
    }


class FrameWriter(Protocol):
    def add_frame(self, frame: dict[str, Any]) -> None: ...
    def close(self) -> None: ...


class DebugJsonlWriter:
    """Validation-only writer; images are summarized, not persisted."""

    def __init__(self, root: str | Path, schema: dict[str, Any]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "schema.json").write_text(json.dumps(schema, indent=2, default=list), encoding="utf-8")
        self.handle = (self.root / "frames.jsonl").open("w", encoding="utf-8")
        self.count = 0

    def add_frame(self, frame: dict[str, Any]) -> None:
        serializable = {}
        for key, value in frame.items():
            if key.startswith("observation.images."):
                # 调试格式保留各颜色通道均值，便于确认三路相机没有被错误复用。
                serializable[key] = {
                    "shape": list(value.shape),
                    "mean": float(value.mean()),
                    "channel_mean": [
                        float(value[:, :, channel].mean()) for channel in range(value.shape[2])
                    ],
                }
            elif isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            else:
                serializable[key] = value
        self.handle.write(json.dumps(serializable, separators=(",", ":")) + "\n")
        self.count += 1

    def close(self) -> None:
        self.handle.flush()
        self.handle.close()
        (self.root / "summary.json").write_text(
            json.dumps({"frames": self.count, "format": "debug_jsonl_not_lerobot"}, indent=2),
            encoding="utf-8",
        )


class LeRobotV3Writer:
    def __init__(self, cfg: dict[str, Any], schema: dict[str, Any], fps: int) -> None:
        # 多进程模式下只有采集/写盘进程需要加载 LeRobot 及编码依赖。
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = Path(cfg["root"])
        if root.exists():
            # self.dataset = LeRobotDataset(repo_id=cfg["repo_id"], root=root)
            self.dataset = LeRobotDataset.resume(
                repo_id=cfg["repo_id"], root=root, image_writer_threads=4, image_writer_processes=0
            )
        else:
            self.dataset = LeRobotDataset.create(
                repo_id=cfg["repo_id"], root=root, fps=fps,
                robot_type="dual_piper_dual_aerohand", features=schema, use_videos=True,
                image_writer_threads=4, image_writer_processes=0,
            )
        self.task = cfg["task"]

    def add_frame(self, frame: dict[str, Any]) -> None:
        value = dict(frame)
        value["task"] = self.task
        self.dataset.add_frame(value)

    def close(self) -> None:
        self.dataset.save_episode()
        self.dataset.finalize()


def make_writer(cfg: dict[str, Any], schema: dict[str, Any], fps: int) -> FrameWriter:
    if cfg["writer"] == "debug_jsonl":
        return DebugJsonlWriter(cfg["root"], schema)
    if cfg["writer"] == "lerobot_v3":
        return LeRobotV3Writer(cfg, schema, fps)
    raise ValueError(f"unknown dataset.writer: {cfg['writer']}")
