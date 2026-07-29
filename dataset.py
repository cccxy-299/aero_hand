from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np

LOG = logging.getLogger(__name__)


def feature_schema(height: int, width: int, hand_dof: int) -> dict[str, dict[str, Any]]:
    """构造双臂、双手、双腕相机和全景相机的 LeRobot v3 schema。"""
    per_side_size = 12 + hand_dof  # 末端 6 + 关节 6 + 灵巧手 7
    state_size = action_size = 2 * per_side_size
    modalities = [
        "scene",
        "wrist_left",
        "wrist_right",
        "robot_state",
        "control_action",
        "teleop",
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
            "dtype": "float32",
            "shape": (state_size,),
            "names": vector_names("observation"),
        },
        "action": {
            "dtype": "float32",
            "shape": (action_size,),
            "names": vector_names("action"),
        },
        "observation.images.scene": {
            "dtype": "video",
            "shape": (height, width, 3),
        },
        "observation.images.wrist_left": {
            "dtype": "video",
            "shape": (height, width, 3),
        },
        "observation.images.wrist_right": {
            "dtype": "video",
            "shape": (height, width, 3),
        },
        "alignment.lag_s": {
            "dtype": "float32",
            "shape": (len(modalities),),
            "names": modalities,
        },
        "alignment.valid": {
            "dtype": "float32",
            "shape": (len(modalities),),
            "names": modalities,
        },
        "diagnostics.source_seq": {"dtype": "int64", "shape": (1,)},
        "diagnostics.safety_flags": {
            "dtype": "int64",
            "shape": (2,),
            "names": ["left", "right"],
        },
    }


class FrameWriter(Protocol):
    total_episodes: int
    video_mode: str

    def begin_episode(self, task: str | None = None) -> None: ...
    def add_frame(self, frame: dict[str, Any]) -> None: ...
    def save_episode(self) -> int: ...
    def discard_episode(self) -> None: ...
    def pending_frames(self) -> int: ...
    def close(self) -> None: ...


def _frame_to_json(frame: dict[str, Any]) -> dict[str, Any]:
    serializable: dict[str, Any] = {}
    for key, value in frame.items():
        if key.startswith("observation.images."):
            serializable[key] = {
                "shape": list(value.shape),
                "mean": float(value.mean()),
                "channel_mean": [
                    float(value[:, :, channel].mean())
                    for channel in range(value.shape[2])
                ],
            }
        elif isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        else:
            serializable[key] = value
    return serializable


class DebugJsonlWriter:
    """仿真验证写入器；每次 save_episode 都生成一个独立 JSONL。"""

    video_mode = "debug-no-video"

    def __init__(self, root: str | Path, schema: dict[str, Any], task: str) -> None:
        self.root = Path(root)
        self.episodes_dir = self.root / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        (self.root / "schema.json").write_text(
            json.dumps(schema, indent=2, default=list), encoding="utf-8"
        )
        self.total_episodes = len(list(self.episodes_dir.glob("episode_*.jsonl")))
        self.task = task
        self.handle: Any | None = None
        self._pending_frames = 0
        self.last_save_report: dict[str, Any] = {}

    def begin_episode(self, task: str | None = None) -> None:
        if self.handle is not None:
            raise RuntimeError("已有正在录制的 debug episode")
        self.task = task or self.task
        path = self.episodes_dir / f"episode_{self.total_episodes:06d}.jsonl"
        self.handle = path.open("w", encoding="utf-8")
        self._pending_frames = 0

    def add_frame(self, frame: dict[str, Any]) -> None:
        if self.handle is None:
            raise RuntimeError("必须先 begin_episode()")
        value = _frame_to_json(frame)
        value["task"] = self.task
        self.handle.write(json.dumps(value, separators=(",", ":")) + "\n")
        self._pending_frames += 1

    def save_episode(self) -> int:
        if self.handle is None or self._pending_frames <= 0:
            raise RuntimeError("当前 episode 没有可保存的帧")
        self.handle.flush()
        self.handle.close()
        self.handle = None
        saved_index = self.total_episodes
        saved_frames = self._pending_frames
        self.total_episodes += 1
        self._pending_frames = 0
        self.last_save_report = {
            "episode_index": saved_index,
            "dataset_frame_delta": saved_frames,
            "total_episodes": self.total_episodes,
        }
        self._write_summary()
        return saved_index

    def discard_episode(self) -> None:
        if self.handle is None:
            return
        path = Path(self.handle.name)
        self.handle.close()
        self.handle = None
        self._pending_frames = 0
        if path.is_file():
            path.unlink()

    def pending_frames(self) -> int:
        return self._pending_frames

    def _write_summary(self) -> None:
        (self.root / "summary.json").write_text(
            json.dumps(
                {
                    "episodes": self.total_episodes,
                    "format": "debug_jsonl_not_lerobot",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def close(self) -> None:
        # close 只做资源回收，不隐式把半个 episode 当成有效数据。
        if self.handle is not None:
            self.discard_episode()
        self._write_summary()


def _supported_kwargs(callable_obj: Any, values: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(callable_obj).parameters
    return {key: value for key, value in values.items() if key in parameters}


class LeRobotV3Writer:
    """兼容 LeRobot 0.4 和新版 writer API 的多 episode 写入器。"""

    def __init__(self, cfg: dict[str, Any], schema: dict[str, Any], fps: int) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.cfg = cfg
        self.root = Path(cfg["root"]).resolve()
        self.task = str(cfg["task"])
        self._episode_open = False
        self.last_save_report: dict[str, Any] = {}
        info_path = self.root / "meta" / "info.json"
        dataset_exists = info_path.is_file()

        if self.root.exists() and not dataset_exists:
            entries = list(self.root.iterdir())
            if entries:
                raise RuntimeError(
                    f"数据集目录存在但缺少 meta/info.json，拒绝覆盖可能损坏的数据：{self.root}"
                )
            # LeRobot 0.4 create() 要求根目录本身尚不存在。
            self.root.rmdir()

        image_processes = int(cfg.get("image_writer_processes", 2))
        image_threads = int(cfg.get("image_writer_threads", 2))
        streaming_requested = bool(cfg.get("streaming_encoding", True))
        common = {
            "repo_id": cfg["repo_id"],
            "root": self.root,
            "image_writer_processes": image_processes,
            "image_writer_threads": image_threads,
            "batch_encoding_size": 1,
            "streaming_encoding": streaming_requested,
            "encoder_queue_maxsize": int(cfg.get("encoder_queue_maxsize", 90)),
            "encoder_threads": cfg.get("encoder_threads"),
        }

        if dataset_exists:
            if hasattr(LeRobotDataset, "resume"):
                kwargs = _supported_kwargs(LeRobotDataset.resume, common)
                self.dataset = LeRobotDataset.resume(**kwargs)
            else:
                # LeRobot 0.4 没有 resume()；构造现有数据集即进入可续写状态。
                constructor_values = {
                    "repo_id": cfg["repo_id"],
                    "root": self.root,
                    "download_videos": False,
                    "batch_encoding_size": 1,
                }
                self.dataset = LeRobotDataset(
                    **_supported_kwargs(LeRobotDataset, constructor_values)
                )
                if image_processes or image_threads:
                    self.dataset.start_image_writer(image_processes, image_threads)
        else:
            create_values = {
                **common,
                "fps": fps,
                "robot_type": "dual_piper_dual_aerohand",
                "features": schema,
                "use_videos": True,
            }
            self.dataset = LeRobotDataset.create(
                **_supported_kwargs(LeRobotDataset.create, create_values)
            )

        self._validate_existing_schema(schema, fps)
        self.total_episodes = int(self.dataset.meta.total_episodes)
        supports_streaming = (
            "streaming_encoding"
            in inspect.signature(
                LeRobotDataset.resume
                if dataset_exists and hasattr(LeRobotDataset, "resume")
                else LeRobotDataset.create
            ).parameters
        )
        self.video_mode = (
            "streaming-mp4"
            if streaming_requested and supports_streaming
            else "png-staging-then-mp4"
        )
        if streaming_requested and not supports_streaming:
            LOG.warning(
                "当前 LeRobot 不支持 streaming_encoding；录制时暂存 PNG，"
                "stop 时仍会编码为最终 MP4。升级到支持该参数的版本可直接流式编码。"
            )
        LOG.info(
            "LeRobot 数据集已%s：root=%s, existing_episodes=%d, video_mode=%s",
            "续写" if dataset_exists else "创建",
            self.root,
            self.total_episodes,
            self.video_mode,
        )

    def _validate_existing_schema(self, schema: dict[str, Any], fps: int) -> None:
        if int(self.dataset.meta.fps) != int(fps):
            raise ValueError(
                f"已有数据集 fps={self.dataset.meta.fps}，当前配置 fps={fps}，不能续写"
            )
        existing = self.dataset.meta.features
        for key, expected in schema.items():
            if key not in existing:
                raise ValueError(f"已有数据集缺少 feature：{key}")
            actual = existing[key]
            if actual["dtype"] != expected["dtype"] or tuple(actual["shape"]) != tuple(
                expected["shape"]
            ):
                raise ValueError(
                    f"已有数据集 feature 不兼容：{key}, existing={actual}, expected={expected}"
                )

    def begin_episode(self, task: str | None = None) -> None:
        if self._episode_open or self.pending_frames() > 0:
            raise RuntimeError("已有正在录制的 LeRobot episode")
        self.task = task or str(self.cfg["task"])
        self._episode_open = True

    def add_frame(self, frame: dict[str, Any]) -> None:
        if not self._episode_open:
            raise RuntimeError("必须先 begin_episode()")
        value = dict(frame)
        value["task"] = self.task
        self.dataset.add_frame(value)

    def save_episode(self) -> int:
        pending = self.pending_frames()
        if not self._episode_open or pending <= 0:
            raise RuntimeError("当前 episode 没有可保存的帧")
        saved_index = int(self.dataset.meta.total_episodes)
        frames_before = int(self.dataset.meta.total_frames)
        save_values: dict[str, Any] = {}
        if "parallel_encoding" in inspect.signature(self.dataset.save_episode).parameters:
            save_values["parallel_encoding"] = True
        self.dataset.save_episode(**save_values)
        actual_total = int(self.dataset.meta.total_episodes)
        frames_after = int(self.dataset.meta.total_frames)
        if actual_total != saved_index + 1:
            raise RuntimeError(
                f"episode 元数据未递增：before={saved_index}, after={actual_total}"
            )
        if frames_after - frames_before != pending:
            raise RuntimeError(
                "LeRobot 元数据帧增量与 Writer 缓冲不一致："
                f"metadata_delta={frames_after - frames_before}, pending={pending}"
            )
        self.total_episodes = actual_total
        self._episode_open = False
        self.last_save_report = {
            "episode_index": saved_index,
            "pending_frames": pending,
            "dataset_frame_delta": frames_after - frames_before,
            "total_frames": frames_after,
            "total_episodes": actual_total,
            "video_mode": self.video_mode,
        }
        return saved_index

    def discard_episode(self) -> None:
        if self.pending_frames() > 0:
            self.dataset.clear_episode_buffer(delete_images=True)
        self._episode_open = False

    def pending_frames(self) -> int:
        if hasattr(self.dataset, "has_pending_frames"):
            if not self.dataset.has_pending_frames():
                return 0
        buffer = getattr(self.dataset, "episode_buffer", None)
        if buffer is None:
            writer = getattr(self.dataset, "writer", None)
            buffer = getattr(writer, "episode_buffer", None)
        return int(buffer.get("size", 0)) if isinstance(buffer, dict) else 0

    def close(self) -> None:
        if self.pending_frames() > 0:
            raise RuntimeError("存在未保存 episode；必须先 save_episode() 或 discard_episode()")
        if hasattr(self.dataset, "stop_image_writer"):
            self.dataset.stop_image_writer()
        self.dataset.finalize()


def make_writer(
    cfg: dict[str, Any], schema: dict[str, Any], fps: int
) -> FrameWriter:
    if cfg["writer"] == "debug_jsonl":
        return DebugJsonlWriter(cfg["root"], schema, str(cfg["task"]))
    if cfg["writer"] == "lerobot_v3":
        return LeRobotV3Writer(cfg, schema, fps)
    raise ValueError(f"unknown dataset.writer: {cfg['writer']}")
