from __future__ import annotations

import json
import logging
import multiprocessing as mp
import queue
import signal
import threading
import time
import traceback
from typing import Any, Callable

import numpy as np

from buffer import TimeBuffer
from clock import ClockMapper
from model import (
    BimanualControlCommand,
    BimanualRobotState,
    ControlCommand,
    TeleopCommand,
    TimedSample,
)
from network import UdpReceiver
from retarget import HardwareBimanualRetargeter, PassthroughRetargeter, SideRetargetConfig
from safety import SafetyConfig, SafetyGate

LOG = logging.getLogger(__name__)
SIDES = ("left", "right")
CAMERA_NAMES = ("scene", "wrist_left", "wrist_right")
ALIGNMENT_NAMES = CAMERA_NAMES + ("robot_state", "control_action", "teleop")


def _configure_child_signals(stop_event: Any) -> None:
    """Ctrl+C 由父进程统一处理；SIGTERM 仍走子进程的安全清理路径。"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())


def _status_put(status_queue: Any, process: str, kind: str, **values: Any) -> None:
    """状态队列只用于诊断，队列满时不得反向阻塞实时路径。"""
    item = {"process": process, "kind": kind, "time_ns": time.perf_counter_ns(), **values}
    try:
        status_queue.put_nowait(item)
    except queue.Full:
        pass


def _put_latest(ipc_queue: Any, sample: TimedSample) -> None:
    """有界 IPC 满时丢弃最旧样本，控制线程永不等待采集进程。"""
    try:
        ipc_queue.put_nowait(sample)
        return
    except queue.Full:
        pass
    try:
        ipc_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        ipc_queue.put_nowait(sample)
    except queue.Full:
        # 多生产线程并发时，另一个线程可能已经填满队列。
        pass


def _periodic(stop_event: Any, hz: float, callback: Callable[[int], None]) -> None:
    period_ns = int(1e9 / hz)
    deadline_ns = time.perf_counter_ns()
    while not stop_event.is_set():
        callback(time.perf_counter_ns())
        deadline_ns += period_ns
        remaining_ns = deadline_ns - time.perf_counter_ns()
        if remaining_ns > 0:
            stop_event.wait(remaining_ns / 1e9)
        else:
            deadline_ns = time.perf_counter_ns()


def _make_safety(cfg: dict[str, Any]) -> dict[str, SafetyGate]:
    result: dict[str, SafetyGate] = {}
    for side in SIDES:
        side_cfg = cfg["robot"][side]
        result[side] = SafetyGate(
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
    return result


def _make_control_components(cfg: dict[str, Any]) -> tuple[Any, Any, dict[str, SafetyGate]]:
    if bool(cfg["robot"]["enabled"]):
        # 真机 SDK 只在控制子进程内加载，避免继承 CAN/串口内部状态。
        from hardware_adapters import DualPiperAerohand

        robot = DualPiperAerohand(cfg["robot"])
        retargeter = HardwareBimanualRetargeter(
            {
                side: SideRetargetConfig(
                    np.asarray(cfg["robot"][side]["initial_pose"], np.float32),
                    float(cfg["robot"][side].get("vive_scale", 0.6)),
                    np.asarray(cfg["robot"][side]["fixed_orientation"], np.float32),
                )
                for side in SIDES
            }
        )
    else:
        from adapters import SimRobot

        robot = SimRobot(int(cfg["robot"]["hand_dof"]))
        retargeter = PassthroughRetargeter()
    return robot, retargeter, _make_safety(cfg)


def _control_process(
    cfg: dict[str, Any],
    stop_event: Any,
    sample_queue: Any,
    control_queue: Any,
    status_queue: Any,
) -> None:
    """控制进程：空闲时只监听 UDP/命令，start 后才连接并控制机器人。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [device-b/control] %(message)s",
    )
    _configure_child_signals(stop_event)
    # 采集进程异常退出时，不等待 IPC 管道刷完残留样本。
    sample_queue.cancel_join_thread()
    teleop_buffer = TimeBuffer(int(cfg["alignment"]["buffer_capacity"]))
    metrics = {
        "control_ticks": 0,
        "state_ticks": 0,
        "stale_teleop": 0,
        "control_overruns": 0,
        "control_sessions": 0,
    }
    robot = None
    retargeter = None
    safety = None
    active = False
    receiver = None
    receiver_thread: threading.Thread | None = None
    state_thread: threading.Thread | None = None
    state_stop: threading.Event | None = None

    def ingest(sample: TimedSample) -> None:
        # 空闲时只完成网络时钟同步，不积压无效遥操作样本。
        if sample.source == "teleop" and active:
            teleop_buffer.append(sample)
            _put_latest(sample_queue, sample)

    def deactivate(reason: str) -> None:
        nonlocal robot, retargeter, safety, active, state_thread, state_stop
        if not active and robot is None:
            return
        # 先关闭控制入口，再停止状态读取和硬件工作线程。
        active = False
        if state_stop is not None:
            state_stop.set()
        if state_thread is not None:
            state_thread.join(timeout=3)
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                LOG.exception("机器人停止失败")
            try:
                robot.disconnect()
            except Exception:
                LOG.exception("机器人断开失败")
        robot = None
        retargeter = None
        safety = None
        state_thread = None
        state_stop = None
        teleop_buffer.clear()
        _status_put(
            status_queue,
            "control",
            "control_stopped",
            reason=reason,
            metrics=dict(metrics),
        )

    def activate() -> None:
        nonlocal robot, retargeter, safety, active, state_thread, state_stop
        if active:
            _status_put(
                status_queue,
                "control",
                "control_rejected",
                command="start",
                reason="already_active",
            )
            return
        candidate = None
        try:
            candidate, candidate_retargeter, candidate_safety = _make_control_components(cfg)
            candidate.connect()
            robot = candidate
            retargeter = candidate_retargeter
            safety = candidate_safety
            teleop_buffer.clear()
            state_stop = threading.Event()

            def state_loop(
                session_robot: Any, session_stop: threading.Event
            ) -> None:
                state_seq = 0

                def read_state(_: int) -> None:
                    nonlocal state_seq
                    stamp_ns = time.perf_counter_ns()
                    state = session_robot.read_state()
                    _put_latest(
                        sample_queue,
                        TimedSample(
                            "robot_state", state_seq, stamp_ns, stamp_ns, state
                        ),
                    )
                    state_seq += 1
                    metrics["state_ticks"] += 1

                try:
                    _periodic(
                        session_stop, float(cfg["rates"]["state_hz"]), read_state
                    )
                except BaseException as exc:
                    _status_put(
                        status_queue,
                        "control",
                        "error",
                        component="robot_state",
                        error=repr(exc),
                        traceback=traceback.format_exc(),
                    )
                    stop_event.set()

            active = True
            state_thread = threading.Thread(
                target=state_loop,
                args=(robot, state_stop),
                name="robot-state",
                daemon=True,
            )
            state_thread.start()
            metrics["control_sessions"] += 1
            _status_put(
                status_queue,
                "control",
                "control_started",
                session=metrics["control_sessions"],
            )
        except BaseException:
            if candidate is not None:
                try:
                    candidate.stop()
                except Exception:
                    pass
                try:
                    candidate.disconnect()
                except Exception:
                    pass
            raise

    def handle_command(command: dict[str, Any]) -> None:
        kind = str(command.get("kind", "")).lower()
        if kind == "start":
            activate()
        elif kind in {"stop", "discard"}:
            deactivate(kind)
        elif kind == "status":
            _status_put(
                status_queue,
                "control",
                "control_status",
                active=active,
                metrics=dict(metrics),
            )

    try:
        # UDP/时钟同步是空闲态唯一常驻 I/O，不会触发机器人动作。
        receiver = UdpReceiver(
            int(cfg["network"]["data_port"]),
            ClockMapper(),
            ingest,
            int(cfg["network"]["max_packet_bytes"]),
        )
        receiver_thread = threading.Thread(
            target=receiver.run, name="udp-receiver", daemon=True
        )
        receiver_thread.start()
        _status_put(
            status_queue,
            "control",
            "ready",
            udp_port=cfg["network"]["data_port"],
            active=False,
        )

        last_report_ns = time.perf_counter_ns()
        period_ns = int(1e9 / float(cfg["rates"]["control_hz"]))
        deadline_ns = time.perf_counter_ns()
        while not stop_event.is_set():
            if not active:
                try:
                    handle_command(control_queue.get(timeout=0.2))
                except queue.Empty:
                    pass
                deadline_ns = time.perf_counter_ns()
                continue

            while True:
                try:
                    handle_command(control_queue.get_nowait())
                except queue.Empty:
                    break
            if not active:
                continue

            tick_start_ns = time.perf_counter_ns()
            selected = teleop_buffer.latest()
            if selected is None:
                metrics["stale_teleop"] += 1
            else:
                try:
                    command = retargeter.retarget(selected.value, selected.seq)
                    safe_by_side: dict[str, ControlCommand] = {}
                    for side in SIDES:
                        side_command: TeleopCommand = getattr(command, side)
                        safe_by_side[side] = safety[side].apply(
                            side_command, tick_start_ns - selected.local_mono_ns
                        )
                    safe = BimanualControlCommand(
                        safe_by_side["left"], safe_by_side["right"], selected.seq
                    )
                    robot.command(safe)
                    action_stamp_ns = time.perf_counter_ns()
                    _put_latest(
                        sample_queue,
                        TimedSample(
                            "control_action",
                            selected.seq,
                            selected.source_mono_ns,
                            action_stamp_ns,
                            safe,
                        ),
                    )
                    if safe.left.safety_flags & 1 or safe.right.safety_flags & 1:
                        metrics["stale_teleop"] += 1
                except (KeyError, TypeError, ValueError):
                    metrics["stale_teleop"] += 1
                    LOG.warning(
                        "丢弃非法双侧遥操作包 seq=%s",
                        selected.seq,
                        exc_info=True,
                    )

            metrics["control_ticks"] += 1
            now_ns = time.perf_counter_ns()
            if now_ns - last_report_ns >= 1_000_000_000:
                _status_put(
                    status_queue,
                    "control",
                    "metrics",
                    metrics=dict(metrics),
                    active=True,
                )
                last_report_ns = now_ns
            deadline_ns += period_ns
            remaining_ns = deadline_ns - time.perf_counter_ns()
            if remaining_ns > 0:
                stop_event.wait(remaining_ns / 1e9)
            else:
                metrics["control_overruns"] += 1
                deadline_ns = time.perf_counter_ns()
    except BaseException as exc:
        _status_put(
            status_queue,
            "control",
            "error",
            error=repr(exc),
            traceback=traceback.format_exc(),
        )
        stop_event.set()
    finally:
        if receiver is not None:
            receiver.close()
        if receiver_thread is not None:
            receiver_thread.join(timeout=2)
        deactivate("shutdown")
        _status_put(status_queue, "control", "stopped", metrics=dict(metrics))


def _make_cameras(cfg: dict[str, Any]) -> dict[str, Any]:
    cameras_cfg = cfg["cameras"]
    width = int(cameras_cfg["width"])
    height = int(cameras_cfg["height"])
    fps = int(cfg["rates"]["camera_hz"])
    if bool(cfg["robot"]["enabled"]):
        # 相机 SDK 只在采集子进程中加载，绝不进入控制进程。
        from hardware_adapters import IntelRealSenseColorCamera, TechNexionCamera

        return {
            "scene": IntelRealSenseColorCamera(
                cameras_cfg["scene"], "scene", width, height, fps
            ),
            "wrist_left": TechNexionCamera(
                cameras_cfg["wrist_left"], "wrist_left", width, height, fps
            ),
            "wrist_right": TechNexionCamera(
                cameras_cfg["wrist_right"], "wrist_right", width, height, fps
            ),
        }

    from adapters import SimCamera

    return {
        "scene": SimCamera(width, height, 0),
        "wrist_left": SimCamera(width, height, 1),
        "wrist_right": SimCamera(width, height, 2),
    }


def _build_frame(
    buffers: dict[str, TimeBuffer],
    target_ns: int,
    max_lag_ns: int,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    selected = {
        name: buffers[name].select_before(target_ns, max_lag_ns)
        for name in ALIGNMENT_NAMES
    }
    missing = [name for name, value in selected.items() if value.sample is None]
    if missing:
        if diagnostics is not None:
            diagnostics["incomplete_frames"] = diagnostics.get("incomplete_frames", 0) + 1
            for name in missing:
                key = f"missing_{name}"
                diagnostics[key] = diagnostics.get(key, 0) + 1
        return None
    state: BimanualRobotState = selected["robot_state"].sample.value
    action: BimanualControlCommand = selected["control_action"].sample.value

    def state_vector(side: str) -> np.ndarray:
        value = getattr(state, side)
        return np.concatenate((value.arm_pose, value.arm_joints, value.hand_joints))

    def action_vector(side: str) -> np.ndarray:
        value = getattr(action, side)
        return np.concatenate((value.arm_pose, value.arm_joints, value.hand_joints))

    return {
        # 双侧向量固定按 left -> right 排列，与 feature names 一致。
        "observation.state": np.concatenate(
            (state_vector("left"), state_vector("right"))
        ).astype(np.float32),
        "action": np.concatenate(
            (action_vector("left"), action_vector("right"))
        ).astype(np.float32),
        "observation.images.scene": selected["scene"].sample.value,
        "observation.images.wrist_left": selected["wrist_left"].sample.value,
        "observation.images.wrist_right": selected["wrist_right"].sample.value,
        "alignment.lag_s": np.asarray(
            [selected[name].lag_ns / 1e9 for name in ALIGNMENT_NAMES], np.float32
        ),
        "alignment.valid": np.asarray(
            [selected[name].valid for name in ALIGNMENT_NAMES], np.float32
        ),
        "diagnostics.source_seq": np.asarray([action.source_seq], np.int64),
        "diagnostics.safety_flags": np.asarray(
            [action.left.safety_flags, action.right.safety_flags], np.int64
        ),
        # 仅供本进程统计重复画面，入 Writer 队列前会删除。
        "__camera_source_seq": {
            name: selected[name].sample.seq for name in CAMERA_NAMES
        },
    }


def _recorder_process(
    cfg: dict[str, Any],
    stop_event: Any,
    sample_queue: Any,
    episode_queue: Any,
    status_queue: Any,
) -> None:
    """采集进程：相机常开，只有 recording 状态才构建并写入 episode。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [device-b/recorder] %(message)s",
    )
    _configure_child_signals(stop_event)
    from dataset import feature_schema, make_writer

    episode_cfg = cfg.get("episode", {})
    min_frames = int(episode_cfg.get("min_frames", 10))
    save_on_shutdown = bool(episode_cfg.get("save_on_shutdown", True))
    min_camera_fps = float(
        episode_cfg.get("min_camera_fps", float(cfg["rates"]["frame_hz"]) * 0.8)
    )
    min_unique_ratio = float(episode_cfg.get("min_camera_unique_ratio", 0.7))
    reject_low_quality = bool(
        episode_cfg.get("reject_low_quality_episode", True)
    )
    buffers = {
        name: TimeBuffer(int(cfg["alignment"]["buffer_capacity"]))
        for name in ALIGNMENT_NAMES
    }
    # 写队列既传帧也传 episode 边界；FIFO 保证 stop 之前的帧先写入。
    writer_queue: queue.Queue[tuple[str, Any]] = queue.Queue(
        int(cfg["dataset"]["queue_capacity"])
    )
    writer_results: queue.Queue[dict[str, Any]] = queue.Queue()
    metrics = {
        "frame_attempts": 0,
        "frame_ticks": 0,
        "written_frames": 0,
        "writer_drops": 0,
        "saved_episodes": 0,
        "discarded_episodes": 0,
    }
    cameras: dict[str, Any] = {}
    connected_cameras: list[str] = []
    worker_threads: list[threading.Thread] = []
    camera_threads: list[threading.Thread] = []
    capture_stop: threading.Event | None = None
    writer = None
    writer_thread: threading.Thread | None = None
    episode_state = "idle"
    episode_session = 0
    episode_frames = 0
    episode_task = str(cfg["dataset"]["task"])
    episode_start_ns = 0
    episode_diagnostics: dict[str, Any] = {}
    camera_first_ns: dict[str, int] = {}
    camera_last_ns: dict[str, int] = {}
    last_group_camera_seq: dict[str, int] = {}

    def fail(component: str, exc: BaseException) -> None:
        _status_put(
            status_queue,
            "recorder",
            "error",
            component=component,
            error=repr(exc),
            traceback=traceback.format_exc(),
        )
        stop_event.set()

    def publish_episode_status(kind: str, **values: Any) -> None:
        _status_put(
            status_queue,
            "recorder",
            kind,
            state=episode_state,
            session=episode_session,
            frames=episode_frames,
            total_episodes=writer.total_episodes if writer is not None else 0,
            **values,
        )

    def enqueue_boundary(kind: str, payload: dict[str, Any]) -> None:
        # episode 边界不能丢；它不在控制进程中，允许短暂等待写队列腾出空间。
        writer_queue.put((kind, payload), timeout=5)

    def camera_loop(
        name: str, camera: Any, session_stop: threading.Event
    ) -> None:
        seq = 0
        last_stamp_ns = -1
        try:
            def read_camera(_: int) -> None:
                nonlocal seq, last_stamp_ns
                image, stamp_ns = camera.read()
                stamp_ns = int(stamp_ns)
                if stamp_ns <= last_stamp_ns:
                    key = f"duplicate_{name}"
                    episode_diagnostics[key] = episode_diagnostics.get(key, 0) + 1
                    return
                last_stamp_ns = stamp_ns
                buffers[name].append(
                    TimedSample(name, seq, stamp_ns, stamp_ns, image)
                )
                camera_first_ns.setdefault(name, stamp_ns)
                camera_last_ns[name] = stamp_ns
                key = f"camera_{name}"
                episode_diagnostics[key] = episode_diagnostics.get(key, 0) + 1
                seq += 1

            _periodic(
                session_stop, float(cfg["rates"]["camera_hz"]), read_camera
            )
        except BaseException as exc:
            if not session_stop.is_set() and not stop_event.is_set():
                fail(name, exc)

    def start_capture() -> None:
        nonlocal cameras, connected_cameras, camera_threads, capture_stop
        for buffer in buffers.values():
            buffer.clear()
        episode_diagnostics.clear()
        camera_first_ns.clear()
        camera_last_ns.clear()
        last_group_camera_seq.clear()
        cameras = _make_cameras(cfg)
        connected_cameras = []
        try:
            for name in CAMERA_NAMES:
                cameras[name].connect()
                connected_cameras.append(name)
            capture_stop = threading.Event()
            camera_threads = [
                threading.Thread(
                    target=camera_loop,
                    args=(name, cameras[name], capture_stop),
                    name=f"camera-{name}",
                    daemon=True,
                )
                for name in CAMERA_NAMES
            ]
            for thread in camera_threads:
                thread.start()
        except BaseException:
            for name in reversed(connected_cameras):
                try:
                    cameras[name].disconnect()
                except Exception:
                    LOG.exception("%s 相机启动回滚失败", name)
            cameras = {}
            connected_cameras = []
            camera_threads = []
            capture_stop = None
            raise

    def stop_capture() -> None:
        nonlocal cameras, connected_cameras, camera_threads, capture_stop
        if capture_stop is not None:
            capture_stop.set()
        for thread in camera_threads:
            thread.join(timeout=3)
        for name in CAMERA_NAMES:
            count = int(episode_diagnostics.get(f"camera_{name}", 0))
            elapsed_ns = camera_last_ns.get(name, 0) - camera_first_ns.get(name, 0)
            source_fps = (
                (count - 1) * 1e9 / elapsed_ns
                if count > 1 and elapsed_ns > 0
                else 0.0
            )
            episode_diagnostics[f"source_fps_{name}"] = round(source_fps, 3)
        for name in reversed(connected_cameras):
            try:
                cameras[name].disconnect()
            except Exception:
                LOG.exception("%s 相机断开失败", name)
        cameras = {}
        connected_cameras = []
        camera_threads = []
        capture_stop = None

    def start_episode(task: str | None = None) -> None:
        nonlocal episode_state, episode_session, episode_frames
        nonlocal episode_task, episode_start_ns
        if episode_state != "idle":
            publish_episode_status(
                "episode_rejected", command="start", reason=f"state={episode_state}"
            )
            return
        episode_state = "starting"
        publish_episode_status("episode_starting")
        start_capture()
        episode_session += 1
        episode_frames = 0
        episode_task = task.strip() if task and task.strip() else str(cfg["dataset"]["task"])
        enqueue_boundary(
            "begin",
            {"session": episode_session, "task": episode_task},
        )
        episode_start_ns = time.perf_counter_ns()
        episode_state = "recording"
        publish_episode_status(
            "episode_started",
            task=episode_task,
            video_mode=writer.video_mode,
            cameras_connected=list(CAMERA_NAMES),
        )

    def finish_episode(save: bool, reason: str) -> None:
        nonlocal episode_state
        if episode_state != "recording":
            publish_episode_status(
                "episode_rejected",
                command="stop" if save else "discard",
                reason=f"state={episode_state}",
            )
            return
        episode_state = "stopping"
        stop_capture()
        quality_failures: list[str] = []
        for name in CAMERA_NAMES:
            unique = int(episode_diagnostics.get(f"unique_used_{name}", 0))
            ratio = unique / episode_frames if episode_frames > 0 else 0.0
            episode_diagnostics[f"unique_ratio_{name}"] = round(ratio, 4)
            source_fps = float(
                episode_diagnostics.get(f"source_fps_{name}", 0.0)
            )
            if source_fps < min_camera_fps:
                quality_failures.append(
                    f"{name}: source_fps={source_fps:.2f}<{min_camera_fps:.2f}"
                )
            if ratio < min_unique_ratio:
                quality_failures.append(
                    f"{name}: unique_ratio={ratio:.3f}<{min_unique_ratio:.3f}"
                )
        if quality_failures:
            episode_diagnostics["quality_failures"] = quality_failures
            if save and reject_low_quality:
                publish_episode_status(
                    "episode_quality_failed",
                    failures=quality_failures,
                    diagnostics=dict(episode_diagnostics),
                )
        should_save = (
            save
            and episode_frames >= min_frames
            and not (reject_low_quality and quality_failures)
        )
        command = "save" if should_save else "discard"
        episode_state = "saving" if should_save else "discarding"
        duration_s = (
            (time.perf_counter_ns() - episode_start_ns) / 1e9
            if episode_start_ns
            else 0.0
        )
        enqueue_boundary(
            command,
            {
                "session": episode_session,
                "frames": episode_frames,
                "reason": reason,
                "duration_s": duration_s,
                "diagnostics": dict(episode_diagnostics),
            },
        )
        publish_episode_status(
            "episode_stopping",
            requested_save=save,
            action=command,
            reason=reason,
            duration_s=duration_s,
            diagnostics=dict(episode_diagnostics),
            quality_failures=quality_failures,
        )

    def drain_writer_results() -> None:
        nonlocal episode_state
        while True:
            try:
                result = writer_results.get_nowait()
            except queue.Empty:
                return
            kind = result["kind"]
            if kind == "saved":
                episode_state = "idle"
                metrics["saved_episodes"] += 1
                publish_episode_status(
                    "episode_saved",
                    episode_index=result["episode_index"],
                    saved_frames=result["frames"],
                    writer_frames=result["writer_frames"],
                    duration_s=result["duration_s"],
                    diagnostics=result["diagnostics"],
                    save_report=result["save_report"],
                )
            elif kind == "discarded":
                episode_state = "idle"
                metrics["discarded_episodes"] += 1
                publish_episode_status(
                    "episode_discarded",
                    discarded_frames=result["frames"],
                    duration_s=result["duration_s"],
                    diagnostics=result["diagnostics"],
                )

    def handle_episode_commands() -> None:
        while True:
            try:
                command = episode_queue.get_nowait()
            except queue.Empty:
                return
            kind = str(command.get("kind", "")).lower()
            if kind == "start":
                start_episode(command.get("task"))
            elif kind == "stop":
                finish_episode(True, "manual")
            elif kind == "discard":
                finish_episode(False, "manual")
            elif kind == "status":
                publish_episode_status("episode_status", task=episode_task)

    try:
        schema = feature_schema(
            int(cfg["cameras"]["height"]),
            int(cfg["cameras"]["width"]),
            int(cfg["robot"]["hand_dof"]),
        )
        writer = make_writer(cfg["dataset"], schema, int(cfg["rates"]["frame_hz"]))

        def ipc_loop() -> None:
            while not stop_event.is_set():
                try:
                    sample = sample_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if isinstance(sample, TimedSample) and sample.source in buffers:
                    buffers[sample.source].append(sample)

        def writer_loop() -> None:
            try:
                while True:
                    kind, payload = writer_queue.get()
                    if kind == "close":
                        break
                    if kind == "begin":
                        writer.begin_episode(payload["task"])
                    elif kind == "frame":
                        writer.add_frame(payload)
                        metrics["written_frames"] += 1
                    elif kind == "save":
                        pending_frames = writer.pending_frames()
                        if pending_frames != int(payload["frames"]):
                            raise RuntimeError(
                                "Writer 帧数与组帧计数不一致："
                                f"writer={pending_frames}, grouped={payload['frames']}"
                            )
                        episode_index = writer.save_episode()
                        writer_results.put(
                            {
                                "kind": "saved",
                                "episode_index": episode_index,
                                "frames": payload["frames"],
                                "duration_s": payload["duration_s"],
                                "diagnostics": payload["diagnostics"],
                                "writer_frames": pending_frames,
                                "save_report": getattr(
                                    writer, "last_save_report", {}
                                ),
                            }
                        )
                    elif kind == "discard":
                        writer.discard_episode()
                        writer_results.put(
                            {
                                "kind": "discarded",
                                "frames": payload["frames"],
                                "duration_s": payload["duration_s"],
                                "diagnostics": payload["diagnostics"],
                            }
                        )
            except BaseException as exc:
                fail("dataset_writer", exc)
            finally:
                try:
                    writer.close()
                except BaseException as exc:
                    fail("dataset_finalize", exc)

        worker_threads.append(
            threading.Thread(target=ipc_loop, name="ipc-reader", daemon=True)
        )
        writer_thread = threading.Thread(
            target=writer_loop, name="dataset-writer", daemon=False
        )
        worker_threads.append(writer_thread)
        for thread in worker_threads:
            thread.start()

        _status_put(
            status_queue,
            "recorder",
            "ready",
            cameras_connected=[],
            existing_episodes=writer.total_episodes,
            video_mode=writer.video_mode,
            episode_state="idle",
        )
        if bool(episode_cfg.get("auto_start", False)):
            start_episode()

        max_lag_ns = int(float(cfg["alignment"]["max_lag_ms"]) * 1e6)
        frame_period_ns = int(1e9 / float(cfg["rates"]["frame_hz"]))
        deadline_ns = time.perf_counter_ns()
        last_report_ns = deadline_ns
        while not stop_event.is_set():
            handle_episode_commands()
            drain_writer_results()
            if episode_state == "recording":
                metrics["frame_attempts"] += 1
                episode_diagnostics["frame_attempts"] = (
                    episode_diagnostics.get("frame_attempts", 0) + 1
                )
                frame = _build_frame(
                    buffers,
                    time.perf_counter_ns(),
                    max_lag_ns,
                    episode_diagnostics,
                )
                if frame is not None:
                    camera_source_seq = frame.pop("__camera_source_seq")
                    try:
                        writer_queue.put_nowait(("frame", frame))
                        episode_frames += 1
                        episode_diagnostics["grouped_frames"] = episode_frames
                        for name, source_seq in camera_source_seq.items():
                            if last_group_camera_seq.get(name) == source_seq:
                                key = f"reused_in_group_{name}"
                            else:
                                key = f"unique_used_{name}"
                                last_group_camera_seq[name] = source_seq
                            episode_diagnostics[key] = (
                                episode_diagnostics.get(key, 0) + 1
                            )
                    except queue.Full:
                        # 只丢当前新帧，绝不从队列中误删 begin/save 边界。
                        metrics["writer_drops"] += 1
                    metrics["frame_ticks"] += 1

            now_ns = time.perf_counter_ns()
            if now_ns - last_report_ns >= 1_000_000_000:
                _status_put(
                    status_queue,
                    "recorder",
                    "metrics",
                    metrics=dict(metrics),
                    episode_state=episode_state,
                    episode_frames=episode_frames,
                )
                last_report_ns = now_ns
            deadline_ns += frame_period_ns
            remaining_ns = deadline_ns - time.perf_counter_ns()
            if remaining_ns > 0:
                stop_event.wait(remaining_ns / 1e9)
            else:
                deadline_ns = time.perf_counter_ns()
    except BaseException as exc:
        fail("startup", exc)
    finally:
        stop_event.set()
        for thread in worker_threads:
            if thread.name != "dataset-writer":
                thread.join(timeout=3)

        if writer is not None and writer_thread is not None:
            if episode_state == "recording":
                finish_episode(save_on_shutdown, "shutdown")
            shutdown_deadline = time.monotonic() + float(
                cfg.get("runtime", {}).get("writer_shutdown_timeout_s", 30)
            )
            while episode_state in {"saving", "discarding"} and time.monotonic() < shutdown_deadline:
                drain_writer_results()
                time.sleep(0.02)
            if episode_state in {"saving", "discarding"}:
                fail(
                    "episode_shutdown",
                    TimeoutError("episode 在退出期限内未完成保存/丢弃"),
                )
            try:
                writer_queue.put(("close", None), timeout=1)
            except queue.Full:
                fail("dataset_writer_shutdown", TimeoutError("无法提交 writer close"))
            writer_thread.join(
                timeout=float(
                    cfg.get("runtime", {}).get("writer_shutdown_timeout_s", 30)
                )
            )
            if writer_thread.is_alive():
                fail(
                    "dataset_writer_shutdown",
                    TimeoutError("数据集写线程未在限定时间内完成刷新"),
                )
        elif writer is not None:
            try:
                writer.close()
            except BaseException as exc:
                fail("dataset_finalize", exc)

        stop_capture()
        _status_put(
            status_queue,
            "recorder",
            "stopped",
            metrics=dict(metrics),
            total_episodes=writer.total_episodes if writer is not None else 0,
        )


def run_robot_multiprocess(
    cfg: dict[str, Any], run_seconds: float | None = None
) -> None:
    """设备 B 父进程：监督两个子进程，并提供人工 episode 控制台。"""
    import sys

    runtime_cfg = cfg.get("runtime", {})
    episode_cfg = cfg.get("episode", {})
    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    sample_queue = ctx.Queue(maxsize=int(runtime_cfg.get("ipc_queue_capacity", 2048)))
    episode_queue = ctx.Queue(maxsize=int(episode_cfg.get("command_queue_capacity", 32)))
    control_queue = ctx.Queue(maxsize=int(episode_cfg.get("command_queue_capacity", 32)))
    status_queue = ctx.Queue(maxsize=int(runtime_cfg.get("status_queue_capacity", 128)))
    processes = {
        "recorder": ctx.Process(
            name="device-b-recorder",
            target=_recorder_process,
            args=(cfg, stop_event, sample_queue, episode_queue, status_queue),
        ),
        "control": ctx.Process(
            name="device-b-control",
            target=_control_process,
            args=(cfg, stop_event, sample_queue, control_queue, status_queue),
        ),
    }
    errors: list[dict[str, Any]] = []
    desired_active = bool(episode_cfg.get("auto_start", False))

    def request_stop(*_: Any) -> None:
        stop_event.set()

    def submit_episode_command(kind: str, task: str | None = None) -> None:
        nonlocal desired_active
        if kind == "start":
            desired_active = True
        elif kind in {"stop", "discard"}:
            desired_active = False
        try:
            episode_queue.put_nowait({"kind": kind, "task": task})
        except queue.Full:
            LOG.error("episode 命令队列已满，命令被拒绝：%s", kind)
            return
        # start 要等相机全部就绪后再发给控制进程；停止命令则立即广播。
        if kind in {"stop", "discard", "status"}:
            try:
                control_queue.put_nowait({"kind": kind})
            except queue.Full:
                LOG.error("control 命令队列已满，命令被拒绝：%s", kind)

    def console_loop() -> None:
        LOG.info(
            "Episode 控制：start [任务描述] | stop | discard | status | quit"
        )
        while not stop_event.is_set():
            line = sys.stdin.readline()
            if line == "":
                return
            command, _, value = line.strip().partition(" ")
            command = command.lower()
            if command in {"start", "stop", "discard", "status"}:
                submit_episode_command(command, value or None)
            elif command in {"quit", "exit", "q"}:
                stop_event.set()
                return
            elif command:
                LOG.warning("未知命令：%s", command)

    old_sigint = signal.signal(signal.SIGINT, request_stop)
    old_sigterm = signal.signal(signal.SIGTERM, request_stop)
    console_thread: threading.Thread | None = None
    try:
        processes["recorder"].start()
        processes["control"].start()
        if bool(episode_cfg.get("interactive", True)):
            console_thread = threading.Thread(
                target=console_loop, name="episode-console", daemon=True
            )
            console_thread.start()

        end_time = None if run_seconds is None else time.monotonic() + run_seconds
        while not stop_event.is_set():
            if end_time is not None and time.monotonic() >= end_time:
                stop_event.set()
                break
            try:
                status = status_queue.get(timeout=0.5)
                kind = status.get("kind")
                if kind == "error":
                    errors.append(status)
                    LOG.error(
                        "%s 子进程失败（%s）：%s",
                        status.get("process"),
                        status.get("component", "runtime"),
                        status.get("error"),
                    )
                    stop_event.set()
                elif kind == "ready":
                    LOG.info("%s 子进程已就绪：%s", status.get("process"), status)
                elif kind == "metrics":
                    LOG.info("%s", json.dumps(status, ensure_ascii=False))
                elif str(kind).startswith("episode_"):
                    LOG.info("EPISODE %s", json.dumps(status, ensure_ascii=False))
                    if kind == "episode_started":
                        if desired_active:
                            control_queue.put({"kind": "start"}, timeout=2)
                        else:
                            # 相机连接期间用户已经 stop，不能再启动机器人。
                            episode_queue.put({"kind": "stop"}, timeout=2)
                    elif (
                        kind == "episode_rejected"
                        and status.get("command") == "start"
                    ):
                        desired_active = False
                elif str(kind).startswith("control_"):
                    LOG.info("CONTROL %s", json.dumps(status, ensure_ascii=False))
            except queue.Empty:
                pass

            for name, process in processes.items():
                if process.exitcode is not None and process.exitcode != 0:
                    errors.append(
                        {
                            "process": name,
                            "kind": "error",
                            "error": f"exitcode={process.exitcode}",
                        }
                    )
                    stop_event.set()
                elif process.exitcode == 0 and not stop_event.is_set():
                    errors.append(
                        {
                            "process": name,
                            "kind": "error",
                            "error": "unexpected clean exit",
                        }
                    )
                    stop_event.set()
    finally:
        stop_event.set()
        shutdown_timeout_s = float(runtime_cfg.get("shutdown_timeout_s", 8))
        # recorder 可能正在编码视频，给它更长的优雅退出时间。
        recorder_timeout_s = float(
            runtime_cfg.get("writer_shutdown_timeout_s", 30)
        ) + shutdown_timeout_s
        for name, process in processes.items():
            if process.pid is not None:
                process.join(
                    timeout=recorder_timeout_s if name == "recorder" else shutdown_timeout_s
                )
        for process in processes.values():
            if process.is_alive():
                LOG.error("%s 优雅退出超时，执行 terminate", process.name)
                process.terminate()
                process.join(timeout=3)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)

        while True:
            try:
                status = status_queue.get_nowait()
            except queue.Empty:
                break
            if status.get("kind") == "error":
                errors.append(status)
                LOG.error(
                    "%s 子进程失败（%s）：%s",
                    status.get("process"),
                    status.get("component", "runtime"),
                    status.get("error"),
                )
        for ipc_queue in (sample_queue, episode_queue, control_queue, status_queue):
            ipc_queue.close()
            ipc_queue.cancel_join_thread()
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)

    if errors:
        first = errors[0]
        raise RuntimeError(f"{first.get('process')} 子进程失败: {first.get('error')}")
