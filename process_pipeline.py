from __future__ import annotations

import json
import logging
import multiprocessing as mp
from multiprocessing import shared_memory
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
from network import ZmqReceiver
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


def _make_safety(
    cfg: dict[str, Any],
    initial_poses: dict[str, np.ndarray] | None = None,
) -> dict[str, SafetyGate]:
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
            (
                initial_poses[side]
                if initial_poses is not None
                else np.asarray(side_cfg["initial_pose"], np.float32)
            ),
        )
    return result


def _make_robot(
    cfg: dict[str, Any], hardware_channels: dict[str, Any] | None = None
) -> Any:
    if bool(cfg["robot"]["enabled"]):
        if hardware_channels is not None:
            # 正式多进程路径：control 只持有 IPC 代理，不加载任何硬件 SDK。
            from hardware_processes import MultiprocessRobotProxy

            return MultiprocessRobotProxy(cfg["robot"], hardware_channels)
        # 兼容旧单进程入口；正式真机采集不走此分支。
        from hardware_adapters import DualPiperAerohand

        return DualPiperAerohand(cfg["robot"])
    from adapters import SimRobot

    return SimRobot(int(cfg["robot"]["hand_dof"]))


def _episode_initial_poses(
    cfg: dict[str, Any], robot_state: BimanualRobotState
) -> dict[str, np.ndarray]:
    """以 episode 开始时的真实反馈作为 VIVE 和安全限速的参考原点。"""
    result: dict[str, np.ndarray] = {}
    for side in SIDES:
        pose = np.asarray(getattr(robot_state, side).arm_pose, dtype=np.float32)
        if pose.shape != (6,) or not np.all(np.isfinite(pose)):
            if bool(cfg["robot"]["enabled"]):
                raise RuntimeError(f"{side} Piper 当前末端位姿无效，拒绝 start")
            pose = np.asarray(cfg["robot"][side]["initial_pose"], np.float32)
        result[side] = pose.copy()
    return result


def _make_episode_control_components(
    cfg: dict[str, Any], robot_state: BimanualRobotState
) -> tuple[Any, dict[str, SafetyGate]]:
    initial_poses = _episode_initial_poses(cfg, robot_state)
    if bool(cfg["robot"]["enabled"]):
        orientations: dict[str, np.ndarray] = {}
        for side in SIDES:
            side_cfg = cfg["robot"][side]
            orientation_mode = str(
                side_cfg.get("orientation_mode", "current_on_start")
            ).lower()
            if orientation_mode == "current_on_start":
                # 最安全的3DoF遥操作方式：锁定 start 时的真实法兰姿态。
                # 因此首个 VIVE 包只建立位置零点，不会触发姿态跳变。
                orientations[side] = initial_poses[side][3:].copy()
            elif orientation_mode == "configured_fixed":
                orientations[side] = np.asarray(
                    side_cfg["fixed_orientation"], dtype=np.float32
                )
            else:
                raise ValueError(
                    f"robot.{side}.orientation_mode 非法: {orientation_mode}"
                )
            LOG.info(
                "%s episode 姿态策略=%s，固定姿态=%s",
                side,
                orientation_mode,
                orientations[side].tolist(),
            )

        retargeter = HardwareBimanualRetargeter(
            {
                side: SideRetargetConfig(
                    initial_poses[side],
                    float(cfg["robot"][side].get("vive_scale", 0.6)),
                    orientations[side],
                    np.asarray(
                        cfg["robot"][side]["vive_to_robot_matrix"],
                        dtype=np.float32,
                    ),
                )
                for side in SIDES
            }
        )
    else:
        retargeter = PassthroughRetargeter()
    return retargeter, _make_safety(cfg, initial_poses)


def _control_process(
    cfg: dict[str, Any],
    stop_event: Any,
    sample_queue: Any,
    control_queue: Any,
    status_queue: Any,
    hardware_channels: dict[str, Any] | None = None,
) -> None:
    """控制协调进程：不持有真机 SDK，仅通过 IPC 驱动四个常驻硬件进程。"""
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
        "control_commands": 0,
        "unique_teleop_packets": 0,
        "reused_teleop_packets": 0,
    }
    robot = None
    retargeter = None
    safety = None
    active = False
    receiver = None
    receiver_thread: threading.Thread | None = None
    state_thread: threading.Thread | None = None
    state_stop: threading.Event | None = None
    last_selected_teleop_seq: int | None = None

    def ingest(sample: TimedSample) -> None:
        # 空闲时只完成网络时钟同步，不积压无效遥操作样本。
        if sample.source == "teleop" and active:
            teleop_buffer.append(sample)
            _put_latest(sample_queue, sample)

    def deactivate(reason: str) -> None:
        nonlocal robot, retargeter, safety, active, state_thread, state_stop
        nonlocal last_selected_teleop_seq
        if not active:
            hardware_state = (
                robot.status_snapshot().get("state")
                if robot is not None
                else None
            )
            if robot is not None and hardware_state == "prepared_enabled":
                # start 准备完成后双臂会保持 enable。若用户在相机启动期间
                # stop/discard，必须立即撤销准备并进入 hold，不能因 active=False
                # 而提前返回。
                robot.deactivate()
                _status_put(
                    status_queue,
                    "control",
                    "control_start_cancelled",
                    reason=reason,
                    hardware=robot.status_snapshot(),
                )
            teleop_buffer.clear()
            return
        # 先关闭控制入口，再停止状态读取和硬件工作线程。
        active = False
        if state_stop is not None:
            state_stop.set()
        state_thread_stuck = False
        if state_thread is not None:
            state_thread.join(
                timeout=float(cfg["robot"].get("state_stop_timeout_s", 3.0))
            )
            state_thread_stuck = state_thread.is_alive()
        deactivate_error: BaseException | None = None
        if robot is not None:
            try:
                robot.deactivate()
            except BaseException as exc:
                deactivate_error = exc
                LOG.exception("机器人停用失败")
        retargeter = None
        safety = None
        state_thread = None
        state_stop = None
        teleop_buffer.clear()
        last_selected_teleop_seq = None
        _status_put(
            status_queue,
            "control",
            "control_stopped",
            reason=reason,
            metrics=dict(metrics),
            hardware=robot.status_snapshot() if robot is not None else None,
        )
        if state_thread_stuck:
            raise RuntimeError(
                "机器人状态读取线程未及时停止；为避免跨 episode 复用阻塞的 SDK "
                "调用，终止当前 control 硬件会话"
            )
        if deactivate_error is not None:
            raise RuntimeError("机器人停用失败") from deactivate_error

    def activate() -> None:
        nonlocal robot, retargeter, safety, active, state_thread, state_stop
        nonlocal last_selected_teleop_seq
        if active:
            _status_put(
                status_queue,
                "control",
                "control_rejected",
                command="start",
                reason="already_active",
            )
            return
        if receiver is None or receiver.sync_updates < 1:
            _status_put(
                status_queue,
                "control",
                "control_rejected",
                command="start",
                reason="zmq_clock_not_synchronized",
            )
            return
        if robot is None:
            _status_put(
                status_queue,
                "control",
                "control_rejected",
                command="start",
                reason="hardware_not_initialized",
            )
            return
        try:
            # 先完成双臂使能与控制器健康检查，再读取真实法兰反馈。
            # 这与旧方案“机械臂已就绪后 activate_teleop() 建立相对参考”一致，
            # 避免把准备前可能滞后的反馈作为首帧控制基准。
            robot.activate()

            # 每个 episode 都从 start 后的真实当前位置重新建立 VIVE 参考和
            # 安全门历史；机器人 SDK 实例本身不重建。
            current_state = robot.read_state()
            candidate_retargeter, candidate_safety = (
                _make_episode_control_components(cfg, current_state)
            )
            retargeter = candidate_retargeter
            safety = candidate_safety
            teleop_buffer.clear()
            last_selected_teleop_seq = None
            state_stop = threading.Event()

            def state_loop(
                session_robot: Any, session_stop: threading.Event
            ) -> None:
                state_seq = 0

                def read_state(_: int) -> None:
                    nonlocal state_seq
                    state = session_robot.read_state()
                    feedback_stamps = getattr(
                        session_robot, "feedback_timestamps_ns", lambda: {}
                    )()
                    valid_stamps = [
                        int(value)
                        for value in feedback_stamps.values()
                        if int(value) > 0
                    ]
                    # 双侧样本采用较旧一侧的时间，避免把其中一侧的缓存状态
                    # 错标为更新时刻；各侧原始时间和偏差保留在 meta 中。
                    stamp_ns = (
                        min(valid_stamps)
                        if valid_stamps
                        else time.perf_counter_ns()
                    )
                    feedback_skew_ms = (
                        (max(valid_stamps) - min(valid_stamps)) / 1e6
                        if len(valid_stamps) >= 2
                        else 0.0
                    )
                    _put_latest(
                        sample_queue,
                        TimedSample(
                            "robot_state",
                            state_seq,
                            stamp_ns,
                            stamp_ns,
                            state,
                            meta={
                                "side_feedback_mono_ns": feedback_stamps,
                                "feedback_skew_ms": round(
                                    feedback_skew_ms, 3
                                ),
                            },
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
                hardware=robot.status_snapshot(),
            )
        except BaseException as exc:
            try:
                robot.deactivate()
            except Exception:
                LOG.exception("start 失败后的机器人停用失败")
            retargeter = None
            safety = None
            state_stop = None
            state_thread = None
            teleop_buffer.clear()
            _status_put(
                status_queue,
                "control",
                "control_rejected",
                command="start",
                reason=repr(exc),
                hardware=robot.status_snapshot(),
            )

    def handle_command(command: dict[str, Any]) -> None:
        kind = str(command.get("kind", "")).lower()
        if kind == "start":
            activate()
        elif kind == "prepare_start":
            if active:
                _status_put(
                    status_queue,
                    "control",
                    "control_rejected",
                    command="prepare_start",
                    reason="episode_active",
                )
                return
            if receiver is None or receiver.sync_updates < 1:
                _status_put(
                    status_queue,
                    "control",
                    "control_rejected",
                    command="prepare_start",
                    reason="zmq_clock_not_synchronized",
                )
                return
            if robot is None:
                _status_put(
                    status_queue,
                    "control",
                    "control_rejected",
                    command="prepare_start",
                    reason="hardware_not_initialized",
                )
                return
            _status_put(
                status_queue,
                "control",
                "control_start_preparing",
                hardware=robot.status_snapshot(),
            )
            try:
                robot.prepare_start()
            except BaseException as exc:
                _status_put(
                    status_queue,
                    "control",
                    "control_start_prepare_failed",
                    error=repr(exc),
                    hardware=robot.status_snapshot(),
                )
                return
            _status_put(
                status_queue,
                "control",
                "control_start_prepared",
                hardware=robot.status_snapshot(),
            )
        elif kind == "home":
            if active:
                _status_put(
                    status_queue,
                    "control",
                    "control_rejected",
                    command="home",
                    reason="episode_active",
                )
                return
            if robot is None:
                _status_put(
                    status_queue,
                    "control",
                    "control_rejected",
                    command="home",
                    reason="hardware_not_initialized",
                )
                return
            _status_put(
                status_queue,
                "control",
                "control_home_started",
                hardware=robot.status_snapshot(),
            )
            try:
                robot.home()
            except BaseException as exc:
                _status_put(
                    status_queue,
                    "control",
                    "control_home_failed",
                    error=repr(exc),
                    hardware=robot.status_snapshot(),
                )
                return
            _status_put(
                status_queue,
                "control",
                "control_home_completed",
                hardware=robot.status_snapshot(),
            )
        elif kind in {"stop", "discard", "cancel_prepare"}:
            deactivate(kind)
        elif kind == "status":
            _status_put(
                status_queue,
                "control",
                "control_status",
                active=active,
                metrics=dict(metrics),
                hardware=robot.status_snapshot() if robot is not None else None,
                network=(
                    {
                        "transport": "zmq",
                        "data_endpoint": receiver.data_endpoint,
                        "sync_endpoint": receiver.sync_endpoint,
                        "bad_packets": receiver.bad_packets,
                        "sequence_gaps": receiver.sequence_gaps,
                        "sync_updates": receiver.sync_updates,
                        "clock_offset_ns": receiver.mapper.offset_ns,
                    }
                    if receiver is not None
                    else None
                ),
            )
        else:
            _status_put(
                status_queue,
                "control",
                "control_rejected",
                command=kind,
                reason="unknown_command",
            )

    try:
        # SDK 对象和 CAN/串口句柄只在本 control 子进程中创建，并跨 episode 常驻。
        robot = _make_robot(cfg, hardware_channels)
        _status_put(status_queue, "control", "hardware_initializing")
        robot.initialize()
        _status_put(
            status_queue,
            "control",
            "hardware_ready",
            hardware=robot.status_snapshot(),
        )

        # 空闲态保留 ZMQ/时钟同步；4个硬件进程继续常驻并保持最后状态。
        data_port = int(cfg["network"]["data_port"])
        sync_port = int(cfg["network"].get("sync_port", data_port + 1))
        receiver = ZmqReceiver(
            data_port,
            sync_port,
            ClockMapper(),
            ingest,
            int(cfg["network"]["max_packet_bytes"]),
        )
        receiver_thread = threading.Thread(
            target=receiver.run, name="zmq-receiver", daemon=True
        )
        receiver_thread.start()
        receiver.wait_ready(
            timeout_s=float(cfg.get("network", {}).get("startup_timeout_s", 5.0))
        )
        _status_put(
            status_queue,
            "control",
            "ready",
            transport="zmq",
            data_endpoint=receiver.data_endpoint,
            sync_endpoint=receiver.sync_endpoint,
            active=False,
            hardware=robot.status_snapshot(),
        )

        last_report_ns = time.perf_counter_ns()
        period_ns = int(1e9 / float(cfg["rates"]["control_hz"]))
        teleop_timeout_ns = int(
            float(cfg["alignment"]["teleop_timeout_ms"]) * 1e6
        )
        deadline_ns = time.perf_counter_ns()
        while not stop_event.is_set():
            if receiver.error is not None:
                raise RuntimeError("ZMQ 接收线程异常退出") from receiver.error
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
                    if selected.seq == last_selected_teleop_seq:
                        metrics["reused_teleop_packets"] += 1
                    else:
                        metrics["unique_teleop_packets"] += 1
                        last_selected_teleop_seq = selected.seq
                    teleop_age_ns = tick_start_ns - selected.local_mono_ns
                    if teleop_age_ns > teleop_timeout_ns:
                        # 过期包不能用于建立 VIVE 零点，也不应继续刷新手部命令。
                        # 硬件命令线程会自然保持最后一条已接受的安全目标。
                        metrics["stale_teleop"] += 1
                    else:
                        command = retargeter.retarget(
                            selected.value, selected.seq
                        )
                        if not command.left.valid or not command.right.valid:
                            # 与旧方案的“Tracker 丢失时不发送新目标”一致；
                            # 双侧系统采用左右联锁，任一侧无效时均保持上一目标。
                            metrics["stale_teleop"] += 1
                        else:
                            safe_by_side: dict[str, ControlCommand] = {}
                            for side in SIDES:
                                side_command: TeleopCommand = getattr(
                                    command, side
                                )
                                safe_by_side[side] = safety[side].apply(
                                    side_command,
                                    teleop_age_ns,
                                )
                            safe = BimanualControlCommand(
                                safe_by_side["left"],
                                safe_by_side["right"],
                                selected.seq,
                            )
                            robot.command(safe)
                            metrics["control_commands"] += 1
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
                            if (
                                safe.left.safety_flags & 1
                                or safe.right.safety_flags & 1
                            ):
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
                    hardware=robot.status_snapshot(),
                    teleop_reference_ready=getattr(
                        retargeter, "references_ready", None
                    ),
                    teleop_reference_source_seq=getattr(
                        retargeter, "reference_source_seq", None
                    ),
                    teleop_reference=getattr(
                        retargeter, "reference_snapshot", lambda: {}
                    )(),
                    teleop_mapping=getattr(
                        retargeter, "mapping_snapshot", lambda: {}
                    )(),
                    network={
                        "transport": "zmq",
                        "bad_packets": receiver.bad_packets,
                        "sequence_gaps": receiver.sequence_gaps,
                        "sync_updates": receiver.sync_updates,
                        "clock_offset_ns": receiver.mapper.offset_ns,
                    },
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
        try:
            deactivate("shutdown")
        except Exception:
            LOG.exception("shutdown 时机器人停用失败")
        if robot is not None:
            try:
                robot.close()
            except Exception:
                LOG.exception("机器人最终释放失败")
        _status_put(status_queue, "control", "stopped", metrics=dict(metrics))


def _make_camera(cfg: dict[str, Any], name: str) -> Any:
    """只创建一路相机；该函数在对应的相机子进程内调用。"""
    cameras_cfg = cfg["cameras"]
    width = int(cameras_cfg["width"])
    height = int(cameras_cfg["height"])
    fps = int(cfg["rates"]["camera_hz"])
    if bool(cfg["robot"]["enabled"]):
        from hardware_adapters import IntelRealSenseColorCamera, OpenCVWristCamera

        if name == "scene":
            return IntelRealSenseColorCamera(
                cameras_cfg["scene"], "scene", width, height, fps
            )
        if name in {"wrist_left", "wrist_right"}:
            return OpenCVWristCamera(
                cameras_cfg[name], name, width, height, fps
            )
        raise KeyError(f"未知相机名称: {name}")

    from adapters import SimCamera

    return SimCamera(width, height, CAMERA_NAMES.index(name))


def _make_cameras(cfg: dict[str, Any]) -> dict[str, Any]:
    """兼容测试工具；正式采集路径会为每路相机单独创建进程。"""
    return {name: _make_camera(cfg, name) for name in CAMERA_NAMES}


def _camera_service_process(
    cfg: dict[str, Any],
    name: str,
    shm_name: str,
    image_shape: tuple[int, int, int],
    frame_seq: Any,
    frame_stamp_ns: Any,
    frame_phase: Any,
    frame_lock: Any,
    stop_event: Any,
    session_active: Any,
    session_id: Any,
    camera_status_queue: Any,
) -> None:
    """长期驻留的一路相机服务，通过共享内存发布最新 RGB 帧。

    服务进程由设备 B 主进程直接创建，与 recorder/control 为同级进程。idle 时
    不打开相机，只等待 ``session_active``；episode start 后连接并持续取流，stop
    后释放设备并等待下一次 start。
    """
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s [device-b/camera-{name}] %(message)s",
    )
    _configure_child_signals(stop_event)
    shm = shared_memory.SharedMemory(name=shm_name)
    image_view = np.ndarray(image_shape, dtype=np.uint8, buffer=shm.buf)
    episode_cfg = cfg.get("episode", {})
    camera_cfg = cfg.get("cameras", {}).get(name, {})
    startup_delay_ms = max(
        0.0, float(camera_cfg.get("startup_delay_ms", 0.0))
    )
    failure_timeout_ms = float(
        camera_cfg.get(
            "failure_timeout_ms",
            episode_cfg.get("camera_failure_timeout_ms", 1500),
        )
    )
    retry_delay_ms = max(0.0, float(
        camera_cfg.get(
            "retry_delay_ms",
            episode_cfg.get("camera_retry_delay_ms", 10),
        )
    ))
    report_interval = max(1, int(
        camera_cfg.get(
            "error_report_interval",
            episode_cfg.get("camera_error_report_interval", 30),
        )
    ))
    active_session = 0

    def report(kind: str, *, session: int, **values: Any) -> None:
        try:
            camera_status_queue.put_nowait(
                {
                    "camera": name,
                    "kind": kind,
                    "session": session,
                    "time_ns": time.perf_counter_ns(),
                    **values,
                }
            )
        except queue.Full:
            pass

    try:
        while not stop_event.is_set():
            if not session_active.wait(0.2):
                continue
            if stop_event.is_set():
                break

            active_session = int(session_id.value)
            # 左右腕和 RealSense 错峰启动，避免三个 USB/V4L2 后端同时初始化。
            if startup_delay_ms > 0 and stop_event.wait(
                startup_delay_ms / 1000.0
            ):
                break
            if not session_active.is_set():
                report(
                    "stopped",
                    session=active_session,
                    published_frames=0,
                    total_frame_errors=0,
                    frame_errors_by_reason={},
                )
                continue

            camera = None
            published_frames = 0
            total_frame_errors = 0
            consecutive_frame_errors = 0
            frame_errors_by_reason: dict[str, int] = {}
            last_valid_ns = time.perf_counter_ns()
            last_error_report_ns = 0
            outage_reported = False
            try:
                camera = _make_camera(cfg, name)
                camera.connect()
                ready_reported = False
                while session_active.is_set() and not stop_event.is_set():
                    try:
                        # phase: 1=阻塞取帧，2=转换/发布，0=等待下一次取帧。
                        frame_phase.value = 1
                        image, stamp_ns = camera.read()
                        frame_phase.value = 2
                    except Exception as exc:
                        frame_phase.value = 0
                        now_ns = time.perf_counter_ns()
                        total_frame_errors += 1
                        consecutive_frame_errors += 1
                        reason = str(getattr(exc, "reason", "read_error"))
                        frame_errors_by_reason[reason] = (
                            frame_errors_by_reason.get(reason, 0) + 1
                        )
                        last_valid_age_ms = (now_ns - last_valid_ns) / 1e6
                        should_report = (
                            now_ns - last_error_report_ns >= 1_000_000_000
                            or consecutive_frame_errors % report_interval == 0
                        )
                        if should_report:
                            report(
                                "frame_error",
                                session=active_session,
                                error=repr(exc),
                                error_type=type(exc).__name__,
                                reason=reason,
                                reason_total=frame_errors_by_reason[reason],
                                consecutive=consecutive_frame_errors,
                                total=total_frame_errors,
                                last_valid_age_ms=round(last_valid_age_ms, 3),
                            )
                            last_error_report_ns = now_ns
                            outage_reported = True
                        if last_valid_age_ms >= failure_timeout_ms:
                            raise RuntimeError(
                                f"{name} 连续 {last_valid_age_ms:.1f}ms "
                                f"没有有效图像，超过阈值 "
                                f"{failure_timeout_ms:.1f}ms；"
                                f"last_error={exc!r}"
                            ) from exc
                        stop_event.wait(retry_delay_ms / 1000.0)
                        continue

                    recovered_errors = consecutive_frame_errors
                    recovered_age_ms = (
                        time.perf_counter_ns() - last_valid_ns
                    ) / 1e6
                    consecutive_frame_errors = 0
                    image = np.asarray(image)
                    if image.shape != image_shape:
                        raise ValueError(
                            f"{name} 图像尺寸错误: {image.shape}, "
                            f"期望 {image_shape}"
                        )
                    if image.dtype != np.uint8:
                        image = image.astype(np.uint8, copy=False)
                    if not image.flags.c_contiguous:
                        image = np.ascontiguousarray(image)
                    with frame_lock:
                        image_view[:] = image
                        frame_stamp_ns.value = int(stamp_ns)
                        frame_seq.value += 1
                    frame_phase.value = 0
                    published_frames += 1
                    last_valid_ns = time.perf_counter_ns()
                    if recovered_errors and outage_reported:
                        report(
                            "recovered",
                            session=active_session,
                            recovered_errors=recovered_errors,
                            outage_ms=round(recovered_age_ms, 3),
                            total=total_frame_errors,
                        )
                    outage_reported = False
                    if not ready_reported:
                        describe = getattr(camera, "describe", None)
                        report(
                            "ready",
                            session=active_session,
                            details=(
                                describe() if callable(describe) else {}
                            ),
                        )
                        ready_reported = True
                    # 真实相机由 V4L2/RealSense 帧到达自然限速，必须持续排空。
                    # 仿真相机才需要显式等待，避免空转占满 CPU。
                    if not bool(cfg["robot"]["enabled"]):
                        stop_event.wait(
                            1.0 / float(cfg["rates"]["camera_hz"])
                        )
            except BaseException as exc:
                frame_phase.value = 0
                if not stop_event.is_set():
                    report(
                        "error",
                        session=active_session,
                        error=repr(exc),
                        traceback=traceback.format_exc(),
                        published_frames=published_frames,
                        total_frame_errors=total_frame_errors,
                        frame_errors_by_reason=dict(frame_errors_by_reason),
                    )
            finally:
                frame_phase.value = 3
                if camera is not None:
                    try:
                        camera.disconnect()
                    except BaseException as exc:
                        report(
                            "disconnect_error",
                            session=active_session,
                            error=repr(exc),
                        )
                frame_phase.value = 0
                report(
                    "stopped",
                    session=active_session,
                    published_frames=published_frames,
                    total_frame_errors=total_frame_errors,
                    frame_errors_by_reason=dict(frame_errors_by_reason),
                )
            # error 后由 recorder 设置全局 stop；正常 stop 时等待 active 被清除，
            # 避免在同一 episode 内立即重新打开设备。
            while session_active.is_set() and not stop_event.is_set():
                stop_event.wait(0.05)
    finally:
        frame_phase.value = 0
        shm.close()


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
    camera_channel_specs: dict[str, dict[str, Any]],
    camera_status_queue: Any,
    camera_session_active: Any,
    camera_session_id: Any,
) -> None:
    """记录进程：控制相机 session，recording 时轮询共享内存并写 episode。"""
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
    max_camera_error_ratio = float(
        episode_cfg.get("max_camera_error_ratio", 0.10)
    )
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
    worker_threads: list[threading.Thread] = []
    camera_channels: dict[str, dict[str, Any]] = {
        name: {
            **spec,
            "shm": shared_memory.SharedMemory(name=spec["shm_name"]),
        }
        for name, spec in camera_channel_specs.items()
    }
    active_camera_session = 0
    last_shared_camera_seq: dict[str, int] = {}
    writer = None
    writer_thread: threading.Thread | None = None
    recorder_ready = False
    episode_state = "idle"
    episode_session = 0
    episode_frames = 0
    episode_task = str(cfg["dataset"]["task"])
    episode_start_ns = 0
    episode_diagnostics: dict[str, Any] = {}
    camera_first_ns: dict[str, int] = {}
    camera_last_ns: dict[str, int] = {}
    camera_first_seq: dict[str, int] = {}
    camera_last_seq: dict[str, int] = {}
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

    def record_camera_status(status: dict[str, Any]) -> None:
        """合并相机进程诊断；瞬时坏帧只告警，不中断 episode。"""
        name = str(status.get("camera", "unknown"))
        kind = str(status.get("kind", ""))
        if "total" in status:
            key = f"camera_frame_errors_{name}"
            episode_diagnostics[key] = max(
                int(episode_diagnostics.get(key, 0)),
                int(status["total"]),
            )
        if "total_frame_errors" in status:
            key = f"camera_frame_errors_{name}"
            episode_diagnostics[key] = max(
                int(episode_diagnostics.get(key, 0)),
                int(status["total_frame_errors"]),
            )
        if "published_frames" in status:
            key = f"camera_published_frames_{name}"
            episode_diagnostics[key] = max(
                int(episode_diagnostics.get(key, 0)),
                int(status["published_frames"]),
            )
        for reason, count in status.get("frame_errors_by_reason", {}).items():
            key = f"camera_{reason}_{name}"
            episode_diagnostics[key] = max(
                int(episode_diagnostics.get(key, 0)),
                int(count),
            )

        if kind == "frame_error":
            reason = str(status.get("reason", "read_error"))
            reason_key = f"camera_{reason}_{name}"
            episode_diagnostics[reason_key] = max(
                int(episode_diagnostics.get(reason_key, 0)),
                int(status.get("reason_total", 1)),
            )
            episode_diagnostics[f"camera_last_valid_age_ms_{name}"] = float(
                status.get("last_valid_age_ms", 0.0)
            )
            LOG.warning(
                "%s 相机瞬时坏帧：reason=%s consecutive=%s total=%s "
                "last_valid_age_ms=%s error=%s",
                name,
                reason,
                status.get("consecutive"),
                status.get("total"),
                status.get("last_valid_age_ms"),
                status.get("error"),
            )
        elif kind == "recovered":
            key = f"camera_recoveries_{name}"
            episode_diagnostics[key] = (
                int(episode_diagnostics.get(key, 0)) + 1
            )
            episode_diagnostics[f"camera_last_recovery_ms_{name}"] = float(
                status.get("outage_ms", 0.0)
            )
            LOG.info(
                "%s 相机已恢复：连续坏帧=%s outage_ms=%s total=%s",
                name,
                status.get("recovered_errors"),
                status.get("outage_ms"),
                status.get("total"),
            )

    def start_capture() -> None:
        nonlocal active_camera_session
        for buffer in buffers.values():
            buffer.clear()
        episode_diagnostics.clear()
        camera_first_ns.clear()
        camera_last_ns.clear()
        camera_first_seq.clear()
        camera_last_seq.clear()
        last_group_camera_seq.clear()
        last_shared_camera_seq.clear()

        # 清除上一会话晚到的状态，并重置共享帧元数据。相机服务进程始终由
        # 设备 B 主进程持有，recorder 只通过 Event/Value 控制本次 session。
        while True:
            try:
                camera_status_queue.get_nowait()
            except queue.Empty:
                break
        for channel in camera_channels.values():
            with channel["lock"]:
                channel["seq"].value = -1
                channel["stamp_ns"].value = 0
            channel["phase"].value = 0
        with camera_session_id.get_lock():
            camera_session_id.value += 1
            active_camera_session = int(camera_session_id.value)

        try:
            camera_session_active.set()
            # 只有三路相机全部 connect 成功后，才进入 recording 并启动机器人。
            ready: set[str] = set()
            deadline = time.monotonic() + float(
                episode_cfg.get("camera_start_timeout_s", 15)
            )
            while ready != set(CAMERA_NAMES):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    missing = sorted(set(CAMERA_NAMES) - ready)
                    raise TimeoutError(f"相机启动超时，未就绪: {missing}")
                try:
                    status = camera_status_queue.get(timeout=min(0.2, remaining))
                except queue.Empty:
                    continue
                if int(status.get("session", -1)) != active_camera_session:
                    continue
                if status["kind"] == "ready":
                    camera_name = str(status["camera"])
                    ready.add(camera_name)
                    episode_diagnostics[
                        f"camera_properties_{camera_name}"
                    ] = status.get("details", {})
                    LOG.info(
                        "%s 相机进程就绪: %s",
                        camera_name,
                        status.get("details", {}),
                    )
                elif status["kind"] in {"frame_error", "recovered"}:
                    record_camera_status(status)
                elif status["kind"] == "error":
                    record_camera_status(status)
                    raise RuntimeError(
                        f"{status['camera']} 相机启动失败: {status.get('error')}"
                    )
        except BaseException:
            stop_capture(require_stopped=False)
            raise

    def stop_capture(*, require_stopped: bool = False) -> None:
        nonlocal active_camera_session
        if active_camera_session <= 0:
            return
        session = active_camera_session
        camera_session_active.clear()
        join_timeout_s = float(
            episode_cfg.get("camera_shutdown_timeout_s", 5)
        )
        stopped: set[str] = set()
        deadline = time.monotonic() + join_timeout_s
        while stopped != set(CAMERA_NAMES):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                status = camera_status_queue.get(
                    timeout=min(0.05, remaining)
                )
            except queue.Empty:
                continue
            if int(status.get("session", -1)) != session:
                continue
            record_camera_status(status)
            if status.get("kind") == "stopped":
                stopped.add(str(status.get("camera")))
            elif status.get("kind") == "disconnect_error":
                LOG.warning(
                    "%s 相机断开异常: %s",
                    status.get("camera"),
                    status.get("error"),
                )

        missing_stopped = sorted(set(CAMERA_NAMES) - stopped)
        for name in CAMERA_NAMES:
            source_frames = (
                camera_last_seq.get(name, -1) - camera_first_seq.get(name, 0) + 1
            )
            elapsed_ns = camera_last_ns.get(name, 0) - camera_first_ns.get(name, 0)
            source_fps = (
                (source_frames - 1) * 1e9 / elapsed_ns
                if source_frames > 1 and elapsed_ns > 0
                else 0.0
            )
            episode_diagnostics[f"source_frames_{name}"] = max(0, source_frames)
            episode_diagnostics[f"source_fps_{name}"] = round(source_fps, 3)
            frame_errors = int(
                episode_diagnostics.get(f"camera_frame_errors_{name}", 0)
            )
            published_frames = int(
                episode_diagnostics.get(
                    f"camera_published_frames_{name}",
                    max(0, source_frames),
                )
            )
            error_denominator = published_frames + frame_errors
            error_ratio = (
                frame_errors / error_denominator
                if error_denominator > 0
                else 0.0
            )
            episode_diagnostics[f"camera_error_ratio_{name}"] = round(
                error_ratio, 4
            )
        active_camera_session = 0
        if missing_stopped:
            message = (
                f"相机服务未在 {join_timeout_s:.1f}s 内结束 session "
                f"{session}: {missing_stopped}"
            )
            if require_stopped:
                raise RuntimeError(message)
            LOG.warning(message)

    def poll_camera_processes() -> None:
        """把共享内存中的新帧复制到 recorder 的时间缓冲区。"""
        while True:
            try:
                status = camera_status_queue.get_nowait()
            except queue.Empty:
                break
            if int(status.get("session", -1)) != active_camera_session:
                continue
            record_camera_status(status)
            if status["kind"] == "error":
                raise RuntimeError(
                    f"{status['camera']} 相机采集失败: {status.get('error')}；"
                    f"frame_errors={status.get('frame_errors_by_reason', {})}"
                )
            if status["kind"] == "disconnect_error":
                LOG.warning(
                    "%s 相机断开异常: %s",
                    status["camera"],
                    status.get("error"),
                )
            if status["kind"] == "stopped" and episode_state == "recording":
                raise RuntimeError(f"{status['camera']} 相机服务意外停止")

        for name, channel in camera_channels.items():
            with channel["lock"]:
                seq = int(channel["seq"].value)
                stamp_ns = int(channel["stamp_ns"].value)
                is_new = (
                    seq >= 0
                    and seq != last_shared_camera_seq.get(name, -1)
                )
                image = (
                    np.ndarray(
                        channel["shape"],
                        dtype=np.uint8,
                        buffer=channel["shm"].buf,
                    ).copy()
                    if is_new
                    else None
                )

            # 必须在读取共享时间戳之后再读取本机单调时钟。相机进程可能在
            # recorder 进入本轮循环后发布新帧；若复用循环开始前的 now_ns，
            # 就可能出现 now_ns < stamp_ns，从而得到负的帧年龄。
            snapshot_ns = time.perf_counter_ns()

            # OpenCV V4L2 通常不接受 CAP_PROP_READ_TIMEOUT_MSEC。若子进程
            # 永久阻塞在 VideoCapture.read()，它仍然存活且不会报告异常，
            # 因此 recorder 必须根据共享帧时间戳独立检测停流。
            if seq >= 0 and stamp_ns > 0:
                camera_cfg = cfg.get("cameras", {}).get(name, {})
                stall_timeout_ms = float(
                    camera_cfg.get(
                        "failure_timeout_ms",
                        episode_cfg.get("camera_failure_timeout_ms", 1500),
                    )
                )
                frame_age_ms = max(0.0, (snapshot_ns - stamp_ns) / 1e6)
                if frame_age_ms >= stall_timeout_ms:
                    phase = int(channel["phase"].value)
                    phase_name = {
                        0: "idle",
                        1: "read",
                        2: "publish",
                        3: "disconnect",
                    }.get(phase, f"unknown({phase})")
                    episode_diagnostics[
                        f"camera_stall_age_ms_{name}"
                    ] = round(frame_age_ms, 3)
                    episode_diagnostics[
                        f"camera_stall_phase_{name}"
                    ] = phase_name
                    raise RuntimeError(
                        f"{name} 相机共享帧已停滞 {frame_age_ms:.1f}ms，"
                        f"超过阈值 {stall_timeout_ms:.1f}ms；"
                        f"相机服务阶段={phase_name}"
                    )

            if not is_new or image is None:
                continue
            last_shared_camera_seq[name] = seq
            if stamp_ns <= camera_last_ns.get(name, -1):
                key = f"duplicate_{name}"
                episode_diagnostics[key] = episode_diagnostics.get(key, 0) + 1
                continue
            buffers[name].append(
                TimedSample(name, seq, stamp_ns, stamp_ns, image)
            )
            camera_first_ns.setdefault(name, stamp_ns)
            camera_last_ns[name] = stamp_ns
            camera_first_seq.setdefault(name, seq)
            camera_last_seq[name] = seq
            key = f"camera_{name}"
            episode_diagnostics[key] = episode_diagnostics.get(key, 0) + 1

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
        stop_capture(require_stopped=not stop_event.is_set())
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
            error_ratio = float(
                episode_diagnostics.get(f"camera_error_ratio_{name}", 0.0)
            )
            if error_ratio > max_camera_error_ratio:
                quality_failures.append(
                    f"{name}: error_ratio={error_ratio:.3f}>"
                    f"{max_camera_error_ratio:.3f}"
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

        recorder_ready = True
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
                poll_camera_processes()
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
                camera_runtime: dict[str, Any] = {}
                for name, channel in camera_channels.items():
                    with channel["lock"]:
                        seq = int(channel["seq"].value)
                        stamp_ns = int(channel["stamp_ns"].value)
                    # 各相机共享时间戳读取完成后分别采样当前时间，避免相机
                    # 在本轮 metrics 开始后更新 stamp_ns 而产生负的 age。
                    snapshot_ns = time.perf_counter_ns()
                    phase = int(channel["phase"].value)
                    camera_runtime[name] = {
                        "seq": seq,
                        "last_frame_age_ms": (
                            round(max(0, snapshot_ns - stamp_ns) / 1e6, 3)
                            if stamp_ns > 0
                            else None
                        ),
                        "phase": {
                            0: "idle",
                            1: "read",
                            2: "publish",
                            3: "disconnect",
                        }.get(phase, f"unknown({phase})"),
                    }
                _status_put(
                    status_queue,
                    "recorder",
                    "metrics",
                    metrics=dict(metrics),
                    cameras=camera_runtime,
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
        component = "startup" if not recorder_ready else f"runtime_{episode_state}"
        fail(component, exc)
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

        stop_capture(require_stopped=False)
        for name, channel in camera_channels.items():
            try:
                channel["shm"].close()
            except Exception:
                LOG.exception("%s recorder共享内存句柄关闭失败", name)
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
    """设备 B 父进程：监督控制、记录及三路相机进程，并提供人工控制台。"""
    import sys

    runtime_cfg = cfg.get("runtime", {})
    episode_cfg = cfg.get("episode", {})
    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    sample_queue = ctx.Queue(maxsize=int(runtime_cfg.get("ipc_queue_capacity", 2048)))
    episode_queue = ctx.Queue(maxsize=int(episode_cfg.get("command_queue_capacity", 32)))
    control_queue = ctx.Queue(maxsize=int(episode_cfg.get("command_queue_capacity", 32)))
    status_queue = ctx.Queue(maxsize=int(runtime_cfg.get("status_queue_capacity", 128)))
    camera_status_queue = ctx.Queue(
        maxsize=int(runtime_cfg.get("status_queue_capacity", 128))
    )
    camera_session_active = ctx.Event()
    camera_session_id = ctx.Value("q", 0)
    hardware_channels: dict[str, Any] | None = None
    if bool(cfg["robot"].get("enabled", False)):
        from hardware_processes import create_hardware_channels

        hardware_channels = create_hardware_channels(ctx, cfg["robot"])
    image_shape = (
        int(cfg["cameras"]["height"]),
        int(cfg["cameras"]["width"]),
        3,
    )
    image_nbytes = int(np.prod(image_shape, dtype=np.int64))
    camera_shms: dict[str, Any] = {}
    camera_channel_specs: dict[str, dict[str, Any]] = {}
    for name in CAMERA_NAMES:
        shm = shared_memory.SharedMemory(create=True, size=image_nbytes)
        camera_shms[name] = shm
        camera_channel_specs[name] = {
            "shm_name": shm.name,
            "shape": image_shape,
            "seq": ctx.Value("q", -1),
            "stamp_ns": ctx.Value("q", 0),
            "phase": ctx.Value("i", 0),
            "lock": ctx.Lock(),
        }

    processes = {
        "recorder": ctx.Process(
            name="device-b-recorder",
            target=_recorder_process,
            args=(
                cfg,
                stop_event,
                sample_queue,
                episode_queue,
                status_queue,
                camera_channel_specs,
                camera_status_queue,
                camera_session_active,
                camera_session_id,
            ),
        ),
        "control": ctx.Process(
            name="device-b-control",
            target=_control_process,
            args=(
                cfg,
                stop_event,
                sample_queue,
                control_queue,
                status_queue,
                hardware_channels,
            ),
        ),
    }
    if hardware_channels is not None:
        from hardware_processes import arm_hardware_process, hand_hardware_process

        for side in SIDES:
            processes[f"arm_{side}"] = ctx.Process(
                name=f"device-b-arm-{side}",
                target=arm_hardware_process,
                args=(
                    cfg["robot"],
                    side,
                    hardware_channels[f"arm_{side}"],
                    stop_event,
                    status_queue,
                ),
            )
            processes[f"hand_{side}"] = ctx.Process(
                name=f"device-b-hand-{side}",
                target=hand_hardware_process,
                args=(
                    cfg["robot"],
                    side,
                    hardware_channels[f"hand_{side}"],
                    stop_event,
                    status_queue,
                ),
            )
    for name in CAMERA_NAMES:
        channel = camera_channel_specs[name]
        processes[f"camera_{name}"] = ctx.Process(
            name=f"device-b-camera-{name}",
            target=_camera_service_process,
            args=(
                cfg,
                name,
                channel["shm_name"],
                channel["shape"],
                channel["seq"],
                channel["stamp_ns"],
                channel["phase"],
                channel["lock"],
                stop_event,
                camera_session_active,
                camera_session_id,
                camera_status_queue,
            ),
        )
    errors: list[dict[str, Any]] = []
    desired_active = bool(episode_cfg.get("auto_start", False))
    home_in_progress = False
    start_prepare_in_progress = False
    pending_start_task: str | None = None

    def request_stop(*_: Any) -> None:
        stop_event.set()

    def submit_episode_command(kind: str, task: str | None = None) -> None:
        nonlocal desired_active, start_prepare_in_progress
        nonlocal pending_start_task
        if kind == "start" and home_in_progress:
            LOG.warning("home 尚未完成，拒绝 start")
            return
        if kind == "start":
            if start_prepare_in_progress or desired_active:
                LOG.warning("start 正在准备或 episode 已在运行，拒绝重复 start")
                return
            desired_active = True
            start_prepare_in_progress = True
            pending_start_task = task
            try:
                # 先让双臂回到已标定 home_pose。只有准备成功后才启动相机，
                # 因此 move_j 轨迹不会进入本 episode 的训练数据。
                control_queue.put_nowait({"kind": "prepare_start"})
            except queue.Full:
                desired_active = False
                start_prepare_in_progress = False
                pending_start_task = None
                LOG.error("control 命令队列已满，start 准备命令被拒绝")
            return
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

    def submit_control_command(kind: str) -> None:
        nonlocal home_in_progress
        if kind == "home":
            if desired_active:
                LOG.warning("episode 正在启动或运行，拒绝 home")
                return
            if home_in_progress:
                LOG.warning("home 已在执行，拒绝重复命令")
                return
            home_in_progress = True
        try:
            control_queue.put_nowait({"kind": kind})
        except queue.Full:
            if kind == "home":
                home_in_progress = False
            LOG.error("control 命令队列已满，命令被拒绝：%s", kind)

    def console_loop() -> None:
        LOG.info(
            "控制命令：home | start [任务描述] | stop | discard | status | quit"
        )
        while not stop_event.is_set():
            line = sys.stdin.readline()
            if line == "":
                return
            command, _, value = line.strip().partition(" ")
            command = command.lower()
            if command in {"start", "stop", "discard", "status"}:
                submit_episode_command(command, value or None)
            elif command == "home":
                # home 只操作已连接的机器人，不启动相机，也不创建 episode。
                submit_control_command("home")
            elif command in {"quit", "exit", "q"}:
                stop_event.set()
                return
            elif command:
                LOG.warning("未知命令：%s", command)

    old_sigint = signal.signal(signal.SIGINT, request_stop)
    old_sigterm = signal.signal(signal.SIGTERM, request_stop)
    console_thread: threading.Thread | None = None
    try:
        # 四个硬件进程最先启动并跨 episode 常驻；每个进程独占一个 SDK 句柄。
        for name in ("arm_left", "arm_right", "hand_left", "hand_right"):
            if name in processes:
                processes[name].start()
        # 相机服务先启动，但在 session_active 置位前不会连接或采集硬件。
        for name in CAMERA_NAMES:
            processes[f"camera_{name}"].start()
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
                elif kind in {"hardware_initializing", "hardware_ready"}:
                    LOG.info("HARDWARE %s", json.dumps(status, ensure_ascii=False))
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
                        try:
                            control_queue.put_nowait(
                                {"kind": "cancel_prepare"}
                            )
                        except queue.Full:
                            LOG.error(
                                "episode start 被拒绝，但无法提交双臂 hold"
                            )
                elif str(kind).startswith("control_"):
                    LOG.info("CONTROL %s", json.dumps(status, ensure_ascii=False))
                    if kind == "control_start_prepared":
                        start_prepare_in_progress = False
                        if desired_active:
                            try:
                                episode_queue.put(
                                    {
                                        "kind": "start",
                                        "task": pending_start_task,
                                    },
                                    timeout=2,
                                )
                            except queue.Full:
                                desired_active = False
                                LOG.error(
                                    "机械臂已回 home，但 episode 命令队列已满"
                                )
                                try:
                                    control_queue.put_nowait(
                                        {"kind": "cancel_prepare"}
                                    )
                                except queue.Full:
                                    LOG.error(
                                        "无法提交双臂 prepared 状态取消命令"
                                    )
                        else:
                            try:
                                control_queue.put_nowait(
                                    {"kind": "cancel_prepare"}
                                )
                            except queue.Full:
                                LOG.error(
                                    "start 已取消，但无法提交双臂 hold"
                                )
                        pending_start_task = None
                    elif kind == "control_start_prepare_failed" or (
                        kind == "control_rejected"
                        and status.get("command") == "prepare_start"
                    ):
                        start_prepare_in_progress = False
                        desired_active = False
                        pending_start_task = None
                    if kind in {
                        "control_home_completed",
                        "control_home_failed",
                    } or (
                        kind == "control_rejected"
                        and status.get("command") == "home"
                    ):
                        home_in_progress = False
                    if (
                        kind == "control_rejected"
                        and status.get("command") == "start"
                        and desired_active
                    ):
                        # 相机已经进入 recording，但机器人未能安全使能；
                        # 自动丢弃本次 episode，避免保存没有真实动作的数据。
                        desired_active = False
                        try:
                            episode_queue.put_nowait({"kind": "discard"})
                        except queue.Full:
                            LOG.error("机器人 start 被拒绝，但 episode 队列已满")
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
        camera_session_active.clear()
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
        for ipc_queue in (
            sample_queue,
            episode_queue,
            control_queue,
            status_queue,
            camera_status_queue,
        ):
            ipc_queue.close()
            ipc_queue.cancel_join_thread()
        if hardware_channels is not None:
            from hardware_processes import close_hardware_channels

            close_hardware_channels(hardware_channels)
        for name, shm in camera_shms.items():
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                LOG.exception("%s 主进程共享内存清理失败", name)
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)

    if errors:
        first = errors[0]
        raise RuntimeError(f"{first.get('process')} 子进程失败: {first.get('error')}")
