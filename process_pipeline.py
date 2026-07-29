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
    cfg: dict[str, Any], stop_event: Any, sample_queue: Any, status_queue: Any
) -> None:
    """控制进程：UDP、重定向、安全门、机器人命令和状态读取。"""
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
    }
    robot = None
    receiver = None
    receiver_thread: threading.Thread | None = None
    state_thread: threading.Thread | None = None

    def ingest(sample: TimedSample) -> None:
        if sample.source == "teleop":
            teleop_buffer.append(sample)
            _put_latest(sample_queue, sample)

    try:
        robot, retargeter, safety = _make_control_components(cfg)
        # 先连接机器人，再绑定 UDP；机器人启动失败时不会占住端口。
        robot.connect()
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

        def state_loop() -> None:
            state_seq = 0

            def read_state(_: int) -> None:
                nonlocal state_seq
                stamp_ns = time.perf_counter_ns()
                state = robot.read_state()
                _put_latest(
                    sample_queue,
                    TimedSample("robot_state", state_seq, stamp_ns, stamp_ns, state),
                )
                state_seq += 1
                metrics["state_ticks"] += 1

            try:
                _periodic(stop_event, float(cfg["rates"]["state_hz"]), read_state)
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

        state_thread = threading.Thread(target=state_loop, name="robot-state", daemon=True)
        state_thread.start()
        _status_put(status_queue, "control", "ready", udp_port=cfg["network"]["data_port"])

        last_report_ns = time.perf_counter_ns()
        period_ns = int(1e9 / float(cfg["rates"]["control_hz"]))
        deadline_ns = time.perf_counter_ns()
        while not stop_event.is_set():
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
                    LOG.warning("丢弃非法双侧遥操作包 seq=%s", selected.seq, exc_info=True)

            metrics["control_ticks"] += 1
            now_ns = time.perf_counter_ns()
            if now_ns - last_report_ns >= 1_000_000_000:
                _status_put(status_queue, "control", "metrics", metrics=dict(metrics))
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
    buffers: dict[str, TimeBuffer], target_ns: int, max_lag_ns: int
) -> dict[str, Any] | None:
    selected = {
        name: buffers[name].select_before(target_ns, max_lag_ns)
        for name in ALIGNMENT_NAMES
    }
    if any(value.sample is None for value in selected.values()):
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
    }


def _recorder_process(
    cfg: dict[str, Any], stop_event: Any, sample_queue: Any, status_queue: Any
) -> None:
    """采集进程：三路相机、时间对齐、组帧、图像编码和 LeRobot 写盘。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [device-b/recorder] %(message)s",
    )
    _configure_child_signals(stop_event)
    # 延迟导入，确保控制进程不加载 LeRobot/PyAV/编码器依赖。
    from dataset import feature_schema, make_writer

    buffers = {
        name: TimeBuffer(int(cfg["alignment"]["buffer_capacity"]))
        for name in ALIGNMENT_NAMES
    }
    writer_queue: queue.Queue[dict[str, Any] | None] = queue.Queue(
        int(cfg["dataset"]["queue_capacity"])
    )
    metrics = {"frame_ticks": 0, "written_frames": 0, "writer_drops": 0}
    cameras: dict[str, Any] = {}
    connected_cameras: list[str] = []
    worker_threads: list[threading.Thread] = []
    writer = None
    writer_thread: threading.Thread | None = None

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

    try:
        cameras = _make_cameras(cfg)
        schema = feature_schema(
            int(cfg["cameras"]["height"]),
            int(cfg["cameras"]["width"]),
            int(cfg["robot"]["hand_dof"]),
        )
        writer = make_writer(cfg["dataset"], schema, int(cfg["rates"]["frame_hz"]))
        for name in CAMERA_NAMES:
            cameras[name].connect()
            connected_cameras.append(name)

        def ipc_loop() -> None:
            while not stop_event.is_set():
                try:
                    sample = sample_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if isinstance(sample, TimedSample) and sample.source in buffers:
                    buffers[sample.source].append(sample)

        def camera_loop(name: str) -> None:
            seq = 0
            try:
                def read_camera(_: int) -> None:
                    nonlocal seq
                    image, stamp_ns = cameras[name].read()
                    buffers[name].append(
                        TimedSample(name, seq, stamp_ns, stamp_ns, image)
                    )
                    seq += 1

                _periodic(
                    stop_event, float(cfg["rates"]["camera_hz"]), read_camera
                )
            except BaseException as exc:
                if not stop_event.is_set():
                    fail(name, exc)

        def writer_loop() -> None:
            try:
                while True:
                    frame = writer_queue.get()
                    if frame is None:
                        break
                    writer.add_frame(frame)
                    metrics["written_frames"] += 1
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
        worker_threads.extend(
            threading.Thread(
                target=camera_loop, args=(name,), name=f"camera-{name}", daemon=True
            )
            for name in CAMERA_NAMES
        )
        writer_thread = threading.Thread(
            target=writer_loop, name="dataset-writer", daemon=False
        )
        worker_threads.append(writer_thread)
        for thread in worker_threads:
            thread.start()

        _status_put(status_queue, "recorder", "ready", cameras=list(CAMERA_NAMES))
        max_lag_ns = int(float(cfg["alignment"]["max_lag_ms"]) * 1e6)
        frame_period_ns = int(1e9 / float(cfg["rates"]["frame_hz"]))
        deadline_ns = time.perf_counter_ns()
        last_report_ns = deadline_ns
        while not stop_event.is_set():
            frame = _build_frame(buffers, time.perf_counter_ns(), max_lag_ns)
            if frame is not None:
                try:
                    writer_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        writer_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        writer_queue.put_nowait(frame)
                    except queue.Full:
                        pass
                    metrics["writer_drops"] += 1
                metrics["frame_ticks"] += 1

            now_ns = time.perf_counter_ns()
            if now_ns - last_report_ns >= 1_000_000_000:
                _status_put(status_queue, "recorder", "metrics", metrics=dict(metrics))
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
            try:
                writer_queue.put(None, timeout=1)
            except queue.Full:
                try:
                    writer_queue.get_nowait()
                    writer_queue.put_nowait(None)
                except (queue.Empty, queue.Full):
                    pass
            writer_thread.join(
                timeout=float(
                    cfg.get("runtime", {}).get("writer_shutdown_timeout_s", 15)
                )
            )
            if writer_thread.is_alive():
                fail(
                    "dataset_writer_shutdown",
                    TimeoutError("数据集写线程未在限定时间内完成刷新"),
                )
        elif writer is not None:
            # 相机启动失败发生在写线程创建之前时，也必须释放数据集资源。
            try:
                writer.close()
            except BaseException as exc:
                fail("dataset_finalize", exc)
        for name in reversed(connected_cameras):
            try:
                cameras[name].disconnect()
            except Exception:
                LOG.exception("%s 相机断开失败", name)
        _status_put(status_queue, "recorder", "stopped", metrics=dict(metrics))


def run_robot_multiprocess(
    cfg: dict[str, Any], run_seconds: float | None = None
) -> None:
    """设备 B 多进程入口；父进程只负责监督和统一退出。"""
    runtime_cfg = cfg.get("runtime", {})
    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    sample_queue = ctx.Queue(maxsize=int(runtime_cfg.get("ipc_queue_capacity", 2048)))
    status_queue = ctx.Queue(maxsize=int(runtime_cfg.get("status_queue_capacity", 128)))
    processes = {
        "recorder": ctx.Process(
            name="device-b-recorder",
            target=_recorder_process,
            args=(cfg, stop_event, sample_queue, status_queue),
        ),
        "control": ctx.Process(
            name="device-b-control",
            target=_control_process,
            args=(cfg, stop_event, sample_queue, status_queue),
        ),
    }
    errors: list[dict[str, Any]] = []

    def request_stop(*_: Any) -> None:
        stop_event.set()

    old_sigint = signal.signal(signal.SIGINT, request_stop)
    old_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        processes["recorder"].start()
        processes["control"].start()
        end_time = None if run_seconds is None else time.monotonic() + run_seconds
        while not stop_event.is_set():
            if end_time is not None and time.monotonic() >= end_time:
                stop_event.set()
                break
            try:
                status = status_queue.get(timeout=0.5)
                if status.get("kind") == "error":
                    errors.append(status)
                    LOG.error(
                        "%s 子进程失败（%s）：%s",
                        status.get("process"),
                        status.get("component", "runtime"),
                        status.get("error"),
                    )
                    stop_event.set()
                elif status.get("kind") == "ready":
                    LOG.info("%s 子进程已就绪", status.get("process"))
                elif status.get("kind") == "metrics":
                    LOG.info("%s", json.dumps(status, ensure_ascii=False))
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
        # stop_event.set()
        # shutdown_timeout_s = float(runtime_cfg.get("shutdown_timeout_s", 8))
        # for process in processes.values():
        #     if process.pid is not None:
        #         process.join(timeout=shutdown_timeout_s)
        # for process in processes.values():
        #     if process.is_alive():
        #         LOG.error(
        #             "%s 未在 %.1fs 内退出，执行强制终止",
        #             process.name,
        #             shutdown_timeout_s,
        #         )
        #         process.terminate()
        #         process.join(timeout=3)
        #         if process.is_alive():
        #             LOG.error("%s 仍未退出，执行 kill", process.name)
        #             process.kill()
        #             process.join(timeout=2)
        # 子进程可能先设置共享停止事件再投递错误；退出前必须排空诊断队列。
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
        sample_queue.close()
        sample_queue.cancel_join_thread()
        status_queue.close()
        status_queue.cancel_join_thread()
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)

    if errors:
        first = errors[0]
        raise RuntimeError(f"{first.get('process')} 子进程失败: {first.get('error')}")
