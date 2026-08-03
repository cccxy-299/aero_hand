from __future__ import annotations

"""四个常驻硬件进程及 control 侧代理。

每个真实硬件实例只在所属 spawn 子进程中创建，并且只由该进程主线程访问。
机械臂进程退出、发生异常或收到 stop 时不会调用 ``disable()``，而是停止发送
新目标并保持控制器当前使能状态。机械臂承重安全必须另外依靠硬件急停、制动器和
机械限位，不能依赖 Python 进程退出逻辑。
"""

import logging
import os
import queue
import signal
import time
import traceback
from typing import Any

import numpy as np

from model import (
    BimanualControlCommand,
    BimanualRobotState,
    ControlCommand,
    RobotState,
)


SIDES = ("left", "right")
PHASE_STARTING = 0
PHASE_HOLD = 1
PHASE_PREPARED = 2
PHASE_ACTIVE = 3
PHASE_FAULT_HOLD = 4
PHASE_EXIT_HOLD = 5
PHASE_NAMES = {
    PHASE_STARTING: "starting",
    PHASE_HOLD: "hold_enabled",
    PHASE_PREPARED: "prepared_enabled",
    PHASE_ACTIVE: "active",
    PHASE_FAULT_HOLD: "fault_hold_enabled",
    PHASE_EXIT_HOLD: "exit_hold_enabled",
}
ERROR_TEXT_BYTES = 4096


def create_hardware_channels(
    ctx: Any,
    cfg: dict[str, Any],
    *,
    include_hands: bool = True,
) -> dict[str, Any]:
    """创建硬件进程所需的有界 IPC；测试时可只创建双臂通道。"""
    capacity = int(cfg.get("hardware_command_queue_capacity", 16))
    channels: dict[str, Any] = {}
    for side in SIDES:
        channels[f"arm_{side}"] = {
            "kind": "arm",
            "side": side,
            "control_queue": ctx.Queue(maxsize=capacity),
            "target_queue": ctx.Queue(maxsize=1),
            "response_queue": ctx.Queue(maxsize=capacity),
            "ready_event": ctx.Event(),
            # control可跨进程立即置位；即使SDK线程暂时阻塞，也不会再接受后续目标。
            "hold_event": ctx.Event(),
            "fault": ctx.Value("b", False),
            "error_text": ctx.Array("B", ERROR_TEXT_BYTES, lock=True),
            "phase": ctx.Value("i", PHASE_STARTING),
            "state_lock": ctx.Lock(),
            "pose": ctx.Array("d", 6, lock=False),
            "joints": ctx.Array("d", 6, lock=False),
            "stamp_ns": ctx.Value("q", 0),
            "applied_seq": ctx.Value("q", -1),
            "submitted": ctx.Value("q", 0),
            "received": ctx.Value("q", 0),
            "applied": ctx.Value("q", 0),
            "failed": ctx.Value("q", 0),
            "last_io_ms": ctx.Value("d", 0.0),
            "max_io_ms": ctx.Value("d", 0.0),
        }
        if not include_hands:
            continue
        channels[f"hand_{side}"] = {
            "kind": "hand",
            "side": side,
            "control_queue": ctx.Queue(maxsize=capacity),
            "target_queue": ctx.Queue(maxsize=1),
            "response_queue": ctx.Queue(maxsize=capacity),
            "ready_event": ctx.Event(),
            "hold_event": ctx.Event(),
            "fault": ctx.Value("b", False),
            "error_text": ctx.Array("B", ERROR_TEXT_BYTES, lock=True),
            "phase": ctx.Value("i", PHASE_STARTING),
            "state_lock": ctx.Lock(),
            "joints": ctx.Array("d", 7, lock=False),
            "stamp_ns": ctx.Value("q", 0),
            "applied_seq": ctx.Value("q", -1),
            "submitted": ctx.Value("q", 0),
            "received": ctx.Value("q", 0),
            "applied": ctx.Value("q", 0),
            "failed": ctx.Value("q", 0),
            "last_io_ms": ctx.Value("d", 0.0),
            "max_io_ms": ctx.Value("d", 0.0),
        }
    return channels


def close_hardware_channels(channels: dict[str, Any]) -> None:
    """只在设备B父进程的最终清理阶段调用。"""
    for channel in channels.values():
        for name in ("control_queue", "target_queue", "response_queue"):
            ipc_queue = channel[name]
            ipc_queue.close()
            ipc_queue.cancel_join_thread()


def _configure_signals(stop_event: Any) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())


def _status_put(
    status_queue: Any, process: str, kind: str, **values: Any
) -> None:
    item = {
        "process": process,
        "kind": kind,
        "time_ns": time.perf_counter_ns(),
        **values,
    }
    try:
        status_queue.put_nowait(item)
    except queue.Full:
        pass


def _put_response(channel: dict[str, Any], request: dict[str, Any], **values: Any) -> None:
    response = {
        "request_id": int(request.get("request_id", -1)),
        "command": str(request.get("kind", "")),
        **values,
    }
    try:
        channel["response_queue"].put_nowait(response)
    except queue.Full:
        # 生命周期 ACK 不能静默覆盖；让设备进程进入 fault，父进程会终止会话。
        channel["fault"].value = True
        raise RuntimeError("硬件生命周期响应队列已满")


def _put_latest_target(target_queue: Any, value: dict[str, Any]) -> None:
    try:
        target_queue.put_nowait(value)
        return
    except queue.Full:
        pass
    try:
        target_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        target_queue.put_nowait(value)
    except queue.Full:
        pass


def _drain_latest(target_queue: Any) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    while True:
        try:
            latest = target_queue.get_nowait()
        except queue.Empty:
            return latest


def _array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    raw = getattr(value, "msg", value)
    result = np.asarray(raw, dtype=np.float32)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise RuntimeError(f"{label} 维度或数值非法: {result.shape}")
    return result


def _set_phase(channel: dict[str, Any], phase: int) -> None:
    channel["phase"].value = int(phase)


def _set_channel_error(channel: dict[str, Any], error: BaseException | str) -> None:
    """错误写入共享内存，避免子进程快速退出时Queue诊断尚未刷新。"""
    text = error if isinstance(error, str) else repr(error)
    encoded = text.encode("utf-8", errors="replace")[: ERROR_TEXT_BYTES - 1]
    shared = channel["error_text"]
    with shared.get_lock():
        shared[:] = [0] * ERROR_TEXT_BYTES
        shared[: len(encoded)] = list(encoded)


def _get_channel_error(channel: dict[str, Any]) -> str | None:
    shared = channel["error_text"]
    with shared.get_lock():
        raw = bytes(shared[:])
    value = raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")
    return value or None


def _update_arm_feedback(
    channel: dict[str, Any], arm: Any, applied_seq: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    pose = _array(arm.get_flange_pose(), (6,), "Piper flange_pose")
    joints = _array(arm.get_joint_angles(), (6,), "Piper joint_angles")
    stamp_ns = time.perf_counter_ns()
    with channel["state_lock"]:
        channel["pose"][:] = pose.astype(np.float64)
        channel["joints"][:] = joints.astype(np.float64)
        channel["stamp_ns"].value = stamp_ns
        if applied_seq is not None:
            channel["applied_seq"].value = int(applied_seq)
    return pose, joints


def _update_hand_feedback(
    channel: dict[str, Any], hand: Any, applied_seq: int | None = None
) -> np.ndarray:
    joints = _array(hand.get_joint_positions_compact(), (7,), "Aerohand joints")
    stamp_ns = time.perf_counter_ns()
    with channel["state_lock"]:
        channel["joints"][:] = joints.astype(np.float64)
        channel["stamp_ns"].value = stamp_ns
        if applied_seq is not None:
            channel["applied_seq"].value = int(applied_seq)
    return joints


def _wait_initial_arm_feedback(
    channel: dict[str, Any],
    arm: Any,
    side_cfg: dict[str, Any],
    side: str,
) -> None:
    """connect后SDK/CAN反馈可能尚未就绪，必须在健康超时内重试。"""
    timeout_s = float(side_cfg.get("health_timeout_s", 3.0))
    deadline = time.monotonic() + timeout_s
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            _update_arm_feedback(channel, arm)
            return
        except BaseException as exc:
            last_error = exc
            time.sleep(0.05)
    raise RuntimeError(
        f"{side} Piper connect后{timeout_s:.1f}s内没有有效反馈；"
        f"last_error={last_error!r}"
    )


def _wait_initial_hand_feedback(
    channel: dict[str, Any],
    hand: Any,
    side_cfg: dict[str, Any],
    side: str,
) -> None:
    timeout_s = float(side_cfg.get("health_timeout_s", 3.0))
    deadline = time.monotonic() + timeout_s
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            _update_hand_feedback(channel, hand)
            return
        except BaseException as exc:
            last_error = exc
            time.sleep(0.05)
    raise RuntimeError(
        f"{side} Aerohand连接后{timeout_s:.1f}s内没有有效反馈；"
        f"last_error={last_error!r}"
    )


def _enable_arm(arm: Any, side_cfg: dict[str, Any], side: str) -> None:
    deadline = time.monotonic() + float(side_cfg.get("enable_timeout_s", 5.0))
    while not arm.enable():
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{side} Piper 使能超时")
        time.sleep(0.05)


def _wait_arm_home(
    arm: Any,
    channel: dict[str, Any],
    side_cfg: dict[str, Any],
    side: str,
    target: np.ndarray,
) -> None:
    deadline = time.monotonic() + float(side_cfg.get("home_timeout_s", 10.0))
    tolerance = float(side_cfg.get("home_joint_tolerance_rad", 0.10))
    time.sleep(0.3)
    last_motion: Any = None
    last_error = float("inf")
    while time.monotonic() < deadline:
        status = arm.get_arm_status()
        message = getattr(status, "msg", status)
        last_motion = getattr(message, "motion_status", None)
        _, joints = _update_arm_feedback(channel, arm)
        delta = np.arctan2(np.sin(joints - target), np.cos(joints - target))
        last_error = float(np.max(np.abs(delta)))
        if last_motion == 0 and last_error <= tolerance:
            return
        time.sleep(0.05)
    raise RuntimeError(
        f"{side} Piper 未到达 home_pose: motion_status={last_motion}, "
        f"max_joint_error={last_error:.4f}rad"
    )


def arm_hardware_process(
    robot_cfg: dict[str, Any],
    side: str,
    channel: dict[str, Any],
    stop_event: Any,
    status_queue: Any,
) -> None:
    """单侧 Piper 常驻进程；该进程主线程独占全部 Piper SDK 调用。"""
    process_name = f"arm-{side}"
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s [device-b/{process_name}] %(message)s",
    )
    _configure_signals(stop_event)
    side_cfg = robot_cfg[side]
    arm = None
    active = False
    last_command_ns = 0
    last_feedback_ns = 0
    last_status_ns = 0
    exit_code = 0
    command_interval_ns = int(1e9 / float(robot_cfg.get("arm_command_hz", 30.0)))
    feedback_interval_ns = int(1e9 / float(robot_cfg.get("arm_feedback_hz", 30.0)))
    try:
        logging.info("%s 正在加载 pyAgxArm", side)
        from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config

        arm_config = create_agx_arm_config(
            robot=ArmModel.PIPER,
            firmeware_version=PiperFW.DEFAULT,
            interface=side_cfg["interface"],
            channel=str(side_cfg.get("channel", "0")),
            bitrate=int(side_cfg.get("bitrate", 1_000_000)),
        )
        arm = AgxArmFactory.create_arm(arm_config)
        logging.info(
            "%s Piper 正在连接 interface=%s channel=%s bitrate=%s",
            side,
            side_cfg["interface"],
            side_cfg.get("channel", "0"),
            side_cfg.get("bitrate", 1_000_000),
        )
        arm.connect()
        # 不在启动阶段自动 disable：如果控制器此前正在承重，disable 可能造成跌落。
        arm.set_speed_percent(int(side_cfg.get("speed_percent", 20)))
        logging.info(
            "%s Piper connect已返回，等待首帧法兰/关节反馈，超时=%.1fs",
            side,
            float(side_cfg.get("health_timeout_s", 3.0)),
        )
        _wait_initial_arm_feedback(channel, arm, side_cfg, side)
        last_feedback_ns = time.perf_counter_ns()
        channel["hold_event"].set()
        _set_phase(channel, PHASE_HOLD)
        channel["ready_event"].set()
        _status_put(
            status_queue,
            process_name,
            "ready",
            state=PHASE_NAMES[PHASE_HOLD],
            note="未自动 disable；保持控制器当前使能状态",
        )

        while not stop_event.is_set():
            if channel["hold_event"].is_set() and active:
                # 共享门控优先于生命周期Queue ACK：立即停止消费后续目标。
                active = False
                _drain_latest(channel["target_queue"])
                _set_phase(channel, PHASE_HOLD)
            request: dict[str, Any] | None = None
            try:
                request = channel["control_queue"].get_nowait()
            except queue.Empty:
                pass

            if request is not None:
                kind = str(request.get("kind", ""))
                try:
                    if kind in {"home", "prepare_start"}:
                        channel["hold_event"].set()
                        active = False
                        target = _array(
                            side_cfg["home_pose"], (6,), f"{side} home_pose"
                        )
                        _enable_arm(arm, side_cfg, side)
                        arm.set_speed_percent(
                            int(side_cfg.get("home_speed_percent", 10))
                        )
                        arm.move_j(target.tolist())
                        _wait_arm_home(arm, channel, side_cfg, side, target)
                        # 到达 home 后保持 enable，不执行 disable。
                        _set_phase(
                            channel,
                            PHASE_PREPARED if kind == "prepare_start" else PHASE_HOLD,
                        )
                        _put_response(
                            channel,
                            request,
                            ok=True,
                            state=PHASE_NAMES[int(channel["phase"].value)],
                        )
                    elif kind == "activate":
                        if int(channel["phase"].value) != PHASE_PREPARED:
                            raise RuntimeError(
                                f"{side} Piper 尚未完成 prepare_start"
                            )
                        arm.set_speed_percent(int(side_cfg.get("speed_percent", 20)))
                        channel["hold_event"].clear()
                        active = True
                        _set_phase(channel, PHASE_ACTIVE)
                        _put_response(channel, request, ok=True, state="active")
                    elif kind in {"stop", "cancel_prepare"}:
                        # 停止消费新目标，但保持当前控制器使能状态和最后目标。
                        channel["hold_event"].set()
                        active = False
                        _drain_latest(channel["target_queue"])
                        _set_phase(channel, PHASE_HOLD)
                        _put_response(
                            channel,
                            request,
                            ok=True,
                            state=PHASE_NAMES[PHASE_HOLD],
                            arm_disabled=False,
                        )
                    elif kind == "status":
                        _put_response(
                            channel,
                            request,
                            ok=True,
                            state=PHASE_NAMES.get(int(channel["phase"].value)),
                        )
                    else:
                        raise RuntimeError(f"未知机械臂命令: {kind}")
                except BaseException as exc:
                    active = False
                    channel["fault"].value = True
                    _set_phase(channel, PHASE_FAULT_HOLD)
                    _put_response(
                        channel,
                        request,
                        ok=False,
                        error=repr(exc),
                        arm_disabled=False,
                    )
                    raise

            now_ns = time.perf_counter_ns()
            target = _drain_latest(channel["target_queue"]) if active else None
            if target is not None:
                channel["received"].value += 1
                remaining_ns = last_command_ns + command_interval_ns - now_ns
                if remaining_ns > 0:
                    stop_event.wait(remaining_ns / 1e9)
                    # 限频等待期间可能到达更新目标；硬件永远执行截止时最新值。
                    newer = _drain_latest(channel["target_queue"])
                    if newer is not None:
                        target = newer
                if channel["hold_event"].is_set():
                    # 限频等待期间收到stop，不再执行已经取出的旧目标。
                    active = False
                    continue
                pose = _array(target["target"], (6,), f"{side} move_p target")
                started_ns = time.perf_counter_ns()
                arm.move_p(pose.tolist())
                _update_arm_feedback(
                    channel, arm, applied_seq=int(target.get("source_seq", -1))
                )
                finished_ns = time.perf_counter_ns()
                duration_ms = (finished_ns - started_ns) / 1e6
                channel["last_io_ms"].value = duration_ms
                channel["max_io_ms"].value = max(
                    float(channel["max_io_ms"].value), duration_ms
                )
                channel["applied"].value += 1
                last_command_ns = started_ns
                last_feedback_ns = finished_ns
                now_ns = finished_ns
            elif now_ns - last_feedback_ns >= feedback_interval_ns:
                _update_arm_feedback(channel, arm)
                last_feedback_ns = time.perf_counter_ns()
                now_ns = last_feedback_ns

            if active and now_ns - last_status_ns >= int(
                float(side_cfg.get("arm_status_poll_interval_s", 1.0)) * 1e9
            ):
                status = arm.get_arm_status()
                message = getattr(status, "msg", status)
                arm_status = getattr(message, "arm_status", None)
                if isinstance(arm_status, int) and arm_status != 0:
                    raise RuntimeError(
                        f"{side} Piper 控制器状态异常: arm_status={arm_status}, "
                        f"mode_feedback={getattr(message, 'mode_feedback', None)}, "
                        f"err_status={getattr(message, 'err_status', None)}"
                    )
                last_status_ns = now_ns
            stop_event.wait(0.001)
    except BaseException as exc:
        exit_code = 1
        _set_channel_error(channel, exc)
        channel["fault"].value = True
        _set_phase(channel, PHASE_FAULT_HOLD)
        channel["ready_event"].set()
        channel["failed"].value += 1
        _status_put(
            status_queue,
            process_name,
            "error",
            component="hardware",
            error=repr(exc),
            traceback=traceback.format_exc(),
            arm_disabled=False,
            note="异常后未调用 disable，保持控制器当前使能状态",
        )
        stop_event.set()
    finally:
        # 用户明确要求：异常、终止和普通退出都不允许自动 disable。
        # 也不主动调用 disconnect，避免厂商 SDK 在 disconnect 内部隐式失能。
        active = False
        channel["hold_event"].set()
        _set_phase(channel, PHASE_EXIT_HOLD)
        _status_put(
            status_queue,
            process_name,
            "hardware_exit_hold_enabled",
            arm_disabled=False,
            note="进程退出但未调用 disable/disconnect",
        )
        # 不能让未知的厂商 SDK __del__ 在解释器清理阶段隐式 disconnect/disable。
        # 该进程没有需要落盘的数据；诊断至少可由父进程的非零 exitcode 保留。
        os._exit(exit_code)


def hand_hardware_process(
    robot_cfg: dict[str, Any],
    side: str,
    channel: dict[str, Any],
    stop_event: Any,
    status_queue: Any,
) -> None:
    """单侧 Aerohand 常驻进程；串口 SDK 只在本进程主线程调用。"""
    process_name = f"hand-{side}"
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s [device-b/{process_name}] %(message)s",
    )
    _configure_signals(stop_event)
    side_cfg = robot_cfg[side]
    hand = None
    active = False
    last_command_ns = 0
    last_feedback_ns = 0
    command_interval_ns = int(1e9 / float(robot_cfg.get("hand_command_hz", 60.0)))
    feedback_interval_ns = int(1e9 / float(robot_cfg.get("hand_feedback_hz", 30.0)))
    try:
        from aero_open_sdk.aero_hand import AeroHand

        hand = AeroHand(port=side_cfg["hand_port"])
        _wait_initial_hand_feedback(channel, hand, side_cfg, side)
        last_feedback_ns = time.perf_counter_ns()
        channel["hold_event"].set()
        _set_phase(channel, PHASE_HOLD)
        channel["ready_event"].set()
        _status_put(status_queue, process_name, "ready", state="hold")

        while not stop_event.is_set():
            if channel["hold_event"].is_set() and active:
                active = False
                _drain_latest(channel["target_queue"])
                _set_phase(channel, PHASE_HOLD)
            request: dict[str, Any] | None = None
            try:
                request = channel["control_queue"].get_nowait()
            except queue.Empty:
                pass
            if request is not None:
                kind = str(request.get("kind", ""))
                try:
                    if kind == "home":
                        channel["hold_event"].set()
                        active = False
                        result = hand.send_homing()
                        if result is False:
                            raise RuntimeError(f"{side} Aerohand homing 返回失败")
                        wait_s = float(side_cfg.get("hand_home_wait_s", 8.0))
                        if wait_s > 0:
                            stop_event.wait(wait_s)
                        _update_hand_feedback(channel, hand)
                        _set_phase(channel, PHASE_HOLD)
                        _put_response(channel, request, ok=True, state="hold")
                    elif kind == "prepare_start":
                        channel["hold_event"].set()
                        _set_phase(channel, PHASE_PREPARED)
                        _put_response(channel, request, ok=True, state="prepared")
                    elif kind == "activate":
                        if int(channel["phase"].value) != PHASE_PREPARED:
                            raise RuntimeError(
                                f"{side} Aerohand 尚未完成 prepare_start"
                            )
                        active = True
                        channel["hold_event"].clear()
                        _set_phase(channel, PHASE_ACTIVE)
                        _put_response(channel, request, ok=True, state="active")
                    elif kind in {"stop", "cancel_prepare"}:
                        channel["hold_event"].set()
                        active = False
                        _drain_latest(channel["target_queue"])
                        _set_phase(channel, PHASE_HOLD)
                        _put_response(channel, request, ok=True, state="hold")
                    elif kind == "status":
                        _put_response(
                            channel,
                            request,
                            ok=True,
                            state=PHASE_NAMES.get(int(channel["phase"].value)),
                        )
                    else:
                        raise RuntimeError(f"未知灵巧手命令: {kind}")
                except BaseException as exc:
                    active = False
                    channel["fault"].value = True
                    _set_phase(channel, PHASE_FAULT_HOLD)
                    _put_response(channel, request, ok=False, error=repr(exc))
                    raise

            now_ns = time.perf_counter_ns()
            target = _drain_latest(channel["target_queue"]) if active else None
            if target is not None:
                channel["received"].value += 1
                remaining_ns = last_command_ns + command_interval_ns - now_ns
                if remaining_ns > 0:
                    stop_event.wait(remaining_ns / 1e9)
                    newer = _drain_latest(channel["target_queue"])
                    if newer is not None:
                        target = newer
                if channel["hold_event"].is_set():
                    active = False
                    continue
                joints = _array(target["target"], (7,), f"{side} hand target")
                started_ns = time.perf_counter_ns()
                hand.set_joint_positions(joints.tolist())
                finished_ns = time.perf_counter_ns()
                duration_ms = (finished_ns - started_ns) / 1e6
                channel["last_io_ms"].value = duration_ms
                channel["max_io_ms"].value = max(
                    float(channel["max_io_ms"].value), duration_ms
                )
                channel["applied_seq"].value = int(target.get("source_seq", -1))
                channel["applied"].value += 1
                last_command_ns = started_ns
                now_ns = finished_ns
            if now_ns - last_feedback_ns >= feedback_interval_ns:
                _update_hand_feedback(channel, hand)
                last_feedback_ns = time.perf_counter_ns()
            stop_event.wait(0.001)
    except BaseException as exc:
        _set_channel_error(channel, exc)
        channel["fault"].value = True
        _set_phase(channel, PHASE_FAULT_HOLD)
        channel["ready_event"].set()
        channel["failed"].value += 1
        _status_put(
            status_queue,
            process_name,
            "error",
            component="hardware",
            error=repr(exc),
            traceback=traceback.format_exc(),
        )
        stop_event.set()
    finally:
        active = False
        channel["hold_event"].set()
        if hand is not None:
            try:
                if hasattr(hand, "close"):
                    hand.close()
                elif hasattr(hand, "disconnect"):
                    hand.disconnect()
            except Exception:
                logging.exception("%s Aerohand 释放失败", side)
        _set_phase(channel, PHASE_EXIT_HOLD)


class MultiprocessRobotProxy:
    """control 子进程中的轻量代理；不加载任何真实硬件 SDK。"""

    def __init__(self, cfg: dict[str, Any], channels: dict[str, Any]) -> None:
        self.cfg = cfg
        self.channels = channels
        self.state = "new"
        self.homed = False
        self.start_prepared = False
        self.start_home_count = 0
        self.last_start_home: dict[str, Any] = {}
        self._request_id = 0
        self._last_command: BimanualControlCommand | None = None

    @property
    def initialized(self) -> bool:
        return self.state not in {"new", "closed"}

    @property
    def active(self) -> bool:
        return self.state == "active"

    def _request(self, device: str, kind: str, timeout_s: float) -> dict[str, Any]:
        channel = self.channels[device]
        self._request_id += 1
        request_id = self._request_id
        channel["control_queue"].put(
            {"kind": kind, "request_id": request_id}, timeout=timeout_s
        )
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if bool(channel["fault"].value) and channel["response_queue"].empty():
                detail = _get_channel_error(channel)
                raise RuntimeError(
                    f"{device} 硬件进程处于fault: {detail or '未提供详细错误'}"
                )
            try:
                response = channel["response_queue"].get(
                    timeout=min(0.1, max(0.001, deadline - time.monotonic()))
                )
            except queue.Empty:
                continue
            if int(response.get("request_id", -1)) != request_id:
                continue
            if not bool(response.get("ok", False)):
                raise RuntimeError(
                    f"{device} {kind} 失败: {response.get('error')}"
                )
            return response
        raise TimeoutError(f"等待 {device} {kind} ACK 超时 {timeout_s:.1f}s")

    def _request_no_wait(self, device: str, kind: str) -> None:
        """投递无需ACK阻塞的hold命令；共享hold_event才是停止入口。"""
        channel = self.channels[device]
        self._request_id += 1
        request = {"kind": kind, "request_id": self._request_id}
        try:
            channel["control_queue"].put_nowait(request)
        except queue.Full:
            logging.getLogger(__name__).warning(
                "%s 生命周期队列已满，未排入%s；共享hold门控仍已生效",
                device,
                kind,
            )

    def initialize(self) -> None:
        if self.state != "new":
            return
        timeout_s = float(self.cfg.get("hardware_start_timeout_s", 15.0))
        deadline = time.monotonic() + timeout_s
        for device, channel in self.channels.items():
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not channel["ready_event"].wait(remaining):
                raise TimeoutError(f"等待 {device} 常驻硬件进程就绪超时")
            if bool(channel["fault"].value):
                detail = _get_channel_error(channel)
                raise RuntimeError(
                    f"{device} 常驻硬件进程启动失败: "
                    f"{detail or '未提供详细错误'}"
                )
        self.state = "idle_enabled"

    def home(self) -> None:
        if self.state != "idle_enabled":
            raise RuntimeError(f"当前状态 {self.state} 不允许 home")
        timeout_s = float(self.cfg.get("hardware_request_timeout_s", 30.0))
        for device in ("arm_right", "arm_left", "hand_left", "hand_right"):
            self._request(device, "home", timeout_s)
        self.homed = True

    def prepare_start(self) -> None:
        if self.state != "idle_enabled":
            raise RuntimeError(f"当前状态 {self.state} 不允许 prepare_start")
        timeout_s = float(self.cfg.get("hardware_request_timeout_s", 30.0))
        self.state = "preparing_start"
        try:
            # 双臂仍按已验证顺序依次回 home；四个进程常驻不代表双臂同时运动。
            for device in ("arm_right", "arm_left"):
                self._request(device, "prepare_start", timeout_s)
            for device in ("hand_left", "hand_right"):
                self._request(device, "prepare_start", timeout_s)
            self.homed = True
            self.start_prepared = True
            self.start_home_count += 1
            self.state = "prepared_enabled"
        except BaseException:
            # 不 disable 已经准备成功的机械臂，只停止后续目标并保持使能。
            self._best_effort_hold("cancel_prepare")
            self.start_prepared = False
            self.state = "fault_hold_enabled"
            raise

    def activate(self) -> None:
        if self.state != "prepared_enabled" or not self.start_prepared:
            raise RuntimeError(f"当前状态 {self.state} 不允许 activate")
        timeout_s = float(self.cfg.get("hardware_request_timeout_s", 30.0))
        try:
            for device in ("arm_left", "arm_right", "hand_left", "hand_right"):
                self._request(device, "activate", timeout_s)
            self.state = "active"
            self.start_prepared = False
        except BaseException:
            self._best_effort_hold("stop")
            self.state = "fault_hold_enabled"
            raise

    def command(self, value: BimanualControlCommand) -> None:
        if not self.active:
            raise RuntimeError(f"当前状态 {self.state} 不允许下发控制命令")
        for side in SIDES:
            side_value: ControlCommand = getattr(value, side)
            for kind, target in (
                ("arm", side_value.arm_pose),
                ("hand", side_value.hand_joints),
            ):
                device = f"{kind}_{side}"
                channel = self.channels[device]
                if bool(channel["fault"].value):
                    self.state = "fault_hold_enabled"
                    detail = _get_channel_error(channel)
                    raise RuntimeError(
                        f"{device} 硬件进程发生异常: "
                        f"{detail or '未提供详细错误'}"
                    )
                channel["submitted"].value += 1
                _put_latest_target(
                    channel["target_queue"],
                    {
                        "session_id": self.start_home_count,
                        "source_seq": int(value.source_seq),
                        "target": np.asarray(target, np.float32),
                        "target_mono_ns": time.perf_counter_ns(),
                    },
                )
        self._last_command = value

    def _read_arm(self, side: str) -> tuple[np.ndarray, np.ndarray, int]:
        channel = self.channels[f"arm_{side}"]
        with channel["state_lock"]:
            pose = np.frombuffer(channel["pose"], dtype=np.float64).copy().astype(np.float32)
            joints = np.frombuffer(channel["joints"], dtype=np.float64).copy().astype(np.float32)
            stamp_ns = int(channel["stamp_ns"].value)
        return pose, joints, stamp_ns

    def _read_hand(self, side: str) -> tuple[np.ndarray, int]:
        channel = self.channels[f"hand_{side}"]
        with channel["state_lock"]:
            joints = np.frombuffer(channel["joints"], dtype=np.float64).copy().astype(np.float32)
            stamp_ns = int(channel["stamp_ns"].value)
        return joints, stamp_ns

    def read_state(self) -> BimanualRobotState:
        states: dict[str, RobotState] = {}
        for side in SIDES:
            pose, arm_joints, _ = self._read_arm(side)
            hand_joints, _ = self._read_hand(side)
            states[side] = RobotState(pose, arm_joints, hand_joints)
        return BimanualRobotState(states["left"], states["right"])

    def feedback_timestamps_ns(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for side in SIDES:
            result[f"arm_{side}"] = self._read_arm(side)[2]
            result[f"hand_{side}"] = self._read_hand(side)[1]
        return result

    def _best_effort_hold(self, kind: str = "stop") -> None:
        devices = ("arm_left", "arm_right", "hand_left", "hand_right")
        # 先原子式门控全部设备，再异步通知设备更新phase并清空本地目标。
        for device in devices:
            self.channels[device]["hold_event"].set()
        for device in devices:
            self._request_no_wait(device, kind)

    def deactivate(self) -> None:
        if self.state in {"new", "closed", "idle_enabled"}:
            return
        previous = self.state
        self._best_effort_hold("stop")
        self._last_command = None
        self.start_prepared = False
        self.state = (
            "fault_hold_enabled"
            if previous == "fault_hold_enabled"
            else "idle_enabled"
        )

    def close(self) -> None:
        # 硬件进程由设备B父进程监督。这里只请求保持，不关闭句柄、不失能。
        self.deactivate()
        self.state = "closed"

    def status_snapshot(self) -> dict[str, Any]:
        devices: dict[str, Any] = {}
        now_ns = time.perf_counter_ns()
        for name, channel in self.channels.items():
            stamp_ns = int(channel["stamp_ns"].value)
            devices[name] = {
                "phase": PHASE_NAMES.get(int(channel["phase"].value), "unknown"),
                "hold_requested": channel["hold_event"].is_set(),
                "ready": channel["ready_event"].is_set(),
                "fault": bool(channel["fault"].value),
                "error": _get_channel_error(channel),
                "submitted": int(channel["submitted"].value),
                "received": int(channel["received"].value),
                "applied": int(channel["applied"].value),
                "failed": int(channel["failed"].value),
                "applied_seq": int(channel["applied_seq"].value),
                "feedback_age_ms": (
                    round((now_ns - stamp_ns) / 1e6, 3)
                    if stamp_ns > 0
                    else None
                ),
                "last_io_ms": round(float(channel["last_io_ms"].value), 3),
                "max_io_ms": round(float(channel["max_io_ms"].value), 3),
            }
        return {
            "state": self.state,
            "initialized": self.initialized,
            "active": self.active,
            "homed": self.homed,
            "start_prepared": self.start_prepared,
            "start_home_count": self.start_home_count,
            "arm_auto_disable": False,
            "devices": devices,
        }


class DualArmProcessProxy(MultiprocessRobotProxy):
    """仅包含左右 Piper 的测试代理，不创建或访问 Aerohand。"""

    def home(self) -> None:
        if self.state != "idle_enabled":
            raise RuntimeError(f"当前状态 {self.state} 不允许 home")
        timeout_s = float(self.cfg.get("hardware_request_timeout_s", 30.0))
        for device in ("arm_right", "arm_left"):
            self._request(device, "home", timeout_s)
        self.homed = True

    def prepare_start(self) -> None:
        if self.state != "idle_enabled":
            raise RuntimeError(f"当前状态 {self.state} 不允许 prepare_start")
        timeout_s = float(self.cfg.get("hardware_request_timeout_s", 30.0))
        self.state = "preparing_start"
        try:
            for device in ("arm_right", "arm_left"):
                self._request(device, "prepare_start", timeout_s)
            self.homed = True
            self.start_prepared = True
            self.start_home_count += 1
            self.state = "prepared_enabled"
        except BaseException:
            self._best_effort_hold("cancel_prepare")
            self.start_prepared = False
            self.state = "fault_hold_enabled"
            raise

    def activate(self) -> None:
        if self.state != "prepared_enabled" or not self.start_prepared:
            raise RuntimeError(f"当前状态 {self.state} 不允许 activate")
        timeout_s = float(self.cfg.get("hardware_request_timeout_s", 30.0))
        try:
            for device in ("arm_left", "arm_right"):
                self._request(device, "activate", timeout_s)
            self.state = "active"
            self.start_prepared = False
        except BaseException:
            self._best_effort_hold("stop")
            self.state = "fault_hold_enabled"
            raise

    def command_poses(
        self,
        left: np.ndarray,
        right: np.ndarray,
        source_seq: int,
    ) -> None:
        if not self.active:
            raise RuntimeError(f"当前状态 {self.state} 不允许下发双臂目标")
        for side, target in (("left", left), ("right", right)):
            pose = np.asarray(target, dtype=np.float32)
            if pose.shape != (6,) or not np.all(np.isfinite(pose)):
                raise ValueError(f"{side} 双臂测试目标必须是有效6维末端位姿")
            device = f"arm_{side}"
            channel = self.channels[device]
            if bool(channel["fault"].value):
                self.state = "fault_hold_enabled"
                detail = _get_channel_error(channel)
                raise RuntimeError(
                    f"{device} 硬件进程发生异常: "
                    f"{detail or '未提供详细错误'}"
                )
            channel["submitted"].value += 1
            _put_latest_target(
                channel["target_queue"],
                {
                    "session_id": self.start_home_count,
                    "source_seq": int(source_seq),
                    "target": pose,
                    "target_mono_ns": time.perf_counter_ns(),
                },
            )

    def read_state(self) -> BimanualRobotState:
        states: dict[str, RobotState] = {}
        for side in SIDES:
            pose, joints, _ = self._read_arm(side)
            states[side] = RobotState(
                pose,
                joints,
                np.zeros(7, dtype=np.float32),
            )
        return BimanualRobotState(states["left"], states["right"])

    def feedback_timestamps_ns(self) -> dict[str, int]:
        return {side: self._read_arm(side)[2] for side in SIDES}

    def _best_effort_hold(self, kind: str = "stop") -> None:
        devices = ("arm_left", "arm_right")
        for device in devices:
            self.channels[device]["hold_event"].set()
        for device in devices:
            self._request_no_wait(device, kind)
