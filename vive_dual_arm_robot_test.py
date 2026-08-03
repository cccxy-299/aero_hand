from __future__ import annotations

"""设备B：ZMQ接收双 VIVE Tracker，并仅遥操作左右 Piper。"""

import argparse
import logging
import multiprocessing as mp
from pathlib import Path
import queue
import signal
import threading
import time
from typing import Any

import numpy as np
import yaml
import zmq

from hardware_processes import (
    DualArmProcessProxy,
    arm_hardware_process,
    close_hardware_channels,
    create_hardware_channels,
)
from retarget import HardwareBimanualRetargeter, SideRetargetConfig
from safety import SafetyConfig, SafetyGate


LOG = logging.getLogger("vive-dual-arm-robot-test")
SIDES = ("left", "right")


def _load_robot_test_config(path: str) -> dict[str, Any]:
    """只校验双臂测试实际使用的字段，不要求相机、手或Dataset配置。"""
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("配置根节点必须是对象")
    if not bool(cfg.get("robot", {}).get("enabled", False)):
        raise ValueError("双臂真机测试要求 robot.enabled=true")
    if float(cfg.get("rates", {}).get("control_hz", 0)) <= 0:
        raise ValueError("rates.control_hz必须大于0")
    if float(cfg.get("alignment", {}).get("teleop_timeout_ms", 0)) <= 0:
        raise ValueError("alignment.teleop_timeout_ms必须大于0")
    for side in SIDES:
        side_cfg = cfg["robot"].get(side)
        if not isinstance(side_cfg, dict):
            raise ValueError(f"缺少robot.{side}")
        for name, size in (
            ("home_pose", 6),
            ("fixed_orientation", 3),
            ("workspace_min", 3),
            ("workspace_max", 3),
        ):
            value = np.asarray(side_cfg.get(name), np.float32)
            if value.shape != (size,) or not np.all(np.isfinite(value)):
                raise ValueError(f"robot.{side}.{name}必须是有效{size}维数组")
        matrix = np.asarray(side_cfg.get("vive_to_robot_matrix"), np.float32)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"robot.{side}.vive_to_robot_matrix必须是有效3x3矩阵"
            )
        if np.any(
            np.asarray(side_cfg["workspace_min"], np.float32)
            >= np.asarray(side_cfg["workspace_max"], np.float32)
        ):
            raise ValueError(f"robot.{side}工作空间上下界非法")
        if float(side_cfg.get("max_linear_step_m", 0)) <= 0:
            raise ValueError(f"robot.{side}.max_linear_step_m必须大于0")
        for name in ("interface", "channel"):
            if not str(side_cfg.get(name, "")).strip():
                raise ValueError(f"robot.{side}.{name}不能为空")
    return cfg


def _load_axis_calibration_config(path: str) -> dict[str, Any]:
    """标定模式只读取现有增益，不校验或连接任何机械臂硬件字段。"""
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict) or not isinstance(cfg.get("robot"), dict):
        raise ValueError("标定配置必须包含 robot 对象")
    for side in SIDES:
        side_cfg = cfg["robot"].get(side)
        if not isinstance(side_cfg, dict):
            raise ValueError(f"标定配置缺少 robot.{side}")
        scale = float(side_cfg.get("vive_scale", 0))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"robot.{side}.vive_scale 必须大于0")
    return cfg


def _make_control(
    cfg: dict[str, Any],
    proxy: DualArmProcessProxy,
    scale: float,
) -> tuple[HardwareBimanualRetargeter, dict[str, SafetyGate]]:
    state = proxy.read_state()
    side_configs: dict[str, SideRetargetConfig] = {}
    safety: dict[str, SafetyGate] = {}
    for side in SIDES:
        side_cfg = cfg["robot"][side]
        initial = np.asarray(getattr(state, side).arm_pose, np.float32)
        workspace_min = np.asarray(side_cfg["workspace_min"], np.float32)
        workspace_max = np.asarray(side_cfg["workspace_max"], np.float32)
        if np.any(initial[:3] < workspace_min) or np.any(
            initial[:3] > workspace_max
        ):
            raise RuntimeError(
                f"{side} start法兰位置{initial[:3].tolist()}不在配置工作空间"
                f"[{workspace_min.tolist()}, {workspace_max.tolist()}]内，拒绝遥操作"
            )
        orientation_mode = str(
            side_cfg.get("orientation_mode", "current_on_start")
        ).lower()
        orientation = (
            initial[3:].copy()
            if orientation_mode == "current_on_start"
            else np.asarray(side_cfg["fixed_orientation"], np.float32)
        )
        side_configs[side] = SideRetargetConfig(
            initial_pose=initial,
            scale=scale,
            fixed_orientation=orientation,
            position_map=np.asarray(
                side_cfg["vive_to_robot_matrix"], np.float32
            ),
        )
        safety[side] = SafetyGate(
            SafetyConfig(
                workspace_min=workspace_min,
                workspace_max=workspace_max,
                max_linear_step_m=float(side_cfg["max_linear_step_m"]),
                hand_min=np.zeros(7, np.float32),
                hand_max=np.zeros(7, np.float32),
                stale_timeout_ns=int(
                    float(cfg["alignment"]["teleop_timeout_ms"]) * 1e6
                ),
            ),
            initial,
        )
    return HardwareBimanualRetargeter(side_configs), safety


def _validate_packet(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("ZMQ消息不是对象")
    if value.get("version") != 1 or value.get("kind") != "vive_dual_arm_test":
        raise ValueError("ZMQ消息版本或kind不匹配")
    if not isinstance(value.get("seq"), int) or value["seq"] < 0:
        raise ValueError("seq非法")
    sides = value.get("sides")
    if not isinstance(sides, dict):
        raise ValueError("缺少sides")
    for side in SIDES:
        item = sides.get(side)
        if not isinstance(item, dict):
            raise ValueError(f"缺少{side}数据")
        pose = np.asarray(item.get("vive_pose"), np.float32)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise ValueError(f"{side} VIVE位姿非法")
        if not isinstance(item.get("valid"), bool):
            raise ValueError(f"{side}.valid非法")
        age_ms = item.get("age_ms")
        if age_ms is not None and (
            not isinstance(age_ms, (int, float))
            or not np.isfinite(float(age_ms))
            or float(age_ms) < 0
        ):
            raise ValueError(f"{side}.age_ms非法")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="设备B：ZMQ双 VIVE Tracker -> 双 Piper测试"
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "cfg" / "robot.yaml"),
    )
    parser.add_argument("--bind", default="tcp://*:17861")
    parser.add_argument(
        "--scale",
        type=float,
        default=0.10,
        help="测试缩放系数，默认0.10，确认方向后再逐步增大",
    )
    parser.add_argument("--max-source-age-ms", type=float, default=100.0)
    parser.add_argument(
        "--calibrate-axes",
        action="store_true",
        help="进入无机械臂轴标定模式，只接收 ZMQ 并拟合左右坐标映射",
    )
    parser.add_argument(
        "--calibration-sides",
        choices=("both", "left", "right"),
        default="both",
        help="需要标定的 Tracker，默认左右依次标定",
    )
    parser.add_argument(
        "--calibration-output",
        default=str(
            Path(__file__).resolve().parent
            / "cfg"
            / "vive_axis_calibration.yaml"
        ),
        help="标定 YAML 输出路径；若已存在会自动添加时间戳，不覆盖旧文件",
    )
    parser.add_argument(
        "--calibration-distance-m",
        type=float,
        default=0.10,
        help="每次引导移动的物理距离，默认0.10m",
    )
    parser.add_argument(
        "--calibration-min-displacement-m",
        type=float,
        default=0.04,
        help="单次有效移动的最小距离，默认0.04m",
    )
    parser.add_argument(
        "--calibration-window-s",
        type=float,
        default=0.60,
        help="每个静止端点的采样窗口，默认0.60s",
    )
    parser.add_argument(
        "--calibration-min-samples",
        type=int,
        default=15,
        help="每个静止端点最少有效样本数，默认15",
    )
    parser.add_argument(
        "--calibration-max-std-m",
        type=float,
        default=0.003,
        help="端点任一坐标允许的最大标准差，默认0.003m",
    )
    parser.add_argument(
        "--calibration-max-fit-error-deg",
        type=float,
        default=20.0,
        help="任一方向允许的最大拟合角误差，默认20度",
    )
    parser.add_argument(
        "--calibration-max-pair-error-deg",
        type=float,
        default=25.0,
        help="同一轴正反方向允许的最大不一致角度，默认25度",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [vive-test/device-b] %(message)s",
    )

    if args.max_source_age_ms <= 0:
        parser.error("--max-source-age-ms 必须大于0")
    if args.calibrate_axes:
        calibration_values = (
            args.calibration_distance_m,
            args.calibration_min_displacement_m,
            args.calibration_window_s,
            args.calibration_max_std_m,
            args.calibration_max_fit_error_deg,
            args.calibration_max_pair_error_deg,
        )
        if any(value <= 0 for value in calibration_values):
            parser.error("所有 calibration 数值参数必须大于0")
        if args.calibration_min_samples < 3:
            parser.error("--calibration-min-samples 必须至少为3")
        if args.calibration_min_displacement_m >= args.calibration_distance_m:
            parser.error(
                "--calibration-min-displacement-m 必须小于 "
                "--calibration-distance-m"
            )
        cfg = _load_axis_calibration_config(args.config)
        selected_sides = (
            SIDES
            if args.calibration_sides == "both"
            else (args.calibration_sides,)
        )
        from vive_axis_calibration import run_axis_calibration

        try:
            run_axis_calibration(
                bind=args.bind,
                validator=_validate_packet,
                sides=selected_sides,
                output_path=Path(args.calibration_output).expanduser().resolve(),
                configured_scales={
                    side: float(cfg["robot"][side]["vive_scale"])
                    for side in selected_sides
                },
                max_source_age_ms=args.max_source_age_ms,
                expected_distance_m=args.calibration_distance_m,
                min_displacement_m=args.calibration_min_displacement_m,
                window_s=args.calibration_window_s,
                min_samples=args.calibration_min_samples,
                max_std_m=args.calibration_max_std_m,
                max_fit_error_deg=args.calibration_max_fit_error_deg,
                max_pair_error_deg=args.calibration_max_pair_error_deg,
            )
        except KeyboardInterrupt:
            LOG.warning("用户取消轴标定；未生成配置")
        return

    if args.scale <= 0:
        parser.error("--scale 必须大于0")
    cfg = _load_robot_test_config(args.config)

    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    status_queue = ctx.Queue(maxsize=128)
    channels = create_hardware_channels(
        ctx, cfg["robot"], include_hands=False
    )
    processes = {
        side: ctx.Process(
            name=f"vive-test-arm-{side}",
            target=arm_hardware_process,
            args=(
                cfg["robot"],
                side,
                channels[f"arm_{side}"],
                stop_event,
                status_queue,
            ),
        )
        for side in SIDES
    }
    for process in processes.values():
        process.start()
    proxy = DualArmProcessProxy(cfg["robot"], channels)

    context = zmq.Context.instance()
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(args.bind)

    console_queue: queue.Queue[str] = queue.Queue(maxsize=8)
    console_stop = threading.Event()

    def console_loop() -> None:
        LOG.info("命令：start | stop | status | quit")
        while not console_stop.is_set():
            try:
                line = input().strip().lower()
            except EOFError:
                return
            if not line:
                continue
            try:
                console_queue.put_nowait(line)
            except queue.Full:
                LOG.warning("控制台命令队列已满，丢弃：%s", line)

    thread = threading.Thread(
        target=console_loop, name="vive-test-console", daemon=True
    )
    thread.start()
    old_sigint = signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    old_sigterm = signal.signal(signal.SIGTERM, lambda *_: stop_event.set())

    active = False
    retargeter: HardwareBimanualRetargeter | None = None
    safety: dict[str, SafetyGate] | None = None
    latest: dict[str, Any] | None = None
    latest_receive_ns = 0
    last_executed_seq = -1
    last_received_seq = -1
    received = 0
    invalid = 0
    sequence_gaps = 0
    executed = 0
    stale = 0
    last_report_ns = time.perf_counter_ns()
    control_period_ns = int(1e9 / float(cfg["rates"]["control_hz"]))
    deadline_ns = last_report_ns
    teleop_timeout_ns = int(
        float(cfg["alignment"]["teleop_timeout_ms"]) * 1e6
    )
    max_source_age_ns = int(args.max_source_age_ms * 1e6)

    try:
        proxy.initialize()
        LOG.info(
            "双 Piper 常驻进程就绪；ZMQ绑定=%s，测试scale=%.3f。"
            "输入start后双臂先依次回home_pose",
            args.bind,
            args.scale,
        )
        while not stop_event.is_set():
            while True:
                try:
                    command = console_queue.get_nowait()
                except queue.Empty:
                    break
                if command == "start":
                    if active:
                        LOG.warning("已经处于ACTIVE")
                        continue
                    proxy.prepare_start()
                    proxy.activate()
                    try:
                        retargeter, safety = _make_control(
                            cfg, proxy, args.scale
                        )
                    except BaseException:
                        # activate后任何初始化失败都必须停止新目标，但按项目策略
                        # 保持机械臂使能，不调用disable。
                        proxy.deactivate()
                        raise
                    latest = None
                    latest_receive_ns = 0
                    last_executed_seq = -1
                    active = True
                    LOG.info(
                        "ACTIVE：请缓慢单轴移动Tracker；第一帧只建立相对参考"
                    )
                elif command == "stop":
                    if active:
                        proxy.deactivate()
                    active = False
                    retargeter = None
                    safety = None
                    LOG.info("HOLD：停止新目标，双臂不自动disable")
                elif command == "status":
                    LOG.info("STATUS %s", proxy.status_snapshot())
                elif command in {"quit", "exit", "q"}:
                    stop_event.set()
                else:
                    LOG.warning("未知命令：%s", command)

            while True:
                try:
                    message = socket.recv_json(flags=zmq.DONTWAIT)
                except zmq.Again:
                    break
                try:
                    message = _validate_packet(message)
                    seq = int(message["seq"])
                    if last_received_seq >= 0 and seq > last_received_seq + 1:
                        sequence_gaps += seq - last_received_seq - 1
                    if seq <= last_received_seq:
                        continue
                    last_received_seq = seq
                    latest = message
                    latest_receive_ns = time.perf_counter_ns()
                    received += 1
                except (ValueError, TypeError, KeyError):
                    invalid += 1
                    LOG.warning("丢弃非法ZMQ VIVE消息", exc_info=True)

            now_ns = time.perf_counter_ns()
            if active and latest is not None and int(latest["seq"]) != last_executed_seq:
                sides = latest["sides"]
                source_ages_ns = [
                    int(
                        float(sides[side]["age_ms"]) * 1e6
                        if sides[side].get("age_ms") is not None
                        else 1e18
                    )
                    for side in SIDES
                ]
                receive_age_ns = now_ns - latest_receive_ns
                valid = all(bool(sides[side].get("valid", False)) for side in SIDES)
                age_ns = max([receive_age_ns, *source_ages_ns])
                if not valid or age_ns > min(teleop_timeout_ns, max_source_age_ns):
                    stale += 1
                    last_executed_seq = int(latest["seq"])
                else:
                    payload = {
                        side: {
                            "vive_pose": sides[side]["vive_pose"],
                            "hand_joints": [0.0] * 7,
                            "valid": True,
                        }
                        for side in SIDES
                    }
                    command = retargeter.retarget(payload, int(latest["seq"]))
                    if command.left.valid and command.right.valid:
                        safe = {
                            side: safety[side].apply(
                                getattr(command, side), age_ns
                            )
                            for side in SIDES
                        }
                        proxy.command_poses(
                            safe["left"].arm_pose,
                            safe["right"].arm_pose,
                            int(latest["seq"]),
                        )
                        executed += 1
                    else:
                        stale += 1
                    last_executed_seq = int(latest["seq"])

            while True:
                try:
                    status = status_queue.get_nowait()
                except queue.Empty:
                    break
                if status.get("kind") == "error":
                    raise RuntimeError(
                        f"{status.get('process')}失败: {status.get('error')}"
                    )
                LOG.info("HARDWARE %s", status)
            for side, process in processes.items():
                if process.exitcode is not None:
                    raise RuntimeError(
                        f"arm_{side}进程意外退出: exitcode={process.exitcode}"
                    )

            if now_ns - last_report_ns >= 1_000_000_000:
                LOG.info(
                    "METRICS received=%d executed=%d stale=%d invalid=%d gaps=%d "
                    "mapping=%s hardware=%s",
                    received,
                    executed,
                    stale,
                    invalid,
                    sequence_gaps,
                    retargeter.mapping_snapshot() if retargeter else {},
                    proxy.status_snapshot(),
                )
                last_report_ns = now_ns
            deadline_ns += control_period_ns
            remaining_ns = deadline_ns - time.perf_counter_ns()
            if remaining_ns > 0:
                stop_event.wait(remaining_ns / 1e9)
            else:
                deadline_ns = time.perf_counter_ns()
    finally:
        console_stop.set()
        if active:
            try:
                proxy.deactivate()
            except Exception:
                LOG.exception("双臂未确认HOLD；不会自动disable")
        stop_event.set()
        socket.close(linger=0)
        for process in processes.values():
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        close_hardware_channels(channels)
        status_queue.close()
        status_queue.cancel_join_thread()
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        LOG.info("双臂VIVE测试已退出，机械臂未自动disable")


if __name__ == "__main__":
    mp.freeze_support()
    main()
