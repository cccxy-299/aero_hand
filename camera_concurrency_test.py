"""TechNexion 双相机并发诊断工具。

本脚本不连接机器人、不创建 LeRobot 数据集、不编码视频，只验证 VizionSDK 取流。
推荐先分别单测左右相机，再比较同进程双线程与独立双进程。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import logging
import multiprocessing as mp
import queue
import threading
import time
import traceback
from typing import Any


@dataclass(frozen=True)
class CameraSpec:
    name: str
    cam_num: int
    width: int
    height: int
    fps: int
    format_idx: int | None
    timeout_ms: int
    poll_hz: float
    strict_fps: bool
    fps_tolerance: float


def _print_event(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def _classify_error(exc: Exception) -> str:
    reason = str(getattr(exc, "reason", "read_error"))
    if reason == "sdk_error" and "VX_TIMEOUT" in repr(exc):
        return "sdk_timeout"
    return reason


def _camera_worker(
    spec: CameraSpec,
    duration_s: float,
    progress_s: float,
    start_event: Any,
    stop_event: Any,
    status_queue: Any,
) -> None:
    """相机工作单元；线程和 spawn 子进程共用同一实现。"""
    camera = None
    valid_frames = 0
    total_errors = 0
    consecutive_errors = 0
    max_consecutive_errors = 0
    errors_by_reason: dict[str, int] = {}
    first_frame_ns: int | None = None
    last_frame_ns: int | None = None
    max_gap_ms = 0.0
    total_read_ms = 0.0
    max_read_ms = 0.0
    started_ns = 0
    selected_format: dict[str, Any] | None = None

    logging.basicConfig(
        level=logging.INFO,
        format=(
            f"%(asctime)s %(levelname)s "
            f"[camera-test/{spec.name}] %(message)s"
        ),
    )

    def send(kind: str, **values: Any) -> None:
        status_queue.put(
            {
                "kind": kind,
                "camera": spec.name,
                "time_ns": time.perf_counter_ns(),
                **values,
            }
        )

    try:
        # 延迟导入，保证没有安装 VizionSDK 的电脑仍可查看 --help。
        from camera import CameraInterface

        camera = CameraInterface(
            cam_num=spec.cam_num,
            fps=spec.fps,
            format_idx=spec.format_idx,
            name=f"concurrency-{spec.name}",
            target_width=spec.width,
            target_height=spec.height,
            timeout_ms=spec.timeout_ms,
            strict_fps=spec.strict_fps,
            fps_tolerance=spec.fps_tolerance,
            # 工作单元本身已经是唯一读取者，不再创建内部采集线程。
            background_capture=False,
        )
        camera.connect()
        selected_format = {
            "width": int(camera.format.width),
            "height": int(camera.format.height),
            "fps": float(camera.format.framerate),
            "format_idx_requested": spec.format_idx,
        }
        send(
            "ready",
            spec=asdict(spec),
            selected_format=selected_format,
        )

        while not stop_event.is_set() and not start_event.wait(0.05):
            pass
        if stop_event.is_set():
            return

        started_ns = time.perf_counter_ns()
        finish_ns = started_ns + int(duration_s * 1e9)
        next_progress_ns = started_ns + int(progress_s * 1e9)
        period_ns = (
            int(1e9 / spec.poll_hz)
            if spec.poll_hz > 0
            else 0
        )
        deadline_ns = started_ns

        while not stop_event.is_set() and time.perf_counter_ns() < finish_ns:
            read_start_ns = time.perf_counter_ns()
            try:
                _, frame_timestamp_s = camera.get_rgb_with_timestamp()
            except Exception as exc:
                read_end_ns = time.perf_counter_ns()
                read_ms = (read_end_ns - read_start_ns) / 1e6
                total_read_ms += read_ms
                max_read_ms = max(max_read_ms, read_ms)
                total_errors += 1
                consecutive_errors += 1
                max_consecutive_errors = max(
                    max_consecutive_errors, consecutive_errors
                )
                reason = _classify_error(exc)
                errors_by_reason[reason] = errors_by_reason.get(reason, 0) + 1
            else:
                read_end_ns = time.perf_counter_ns()
                read_ms = (read_end_ns - read_start_ns) / 1e6
                total_read_ms += read_ms
                max_read_ms = max(max_read_ms, read_ms)
                frame_ns = int(frame_timestamp_s * 1e9)
                if last_frame_ns is not None:
                    max_gap_ms = max(
                        max_gap_ms, (frame_ns - last_frame_ns) / 1e6
                    )
                first_frame_ns = (
                    frame_ns if first_frame_ns is None else first_frame_ns
                )
                last_frame_ns = frame_ns
                valid_frames += 1
                consecutive_errors = 0

            now_ns = time.perf_counter_ns()
            if now_ns >= next_progress_ns:
                elapsed_s = (now_ns - started_ns) / 1e9
                send(
                    "progress",
                    elapsed_s=round(elapsed_s, 3),
                    valid_frames=valid_frames,
                    total_errors=total_errors,
                    errors_by_reason=dict(errors_by_reason),
                    wall_fps=round(valid_frames / max(elapsed_s, 1e-9), 3),
                    max_gap_ms=round(max_gap_ms, 3),
                )
                next_progress_ns = now_ns + int(progress_s * 1e9)

            if period_ns > 0:
                deadline_ns += period_ns
                remaining_ns = deadline_ns - time.perf_counter_ns()
                if remaining_ns > 0:
                    stop_event.wait(remaining_ns / 1e9)
                else:
                    deadline_ns = time.perf_counter_ns()
    except BaseException as exc:
        send(
            "fatal",
            error=repr(exc),
            traceback=traceback.format_exc(),
            valid_frames=valid_frames,
            total_errors=total_errors,
            errors_by_reason=dict(errors_by_reason),
        )
    finally:
        if camera is not None:
            try:
                camera.disconnect()
            except Exception as exc:
                send("disconnect_error", error=repr(exc))

        ended_ns = time.perf_counter_ns()
        elapsed_s = (
            (ended_ns - started_ns) / 1e9
            if started_ns
            else 0.0
        )
        attempts = valid_frames + total_errors
        source_fps = (
            (valid_frames - 1) * 1e9 / (last_frame_ns - first_frame_ns)
            if (
                valid_frames > 1
                and first_frame_ns is not None
                and last_frame_ns is not None
                and last_frame_ns > first_frame_ns
            )
            else 0.0
        )
        send(
            "summary",
            elapsed_s=round(elapsed_s, 3),
            valid_frames=valid_frames,
            total_errors=total_errors,
            attempts=attempts,
            errors_by_reason=dict(errors_by_reason),
            error_ratio=round(
                total_errors / attempts if attempts else 0.0, 6
            ),
            source_fps=round(source_fps, 3),
            wall_fps=round(
                valid_frames / elapsed_s if elapsed_s > 0 else 0.0, 3
            ),
            max_gap_ms=round(max_gap_ms, 3),
            mean_read_ms=round(
                total_read_ms / attempts if attempts else 0.0, 3
            ),
            max_read_ms=round(max_read_ms, 3),
            max_consecutive_errors=max_consecutive_errors,
            last_frame_age_ms=round(
                (ended_ns - last_frame_ns) / 1e6
                if last_frame_ns is not None
                else elapsed_s * 1000.0,
                3,
            ),
            selected_format=selected_format,
        )


def _run_group(
    specs: list[CameraSpec],
    *,
    backend: str,
    duration_s: float,
    progress_s: float,
    stagger_start_ms: float,
    startup_timeout_s: float,
) -> list[dict[str, Any]]:
    if backend == "process":
        context = mp.get_context("spawn")
        start_event = context.Event()
        stop_event = context.Event()
        status_queue = context.Queue(maxsize=256)
        workers = [
            context.Process(
                name=f"camera-test-{spec.name}",
                target=_camera_worker,
                args=(
                    spec,
                    duration_s,
                    progress_s,
                    start_event,
                    stop_event,
                    status_queue,
                ),
            )
            for spec in specs
        ]
    else:
        start_event = threading.Event()
        stop_event = threading.Event()
        status_queue = queue.Queue(maxsize=256)
        workers = [
            threading.Thread(
                name=f"camera-test-{spec.name}",
                target=_camera_worker,
                args=(
                    spec,
                    duration_s,
                    progress_s,
                    start_event,
                    stop_event,
                    status_queue,
                ),
                daemon=False,
            )
            for spec in specs
        ]

    summaries: dict[str, dict[str, Any]] = {}
    ready: set[str] = set()
    fatal = False
    try:
        for index, worker in enumerate(workers):
            worker.start()
            if index + 1 < len(workers) and stagger_start_ms > 0:
                time.sleep(stagger_start_ms / 1000.0)

        startup_deadline = time.monotonic() + startup_timeout_s
        while len(ready) < len(specs):
            remaining = startup_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"相机启动超时，未就绪: "
                    f"{sorted({value.name for value in specs} - ready)}"
                )
            try:
                event = status_queue.get(timeout=min(0.2, remaining))
            except queue.Empty:
                continue
            _print_event(event)
            if event["kind"] == "ready":
                ready.add(str(event["camera"]))
            elif event["kind"] == "fatal":
                fatal = True
                raise RuntimeError(
                    f"{event['camera']} 启动失败: {event.get('error')}"
                )

        _print_event(
            {
                "kind": "concurrent_start",
                "backend": backend,
                "cameras": sorted(ready),
                "duration_s": duration_s,
            }
        )
        start_event.set()

        result_deadline = time.monotonic() + duration_s + startup_timeout_s
        while len(summaries) < len(specs):
            remaining = result_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("等待相机测试结果超时")
            try:
                event = status_queue.get(timeout=min(0.5, remaining))
            except queue.Empty:
                continue
            _print_event(event)
            if event["kind"] == "fatal":
                fatal = True
            elif event["kind"] == "summary":
                summaries[str(event["camera"])] = event
    except KeyboardInterrupt:
        _print_event({"kind": "interrupted"})
        fatal = True
    finally:
        stop_event.set()
        start_event.set()
        for worker in workers:
            worker.join(timeout=max(3.0, startup_timeout_s))
        if backend == "process":
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=2)
                if worker.is_alive():
                    worker.kill()
                    worker.join(timeout=2)
                worker.close()
            status_queue.close()
            status_queue.cancel_join_thread()

    if fatal:
        raise RuntimeError("相机测试出现 fatal 事件，请检查上方JSON日志")
    return [summaries[spec.name] for spec in specs]


def _parse_format_idx(value: str) -> int | None:
    if value.strip().lower() in {"auto", "none", "null"}:
        return None
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("format_idx 必须非负或为 auto")
    return result


def _build_spec(args: argparse.Namespace, side: str) -> CameraSpec:
    return CameraSpec(
        name=side,
        cam_num=int(getattr(args, f"{side}_cam_num")),
        width=args.width,
        height=args.height,
        fps=args.fps,
        format_idx=getattr(args, f"{side}_format_idx"),
        timeout_ms=args.timeout_ms,
        poll_hz=args.poll_hz,
        strict_fps=args.strict_fps,
        fps_tolerance=args.fps_tolerance,
    )


def _discover_format_indices(cam_num: int) -> list[int]:
    """枚举指定相机的 MJPG format 索引，供逐格式扫描使用。"""
    import camera as camera_module

    camera_module.detect_connected_cameras()
    if cam_num not in camera_module.camera_num_to_index:
        raise RuntimeError(f"未发现 cam_num={cam_num}")
    handle = camera_module.pyvizionsdk.VxInitialCameraDevice(
        camera_module.camera_num_to_index[cam_num]
    )
    opened = False
    try:
        result = camera_module.pyvizionsdk.VxOpen(handle)
        if not camera_module._sdk_call_succeeded(result):
            raise RuntimeError(f"VxOpen失败: {result!r}")
        opened = True
        result, formats = camera_module.pyvizionsdk.VxGetFormatList(handle)
        if not camera_module._sdk_call_succeeded(result):
            raise RuntimeError(f"VxGetFormatList失败: {result!r}")
        mjpg = [
            value
            for value in formats
            if (
                value.format
                == camera_module.VX_IMAGE_FORMAT.VX_IMAGE_FORMAT_MJPG
            )
        ]
        _print_event(
            {
                "kind": "format_list",
                "cam_num": cam_num,
                "formats": [
                    {
                        "index": index,
                        "width": int(value.width),
                        "height": int(value.height),
                        "fps": float(value.framerate),
                    }
                    for index, value in enumerate(mjpg)
                ],
            }
        )
        return list(range(len(mjpg)))
    finally:
        if opened:
            camera_module.pyvizionsdk.VxClose(handle)


def _print_verdict(
    summaries: list[dict[str, Any]], specs: list[CameraSpec]
) -> None:
    for summary, spec in zip(summaries, specs):
        error_ratio = float(summary["error_ratio"])
        source_fps = float(summary["source_fps"])
        selected = summary.get("selected_format") or {}
        selected_fps = float(selected.get("fps", spec.fps))
        expected_fps = (
            min(selected_fps, spec.poll_hz)
            if spec.poll_hz > 0
            else selected_fps
        )
        healthy = (
            int(summary["valid_frames"]) > 0
            and error_ratio <= 0.01
            and source_fps >= expected_fps * 0.8
        )
        _print_event(
            {
                "kind": "verdict",
                "camera": spec.name,
                "healthy": healthy,
                "expected_fps": expected_fps,
                "source_fps": source_fps,
                "error_ratio": error_ratio,
                "errors_by_reason": summary["errors_by_reason"],
                "max_gap_ms": summary["max_gap_ms"],
                "last_frame_age_ms": summary["last_frame_age_ms"],
                "selected_format": selected,
            }
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TechNexion 单/双相机并发与format稳定性测试"
    )
    parser.add_argument(
        "--mode",
        choices=(
            "single-left",
            "single-right",
            "sequential",
            "dual-thread",
            "dual-process",
            "sweep-left",
            "sweep-right",
        ),
        default="dual-process",
    )
    parser.add_argument("--left-cam-num", type=int, default=0)
    parser.add_argument("--right-cam-num", type=int, default=1)
    parser.add_argument(
        "--left-format-idx", type=_parse_format_idx, default=None
    )
    parser.add_argument(
        "--right-format-idx", type=_parse_format_idx, default=None
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--timeout-ms", type=int, default=1000)
    parser.add_argument(
        "--poll-hz",
        type=float,
        default=0.0,
        help="0表示尽快连续取流；30表示模拟生产代码30Hz轮询",
    )
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument(
        "--sweep-duration-s",
        type=float,
        default=10.0,
        help="逐格式扫描时每个format的测试时间",
    )
    parser.add_argument("--progress-s", type=float, default=1.0)
    parser.add_argument("--stagger-start-ms", type=float, default=500.0)
    parser.add_argument("--startup-timeout-s", type=float, default=15.0)
    parser.add_argument(
        "--strict-fps",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--fps-tolerance", type=float, default=1.0)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("width", "height", "fps", "timeout_ms"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须大于0")
    for name in (
        "duration_s",
        "sweep_duration_s",
        "progress_s",
        "startup_timeout_s",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须大于0")
    if args.poll_hz < 0 or args.stagger_start_ms < 0:
        raise ValueError("--poll-hz/--stagger-start-ms 不能为负数")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [camera-test] %(message)s",
    )
    args = build_parser().parse_args()
    _validate_args(args)
    left = _build_spec(args, "left")
    right = _build_spec(args, "right")

    if args.mode == "single-left":
        specs = [left]
        summaries = _run_group(
            specs,
            backend="thread",
            duration_s=args.duration_s,
            progress_s=args.progress_s,
            stagger_start_ms=0,
            startup_timeout_s=args.startup_timeout_s,
        )
        _print_verdict(summaries, specs)
    elif args.mode == "single-right":
        specs = [right]
        summaries = _run_group(
            specs,
            backend="thread",
            duration_s=args.duration_s,
            progress_s=args.progress_s,
            stagger_start_ms=0,
            startup_timeout_s=args.startup_timeout_s,
        )
        _print_verdict(summaries, specs)
    elif args.mode == "sequential":
        for spec in (left, right):
            summaries = _run_group(
                [spec],
                backend="thread",
                duration_s=args.duration_s,
                progress_s=args.progress_s,
                stagger_start_ms=0,
                startup_timeout_s=args.startup_timeout_s,
            )
            _print_verdict(summaries, [spec])
    elif args.mode in {"dual-thread", "dual-process"}:
        specs = [left, right]
        summaries = _run_group(
            specs,
            backend=(
                "thread" if args.mode == "dual-thread" else "process"
            ),
            duration_s=args.duration_s,
            progress_s=args.progress_s,
            stagger_start_ms=args.stagger_start_ms,
            startup_timeout_s=args.startup_timeout_s,
        )
        _print_verdict(summaries, specs)
    else:
        base = left if args.mode == "sweep-left" else right
        indices = _discover_format_indices(base.cam_num)
        for index in indices:
            spec = CameraSpec(
                **{
                    **asdict(base),
                    "name": f"{base.name}-format-{index}",
                    "format_idx": index,
                    "strict_fps": False,
                }
            )
            summaries = _run_group(
                [spec],
                backend="thread",
                duration_s=args.sweep_duration_s,
                progress_s=args.progress_s,
                stagger_start_ms=0,
                startup_timeout_s=args.startup_timeout_s,
            )
            _print_verdict(summaries, [spec])


if __name__ == "__main__":
    main()
