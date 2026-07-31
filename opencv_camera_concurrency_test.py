"""纯 OpenCV/V4L2 双相机并发测试。

不启动机器人、不导入 pyvizionsdk、不写 LeRobot 数据集，也不编码视频。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import multiprocessing as mp
import queue
import threading
import time
import traceback
from typing import Any
import zlib


@dataclass(frozen=True)
class WorkerSpec:
    name: str
    device: str | int
    width: int
    height: int
    fps: float
    fourcc: str
    backend: str
    buffer_size: int
    open_timeout_ms: int
    read_timeout_ms: int
    strict_resolution: bool
    fps_tolerance: float
    failure_backoff_ms: float


def _print_event(event: dict[str, Any]) -> None:
    print(json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True)


def _frame_signature(frame: Any) -> int:
    """低开销帧指纹，用于识别静止缓存帧反复返回。"""
    sampled = frame[::16, ::16]
    return zlib.crc32(memoryview(sampled.copy()).cast("B"))


def _worker(
    spec: WorkerSpec,
    duration_s: float,
    progress_s: float,
    start_event: Any,
    stop_event: Any,
    status_queue: Any,
) -> None:
    camera = None

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
        # 子进程/工作线程内延迟导入，避免 fork 继承 OpenCV 内部状态。
        import cv2
        from opencv_camera import OpenCVCamera, OpenCVCameraConfig

        cv2.setNumThreads(1)
        camera = OpenCVCamera(
            OpenCVCameraConfig(
                device=spec.device,
                width=spec.width,
                height=spec.height,
                fps=spec.fps,
                fourcc=spec.fourcc,
                backend=spec.backend,
                buffer_size=spec.buffer_size,
                open_timeout_ms=spec.open_timeout_ms,
                read_timeout_ms=spec.read_timeout_ms,
                strict_resolution=spec.strict_resolution,
                fps_tolerance=spec.fps_tolerance,
                name=f"opencv-{spec.name}",
            )
        )
        camera.connect()
        send("opened", description=camera.describe())

        # 两路计时同步前必须持续取帧，避免打开设备后停在屏障处塞满缓冲区。
        warmup_frames = 0
        warmup_errors = 0
        ready_reported = False
        while not stop_event.is_set() and not start_event.is_set():
            try:
                camera.read_bgr()
            except Exception as exc:
                warmup_errors += 1
                if spec.failure_backoff_ms > 0:
                    stop_event.wait(spec.failure_backoff_ms / 1000.0)
                if warmup_errors == 1 or warmup_errors % 30 == 0:
                    send(
                        "warmup_error",
                        error=repr(exc),
                        warmup_errors=warmup_errors,
                    )
            else:
                warmup_frames += 1
                if not ready_reported:
                    send(
                        "ready",
                        warmup_frames=warmup_frames,
                        warmup_errors=warmup_errors,
                    )
                    ready_reported = True
        if stop_event.is_set():
            return

        started_ns = time.perf_counter_ns()
        finish_ns = started_ns + int(duration_s * 1e9)
        next_progress_ns = started_ns + int(progress_s * 1e9)
        valid_frames = 0
        read_errors = 0
        consecutive_errors = 0
        max_consecutive_errors = 0
        first_frame_ns: int | None = None
        last_frame_ns: int | None = None
        max_gap_ms = 0.0
        total_read_ms = 0.0
        max_read_ms = 0.0
        last_signature: int | None = None
        consecutive_duplicates = 0
        max_consecutive_duplicates = 0
        duplicate_frames = 0
        unique_signatures: set[int] = set()

        send(
            "measurement_started",
            warmup_frames=warmup_frames,
            warmup_errors=warmup_errors,
        )
        while (
            not stop_event.is_set()
            and time.perf_counter_ns() < finish_ns
        ):
            read_started_ns = time.perf_counter_ns()
            try:
                frame, stamp_ns = camera.read_bgr()
            except Exception as exc:
                read_ended_ns = time.perf_counter_ns()
                read_ms = (read_ended_ns - read_started_ns) / 1e6
                total_read_ms += read_ms
                max_read_ms = max(max_read_ms, read_ms)
                read_errors += 1
                consecutive_errors += 1
                max_consecutive_errors = max(
                    max_consecutive_errors, consecutive_errors
                )
                if read_errors == 1 or read_errors % 30 == 0:
                    send(
                        "read_error",
                        error=repr(exc),
                        read_errors=read_errors,
                        consecutive_errors=consecutive_errors,
                    )
                if spec.failure_backoff_ms > 0:
                    stop_event.wait(spec.failure_backoff_ms / 1000.0)
            else:
                read_ended_ns = time.perf_counter_ns()
                read_ms = (read_ended_ns - read_started_ns) / 1e6
                total_read_ms += read_ms
                max_read_ms = max(max_read_ms, read_ms)
                consecutive_errors = 0
                valid_frames += 1
                if first_frame_ns is None:
                    first_frame_ns = stamp_ns
                if last_frame_ns is not None:
                    max_gap_ms = max(
                        max_gap_ms, (stamp_ns - last_frame_ns) / 1e6
                    )
                last_frame_ns = stamp_ns

                signature = _frame_signature(frame)
                unique_signatures.add(signature)
                if signature == last_signature:
                    duplicate_frames += 1
                    consecutive_duplicates += 1
                    max_consecutive_duplicates = max(
                        max_consecutive_duplicates,
                        consecutive_duplicates,
                    )
                else:
                    consecutive_duplicates = 0
                last_signature = signature

            now_ns = time.perf_counter_ns()
            if now_ns >= next_progress_ns:
                elapsed_s = (now_ns - started_ns) / 1e9
                send(
                    "progress",
                    elapsed_s=round(elapsed_s, 3),
                    valid_frames=valid_frames,
                    read_errors=read_errors,
                    wall_fps=round(
                        valid_frames / max(elapsed_s, 1e-9), 3
                    ),
                    max_gap_ms=round(max_gap_ms, 3),
                    duplicate_frames=duplicate_frames,
                    unique_ratio=round(
                        len(unique_signatures)
                        / max(valid_frames, 1),
                        4,
                    ),
                )
                next_progress_ns = now_ns + int(progress_s * 1e9)

        ended_ns = time.perf_counter_ns()
        elapsed_s = (ended_ns - started_ns) / 1e9
        attempts = valid_frames + read_errors
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
            attempts=attempts,
            valid_frames=valid_frames,
            read_errors=read_errors,
            error_ratio=round(
                read_errors / attempts if attempts else 0.0, 6
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
            duplicate_frames=duplicate_frames,
            max_consecutive_duplicates=max_consecutive_duplicates,
            unique_ratio=round(
                len(unique_signatures) / max(valid_frames, 1), 6
            ),
            last_frame_age_ms=round(
                (ended_ns - last_frame_ns) / 1e6
                if last_frame_ns is not None
                else elapsed_s * 1000,
                3,
            ),
            warmup_frames=warmup_frames,
            warmup_errors=warmup_errors,
            actual_properties=camera.actual_properties,
        )
    except BaseException as exc:
        send(
            "fatal",
            error=repr(exc),
            traceback=traceback.format_exc(),
        )
    finally:
        if camera is not None:
            try:
                camera.disconnect()
            except Exception as exc:
                send("disconnect_error", error=repr(exc))


def _run_group(
    specs: list[WorkerSpec],
    *,
    execution: str,
    duration_s: float,
    progress_s: float,
    startup_timeout_s: float,
    shutdown_timeout_s: float,
    stagger_start_ms: float,
) -> list[dict[str, Any]]:
    if execution == "process":
        context = mp.get_context("spawn")
        start_event = context.Event()
        stop_event = context.Event()
        status_queue = context.Queue(maxsize=512)
        workers = [
            context.Process(
                name=f"opencv-camera-{spec.name}",
                target=_worker,
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
        status_queue = queue.Queue(maxsize=512)
        workers = [
            threading.Thread(
                name=f"opencv-camera-{spec.name}",
                target=_worker,
                args=(
                    spec,
                    duration_s,
                    progress_s,
                    start_event,
                    stop_event,
                    status_queue,
                ),
                # V4L2 驱动若永久阻塞在 read()，Python 无法强制结束线程。
                # 守护线程保证诊断程序仍可退出；正式系统应使用进程 watchdog。
                daemon=True,
            )
            for spec in specs
        ]

    ready: set[str] = set()
    summaries: dict[str, dict[str, Any]] = {}
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
                    f"相机启动超时，未取得首帧: "
                    f"{sorted({spec.name for spec in specs} - ready)}"
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
                "execution": execution,
                "cameras": sorted(ready),
                "duration_s": duration_s,
            }
        )
        start_event.set()

        result_deadline = (
            time.monotonic() + duration_s + shutdown_timeout_s
        )
        while len(summaries) < len(specs):
            remaining = result_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("等待相机测试结果超时，可能阻塞在 read()")
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
    except BaseException as exc:
        _print_event(
            {
                "kind": "runner_error",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        fatal = True
    finally:
        stop_event.set()
        start_event.set()
        for worker in workers:
            worker.join(timeout=shutdown_timeout_s)
        if execution == "process":
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
        else:
            for worker in workers:
                if worker.is_alive():
                    _print_event(
                        {
                            "kind": "worker_hung",
                            "worker": worker.name,
                            "reason": "可能阻塞在 VideoCapture.read()",
                        }
                    )
                    fatal = True

    if fatal or len(summaries) != len(specs):
        raise SystemExit(2)
    return [summaries[spec.name] for spec in specs]


def _parse_device(value: str) -> str | int:
    """纯数字参数按 OpenCV index 处理，路径按字符串处理。"""
    stripped = value.strip()
    return int(stripped) if stripped.isdecimal() else stripped


def _make_spec(args: argparse.Namespace, side: str) -> WorkerSpec:
    return WorkerSpec(
        name=side,
        device=getattr(args, f"{side}_device"),
        width=args.width,
        height=args.height,
        fps=args.fps,
        fourcc=args.fourcc.upper(),
        backend=args.backend,
        buffer_size=args.buffer_size,
        open_timeout_ms=args.open_timeout_ms,
        read_timeout_ms=args.read_timeout_ms,
        strict_resolution=args.strict_resolution,
        fps_tolerance=args.fps_tolerance,
        failure_backoff_ms=args.failure_backoff_ms,
    )


def _print_verdict(
    summaries: list[dict[str, Any]],
    specs: list[WorkerSpec],
) -> None:
    for summary, spec in zip(summaries, specs):
        fps_ok = float(summary["source_fps"]) >= spec.fps * 0.8
        errors_ok = float(summary["error_ratio"]) <= 0.01
        recent_ok = (
            float(summary["last_frame_age_ms"])
            <= max(1000.0, 5 * 1000.0 / spec.fps)
        )
        healthy = (
            int(summary["valid_frames"]) > 0
            and fps_ok
            and errors_ok
            and recent_ok
        )
        _print_event(
            {
                "kind": "verdict",
                "camera": spec.name,
                "healthy": healthy,
                "expected_fps": spec.fps,
                "source_fps": summary["source_fps"],
                "error_ratio": summary["error_ratio"],
                "max_gap_ms": summary["max_gap_ms"],
                "last_frame_age_ms": summary["last_frame_age_ms"],
                "unique_ratio": summary["unique_ratio"],
                "actual_properties": summary["actual_properties"],
            }
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="不依赖 pyvizionsdk 的 OpenCV/V4L2 双相机测试"
    )
    parser.add_argument(
        "--mode",
        choices=(
            "single-left",
            "single-right",
            "sequential",
            "dual-thread",
            "dual-process",
        ),
        default="dual-thread",
    )
    parser.add_argument(
        "--left-device", type=_parse_device, default="/dev/video6"
    )
    parser.add_argument(
        "--right-device", type=_parse_device, default="/dev/video8"
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument(
        "--backend", choices=("v4l2", "any"), default="v4l2"
    )
    parser.add_argument("--buffer-size", type=int, default=1)
    parser.add_argument("--open-timeout-ms", type=int, default=5000)
    parser.add_argument("--read-timeout-ms", type=int, default=2000)
    parser.add_argument("--failure-backoff-ms", type=float, default=5.0)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--progress-s", type=float, default=1.0)
    parser.add_argument("--startup-timeout-s", type=float, default=15.0)
    parser.add_argument("--shutdown-timeout-s", type=float, default=5.0)
    parser.add_argument("--stagger-start-ms", type=float, default=300.0)
    parser.add_argument(
        "--strict-resolution",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fps-tolerance", type=float, default=2.0)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "width",
        "height",
        "fps",
        "buffer_size",
        "open_timeout_ms",
        "read_timeout_ms",
        "duration_s",
        "progress_s",
        "startup_timeout_s",
        "shutdown_timeout_s",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须大于 0")
    if len(args.fourcc) != 4:
        raise ValueError("--fourcc 必须是四个字符，例如 MJPG")
    if args.failure_backoff_ms < 0 or args.stagger_start_ms < 0:
        raise ValueError("backoff 和 stagger 不能为负数")
    if (
        args.mode in {"dual-thread", "dual-process"}
        and args.left_device == args.right_device
    ):
        raise ValueError("双相机模式的 left-device 和 right-device 不能相同")


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    left = _make_spec(args, "left")
    right = _make_spec(args, "right")

    groups: list[tuple[list[WorkerSpec], str]]
    if args.mode == "single-left":
        groups = [([left], "thread")]
    elif args.mode == "single-right":
        groups = [([right], "thread")]
    elif args.mode == "sequential":
        groups = [([left], "thread"), ([right], "thread")]
    elif args.mode == "dual-process":
        groups = [([left, right], "process")]
    else:
        groups = [([left, right], "thread")]

    for specs, execution in groups:
        summaries = _run_group(
            specs,
            execution=execution,
            duration_s=args.duration_s,
            progress_s=args.progress_s,
            startup_timeout_s=args.startup_timeout_s,
            shutdown_timeout_s=args.shutdown_timeout_s,
            stagger_start_ms=args.stagger_start_ms,
        )
        _print_verdict(summaries, specs)


if __name__ == "__main__":
    main()
