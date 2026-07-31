from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable

import numpy as np

from model import (
    BimanualControlCommand,
    BimanualRobotState,
    ControlCommand,
    RobotState,
)

LOG = logging.getLogger(__name__)


class LatestCommandWorker:
    """只执行最新命令的非阻塞硬件工作线程。

    控制线程只做 put_nowait，不会被 CAN/串口调用阻塞。队列满时覆盖旧命令。
    """

    def __init__(
        self,
        name: str,
        callback: Callable[[np.ndarray], None],
        max_hz: float | None = None,
    ) -> None:
        self.name = name
        self.callback = callback
        self.max_hz = float(max_hz) if max_hz is not None else None
        if self.max_hz is not None and self.max_hz <= 0:
            raise ValueError(f"{name} max_hz 必须大于0")
        self.min_interval_ns = (
            int(1e9 / self.max_hz) if self.max_hz is not None else 0
        )
        self.queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.last_error: str | None = None
        self.dropped = 0
        self.coalesced = 0
        self.submitted = 0
        self.completed = 0
        self.failed = 0
        self.last_duration_ms = 0.0
        self.max_duration_ms = 0.0
        self.first_completed_ns = 0
        self.last_completed_ns = 0
        self.thread.start()

    def submit(self, value: np.ndarray) -> None:
        self.submitted += 1
        try:
            self.queue.put_nowait(value.copy())
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            self.queue.put_nowait(value.copy())
            self.dropped += 1

    def _run(self) -> None:
        last_started_ns = 0
        while True:
            value = self.queue.get()
            if value is None:
                return

            # 限频等待期间继续消费单槽队列，只保留最近目标。
            # 控制线程可以100Hz计算，真实硬件 SDK 不会被同频率轰炸。
            if self.min_interval_ns > 0 and last_started_ns > 0:
                deadline_ns = last_started_ns + self.min_interval_ns
                while True:
                    remaining_ns = deadline_ns - time.perf_counter_ns()
                    if remaining_ns <= 0:
                        break
                    try:
                        newer = self.queue.get(
                            timeout=remaining_ns / 1e9
                        )
                    except queue.Empty:
                        break
                    if newer is None:
                        return
                    value = newer
                    self.coalesced += 1

            started_ns = time.perf_counter_ns()
            last_started_ns = started_ns
            try:
                self.callback(value)
                self.last_error = None
                self.completed += 1
                completed_ns = time.perf_counter_ns()
                if self.first_completed_ns == 0:
                    self.first_completed_ns = completed_ns
                self.last_completed_ns = completed_ns
            except Exception as exc:  # 硬件异常留在工作线程并进入诊断
                self.last_error = repr(exc)
                self.failed += 1
                LOG.exception("%s 下发失败", self.name)
            finally:
                duration_ms = (time.perf_counter_ns() - started_ns) / 1e6
                self.last_duration_ms = duration_ms
                self.max_duration_ms = max(self.max_duration_ms, duration_ms)

    def status_snapshot(self) -> dict[str, Any]:
        completed_span_ns = self.last_completed_ns - self.first_completed_ns
        effective_hz = (
            (self.completed - 1) * 1e9 / completed_span_ns
            if self.completed > 1 and completed_span_ns > 0
            else 0.0
        )
        return {
            "max_hz": self.max_hz,
            "effective_hz": round(effective_hz, 3),
            "submitted": self.submitted,
            "completed": self.completed,
            "failed": self.failed,
            "dropped": self.dropped,
            "coalesced": self.coalesced,
            "last_duration_ms": round(self.last_duration_ms, 3),
            "max_duration_ms": round(self.max_duration_ms, 3),
            "last_error": self.last_error,
            "alive": self.thread.is_alive(),
        }

    def close(self, timeout_s: float = 3.0) -> bool:
        """丢弃未执行命令并等待工作线程退出。

        返回 False 表示线程仍阻塞在硬件 SDK 中。此时上层必须保持机械臂 disable，
        且不能在同一硬件会话中再次进入 ACTIVE。
        """
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.queue.put(None)
        self.thread.join(timeout=timeout_s)
        return not self.thread.is_alive()


class DualPiperAerohand:
    """设备 B 的双 Piper + 双 Aerohand 真实硬件适配器。"""

    def __init__(self, cfg: dict[str, Any]) -> None:
        # 构造函数只初始化 Python 状态，不打开 CAN/串口，也不触发真实运动。
        self.cfg = cfg
        self.arms: dict[str, Any] = {}
        self.hands: dict[str, Any] = {}
        self.arm_locks = {side: threading.Lock() for side in ("left", "right")}
        self.workers: dict[str, LatestCommandWorker] = {}
        self.last_hand = {side: np.zeros(7, np.float32) for side in ("left", "right")}
        self.last_arm_pose: dict[str, np.ndarray | None] = {
            side: None for side in ("left", "right")
        }
        self.last_arm_target: dict[str, np.ndarray | None] = {
            side: None for side in ("left", "right")
        }
        self.first_arm_target: dict[str, np.ndarray | None] = {
            side: None for side in ("left", "right")
        }
        self.arm_diagnostics: dict[str, dict[str, Any]] = {
            side: {
                "move_p_calls": 0,
                "last_move_p_ms": 0.0,
                "max_move_p_ms": 0.0,
                "last_status_poll_ns": 0,
            }
            for side in ("left", "right")
        }
        self.last_command: BimanualControlCommand | None = None
        self.state = "new"
        self.homed = False
        self.start_home_count = 0
        self.last_start_home: dict[str, dict[str, Any]] = {}
        self.start_prepared = False
        self._lifecycle_lock = threading.RLock()

    @property
    def initialized(self) -> bool:
        return self.state not in {"new", "closed"}

    @property
    def active(self) -> bool:
        return self.state == "active"

    def status_snapshot(self) -> dict[str, Any]:
        arm_status: dict[str, Any] = {}
        for side in ("left", "right"):
            target = self.last_arm_target[side]
            first_target = self.first_arm_target[side]
            actual = self.last_arm_pose[side]
            diagnostics = {
                key: value
                for key, value in self.arm_diagnostics[side].items()
                if key != "last_status_poll_ns"
            }
            diagnostics["target"] = (
                target.tolist() if target is not None else None
            )
            diagnostics["actual"] = (
                actual.tolist() if actual is not None else None
            )
            diagnostics["target_delta_from_first_m"] = (
                round(float(np.linalg.norm(target[:3] - first_target[:3])), 6)
                if target is not None and first_target is not None
                else None
            )
            diagnostics["tracking_error_m"] = (
                round(float(np.linalg.norm(target[:3] - actual[:3])), 6)
                if target is not None and actual is not None
                else None
            )
            arm_status[side] = diagnostics
        return {
            "state": self.state,
            "initialized": self.initialized,
            "active": self.active,
            "homed": self.homed,
            "start_prepared": self.start_prepared,
            "start_home_count": self.start_home_count,
            "last_start_home": dict(self.last_start_home),
            "connected_arms": sorted(self.arms),
            "connected_hands": sorted(self.hands),
            "arms": arm_status,
            "workers": {
                name: worker.status_snapshot()
                for name, worker in self.workers.items()
            },
        }

    @staticmethod
    def _status_value(value: Any) -> int | str | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _message_array(value: Any, expected_shape: tuple[int, ...]) -> np.ndarray:
        raw = getattr(value, "msg", value)
        result = np.asarray(raw, dtype=np.float32)
        if result.shape != expected_shape or not np.all(np.isfinite(result)):
            raise RuntimeError(
                f"硬件反馈维度或数值非法，期望 {expected_shape}，实际 {result.shape}"
            )
        return result

    def _read_arm_feedback(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        arm = self.arms[side]
        flange = arm.get_flange_pose()
        joints = arm.get_joint_angles()
        if flange is None or joints is None:
            raise RuntimeError(f"{side} Piper 未返回完整状态")
        return (
            self._message_array(flange, (6,)),
            self._message_array(joints, (6,)),
        )

    def _read_hand_feedback(self, side: str) -> np.ndarray:
        state = self.hands[side].get_joint_positions_compact()
        result = self._message_array(state, (7,))
        self.last_hand[side] = result.copy()
        return result

    def _wait_for_feedback(self, side: str) -> None:
        timeout_s = float(self.cfg[side].get("health_timeout_s", 3.0))
        deadline = time.monotonic() + timeout_s
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                self._read_arm_feedback(side)
                self._read_hand_feedback(side)
                return
            except BaseException as exc:
                last_error = exc
                time.sleep(0.05)
        raise RuntimeError(
            f"{side} 侧设备在 {timeout_s:.1f}s 内未返回有效状态: {last_error!r}"
        )

    def _disable_all_arms(self) -> None:
        for side in ("left", "right"):
            arm = self.arms.get(side)
            if arm is None:
                continue
            try:
                arm.disable()
            except Exception:
                LOG.exception("%s Piper disable 失败", side)

    def initialize(self) -> None:
        """创建并连接全部硬件，最后确保双臂处于 disable 状态。

        该方法只应在 control 子进程启动时调用一次。任何一侧失败都会回滚已经打开的
        左右臂和灵巧手，禁止以单侧降级方式进入 READY。
        """
        with self._lifecycle_lock:
            if self.state == "idle_disabled":
                return
            if self.state != "new":
                raise RuntimeError(f"当前状态 {self.state} 不允许 initialize")
            self.state = "initializing"
            try:
                from pyAgxArm import (
                    AgxArmFactory,
                    ArmModel,
                    PiperFW,
                    create_agx_arm_config,
                )
                from aero_open_sdk.aero_hand import AeroHand
            except ImportError as exc:
                self.state = "fault"
                raise RuntimeError(
                    "真实机器人模式需要安装 pyAgxArm 和 aero_open_sdk"
                ) from exc

            try:
                # 先连接并立即 disable 两台机械臂，避免第二侧初始化期间第一侧保持使能。
                for side in ("left", "right"):
                    side_cfg = self.cfg[side]
                    arm_config = create_agx_arm_config(
                        robot=ArmModel.PIPER,
                        firmeware_version=PiperFW.DEFAULT,
                        interface=side_cfg["interface"],
                        channel=str(side_cfg.get("channel", "0")),
                        bitrate=int(side_cfg.get("bitrate", 1_000_000)),
                    )
                    LOG.info("%s Piper 正在连接", side)
                    arm = AgxArmFactory.create_arm(arm_config)
                    # 先保存句柄，保证 connect/disable 中途失败时 close() 仍能清理。
                    self.arms[side] = arm
                    arm.connect()
                    arm.disable()
                    arm.set_speed_percent(int(side_cfg.get("speed_percent", 20)))

                for side in ("left", "right"):
                    LOG.info("%s Aerohand 正在连接", side)
                    hand = AeroHand(port=self.cfg[side]["hand_port"])
                    self.hands[side] = hand

                for side in ("left", "right"):
                    self._wait_for_feedback(side)
                self.state = "idle_disabled"
                LOG.info("双 Piper/双 Aerohand 已连接，双臂保持 disable")
            except BaseException:
                self.state = "fault"
                self.close()
                raise

    def _validated_home_pose(self, side: str) -> np.ndarray:
        side_cfg = self.cfg[side]
        if "home_pose" not in side_cfg:
            raise RuntimeError(
                f"robot.{side}.home_pose 未配置；为避免把 initial_pose 的坐标语义"
                "误当作回零目标，拒绝执行真实运动"
            )
        pose = np.asarray(side_cfg["home_pose"], dtype=np.float32)
        if pose.shape != (6,) or not np.all(np.isfinite(pose)):
            raise RuntimeError(f"robot.{side}.home_pose 必须是有效的6维关节角")
        return pose

    def _validated_zero_pose(self, side: str) -> np.ndarray:
        side_cfg = self.cfg[side]
        if "initial_pose" not in side_cfg:
            raise RuntimeError(
                f"robot.{side}.initial_pose 未配置；"
            )
        pose = np.asarray(side_cfg["initial_pose"], dtype=np.float32)
        if pose.shape != (6,) or not np.all(np.isfinite(pose)):
            raise RuntimeError(f"robot.{side}.initial_pose 必须是有效的6维关节角")
        return pose

    def _enable_arm(self, side: str) -> None:
        arm = self.arms[side]
        timeout_s = float(self.cfg[side].get("enable_timeout_s", 5.0))
        deadline = time.monotonic() + timeout_s
        while not arm.enable():
            if time.monotonic() >= deadline:
                raise RuntimeError(f"{side} Piper 使能超时")
            time.sleep(0.05)

    def _wait_motion_done(
        self, side: str, target: np.ndarray, timeout_s: float
    ) -> None:
        """等待 move_j 完成，并用真实关节反馈校验是否到达目标。"""
        time.sleep(0.3)
        deadline = time.monotonic() + timeout_s
        tolerance = float(
            self.cfg[side].get("home_joint_tolerance_rad", 0.10)
        )
        last_motion_status: Any = None
        last_joint_error = float("inf")
        last_joints: np.ndarray | None = None
        while time.monotonic() < deadline:
            status = self.arms[side].get_arm_status()
            last_motion_status = getattr(
                getattr(status, "msg", status), "motion_status", None
            )
            flange, joints = self._read_arm_feedback(side)
            self.last_arm_pose[side] = flange.copy()
            last_joints = joints
            joint_delta = joints - target
            wrapped_delta = np.arctan2(
                np.sin(joint_delta), np.cos(joint_delta)
            )
            last_joint_error = float(np.max(np.abs(wrapped_delta)))
            if last_motion_status == 0 and last_joint_error <= tolerance:
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"{side} Piper move_j 在 {timeout_s:.1f}s 内未到达 home_pose；"
            f"motion_status={last_motion_status}, "
            f"max_joint_error={last_joint_error:.4f}rad, "
            f"tolerance={tolerance:.4f}rad, "
            f"target={target.tolist()}, "
            f"actual={last_joints.tolist() if last_joints is not None else None}"
        )

    def home(self) -> None:
        """显式执行双侧回零；运动完成并校验后再次 disable 双臂。"""
        with self._lifecycle_lock:
            if self.state != "idle_disabled":
                raise RuntimeError(f"当前状态 {self.state} 不允许 home")
            # 在任何机械臂使能之前一次性校验左右目标，避免单侧先动后才发现另一侧配置错误。
            targets = {
                side: self._validated_home_pose(side)
                for side in ("left", "right")
            }
            zero_targets = {
                side: self._validated_zero_pose(side)
                for side in ("left", "right")
            }
            self.state = "homing"
            try:
                # 双臂依次回零，减少双臂轨迹同时运动造成的碰撞风险。
                # for side in ("left", "right"):
                for side in ("right", "left"):
                    arm = self.arms[side]
                    self._enable_arm(side)
                    try:
                        arm.set_speed_percent(
                            int(self.cfg[side].get("home_speed_percent", 10))
                        )
                        arm.move_j(targets[side].tolist())
                        self._wait_motion_done(
                            side,
                            targets[side],
                            float(self.cfg[side].get("home_timeout_s", 10.0)),
                        )

                    finally:
                        arm.move_j(zero_targets[side].tolist())
                        self._wait_motion_done(
                            side,
                            zero_targets[side],
                            float(self.cfg[side].get("home_timeout_s", 10.0)),
                        )
                        arm.disable()
                    LOG.info("%s Piper 回零完成并已 disable", side)

                for side in ("left", "right"):
                    result = self.hands[side].send_homing()
                    if result is False:
                        raise RuntimeError(f"{side} Aerohand homing 返回失败")
                    wait_s = float(self.cfg[side].get("hand_home_wait_s", 8.0))
                    if wait_s > 0:
                        time.sleep(wait_s)
                    self._read_hand_feedback(side)
                    LOG.info("%s Aerohand 回零完成", side)

                self.homed = True
                self.state = "idle_disabled"
            except BaseException:
                self.homed = False
                self._disable_all_arms()
                self.state = "fault"
                raise

    def prepare_start(self) -> None:
        """start 前依次将双臂移动到已标定的关节 home_pose。

        该阶段不启动遥操作命令线程。每侧到位后保持 enable，全部完成后才允许
        相机和 episode 继续启动，因此回 home 的运动不会写入训练数据，也不会
        在正式遥操作前发生一次多余的 disable/enable。
        """
        with self._lifecycle_lock:
            if self.state != "idle_disabled":
                raise RuntimeError(
                    f"当前状态 {self.state} 不允许 prepare_start"
                )

            # 必须在任何机械臂运动前一次性校验左右目标，避免只移动一侧。
            targets = {
                side: self._validated_home_pose(side)
                for side in ("left", "right")
            }
            self.state = "preparing_start"
            self.start_prepared = False
            self.last_start_home = {}
            try:
                # 沿用现有真机验证顺序，双臂绝不同时执行回 home 轨迹。
                for side in ("right", "left"):
                    arm = self.arms[side]
                    self._enable_arm(side)
                    speed = int(
                        self.cfg[side].get("home_speed_percent", 10)
                    )
                    timeout_s = float(
                        self.cfg[side].get("home_timeout_s", 10.0)
                    )
                    arm.set_speed_percent(speed)
                    LOG.info(
                        "%s Piper start 前移动到 home_pose: %s",
                        side,
                        targets[side].tolist(),
                    )
                    started_ns = time.perf_counter_ns()
                    arm.move_j(targets[side].tolist())
                    self._wait_motion_done(
                        side, targets[side], timeout_s
                    )
                    _, actual_joints = self._read_arm_feedback(side)
                    elapsed_s = (
                        time.perf_counter_ns() - started_ns
                    ) / 1e9
                    self.last_start_home[side] = {
                        "target_joints": targets[side].tolist(),
                        "actual_joints": actual_joints.tolist(),
                        "elapsed_s": round(elapsed_s, 3),
                        "speed_percent": speed,
                    }
                    LOG.info(
                        "%s Piper 已到达 start home_pose，保持 enable", side
                    )

                self.homed = True
                self.start_home_count += 1
                self.start_prepared = True
                self.state = "prepared_enabled"
            except BaseException:
                self.homed = False
                self.start_prepared = False
                self._disable_all_arms()
                self.state = "fault"
                raise

    def activate(self) -> None:
        """从保持使能的 home_pose 直接启动非阻塞遥操作命令线程。"""
        with self._lifecycle_lock:
            if self.state != "prepared_enabled":
                raise RuntimeError(f"当前状态 {self.state} 不允许 activate")
            if not self.start_prepared:
                raise RuntimeError(
                    "本次 start 尚未完成双臂 home_pose 准备，拒绝 activate"
                )
            # if bool(self.cfg.get("require_home_before_start", True)) and not self.homed:
            #     raise RuntimeError("尚未成功执行显式 home，拒绝 start")

            for side in ("left", "right"):
                self._wait_for_feedback(side)

            try:
                for side in ("left", "right"):
                    # prepare_start 已保持 enable；这里只切换为遥操作速度，
                    # 不再执行 disable/enable。
                    self.arms[side].set_speed_percent(
                        int(self.cfg[side].get("speed_percent", 20))
                    )
                    self.last_arm_target[side] = None
                    self.first_arm_target[side] = None
                    self.arm_diagnostics[side] = {
                        "move_p_calls": 0,
                        "last_move_p_ms": 0.0,
                        "max_move_p_ms": 0.0,
                        # 不在第一条 move_p 后立即读取，避免拿到上一会话残留状态；
                        # 首次诊断在配置的轮询周期后进行。
                        "last_status_poll_ns": time.perf_counter_ns(),
                    }

                for side in ("left", "right"):
                    self.workers[f"arm_{side}"] = LatestCommandWorker(
                        f"piper-{side}",
                        lambda value, s=side: self._send_arm(s, value),
                        max_hz=float(self.cfg.get("arm_command_hz", 30.0)),
                    )
                    self.workers[f"hand_{side}"] = LatestCommandWorker(
                        f"aerohand-{side}",
                        lambda value, s=side: self._send_hand(s, value),
                        max_hz=float(self.cfg.get("hand_command_hz", 60.0)),
                    )
                self.state = "active"
                # 准备状态只允许消费一次；下一个 episode 必须重新回 home_pose。
                self.start_prepared = False
            except BaseException:
                self.start_prepared = False
                self._disable_all_arms()
                self.state = "fault"
                raise

    def _send_arm(self, side: str, value: np.ndarray) -> None:
        target = np.asarray(value, dtype=np.float32)
        if target.shape != (6,) or not np.all(np.isfinite(target)):
            raise ValueError(
                f"{side} Piper move_p 目标必须是有效6维末端位姿，实际 {target.shape}"
            )

        with self.arm_locks[side]:
            started_ns = time.perf_counter_ns()
            self.arms[side].move_p(target.tolist())
            duration_ms = (time.perf_counter_ns() - started_ns) / 1e6

            diagnostics = self.arm_diagnostics[side]
            diagnostics["move_p_calls"] += 1
            diagnostics["last_move_p_ms"] = round(duration_ms, 3)
            diagnostics["max_move_p_ms"] = round(
                max(float(diagnostics["max_move_p_ms"]), duration_ms), 3
            )
            self.last_arm_target[side] = target.copy()
            if self.first_arm_target[side] is None:
                self.first_arm_target[side] = target.copy()

            # move_p() 只负责发送 CAN 帧，不返回控制器是否接受目标。低频读取
            # arm_status 才能发现无解、奇异点、目标越限、刹车未释放等异步错误。
            now_ns = time.perf_counter_ns()
            poll_interval_ns = int(
                float(self.cfg[side].get("arm_status_poll_interval_s", 1.0))
                * 1e9
            )
            if (
                now_ns - int(diagnostics["last_status_poll_ns"])
                >= poll_interval_ns
            ):
                status = self.arms[side].get_arm_status()
                if status is None:
                    raise RuntimeError(f"{side} Piper 未返回 arm_status")
                message = getattr(status, "msg", status)
                for name in (
                    "ctrl_mode",
                    "arm_status",
                    "mode_feedback",
                    "motion_status",
                    "err_status",
                ):
                    diagnostics[name] = self._status_value(
                        getattr(message, name, None)
                    )
                if hasattr(self.arms[side], "get_joints_enable_status_list"):
                    enabled_feedback = self.arms[
                        side
                    ].get_joints_enable_status_list()
                    if enabled_feedback is None:
                        raise RuntimeError(
                            f"{side} Piper 未返回关节使能状态"
                        )
                    enabled = [
                        bool(item)
                        for item in enabled_feedback
                    ]
                    diagnostics["joints_enabled"] = enabled
                    if enabled and not all(enabled):
                        raise RuntimeError(
                            f"{side} Piper 关节未全部使能: {enabled}"
                        )
                diagnostics["last_status_poll_ns"] = now_ns
                arm_status = diagnostics.get("arm_status")
                if isinstance(arm_status, int) and arm_status != 0:
                    raise RuntimeError(
                        f"{side} Piper 控制器拒绝 move_p: "
                        f"arm_status={arm_status}, "
                        f"mode_feedback={diagnostics.get('mode_feedback')}, "
                        f"err_status={diagnostics.get('err_status')}, "
                        f"target={target.tolist()}"
                    )

    def _send_hand(self, side: str, value: np.ndarray) -> None:
        if value.shape != (7,):
            raise ValueError(f"{side} Aerohand 控制指令必须为7维，实际为 {value.shape}")
        # TODO: 打印并未执行
        # print(f"动作执行，hardware_adapter, _send_hand() side: {side}, value: {value}" )
        self.hands[side].set_joint_positions(value.tolist())

        # state = self.hands[side].get_joint_positions_compact() # 获取7位
        # state = np.asarray(state, np.float32)
        # self.last_hand[side] = state if state.shape == (7,) else value.copy()

    def command(self, value: BimanualControlCommand) -> None:
        if self.state != "active":
            raise RuntimeError(f"当前状态 {self.state} 不允许下发控制命令")
        worker_errors = {
            name: worker.last_error
            for name, worker in self.workers.items()
            if worker.last_error is not None
        }
        if worker_errors:
            self.state = "fault"
            raise RuntimeError(f"硬件命令线程发生异常: {worker_errors}")
        self.last_command = value
        for side in ("left", "right"):
            side_value: ControlCommand = getattr(value, side)
            self.workers[f"arm_{side}"].submit(side_value.arm_pose)
            self.workers[f"hand_{side}"].submit(side_value.hand_joints)

    def read_state(self) -> BimanualRobotState:
        if not self.initialized:
            raise RuntimeError("机器人硬件尚未 initialize")
        states: dict[str, RobotState] = {}
        for side in ("left", "right"):
            with self.arm_locks[side]:
                flange, joints = self._read_arm_feedback(side)
                self.last_arm_pose[side] = flange.copy()

            states[side] = RobotState(flange, joints, self.last_hand[side].copy())
        return BimanualRobotState(states["left"], states["right"])

    def deactivate(self) -> None:
        """episode stop：停止下发并 disable 双臂，保留 CAN/串口硬件会话。"""
        with self._lifecycle_lock:
            if self.state in {"new", "closed", "idle_disabled"} and not self.workers:
                return
            previous_state = self.state
            self.state = "deactivating"
            stuck_workers: list[str] = []
            for name, worker in list(self.workers.items()):
                if not worker.close(
                    float(self.cfg.get("worker_stop_timeout_s", 3.0))
                ):
                    stuck_workers.append(name)
            self.workers.clear()
            self._disable_all_arms()
            self.start_prepared = False
            self.last_command = None
            if stuck_workers:
                self.state = "fault"
                raise RuntimeError(
                    f"硬件命令线程未及时停止: {stuck_workers}；双臂已 disable，"
                    "当前会话禁止再次 activate"
                )
            self.state = (
                "fault" if previous_state == "fault" else "idle_disabled"
            )

    def close(self) -> None:
        """进程退出时最终释放全部硬件句柄。"""
        with self._lifecycle_lock:
            try:
                self.deactivate()
            except Exception:
                LOG.exception("机器人 deactivate 失败，继续执行最终硬件释放")
            self._disable_all_arms()
        for side in ("left", "right"):
            arm = self.arms.get(side)
            if arm is not None:
                try:
                    arm.disconnect()
                except Exception:
                    LOG.exception("%s Piper 断开失败", side)
        self.arms.clear()
        for side, hand in list(self.hands.items()):
            try:
                if hasattr(hand, "close"):
                    hand.close()
                elif hasattr(hand, "disconnect"):
                    hand.disconnect()
            except Exception:
                LOG.exception("%s Aerohand 断开失败", side)
        self.hands.clear()
        self.homed = False
        self.state = "closed"

    # 兼容旧单进程入口；正式真机多进程路径使用显式生命周期方法。
    def connect(self) -> None:
        self.initialize()
        self.activate()

    def stop(self) -> None:
        self.deactivate()

    def disconnect(self) -> None:
        self.close()


class OpenCVWristCamera:
    """基于 OpenCV/V4L2 的腕部相机适配器。

    该适配器自身不创建线程。正式 pipeline 已经为左右腕相机分别创建独立的
    spawn 子进程，因此 ``VideoCapture.read``、MJPG 解码和颜色转换都不会阻塞
    recorder 或控制进程。
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        name: str,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        try:
            from opencv_camera import OpenCVCamera, OpenCVCameraConfig
        except ImportError as exc:
            raise RuntimeError("腕部相机需要安装 opencv-python") from exc

        self.name = name
        self.opencv_threads = int(cfg.get("opencv_threads", 1))
        self.camera = OpenCVCamera(
            OpenCVCameraConfig(
                device=cfg["device"],
                width=int(width),
                height=int(height),
                fps=float(cfg.get("fps", fps)),
                fourcc=str(cfg.get("fourcc", "MJPG")).upper(),
                backend=str(cfg.get("backend", "v4l2")),
                buffer_size=int(cfg.get("buffer_size", 1)),
                open_timeout_ms=int(cfg.get("open_timeout_ms", 5000)),
                read_timeout_ms=int(cfg.get("read_timeout_ms", 2000)),
                strict_fourcc=bool(cfg.get("strict_fourcc", True)),
                strict_resolution=bool(
                    cfg.get("strict_resolution", True)
                ),
                fps_tolerance=float(cfg.get("fps_tolerance", 2.0)),
                name=name,
            )
        )

    def connect(self) -> None:
        # 每个相机位于独立进程；限制 OpenCV 内部线程，避免与视频编码器争抢 CPU。
        import cv2

        cv2.setNumThreads(self.opencv_threads)
        self.camera.connect()
        LOG.info("%s OpenCV相机已连接: %s", self.name, self.camera.describe())

    def read(self) -> tuple[np.ndarray, int]:
        # OpenCV 输出 BGR；在相机子进程内转换为 LeRobot 需要的 RGB。
        return self.camera.read_rgb()

    def disconnect(self) -> None:
        self.camera.disconnect()

    def describe(self) -> dict[str, Any]:
        return self.camera.describe()


class IntelRealSenseColorCamera:
    """Intel RealSense 全景相机适配器，目前只启用彩色流。

    深度流没有被配置，因此不会产生不需要的深度帧和额外 USB 带宽。相机 SDK
    自带时间戳不一定与设备 B 的 ``perf_counter`` 同源，所以在彩色帧到达进程后
    立即记录设备 B 单调时钟，供统一时间缓冲和 LeRobot frame 对齐使用。
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        name: str,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        self.cfg = cfg
        self.name = name
        self.width = int(width)
        self.height = int(height)
        self.fps = int(cfg.get("fps", fps))
        self.timeout_ms = int(cfg.get("timeout_ms", 1000))
        self.warmup_frames = int(cfg.get("warmup_frames", 10))
        self.pipeline: Any | None = None
        self._rs: Any | None = None

    def connect(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "Intel RealSense scene相机需要安装 pyrealsense2"
            ) from exc

        pipeline = rs.pipeline()
        stream_config = rs.config()
        serial = str(self.cfg.get("serial", "")).strip()
        if serial:
            stream_config.enable_device(serial)

        # 明确只启用 RGB8 彩色流，不调用任何 depth stream。
        stream_config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.rgb8,
            self.fps,
        )
        pipeline.start(stream_config)
        self._rs = rs
        self.pipeline = pipeline

        # 丢弃自动曝光尚未稳定的启动帧。
        for _ in range(max(0, self.warmup_frames)):
            pipeline.wait_for_frames(self.timeout_ms)

    def read(self) -> tuple[np.ndarray, int]:
        if self.pipeline is None:
            raise RuntimeError(f"{self.name} RealSense尚未连接")
        frames = self.pipeline.wait_for_frames(self.timeout_ms)
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError(f"{self.name} 未收到彩色帧")

        image = np.asanyarray(color_frame.get_data())
        timestamp_ns = time.perf_counter_ns()
        expected_shape = (self.height, self.width, 3)
        if image.shape != expected_shape:
            raise ValueError(
                f"{self.name} 彩色帧尺寸错误: {image.shape}, 期望 {expected_shape}"
            )
        if image.dtype != np.uint8:
            image = image.astype(np.uint8, copy=False)
        # rs.format.rgb8 已是 LeRobot 需要的 RGB 顺序，不执行 BGR 转换。
        return image.copy(), timestamp_ns

    def disconnect(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            finally:
                self.pipeline = None
