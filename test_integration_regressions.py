from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
import sys
import symtable
import types
import unittest

import numpy as np


# 允许在只安装最小硬件测试依赖的环境中导入 process_pipeline；这些测试不创建
# ZMQ socket。正式运行仍由 requirements.txt 强制提供 pyzmq。
try:
    import zmq  # noqa: F401
except ModuleNotFoundError:
    sys.modules["zmq"] = types.ModuleType("zmq")

from adapters import SimCamera
from clock import ClockMapper
from hardware_processes import (
    HardwareRequestCancelled,
    MultiprocessRobotProxy,
    create_hardware_channels,
    gate_hardware_channels,
)
from process_pipeline import (
    _camera_frame_advances,
    _camera_startup_order,
    _make_camera,
    _teleop_age_ns,
    _teleop_sample_readiness,
)
from model import TimedSample
from safety import SafetyConfig, SafetyGate, effective_control_step_m


class IntegrationRegressionTests(unittest.TestCase):
    def test_rejected_clock_sample_does_not_claim_sync(self) -> None:
        mapper = ClockMapper()
        accepted = mapper.observe_round_trip(
            a_send=100,
            b_recv=200,
            b_send=300,
            a_recv=150,
        )
        self.assertFalse(accepted)
        self.assertEqual(mapper.sample_count, 0)
        self.assertEqual(mapper.offset_ns, 0)

    def test_control_age_uses_local_receive_time(self) -> None:
        now_ns = 2_000_000_000
        sample = TimedSample(
            "teleop",
            1,
            9_000_000_000,
            9_000_000_000,
            {"left": {"valid": True}, "right": {"valid": True}},
            meta={"receive_mono_ns": now_ns - 10_000_000},
        )
        self.assertEqual(_teleop_age_ns(sample, now_ns), 10_000_000)
        ready, reason, age_ms = _teleop_sample_readiness(
            sample, now_ns, 150_000_000
        )
        self.assertTrue(ready)
        self.assertEqual(reason, "ready")
        self.assertAlmostEqual(age_ms, 10.0)

    def test_safety_step_is_scaled_to_real_arm_command_rate(self) -> None:
        self.assertAlmostEqual(
            effective_control_step_m(0.01, control_hz=100, arm_command_hz=5),
            0.0005,
        )

    def test_safety_rejects_initial_pose_outside_workspace(self) -> None:
        cfg = SafetyConfig(
            np.array([0.05, -0.35, 0.05], np.float32),
            np.array([0.40, 0.00, 0.45], np.float32),
            0.0005,
            np.zeros(7, np.float32),
            np.full(7, 100, np.float32),
            150_000_000,
        )
        with self.assertRaisesRegex(ValueError, "初始法兰位置不在工作空间"):
            SafetyGate(
                cfg,
                np.array([0.25, 0.10, 0.25, 0, 0, 0], np.float32),
            )

    def test_start_rejects_fresh_but_invalid_bimanual_teleop(self) -> None:
        now_ns = 2_000_000_000
        sample = TimedSample(
            "teleop",
            1,
            now_ns - 10_000_000,
            now_ns - 10_000_000,
            {
                "left": {"valid": True},
                "right": {"valid": False},
            },
        )
        ready, reason, age_ms = _teleop_sample_readiness(
            sample, now_ns, 150_000_000
        )
        self.assertFalse(ready)
        self.assertIn("right:valid=false", reason)
        self.assertAlmostEqual(age_ms, 10.0)

    def test_start_accepts_fresh_valid_bimanual_teleop(self) -> None:
        now_ns = 2_000_000_000
        sample = TimedSample(
            "teleop",
            1,
            now_ns - 10_000_000,
            now_ns - 10_000_000,
            {"left": {"valid": True}, "right": {"valid": True}},
        )
        ready, reason, _ = _teleop_sample_readiness(
            sample, now_ns, 150_000_000
        )
        self.assertTrue(ready)
        self.assertEqual(reason, "ready")

    def test_start_rejects_stale_bimanual_teleop(self) -> None:
        now_ns = 2_000_000_000
        sample = TimedSample(
            "teleop",
            1,
            now_ns - 200_000_000,
            now_ns - 200_000_000,
            {"left": {"valid": True}, "right": {"valid": True}},
        )
        ready, reason, age_ms = _teleop_sample_readiness(
            sample, now_ns, 150_000_000
        )
        self.assertFalse(ready)
        self.assertIn("teleop_stale", reason)
        self.assertAlmostEqual(age_ms, 200.0)

    def test_arm_command_output_switch_is_local(self) -> None:
        source_path = Path(__file__).with_name("hardware_processes.py")
        table = symtable.symtable(
            source_path.read_text(encoding="utf-8"), str(source_path), "exec"
        )
        arm_table = next(
            child
            for child in table.get_children()
            if child.get_name() == "arm_hardware_process"
        )
        symbol = arm_table.lookup("command_output_enabled")
        self.assertTrue(symbol.is_local())
        self.assertFalse(symbol.is_global())

    def test_camera_startup_order_uses_configured_order_keys(self) -> None:
        cfg = {
            "cameras": {
                "scene": {"startup_delay_ms": 600},
                "wrist_left": {"startup_delay_ms": 0},
                "wrist_right": {"startup_delay_ms": 300},
            }
        }
        self.assertEqual(
            _camera_startup_order(cfg),
            ("wrist_left", "wrist_right", "scene"),
        )

    def test_camera_startup_gate_detects_a_frozen_ready_camera(self) -> None:
        baseline = {"scene": 3, "wrist_left": 17, "wrist_right": 25}
        latest = {"scene": 33, "wrist_left": 17, "wrist_right": 55}
        advances = _camera_frame_advances(baseline, latest)
        self.assertEqual(advances["wrist_left"], 0)
        self.assertGreaterEqual(advances["scene"], 5)
        self.assertGreaterEqual(advances["wrist_right"], 5)

    def test_real_camera_switch_is_independent_from_robot_switch(self) -> None:
        cfg = {
            "robot": {"enabled": True},
            "rates": {"camera_hz": 30},
            "cameras": {
                "hardware_enabled": False,
                "width": 16,
                "height": 12,
                "scene": {"driver": "realsense"},
                "wrist_left": {"driver": "opencv"},
                "wrist_right": {"driver": "opencv"},
            },
        }
        for name in ("scene", "wrist_left", "wrist_right"):
            self.assertIsInstance(_make_camera(cfg, name), SimCamera)

    def test_parent_gate_invalidates_pending_lifecycle_request(self) -> None:
        ctx = mp.get_context("spawn")
        cfg = {"hardware_command_queue_capacity": 4}
        channels = create_hardware_channels(ctx, cfg)
        try:
            before = {
                name: int(channel["cancel_generation"].value)
                for name, channel in channels.items()
            }
            gate_hardware_channels(channels)
            for name, channel in channels.items():
                self.assertTrue(channel["hold_event"].is_set())
                self.assertEqual(
                    int(channel["cancel_generation"].value), before[name] + 1
                )

            proxy = MultiprocessRobotProxy(cfg, channels)
            channels["arm_left"]["response_queue"].put(
                {
                    "request_id": 1,
                    "ok": False,
                    "cancelled": True,
                    "error": "test cancellation",
                }
            )
            with self.assertRaises(HardwareRequestCancelled):
                proxy._request("arm_left", "home", 1.0)
        finally:
            for channel in channels.values():
                for queue_name in (
                    "control_queue",
                    "target_queue",
                    "response_queue",
                ):
                    ipc_queue = channel[queue_name]
                    ipc_queue.close()
                    ipc_queue.cancel_join_thread()


if __name__ == "__main__":
    unittest.main()
