from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import zmq

from clock import ClockMapper
from model import TimedSample
from protocol import Packet


class ZmqSender:
    """设备 A 的最新值 PUSH 发送端；send() 永远不等待网络。"""

    def __init__(self, host: str, data_port: int) -> None:
        context = zmq.Context.instance()
        self.data_endpoint = f"tcp://{host}:{data_port}"
        self.data_socket = context.socket(zmq.PUSH)
        self.data_socket.setsockopt(zmq.SNDHWM, 1)
        self.data_socket.setsockopt(zmq.LINGER, 0)
        self.data_socket.setsockopt(zmq.IMMEDIATE, 1)
        self.data_socket.connect(self.data_endpoint)
        self.sent = 0
        self.dropped = 0

    def send(self, packet: Packet) -> bool:
        try:
            self.data_socket.send(packet.encode(), flags=zmq.DONTWAIT)
            self.sent += 1
            return True
        except zmq.Again:
            # 设备 B 未连接或网络拥塞时不能阻塞 VIVE/MANUS 采集线程。
            self.dropped += 1
            return False

    def close(self) -> None:
        self.data_socket.close(linger=0)


class ZmqClockSynchronizer:
    """独占 REQ socket 的设备 A 时钟同步循环，不阻塞遥操作发送。"""

    def __init__(self, host: str, sync_port: int) -> None:
        self.sync_endpoint = f"tcp://{host}:{sync_port}"
        self.success = 0
        self.failures = 0
        self.error: BaseException | None = None
        self._pending_result: dict[str, int] | None = None

    def run(
        self,
        stop_event: threading.Event,
        interval_s: float = 1.0,
        timeout_ms: int = 50,
    ) -> None:
        socket: Any = None
        try:
            context = zmq.Context.instance()
            socket = context.socket(zmq.REQ)
            socket.setsockopt(zmq.SNDHWM, 1)
            socket.setsockopt(zmq.RCVHWM, 1)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.IMMEDIATE, 1)
            # 超时后允许下一轮发送，并丢弃迟到的旧回复。
            socket.setsockopt(zmq.REQ_RELAXED, 1)
            socket.setsockopt(zmq.REQ_CORRELATE, 1)
            socket.connect(self.sync_endpoint)
            seq = 0
            while not stop_event.is_set():
                started = time.monotonic()
                self._exchange(socket, seq, timeout_ms)
                seq += 1
                remaining = interval_s - (time.monotonic() - started)
                if remaining > 0:
                    stop_event.wait(remaining)
        except BaseException as exc:
            self.error = exc
            stop_event.set()
        finally:
            if socket is not None:
                socket.close(linger=0)

    def _exchange(self, socket: Any, seq: int, timeout_ms: int) -> bool:
        a_send = time.perf_counter_ns()
        request = Packet(
            "sync_request",
            seq,
            a_send,
            {"previous": self._pending_result},
        )
        try:
            socket.send(request.encode(), flags=zmq.DONTWAIT)
            if not socket.poll(timeout=timeout_ms, flags=zmq.POLLIN):
                self.failures += 1
                return False
            reply = Packet.decode(socket.recv())
            a_recv = time.perf_counter_ns()
            if (
                reply.kind != "sync_reply"
                or int(reply.payload.get("a_send", -1)) != a_send
            ):
                self.failures += 1
                return False
            self._pending_result = {
                "a_send": a_send,
                "b_recv": int(reply.payload["b_recv"]),
                "b_send": int(reply.payload["b_send"]),
                "a_recv": a_recv,
            }
            self.success += 1
            return True
        except (zmq.ZMQError, ValueError, KeyError, TypeError):
            self.failures += 1
            return False


class ZmqReceiver:
    """设备 B 的 ZMQ 接收端；所有 socket 只在接收线程内创建和释放。"""

    def __init__(
        self,
        data_port: int,
        sync_port: int,
        mapper: ClockMapper,
        on_sample: Callable[[TimedSample], None],
        max_bytes: int = 65507,
        bind_host: str = "*",
    ) -> None:
        self.data_endpoint = f"tcp://{bind_host}:{data_port}"
        self.sync_endpoint = f"tcp://{bind_host}:{sync_port}"
        self.mapper = mapper
        self.on_sample = on_sample
        self.max_bytes = int(max_bytes)
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.error: BaseException | None = None
        self.bad_packets = 0
        self.sequence_gaps = 0
        self.sync_updates = 0
        self.sync_rejected = 0
        self._last_seq: int | None = None

    def wait_ready(self, timeout_s: float = 5.0) -> None:
        if not self.ready_event.wait(timeout_s):
            raise TimeoutError(f"等待 ZMQ 端口绑定超时 {timeout_s:.1f}s")
        if self.error is not None:
            raise RuntimeError("ZMQ 接收端启动失败") from self.error

    def run(self) -> None:
        data_socket: Any = None
        sync_socket: Any = None
        try:
            context = zmq.Context.instance()
            data_socket = context.socket(zmq.PULL)
            data_socket.setsockopt(zmq.RCVHWM, 1)
            data_socket.setsockopt(zmq.CONFLATE, 1)
            data_socket.setsockopt(zmq.LINGER, 0)
            data_socket.bind(self.data_endpoint)

            sync_socket = context.socket(zmq.REP)
            sync_socket.setsockopt(zmq.RCVHWM, 4)
            sync_socket.setsockopt(zmq.SNDHWM, 4)
            sync_socket.setsockopt(zmq.LINGER, 0)
            sync_socket.bind(self.sync_endpoint)

            poller = zmq.Poller()
            poller.register(data_socket, zmq.POLLIN)
            poller.register(sync_socket, zmq.POLLIN)
            self.ready_event.set()
            while not self.stop_event.is_set():
                events = dict(poller.poll(timeout=100))
                if sync_socket in events:
                    self._handle_sync(sync_socket)
                if data_socket in events:
                    self._handle_data(data_socket)
        except BaseException as exc:
            self.error = exc
            self.ready_event.set()
        finally:
            if data_socket is not None:
                data_socket.close(linger=0)
            if sync_socket is not None:
                sync_socket.close(linger=0)

    def _handle_sync(self, sync_socket: Any) -> None:
        b_recv = time.perf_counter_ns()
        try:
            raw = sync_socket.recv()
            if len(raw) > self.max_bytes:
                raise ValueError("sync packet too large")
            packet = Packet.decode(raw)
            if packet.kind != "sync_request":
                raise ValueError("unexpected sync packet kind")
            previous = packet.payload.get("previous")
            if previous is not None:
                accepted = self.mapper.observe_round_trip(
                    int(previous["a_send"]),
                    int(previous["b_recv"]),
                    int(previous["b_send"]),
                    int(previous["a_recv"]),
                )
                if accepted:
                    self.sync_updates += 1
                else:
                    self.sync_rejected += 1
            b_send = time.perf_counter_ns()
            reply = Packet(
                "sync_reply",
                packet.seq,
                b_send,
                {
                    "a_send": packet.source_mono_ns,
                    "b_recv": b_recv,
                    "b_send": b_send,
                },
            )
        except (ValueError, KeyError, TypeError, UnicodeError):
            self.bad_packets += 1
            b_send = time.perf_counter_ns()
            reply = Packet("sync_error", 0, b_send, {})
        # REP 必须对每个请求回复，否则 socket 会卡在状态机中。
        sync_socket.send(reply.encode())

    def _handle_data(self, data_socket: Any) -> None:
        receive_ns = time.perf_counter_ns()
        try:
            raw = data_socket.recv()
            if len(raw) > self.max_bytes:
                raise ValueError("data packet too large")
            packet = Packet.decode(raw)
            if packet.kind != "teleop":
                raise ValueError(f"unexpected data packet kind: {packet.kind}")
            if self._last_seq is not None and packet.seq > self._last_seq + 1:
                self.sequence_gaps += packet.seq - self._last_seq - 1
            if self._last_seq is not None and packet.seq <= self._last_seq:
                return
            self._last_seq = packet.seq
            mapped = self.mapper.to_local(packet.source_mono_ns)
            self.on_sample(
                TimedSample(
                    packet.kind,
                    packet.seq,
                    packet.source_mono_ns,
                    mapped,
                    packet.payload,
                    True,
                    {
                        "receive_mono_ns": receive_ns,
                        "clock_offset_ns": self.mapper.offset_ns,
                    },
                )
            )
        except (ValueError, KeyError, TypeError, UnicodeError):
            self.bad_packets += 1

    def close(self) -> None:
        # ZMQ socket 不是线程安全的；这里只置事件，由 run() 所在线程负责关闭。
        self.stop_event.set()
