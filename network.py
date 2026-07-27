from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable

from clock import ClockMapper
from model import TimedSample
from protocol import Packet


class UdpSender:
    def __init__(self, host: str, port: int) -> None:
        self.target = (host, port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.settimeout(0.05)

    def send(self, packet: Packet) -> None:
        self.socket.sendto(packet.encode(), self.target)

    def synchronize(self, packet: Packet) -> bool:
        self.send(packet)
        try:
            data, _ = self.socket.recvfrom(4096)
            a_recv = time.perf_counter_ns()
            reply = Packet.decode(data)
            if reply.kind != "sync_reply" or reply.payload.get("a_send") != packet.source_mono_ns:
                return False
            self.send(Packet("sync_result", packet.seq, a_recv, {
                "a_send": packet.source_mono_ns,
                "b_recv": int(reply.payload["b_recv"]),
                "b_send": int(reply.payload["b_send"]),
                "a_recv": a_recv,
            }))
            return True
        except (socket.timeout, ValueError, KeyError):
            return False

    def close(self) -> None:
        self.socket.close()


class UdpReceiver:
    def __init__(
        self,
        port: int,
        mapper: ClockMapper,
        on_sample: Callable[[TimedSample], None],
        max_bytes: int = 65507,
    ) -> None:
        self.mapper, self.on_sample, self.max_bytes = mapper, on_sample, max_bytes
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self.socket.bind(("0.0.0.0", port))
        self.socket.settimeout(0.2)
        self.stop_event = threading.Event()
        self.bad_packets = 0
        self.sequence_gaps = 0
        self._last_seq: int | None = None

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                data, _ = self.socket.recvfrom(self.max_bytes)
                receive_ns = time.perf_counter_ns()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                packet = Packet.decode(data)
                if packet.kind == "sync_request":
                    b_send = time.perf_counter_ns()
                    self.socket.sendto(Packet("sync_reply", packet.seq, b_send, {
                        "a_send": packet.source_mono_ns, "b_recv": receive_ns, "b_send": b_send,
                    }).encode(), _)
                    continue
                if packet.kind == "sync_result":
                    p = packet.payload
                    self.mapper.observe_round_trip(
                        int(p["a_send"]), int(p["b_recv"]), int(p["b_send"]), int(p["a_recv"])
                    )
                    continue
                if self._last_seq is not None and packet.seq > self._last_seq + 1:
                    self.sequence_gaps += packet.seq - self._last_seq - 1
                if self._last_seq is not None and packet.seq <= self._last_seq:
                    continue
                self._last_seq = packet.seq
                mapped = self.mapper.to_local(packet.source_mono_ns)
                self.on_sample(TimedSample(
                    packet.kind, packet.seq, packet.source_mono_ns, mapped,
                    packet.payload, True, {"receive_mono_ns": receive_ns},
                ))
            except (ValueError, UnicodeError):
                self.bad_packets += 1

    def close(self) -> None:
        self.stop_event.set()
        self.socket.close()
