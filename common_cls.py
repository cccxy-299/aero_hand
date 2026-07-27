import json
from dataclasses import dataclass

import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple

import zmq

@dataclass
class RetargetingConfig:
    # MANUS ergonomics 是否已经是 degrees。
    # MANUS ergonomics 文档定义为 degrees；如果你自己 C++ 端转换成了 rad，这里设 False。
    ergonomics_in_degrees: bool = True

    # 输出给 Aero SDK 时建议用 degrees。
    output_degrees: bool = True

    # 若 True，输出 7 维 compact representation。
    # 若 False，输出 16 维完整 joint positions。
    compact_7dof: bool = False

    # 是否启用简单低通滤波。
    enable_filter: bool = True
    filter_alpha: float = 0.25


class LowPass:
    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
        self.y: Optional[np.ndarray] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)

        if self.y is None or self.y.shape != x.shape:
            self.y = x.copy()
        else:
            self.y = self.alpha * x + (1.0 - self.alpha) * self.y

        return self.y.copy()

class ManusZmqReader:
    def __init__(self, address: str = "tcp://127.0.0.1:9000", topic: str = "manus"):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.RCVHWM, 1)
        self.sock.connect(address)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, topic)

        self.poller = zmq.Poller()
        self.poller.register(self.sock, zmq.POLLIN)

    def read_latest(self, timeout_ms: int = 1000) -> Optional[Dict[str, Any]]:
        events = dict(self.poller.poll(timeout_ms))
        if self.sock not in events:
            return None

        latest_payload = None

        while True:
            try:
                parts = self.sock.recv_multipart(flags=zmq.DONTWAIT)
            except zmq.Again:
                break

            if len(parts) == 1:
                latest_payload = parts[0]
            elif len(parts) >= 2:
                # C++ 端当前是 [topic, payload]
                latest_payload = parts[1]

        if latest_payload is None:
            return None

        return json.loads(latest_payload.decode("utf-8"))

    def close(self):
        self.poller.unregister(self.sock)
        self.sock.close(linger=0)