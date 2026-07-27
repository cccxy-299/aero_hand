"""设备 A 的 VIVE Tracker / Lighthouse 三维可视化。

该模块由原 ``vive tracker/vive_teleop/visualization.py`` 中的
``ViveViewer`` 适配而来。Qt、pyqtgraph 和 OpenGL 仅在用户启用可视化时导入，
因此无桌面环境的设备 A 仍可使用 ``--no-visualize`` 正常运行。
"""

from __future__ import annotations

import signal
import time
from typing import Any

import numpy as np

QtCore: Any = None
QtWidgets: Any = None
gl: Any = None


def _load_gui_dependencies() -> None:
    """延迟加载 GUI 依赖，避免无可视化模式被 Qt/OpenGL 安装状态影响。"""
    global QtCore, QtWidgets, gl
    if gl is not None:
        return
    try:
        from pyqtgraph.Qt import QtCore as qt_core, QtWidgets as qt_widgets
        import pyqtgraph.opengl as pyqtgraph_gl
    except ImportError as exc:
        raise RuntimeError(
            "启用 VIVE 可视化需要安装 pyqtgraph、PyOpenGL 以及 PySide6/PyQt5"
        ) from exc
    QtCore, QtWidgets, gl = qt_core, qt_widgets, pyqtgraph_gl


def quat_wxyz_to_rotmat(quaternion: Any) -> np.ndarray:
    """把 VIVE 的 [w, x, y, z] 四元数转换成3×3旋转矩阵。"""
    q = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.eye(3)
    w, x, y, z = q / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def add_line(
    view: Any,
    p0: Any,
    p1: Any,
    color: tuple[float, float, float, float],
    width: int = 2,
) -> Any:
    item = gl.GLLinePlotItem(
        pos=np.vstack([np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)]),
        color=color,
        width=width,
        antialias=True,
    )
    view.addItem(item)
    return item


def try_add_text(view: Any, pos: Any, text: str) -> Any | None:
    if not hasattr(gl, "GLTextItem"):
        return None
    try:
        item = gl.GLTextItem(pos=np.asarray(pos, dtype=float), text=text)
        view.addItem(item)
        return item
    except Exception:
        return None


def add_world_axes(view: Any, axis_length: float = 1.0) -> None:
    origin = np.zeros(3)
    add_line(view, origin, [axis_length, 0, 0], (1, 0, 0, 1), 4)
    add_line(view, origin, [0, axis_length, 0], (0, 1, 0, 1), 4)
    add_line(view, origin, [0, 0, axis_length], (0, 0.4, 1, 1), 4)
    try_add_text(view, [axis_length, 0, 0], "X")
    try_add_text(view, [0, axis_length, 0], "Y")
    try_add_text(view, [0, 0, axis_length], "Z")


class FrameWidget:
    """显示一个设备点以及它的局部 XYZ 坐标轴。"""

    def __init__(
        self,
        view: Any,
        name: str,
        point_color: tuple[float, float, float, float],
        axis_len: float,
        point_size: int,
    ) -> None:
        self.name = name
        self.axis_len = axis_len
        self.point = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)), size=point_size, color=point_color, pxMode=True
        )
        self.x_axis = gl.GLLinePlotItem(width=3, color=(1, 0, 0, 1), antialias=True)
        self.y_axis = gl.GLLinePlotItem(width=3, color=(0, 1, 0, 1), antialias=True)
        self.z_axis = gl.GLLinePlotItem(width=3, color=(0, 0.4, 1, 1), antialias=True)
        for item in (self.point, self.x_axis, self.y_axis, self.z_axis):
            view.addItem(item)
        self.text_item = try_add_text(view, [0, 0, 0], name)

    def update(self, pos: Any, quat_wxyz: Any) -> None:
        pos = np.asarray(pos, dtype=float).reshape(3)
        rotation = quat_wxyz_to_rotmat(quat_wxyz)
        self.point.setData(pos=pos.reshape(1, 3))
        axes = np.eye(3) * self.axis_len
        self.x_axis.setData(pos=np.vstack([pos, pos + rotation @ axes[0]]))
        self.y_axis.setData(pos=np.vstack([pos, pos + rotation @ axes[1]]))
        self.z_axis.setData(pos=np.vstack([pos, pos + rotation @ axes[2]]))
        if self.text_item is not None:
            try:
                self.text_item.setData(pos=pos + 0.02, text=self.name)
            except Exception:
                pass


class ViveViewer:
    """显示双 VIVE Tracker、Lighthouse、世界坐标轴和数据新鲜度。"""

    def __init__(self, cfg: dict[str, Any], reader: Any, stop_event: Any) -> None:
        self._cfg = cfg
        self._reader = reader
        self._stop_event = stop_event
        self._app: Any = None
        self._view: Any = None
        self._status: Any = None
        self._tracker_widgets: dict[str, FrameWidget] = {}
        self._lighthouse_widgets: dict[str, FrameWidget] = {}

    def run(self) -> int:
        _load_gui_dependencies()
        visual_cfg = self._cfg.get("visualization", {})
        world_cfg = visual_cfg.get("world", {})
        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = QtWidgets.QWidget()
        window.setWindowTitle(visual_cfg.get("window_title", "双侧 VIVE 遥操作可视化"))
        layout = QtWidgets.QVBoxLayout(window)
        self._view = gl.GLViewWidget()
        self._view.setCameraPosition(
            distance=float(visual_cfg.get("camera_distance", 3.0)),
            elevation=float(visual_cfg.get("camera_elevation", 25)),
            azimuth=float(visual_cfg.get("camera_azimuth", 45)),
        )
        layout.addWidget(self._view)
        self._status = QtWidgets.QLabel("等待 VIVE 数据...")
        layout.addWidget(self._status)
        window.resize(
            int(visual_cfg.get("window_width", 1100)),
            int(visual_cfg.get("window_height", 850)),
        )
        window.show()

        grid = gl.GLGridItem()
        grid_size = float(world_cfg.get("grid_size", 4.0))
        spacing = float(world_cfg.get("grid_spacing", 0.25))
        grid.setSize(grid_size, grid_size, 1)
        grid.setSpacing(spacing, spacing, spacing)
        self._view.addItem(grid)
        add_world_axes(self._view, float(world_cfg.get("axis_length", 1.0)))

        timer = QtCore.QTimer()
        timer.timeout.connect(self._gui_tick)
        timer.start(max(1, int(1000 / float(visual_cfg.get("update_hz", 60)))))

        def shutdown(*_: Any) -> None:
            self._stop_event.set()
            self._app.quit()

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)
        result = self._app.exec() if hasattr(self._app, "exec") else self._app.exec_()
        self._stop_event.set()
        return int(result)

    def _gui_tick(self) -> None:
        if self._stop_event.is_set():
            self._app.quit()
            return
        tracker_snapshot = self._reader.snapshot()
        lighthouse_snapshot = self._reader.lighthouse_snapshot()
        status: list[str] = []
        self._update_trackers(tracker_snapshot, status)
        self._update_lighthouses(lighthouse_snapshot, status)
        self._status.setText(" | ".join(status) if status else "等待 VIVE 数据...")

    def _update_trackers(self, snapshot: dict[str, Any], status: list[str]) -> None:
        names = self._cfg["operator"]["vive"]
        colors = {"left": (0.2, 0.8, 1.0, 1.0), "right": (1.0, 0.3, 0.3, 1.0)}
        now_ns = time.perf_counter_ns()
        for side in ("left", "right"):
            sample = snapshot.get(side)
            if sample is None:
                status.append(f"{side}:无数据")
                continue
            pose, timestamp_ns = sample
            if side not in self._tracker_widgets:
                tracker_name = names[f"{side}_tracker_name"]
                self._tracker_widgets[side] = FrameWidget(
                    self._view, f"{side}:{tracker_name}", colors[side], 0.18, 16
                )
            self._tracker_widgets[side].update(pose[:3], pose[3:7])
            age_ms = max(0.0, (now_ns - timestamp_ns) / 1e6)
            status.append(
                f"{side}=({pose[0]:+.3f},{pose[1]:+.3f},{pose[2]:+.3f}) {age_ms:.0f}ms"
            )

    def _update_lighthouses(self, snapshot: dict[str, Any], status: list[str]) -> None:
        configured = self._cfg.get("visualization", {}).get("lighthouses", {})
        for name in sorted(set(configured) | set(snapshot)):
            sample = snapshot.get(name)
            if sample is not None:
                pose, _ = sample
                pos, quat = pose[:3], pose[3:7]
                source = "live"
            else:
                value = configured[name]
                pos = value.get("pos", [0, 0, 0])
                quat = value.get("quat_wxyz", [1, 0, 0, 0])
                source = "config"
            if name not in self._lighthouse_widgets:
                self._lighthouse_widgets[name] = FrameWidget(
                    self._view, name, (1.0, 0.8, 0.1, 1.0), 0.25, 18
                )
            self._lighthouse_widgets[name].update(pos, quat)
            status.append(f"{name}:{source}")

