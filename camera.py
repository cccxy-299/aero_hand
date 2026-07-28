import threading
import logging
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
import pyvizionsdk
from pyvizionsdk import VX_IMAGE_FORMAT, VX_ISP_IMAGE_PROPERTIES


# ============================================================
# 1. 已知摄像头编号与序列号映射
# ============================================================

camera_num_to_serial_number: Dict[int, str] = {
    0: "XY-GS-Camera: XY-GS-Camera (/dev/video0)",
    1: "XY-GS-Camera: XY-GS-Camera (/dev/video2)",
    11: "VCI-AR0234-C-25D59",
    12: "VCI-AR0234-C-1C982",
    13: "VCS-AR0234-C-B1C8A",
}

camera_serial_to_num: Dict[str, int] = {
    v: k for k, v in camera_num_to_serial_number.items()
}

# 运行时填充：cam_num -> 设备列表中的物理 index
camera_num_to_index: Dict[int, int] = {}


# ============================================================
# 2. UYVY 解码函数
# ============================================================
def decode_mjpg(image) -> Optional[np.ndarray]:
    if image is None:
        return None

    if isinstance(image, (bytes, bytearray, memoryview)):
        buf = np.frombuffer(image, dtype=np.uint8)
    else:
        buf = np.asarray(image, dtype=np.uint8).reshape(-1)

    # MJPG 是自描述的 JPEG 压缩数据，cv2.imdecode 直接解码为 BGR
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return frame  # 解码失败时返回 None

def decode_uyvy(image, width: int, height: int) -> Optional[np.ndarray]:
    """
    将 TechNexion SDK 返回的 UYVY buffer 解码为 BGR 图像。

    返回：
        frame_bgr: np.ndarray, shape = (H, W, 3), dtype = uint8
    """
    if image is None:
        return None

    if isinstance(image, (bytes, bytearray, memoryview)):
        buf = np.frombuffer(image, dtype=np.uint8)
    else:
        buf = np.asarray(image, dtype=np.uint8).reshape(-1)

    expected = width * height * 2

    if buf.size < expected:
        return None

    # 如果 SDK buffer 比实际图像大，只取前 expected 字节
    uyvy = buf[:expected].reshape((height, width, 2))

    frame_bgr = cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
    return frame_bgr


def detect_connected_cameras() -> List[int]:
    """
    自动发现已连接的 TechNexion 摄像头。

    根据序列号匹配 camera_num，并写入全局 camera_num_to_index。
    """
    logger = logging.getLogger("detect_cameras")

    _, connected_serials = pyvizionsdk.VxDiscoverCameraDevices()
    logger.info(f"Connected serials: {connected_serials}")

    connected_cam_nums: List[int] = []

    for idx, serial in enumerate(connected_serials):
        matched = False

        for known_serial, cam_num in camera_serial_to_num.items():
            if known_serial in serial:
                camera_num_to_index[cam_num] = idx
                connected_cam_nums.append(cam_num)
                matched = True
                logger.info(
                    f"Matched camera: cam_num={cam_num}, "
                    f"serial={serial}, physical_index={idx}"
                )
                break

        if not matched:
            logger.warning(f"Unknown camera serial: {serial}, physical_index={idx}")

    return connected_cam_nums


# ============================================================
# 3. CameraInterface
# ============================================================

class CameraInterface:
    """
    TechNexion 摄像头接口。

    用法示例：
        scene_cam = CameraInterface(cam_num=10, fps=60, format_idx=7, name="scene")
        wrist_cam = CameraInterface(cam_num=11, fps=60, format_idx=7, name="wrist")

        scene_cam.connect()
        wrist_cam.connect()

        img_rgb = scene_cam.get_rgb()

        scene_cam.disconnect()
        wrist_cam.disconnect()

    输出：
        get_rgb() 返回 RGB uint8 图像，shape = (H, W, 3)
        可直接写入 LeRobot Dataset 的 observation.images.*
    """

    def __init__(
        self,
        cam_num: int,
        fps: int = 60,
        format_idx: int = 7,
        name: Optional[str] = None,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        timeout_ms: int = 2500,
        warmup_timeout_s: float = 5.0,
    ):
        self.cam_num = cam_num
        self.fps = fps
        self.format_idx = format_idx
        self.name = name or f"camera_{cam_num}"

        # 如果 LeRobot 数据集要求固定图像大小，可以在这里指定 resize 目标大小
        self.target_width = target_width
        self.target_height = target_height

        self.timeout_ms = timeout_ms
        self.warmup_timeout_s = warmup_timeout_s

        self.logger = logging.getLogger(self.name)

        self.camera = None
        self.format = None
        self.index = None

        self._frame_bgr: Optional[np.ndarray] = None
        self._frame_timestamp: Optional[float] = None
        self._lock = threading.Lock()

        self._running = False
        self._thread: Optional[threading.Thread] = None

    # --------------------------------------------------------
    # 初始化相关函数
    # --------------------------------------------------------

    def connect(self):
        """
        连接并启动摄像头后台采集线程。
        """

        # 如果还没有检测过设备，则自动检测一次
        if self.cam_num not in camera_num_to_index:
            self.logger.info("camera_num_to_index is empty or missing current camera, detecting cameras...")
            detect_connected_cameras()

        if self.cam_num not in camera_num_to_index:
            raise RuntimeError(
                f"未找到 cam_num={self.cam_num} 对应的摄像头。"
                f"请检查 camera_num_to_serial_number 映射表和实际连接设备。"
            )

        self.index = camera_num_to_index[self.cam_num]

        self.logger.info(
            f"Connecting TechNexion camera: "
            f"cam_num={self.cam_num}, physical_index={self.index}, "
            f"fps={self.fps}, format_idx={self.format_idx}"
        )

        self.camera, self.format = self._init_tn_camera(self.index)

        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name=self.name,
        )
        self._thread.start()

        self.logger.info(f"{self.name} capture thread started.")

        # 等待第一帧，避免刚启动时 get_rgb() 返回 None
        self._wait_first_frame()

    def disconnect(self):
        """
        停止采集线程并释放摄像头资源。
        """
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=3.0)

        if self.camera is not None:
            try:
                pyvizionsdk.VxStopStreaming(self.camera)
                pyvizionsdk.VxClose(self.camera)
            except Exception as e:
                self.logger.warning(f"释放摄像头资源时出现异常: {e}")

        self.logger.info(f"{self.name} stopped and camera released.")

    def _init_tn_camera(self, index: int):
        """
        初始化 TechNexion 摄像头，并选择 UYVY 格式。
        """
        camera = pyvizionsdk.VxInitialCameraDevice(index)
        pyvizionsdk.VxOpen(camera)

        _, format_list = pyvizionsdk.VxGetFormatList(camera)

        mjpg_formats = [
            f for f in format_list
            if f.format == VX_IMAGE_FORMAT.VX_IMAGE_FORMAT_MJPG
        ]

        self.logger.info(f"Available MJPG formats ({len(mjpg_formats)} total):")
        for i, fmt in enumerate(mjpg_formats):
            self.logger.info(f"  [{i}] {fmt.width}x{fmt.height} @{fmt.framerate}fps")

        if len(mjpg_formats) == 0:
            raise RuntimeError(f"No MJPG formats found for camera index {index}")

        fmt = mjpg_formats[min(self.format_idx, len(mjpg_formats) - 1)]

        self.logger.info(
            f"Selected MJPG format [{self.format_idx}]: "
            f"{fmt.width}x{fmt.height} @{fmt.framerate}fps"
        )

        pyvizionsdk.VxSetFormat(camera, fmt)
        # self._apply_isp_defaults(camera)
        pyvizionsdk.VxStartStreaming(camera)

        return camera, fmt

    def _apply_isp_defaults(self, camera: pyvizionsdk.VxCamera):
        """
        应用默认 ISP 图像处理参数。
        这些参数沿用你 demo 里的设置。
        """
        pyvizionsdk.VxSetISPImageProcessing(
            camera, VX_ISP_IMAGE_PROPERTIES.ISP_IMAGE_BRIGHTNESS, 0
        )
        pyvizionsdk.VxSetISPImageProcessing(
            camera, VX_ISP_IMAGE_PROPERTIES.ISP_IMAGE_CONTRAST, -50
        )
        pyvizionsdk.VxSetISPImageProcessing(
            camera, VX_ISP_IMAGE_PROPERTIES.ISP_IMAGE_SATURATION, 10
        )
        pyvizionsdk.VxSetISPImageProcessing(
            camera, VX_ISP_IMAGE_PROPERTIES.ISP_IMAGE_SHARPNESS, 0
        )
        pyvizionsdk.VxSetISPImageProcessing(
            camera, VX_ISP_IMAGE_PROPERTIES.ISP_IMAGE_DENOISE, 0
        )
        pyvizionsdk.VxSetISPImageProcessing(
            camera, VX_ISP_IMAGE_PROPERTIES.ISP_IMAGE_BACKLIGHT_COMPENSATION, 0
        )
        pyvizionsdk.VxSetISPImageProcessing(
            camera, VX_ISP_IMAGE_PROPERTIES.ISP_IMAGE_JPEG_QUALITY, 255
        )
        pyvizionsdk.VxSetISPImageProcessing(
            camera, VX_ISP_IMAGE_PROPERTIES.ISP_IMAGE_FLICK_MODE, 2
        )

    # --------------------------------------------------------
    # 后台采集线程
    # --------------------------------------------------------

    def _capture_loop(self):
        """
        后台循环采集最新图像。

        只保存最新一帧，LeRobot 采集主循环调用 get_rgb() 时取当前最新图像。
        """
        bad_count = 0
        ok_count = 0

        while self._running:
            try:
                result, image = pyvizionsdk.VxGetImage(
                    self.camera,
                    self.timeout_ms,
                    self.format,
                )

                # frame_bgr = decode_uyvy(
                #     image,
                #     width=self.format.width,
                #     height=self.format.height,
                # )
                frame_bgr = decode_mjpg(image)

                if frame_bgr is None:
                    bad_count += 1
                    if bad_count % 100 == 0:
                        self.logger.warning(
                            f"Failed to decode UYVY frame. "
                            f"bad_count={bad_count}, ok_count={ok_count}"
                        )
                    continue

                # 如果数据集要求固定分辨率，则 resize
                if self.target_width is not None and self.target_height is not None:
                    frame_bgr = cv2.resize(
                        frame_bgr,
                        (self.target_width, self.target_height),
                        interpolation=cv2.INTER_AREA,
                    )

                ok_count += 1
                frame_timestamp = time.perf_counter()
                with self._lock:
                    self._frame_bgr = frame_bgr
                    self._frame_timestamp = frame_timestamp

            except Exception as e:
                bad_count += 1
                if bad_count % 20 == 0:
                    self.logger.warning(f"Camera capture exception: {e}")
                time.sleep(0.01)

    def _wait_first_frame(self):
        """
        等待第一帧图像到达。
        """
        start = time.perf_counter()

        while time.perf_counter() - start < self.warmup_timeout_s:
            with self._lock:
                if self._frame_bgr is not None:
                    self.logger.info(f"{self.name} first frame received.")
                    return
            time.sleep(0.01)

        raise RuntimeError(
            f"{self.name} 在 {self.warmup_timeout_s}s 内没有获取到第一帧。"
        )

    # --------------------------------------------------------
    # 对外接口
    # --------------------------------------------------------

    def get_bgr(self) -> np.ndarray:
        """
        获取最新一帧 BGR 图像。

        返回：
            frame_bgr: np.ndarray, shape = (H, W, 3), dtype = uint8
        """
        with self._lock:
            if self._frame_bgr is None:
                raise RuntimeError(f"{self.name} 尚未采集到图像")
            return self._frame_bgr.copy()

    def get_rgb(self) -> np.ndarray:
        """
        获取最新一帧 RGB 图像。

        LeRobot Dataset 推荐保存 RGB 图像，因此数据集采集代码应调用该函数。
        """
        # frame_rgb, _ = self.get_rgb_with_timestamp()
        # frame_bgr = self.get_bgr()
        # frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # return frame_rgb.astype(np.uint8)
        frame_rgb, _ = self.get_rgb_with_timestamp()
        return frame_rgb

    def get_rgb_with_timestamp(self):
        """
        获取最新一帧 RGB 图像和该帧的相机采集时间戳。

        返回：
            frame_rgb: np.ndarray, shape = (H, W, 3), dtype = uint8
            timestamp: float, Unix 时间戳，单位秒
        """
        with self._lock:
            if self._frame_bgr is None:
                raise RuntimeError(f"{self.name} 尚未采集到图像")

            frame_bgr = self._frame_bgr.copy()
            timestamp = self._frame_timestamp

        if timestamp is None:
            raise RuntimeError(f"{self.name} 尚未记录图像时间戳")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb.astype(np.uint8), float(timestamp)

    def get_timestamp(self) -> float:
        """
        获取最新一帧图像的时间戳。
        """
        with self._lock:
            if self._frame_timestamp is None:
                raise RuntimeError(f"{self.name} 尚未记录图像时间戳")
            return float(self._frame_timestamp)
    def get_size(self):
        """
        返回当前输出图像尺寸。

        注意：
        如果设置了 target_width/target_height，则返回 resize 后的尺寸。
        否则返回摄像头原始 format 尺寸。
        """
        if self.target_width is not None and self.target_height is not None:
            return self.target_width, self.target_height

        if self.format is None:
            raise RuntimeError("CameraInterface 尚未 connect()，无法获取图像尺寸")

        return self.format.width, self.format.height


class CameraForTestInterface:
    """
    OpenCV 相机读取封装。
    注意：LeRobot 中建议保存 RGB 图像，而 OpenCV 默认是 BGR。
    """

    def __init__(
        self,
        index_or_path,
        width: int,
        height: int,
        fps: int,
        name: str,
    ):
        self.index_or_path = index_or_path
        self.width = width
        self.height = height
        self.fps = fps
        self.name = name
        self.cap = None

    def connect(self):

        print(f"[Camera:{self.name}] connected")

    def disconnect(self):
        print(f"[Camera:{self.name}] disconnected")
    def get_rgb_with_timestamp(self) :
        """
        返回 RGB uint8 图像，shape = (H, W, 3)
        """
        dummy_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        return dummy_img, float(time.perf_counter())
if __name__ == "__main__":
    scene_cam = CameraInterface(
        cam_num=10,
        fps=60,
        format_idx=7,
        name="scene_camera",
        target_width=1280,
        target_height=720,
    )
    scene_cam.connect()

    # wrist_cam = CameraInterface(
    #     cam_num=11,
    #     fps=60,
    #     format_idx=7,
    #     name="wrist_camera",
    #     target_width=640,
    #     target_height=480,
    # )