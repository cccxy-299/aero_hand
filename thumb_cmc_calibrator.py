import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ThumbCMCCalibrator:
    """
    拇指 CMC 标定和运行时映射。

    功能：
    1. 按 P0/P3/P5 三个姿态采集数据
    2. 从 MANUS nodes 计算拇指掌骨方位角
    3. 从 MANUS ergonomics 读取 thumb_cmc_flex
    4. 使用中位数生成标定参数
    5. 保存/加载不同操作者、不同左右手的标定
    6. 运行时直接输出 Aero：
       - thumb_cmc_abd
       - thumb_cmc_flex
    """

    POSES = ("open", "mid", "closed")

    POSE_INSTRUCTIONS = {
        "open": "P0：手掌张开，拇指完全外展，MCP/IP 尽量伸直",
        "mid": "P3：拇指垂直于手掌，MCP/IP 尽量伸直",
        "closed": "P5：拇指指腹接触小拇指指腹",
    }

    def __init__(
        self,
        operator_id: str,
        side: str,
        calibration_file: str = "thumb_calibration.json",
        min_samples: int = 20,
        abd_output: Tuple[float, float, float] = (0.0, 60.0, 100.0),
        flex_output: Tuple[float, float, float] = (0.0, 45.0, 55.0),
        debug: bool = False,
    ):
        self.operator_id = str(operator_id)
        self.side = self._normalize_side(side)
        self.calibration_file = calibration_file
        self.min_samples = int(min_samples)

        # Aero 输出端锚点：
        # open、mid、closed
        self.abd_output = tuple(float(x) for x in abd_output)
        self.flex_output = tuple(float(x) for x in flex_output)

        self.debug = bool(debug)

        self.samples: Dict[str, Dict[str, List[float]]] = {
            pose: {
                "azimuth": [],
                "flex": [],
            }
            for pose in self.POSES
        }

        self.profile: Optional[Dict[str, float]] = None

    # ============================================================
    # 公共接口：执行完整标定
    # ============================================================

    def calibrate(
        self,
        reader: Any,
        seconds_per_pose: float = 2.0,
        timeout_ms: int = 100,
    ) -> Dict[str, float]:
        """
        交互式完成 P0、P3、P5 标定。

        reader 需要提供：
            reader.read_latest(timeout_ms=...)
        """

        self.reset_samples()

        print("=" * 70)
        print(
            f"开始拇指 CMC 标定："
            f"operator={self.operator_id}, side={self.side}"
        )
        print("标定过程中保持手腕和手掌尽量稳定。")
        print("=" * 70)

        for pose in self.POSES:
            print()
            print(self.POSE_INSTRUCTIONS[pose])
            input("动作稳定后按 Enter 开始采样：")

            result = self.collect_pose(
                reader=reader,
                pose=pose,
                duration_sec=seconds_per_pose,
                timeout_ms=timeout_ms,
            )

            print(
                f"{pose}: "
                f"samples={result['sample_count']}, "
                f"azimuth={result['azimuth_median']:.2f}, "
                f"flex={result['flex_median']:.2f}"
            )

        profile = self.build_profile()
        self.save_profile(profile)
        self.profile = profile

        print()
        print("=" * 70)
        print("拇指 CMC 标定完成")
        self.print_profile()
        print(f"标定文件：{self.calibration_file}")
        print("=" * 70)

        return profile

    # ============================================================
    # 公共接口：采集单个姿态
    # ============================================================

    def collect_pose(
        self,
        reader: Any,
        pose: str,
        duration_sec: float = 2.0,
        timeout_ms: int = 100,
    ) -> Dict[str, float]:
        pose = self._normalize_pose(pose)

        self.samples[pose]["azimuth"].clear()
        self.samples[pose]["flex"].clear()

        deadline = time.monotonic() + float(duration_sec)

        while time.monotonic() < deadline:
            frame = reader.read_latest(timeout_ms=timeout_ms)

            if frame is None:
                continue

            try:
                azimuth, manus_flex = self.extract_inputs(frame)
            except (ValueError, KeyError, TypeError):
                continue

            if not np.isfinite(azimuth):
                continue

            if not np.isfinite(manus_flex):
                continue

            self.samples[pose]["azimuth"].append(float(azimuth))
            self.samples[pose]["flex"].append(float(manus_flex))

        count = len(self.samples[pose]["azimuth"])

        if count < self.min_samples:
            raise RuntimeError(
                f"{pose} 有效样本不足："
                f"{count} < {self.min_samples}"
            )

        azimuth_values = np.asarray(
            self.samples[pose]["azimuth"],
            dtype=float,
        )

        flex_values = np.asarray(
            self.samples[pose]["flex"],
            dtype=float,
        )

        return {
            "sample_count": count,
            "azimuth_median": float(np.median(azimuth_values)),
            "azimuth_mad": self._median_absolute_deviation(
                azimuth_values
            ),
            "flex_median": float(np.median(flex_values)),
            "flex_mad": self._median_absolute_deviation(
                flex_values
            ),
        }

    # ============================================================
    # 公共接口：运行时映射
    # ============================================================

    def map_frame(
        self,
        frame: Dict[str, Any],
    ) -> Tuple[float, float]:
        """
        输入 MANUS frame，直接返回：

            aero_thumb_cmc_abd
            aero_thumb_cmc_flex
        """

        if self.profile is None:
            self.load_profile()

        if self.profile is None:
            raise RuntimeError("拇指标定参数没有加载")

        azimuth, manus_flex = self.extract_inputs(frame)

        # 避免角度在 -180/180 附近跳变。
        azimuth = self._align_angle_to_reference(
            value=azimuth,
            reference=self.profile["abd_mid"],
        )

        aero_abd = self._piecewise_map(
            value=azimuth,
            input_open=self.profile["abd_open"],
            input_mid=self.profile["abd_mid"],
            input_closed=self.profile["abd_closed"],
            output_open=self.profile["abd_output_open"],
            output_mid=self.profile["abd_output_mid"],
            output_closed=self.profile["abd_output_closed"],
        )

        aero_flex = self._piecewise_map(
            value=manus_flex,
            input_open=self.profile["flex_open"],
            input_mid=self.profile["flex_mid"],
            input_closed=self.profile["flex_closed"],
            output_open=self.profile["flex_output_open"],
            output_mid=self.profile["flex_output_mid"],
            output_closed=self.profile["flex_output_closed"],
        )

        aero_abd = float(
            np.clip(
                aero_abd,
                min(self.abd_output),
                max(self.abd_output),
            )
        )

        aero_flex = float(
            np.clip(
                aero_flex,
                min(self.flex_output),
                max(self.flex_output),
            )
        )

        if self.debug:
            print(
                "[thumb CMC] "
                f"operator={self.operator_id}, "
                f"side={self.side}, "
                f"azimuth={azimuth:7.2f}, "
                f"manus_flex={manus_flex:7.2f}, "
                f"aero_abd={aero_abd:7.2f}, "
                f"aero_flex={aero_flex:7.2f}"
            )

        return aero_abd, aero_flex

    # ============================================================
    # 公共接口：提取当前帧的两个标定输入
    # ============================================================

    def extract_inputs(
        self,
        frame: Dict[str, Any],
    ) -> Tuple[float, float]:
        """
        返回：
            thumb_azimuth_deg
            manus_thumb_cmc_flex
        """

        azimuth = self._compute_thumb_azimuth(frame)
        ergonomics = self._get_ergonomics(frame)

        if ergonomics is None or len(ergonomics) < 2:
            raise ValueError("当前帧没有有效 ergonomics")

        manus_thumb_cmc_flex = float(ergonomics[1])

        return azimuth, manus_thumb_cmc_flex

    # ============================================================
    # 公共接口：构建标定参数
    # ============================================================

    def build_profile(self) -> Dict[str, float]:
        medians: Dict[str, Dict[str, float]] = {}

        for pose in self.POSES:
            azimuth_samples = self.samples[pose]["azimuth"]
            flex_samples = self.samples[pose]["flex"]

            if len(azimuth_samples) < self.min_samples:
                raise RuntimeError(
                    f"{pose} 的 azimuth 样本不足"
                )

            if len(flex_samples) < self.min_samples:
                raise RuntimeError(
                    f"{pose} 的 flex 样本不足"
                )

            medians[pose] = {
                "azimuth": float(
                    np.median(
                        np.asarray(
                            azimuth_samples,
                            dtype=float,
                        )
                    )
                ),
                "flex": float(
                    np.median(
                        np.asarray(
                            flex_samples,
                            dtype=float,
                        )
                    )
                ),
            }

        # 对三个方位角进行解包，避免跨越 ±180 度。
        azimuth_points = np.asarray(
            [
                medians["open"]["azimuth"],
                medians["mid"]["azimuth"],
                medians["closed"]["azimuth"],
            ],
            dtype=float,
        )

        azimuth_points = np.degrees(
            np.unwrap(
                np.radians(azimuth_points)
            )
        )

        abd_open = float(azimuth_points[0])
        abd_mid = float(azimuth_points[1])
        abd_closed = float(azimuth_points[2])

        flex_open = medians["open"]["flex"]
        flex_mid = medians["mid"]["flex"]
        flex_closed = medians["closed"]["flex"]

        self._validate_three_points(
            name="thumb azimuth",
            open_value=abd_open,
            mid_value=abd_mid,
            closed_value=abd_closed,
            minimum_span=10.0,
        )

        self._validate_three_points(
            name="MANUS thumb flex",
            open_value=flex_open,
            mid_value=flex_mid,
            closed_value=flex_closed,
            minimum_span=5.0,
        )

        profile = {
            "abd_open": abd_open,
            "abd_mid": abd_mid,
            "abd_closed": abd_closed,

            "flex_open": flex_open,
            "flex_mid": flex_mid,
            "flex_closed": flex_closed,

            "abd_output_open": self.abd_output[0],
            "abd_output_mid": self.abd_output[1],
            "abd_output_closed": self.abd_output[2],

            "flex_output_open": self.flex_output[0],
            "flex_output_mid": self.flex_output[1],
            "flex_output_closed": self.flex_output[2],
        }

        return profile

    # ============================================================
    # 公共接口：保存和加载
    # ============================================================

    def save_profile(
        self,
        profile: Optional[Dict[str, float]] = None,
    ) -> None:
        if profile is None:
            profile = self.profile

        if profile is None:
            raise RuntimeError("没有可保存的标定参数")

        database: Dict[str, Any] = {}

        if os.path.exists(self.calibration_file):
            try:
                with open(
                    self.calibration_file,
                    "r",
                    encoding="utf-8",
                ) as file:
                    loaded = json.load(file)

                if isinstance(loaded, dict):
                    database = loaded

            except (OSError, json.JSONDecodeError):
                database = {}

        if self.operator_id not in database:
            database[self.operator_id] = {}

        database[self.operator_id][self.side] = profile

        directory = os.path.dirname(
            os.path.abspath(self.calibration_file)
        )

        os.makedirs(directory, exist_ok=True)

        temporary_file = self.calibration_file + ".tmp"

        with open(
            temporary_file,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                database,
                file,
                ensure_ascii=False,
                indent=2,
            )

        os.replace(
            temporary_file,
            self.calibration_file,
        )

        self.profile = profile

    def load_profile(self) -> Dict[str, float]:
        if not os.path.exists(self.calibration_file):
            raise FileNotFoundError(
                f"找不到标定文件：{self.calibration_file}"
            )

        with open(
            self.calibration_file,
            "r",
            encoding="utf-8",
        ) as file:
            database = json.load(file)

        operator_profiles = database.get(self.operator_id)

        if not isinstance(operator_profiles, dict):
            raise KeyError(
                f"找不到操作者标定：{self.operator_id}"
            )

        profile = operator_profiles.get(self.side)

        if not isinstance(profile, dict):
            raise KeyError(
                f"找不到标定："
                f"operator={self.operator_id}, "
                f"side={self.side}"
            )

        required_keys = {
            "abd_open",
            "abd_mid",
            "abd_closed",
            "flex_open",
            "flex_mid",
            "flex_closed",
            "abd_output_open",
            "abd_output_mid",
            "abd_output_closed",
            "flex_output_open",
            "flex_output_mid",
            "flex_output_closed",
        }

        missing = required_keys.difference(profile.keys())

        if missing:
            raise ValueError(
                f"标定文件缺少字段：{sorted(missing)}"
            )

        self.profile = {
            key: float(profile[key])
            for key in required_keys
        }

        return self.profile

    def print_profile(self) -> None:
        if self.profile is None:
            print("没有加载标定参数")
            return

        print(
            "azimuth: "
            f"{self.profile['abd_open']:.2f} -> "
            f"{self.profile['abd_mid']:.2f} -> "
            f"{self.profile['abd_closed']:.2f}"
        )

        print(
            "MANUS flex: "
            f"{self.profile['flex_open']:.2f} -> "
            f"{self.profile['flex_mid']:.2f} -> "
            f"{self.profile['flex_closed']:.2f}"
        )

    def reset_samples(self) -> None:
        for pose in self.POSES:
            self.samples[pose]["azimuth"].clear()
            self.samples[pose]["flex"].clear()

    # ============================================================
    # MANUS 数据读取
    # ============================================================

    def _get_skeleton(
        self,
        frame: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        skeletons = frame.get("skeletons", [])

        if not isinstance(skeletons, list):
            return None

        for skeleton in skeletons:
            skeleton_side = self._normalize_side_or_none(
                skeleton.get("side")
            )

            if skeleton_side == self.side:
                return skeleton

        # 单手数据但没有 side 时允许回退。
        if len(skeletons) == 1:
            return skeletons[0]

        return None

    def _get_nodes(
        self,
        frame: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        skeleton = self._get_skeleton(frame)

        if skeleton is None:
            return []

        nodes = skeleton.get("nodes", [])

        if not isinstance(nodes, list):
            return []

        return nodes

    def _get_ergonomics(
        self,
        frame: Dict[str, Any],
    ) -> Optional[List[float]]:
        skeleton = self._get_skeleton(frame)

        ergonomics = None

        if skeleton is not None:
            ergonomics = skeleton.get("ergonomics")

        if ergonomics is None:
            ergonomics = frame.get("ergonomics")

        ergonomics = self._normalize_ergonomics(
            ergonomics
        )

        if ergonomics is None:
            return None

        # 双手 40 维格式：
        # 0:20 左手
        # 20:40 右手
        if len(ergonomics) >= 40:
            if self.side == "left":
                return ergonomics[0:20]

            return ergonomics[20:40]

        # 当前 skeleton 已按 side 选择时，
        # 20 维可以直接使用。
        if len(ergonomics) >= 20:
            return ergonomics[0:20]

        return None

    @staticmethod
    def _normalize_ergonomics(
        ergonomics: Any,
    ) -> Optional[List[float]]:
        if ergonomics is None:
            return None

        if not isinstance(ergonomics, list):
            return None

        if not ergonomics:
            return None

        if isinstance(ergonomics[0], dict):
            max_index = max(
                int(item.get("index", 0))
                for item in ergonomics
            )

            values = [0.0] * (max_index + 1)

            for item in ergonomics:
                index = int(item.get("index", 0))
                values[index] = float(
                    item.get("value", 0.0)
                )

            return values

        return [float(value) for value in ergonomics]

    # ============================================================
    # 拇指几何计算
    # ============================================================

    def _compute_thumb_azimuth(
        self,
        frame: Dict[str, Any],
    ) -> float:
        nodes = self._get_nodes(frame)

        if len(nodes) <= 20:
            raise ValueError(
                f"MANUS nodes 数量不足：{len(nodes)}"
            )

        def position(index: int) -> np.ndarray:
            node = nodes[index]

            if "position" not in node:
                raise KeyError(
                    f"node[{index}] 没有 position"
                )

            value = np.asarray(
                node["position"],
                dtype=float,
            )

            if value.shape != (3,):
                raise ValueError(
                    f"node[{index}] position 格式错误"
                )

            return value

        p_root = position(0)

        p_thumb_cmc = position(1)
        p_thumb_ip = position(3)
        p_thumb_tip = position(4)

        p_index = position(5)
        p_middle = position(10)
        p_ring = position(15)
        p_pinky = position(20)

        # 手掌朝向手指方向。
        mcp_center = (
            p_index
            + p_middle
            + p_ring
            + p_pinky
        ) / 4.0

        y_axis = self._unit(
            mcp_center - p_root
        )

        if y_axis is None:
            raise ValueError("无法计算手掌 Y 轴")

        # 小拇指侧指向食指侧。
        x_raw = p_index - p_pinky

        # Gram-Schmidt 正交化。
        x_raw = (
            x_raw
            - np.dot(x_raw, y_axis) * y_axis
        )

        x_axis = self._unit(x_raw)

        if x_axis is None:
            raise ValueError("无法计算手掌 X 轴")

        # 拇指有效方向。
        thumb_direction_1 = self._unit(
            p_thumb_ip - p_thumb_cmc
        )

        thumb_direction_2 = self._unit(
            p_thumb_tip - p_thumb_cmc
        )

        if (
            thumb_direction_1 is None
            or thumb_direction_2 is None
        ):
            raise ValueError("无法计算拇指方向")

        thumb_direction = self._unit(
            0.70 * thumb_direction_1
            + 0.30 * thumb_direction_2
        )

        if thumb_direction is None:
            raise ValueError("拇指方向无效")

        local_x = float(
            np.dot(thumb_direction, x_axis)
        )

        local_y = float(
            np.dot(thumb_direction, y_axis)
        )

        azimuth = math.degrees(
            math.atan2(local_y, local_x)
        )

        return float(azimuth)

    # ============================================================
    # 数学辅助函数
    # ============================================================

    @staticmethod
    def _unit(
        vector: np.ndarray,
        epsilon: float = 1e-9,
    ) -> Optional[np.ndarray]:
        vector = np.asarray(vector, dtype=float)
        norm = float(np.linalg.norm(vector))

        if norm < epsilon:
            return None

        return vector / norm

    @staticmethod
    def _piecewise_map(
        value: float,
        input_open: float,
        input_mid: float,
        input_closed: float,
        output_open: float,
        output_mid: float,
        output_closed: float,
    ) -> float:
        """
        支持输入递增或递减的三点分段线性映射。
        """

        increasing = (
            input_open < input_mid < input_closed
        )

        decreasing = (
            input_open > input_mid > input_closed
        )

        if not increasing and not decreasing:
            raise ValueError(
                "标定输入点不是单调排列："
                f"{input_open}, "
                f"{input_mid}, "
                f"{input_closed}"
            )

        if increasing:
            value = float(
                np.clip(
                    value,
                    input_open,
                    input_closed,
                )
            )

            if value <= input_mid:
                return ThumbCMCCalibrator._linear_map(
                    value,
                    input_open,
                    input_mid,
                    output_open,
                    output_mid,
                )

            return ThumbCMCCalibrator._linear_map(
                value,
                input_mid,
                input_closed,
                output_mid,
                output_closed,
            )

        value = float(
            np.clip(
                value,
                input_closed,
                input_open,
            )
        )

        if value >= input_mid:
            return ThumbCMCCalibrator._linear_map(
                value,
                input_open,
                input_mid,
                output_open,
                output_mid,
            )

        return ThumbCMCCalibrator._linear_map(
            value,
            input_mid,
            input_closed,
            output_mid,
            output_closed,
        )

    @staticmethod
    def _linear_map(
        value: float,
        input_start: float,
        input_end: float,
        output_start: float,
        output_end: float,
    ) -> float:
        denominator = input_end - input_start

        if abs(denominator) < 1e-9:
            return float(output_start)

        ratio = (
            float(value) - input_start
        ) / denominator

        ratio = float(
            np.clip(ratio, 0.0, 1.0)
        )

        return float(
            output_start
            + ratio * (
                output_end - output_start
            )
        )

    @staticmethod
    def _validate_three_points(
        name: str,
        open_value: float,
        mid_value: float,
        closed_value: float,
        minimum_span: float,
    ) -> None:
        increasing = (
            open_value < mid_value < closed_value
        )

        decreasing = (
            open_value > mid_value > closed_value
        )

        if not increasing and not decreasing:
            raise ValueError(
                f"{name} 标定点不单调："
                f"open={open_value:.2f}, "
                f"mid={mid_value:.2f}, "
                f"closed={closed_value:.2f}"
            )

        total_span = abs(
            closed_value - open_value
        )

        if total_span < minimum_span:
            raise ValueError(
                f"{name} 标定范围太小："
                f"{total_span:.2f}°"
            )

    @staticmethod
    def _align_angle_to_reference(
        value: float,
        reference: float,
    ) -> float:
        """
        将 value 加减 360°，使其最接近 reference。
        """

        return float(
            value
            + 360.0
            * round(
                (reference - value) / 360.0
            )
        )

    @staticmethod
    def _median_absolute_deviation(
        values: np.ndarray,
    ) -> float:
        median = float(np.median(values))

        return float(
            np.median(
                np.abs(values - median)
            )
        )

    @staticmethod
    def _normalize_side(side: str) -> str:
        normalized = str(side).strip().lower()

        if normalized not in ("left", "right"):
            raise ValueError(
                f"无效 side：{side}"
            )

        return normalized

    @staticmethod
    def _normalize_side_or_none(
        side: Any,
    ) -> Optional[str]:
        if side is None:
            return None

        normalized = str(side).strip().lower()

        if normalized in ("left", "right"):
            return normalized

        return None

    @classmethod
    def _normalize_pose(
        cls,
        pose: str,
    ) -> str:
        normalized = str(pose).strip().lower()

        aliases = {
            "p0": "open",
            "open": "open",
            "p3": "mid",
            "mid": "mid",
            "p5": "closed",
            "closed": "closed",
        }

        result = aliases.get(normalized)

        if result not in cls.POSES:
            raise ValueError(
                f"无效标定姿态：{pose}"
            )

        return result

if __name__ == "__main__":
    CALIBRATION_MODE = True
    OPERATOR_ID = "operator_01"
    SIDE = "right"
    thumb_calibrator = ThumbCMCCalibrator(
        operator_id=OPERATOR_ID,
        side=SIDE,
        calibration_file="thumb_calibration.json",
        debug=True,
    )
