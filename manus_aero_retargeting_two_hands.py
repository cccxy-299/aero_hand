import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import zmq
from const import AERO_JOINT_LOWER_DEG_16,AERO_JOINT_UPPER_DEG_16,AERO_JOINT_NAMES_16
from common_cls import LowPass, ManusZmqReader, RetargetingConfig
try:
    from manus.thumb_cmc_calibrator import ThumbCMCCalibrator
except ModuleNotFoundError:
    # 允许从当前仓库的 “manus glove” 目录直接复用该参考实现。
    from thumb_cmc_calibrator import ThumbCMCCalibrator



class ManusToAeroRetargeter:
    def __init__(self, config: RetargetingConfig,left_calibrator: ThumbCMCCalibrator,right_calibrator: ThumbCMCCalibrator):
        self.config = config
        self.filter = LowPass(config.filter_alpha)
        # self.thumb_calibrator = thumb_calibrator
        self.left_thumb_calibrator = left_calibrator
        self.right_thumb_calibrator = right_calibrator
        self.filters = {
            "left": LowPass(config.filter_alpha),
            "right": LowPass(config.filter_alpha),
        }

    def retarget(self, frame: Dict[str, Any]) -> Optional[List[float]]:
        """
        输入：ZMQ 收到的 MANUS frame。
        输出：
          - config.compact_7dof=False: 16 维 Aero joint positions
          - config.compact_7dof=True:  7 维 Aero compact joint positions
        单位：
          - 默认 degrees，可直接给 aero_hand.set_joint_positions()
        """
        if self._has_ergonomics(frame):
            q16 = self._retarget_from_ergonomics(frame)
        else:
            q16 = self._fallback_from_nodes(frame)

        if q16 is None:
            return None

        q16 = np.asarray(q16, dtype=float)
        q16 = np.clip(q16, AERO_JOINT_LOWER_DEG_16, AERO_JOINT_UPPER_DEG_16)

        if self.config.enable_filter:
            q16 = self.filter(q16)
            q16 = np.clip(q16, AERO_JOINT_LOWER_DEG_16, AERO_JOINT_UPPER_DEG_16)

        if self.config.compact_7dof:
            q = self.to_compact_7dof(q16)
        else:
            q = q16

        if not self.config.output_degrees:
            q = np.deg2rad(q)

        return q.astype(float).tolist()

    def retarget_two_hands(self, frame: Dict[str, Any]) ->  Dict[str, Any]:
        """
        输入：ZMQ 收到的 MANUS frame。
        输出：
          - config.compact_7dof=False: 16 维 Aero joint positions
          - config.compact_7dof=True:  7 维 Aero compact joint positions
        单位：
          - 默认 degrees，可直接给 aero_hand.set_joint_positions()
        """

        q16_left = self._retarget_from_ergonomics_two_hands(frame, "left")
        q16_right = self._retarget_from_ergonomics_two_hands(frame,"right")


        q16_left = np.asarray(q16_left, dtype=float)
        q16_left = np.clip(q16_left, AERO_JOINT_LOWER_DEG_16, AERO_JOINT_UPPER_DEG_16)

        q16_right = np.asarray(q16_right, dtype=float)
        q16_right = np.clip(q16_right, AERO_JOINT_LOWER_DEG_16, AERO_JOINT_UPPER_DEG_16)

        if self.config.enable_filter:
            q16_left = self.filters["left"](q16_left)
            q16_left = np.clip(q16_left, AERO_JOINT_LOWER_DEG_16, AERO_JOINT_UPPER_DEG_16)
            q16_right = self.filters["right"](q16_right)
            q16_right = np.clip(q16_right, AERO_JOINT_LOWER_DEG_16, AERO_JOINT_UPPER_DEG_16)

        if self.config.compact_7dof:
            q_left = self.to_compact_7dof(q16_left)
            q_right = self.to_compact_7dof(q16_right)
        else:
            q_left = q16_left
            q_right = q16_right

        if not self.config.output_degrees:
            q_left = np.deg2rad(q_left)
            q_right = np.deg2rad(q_right)

        q_left = q_left.astype(float).tolist()
        q_right = q_right.astype(float).tolist()

        return {"left": q_left, "right": q_right}

    # def _retarget_thumb_cmc(
    #         self,
    #         frame: Dict[str, Any],
    #         manus_thumb_flex: float,
    # ) -> Optional[Tuple[float, float, float]]:
    #     """
    #     使用 MANUS nodes 计算 Aero 拇指 CMC 两个关节。
    #
    #     返回：
    #         aero_abd
    #         aero_flex
    #         thumb_azimuth_deg
    #
    #     当前标定针对右手。
    #     """
    #
    #     nodes = self._get_nodes(frame)
    #
    #     # 需要：
    #     # 0      palm/root
    #     # 1-4    thumb
    #     # 5      index base
    #     # 10     middle base
    #     # 15     ring base
    #     # 20     pinky base
    #     if len(nodes) <= 20:
    #         return None
    #
    #     def position(index: int) -> np.ndarray:
    #         return np.asarray(nodes[index]["position"], dtype=float)
    #
    #     p_root = position(0)
    #
    #     p_thumb_cmc = position(1)
    #     p_thumb_ip = position(3)
    #     p_thumb_tip = position(4)
    #
    #     p_index = position(5)
    #     p_middle = position(10)
    #     p_ring = position(15)
    #     p_pinky = position(20)
    #
    #     # ============================================================
    #     # 建立手掌局部坐标系
    #     #
    #     # X：小拇指侧 -> 食指侧
    #     # Y：手腕/掌根 -> 四指 MCP 中心
    #     # Z：手掌法向
    #     # ============================================================
    #
    #     mcp_center = (
    #                          p_index
    #                          + p_middle
    #                          + p_ring
    #                          + p_pinky
    #                  ) / 4.0
    #
    #     y_axis = self._unit_vector(mcp_center - p_root)
    #
    #     if y_axis is None:
    #         return None
    #
    #     x_raw = p_index - p_pinky
    #
    #     # Gram-Schmidt 正交化
    #     x_raw = x_raw - np.dot(x_raw, y_axis) * y_axis
    #     x_axis = self._unit_vector(x_raw)
    #
    #     if x_axis is None:
    #         return None
    #
    #     z_axis = self._unit_vector(np.cross(x_axis, y_axis))
    #
    #     if z_axis is None:
    #         return None
    #
    #     # ============================================================
    #     # 拇指掌骨有效方向
    #     #
    #     # 必须从 node[1]，即 CMC 基部出发。
    #     # 不能使用 root -> thumb，因为会混入拇指基部平移。
    #     # ============================================================
    #
    #     thumb_dir_1 = self._unit_vector(p_thumb_ip - p_thumb_cmc)
    #     thumb_dir_2 = self._unit_vector(p_thumb_tip - p_thumb_cmc)
    #
    #     if thumb_dir_1 is None or thumb_dir_2 is None:
    #         return None
    #
    #     thumb_direction = self._unit_vector(
    #         0.70 * thumb_dir_1
    #         + 0.30 * thumb_dir_2
    #     )
    #
    #     if thumb_direction is None:
    #         return None
    #
    #     thumb_local_x = float(np.dot(thumb_direction, x_axis))
    #     thumb_local_y = float(np.dot(thumb_direction, y_axis))
    #     thumb_local_z = float(np.dot(thumb_direction, z_axis))
    #
    #     # 手掌平面内方位角。
    #     # 你的 P0 -> P5 数据中：
    #     # 54.05 -> 65.96 -> 101.08 -> 126.85 -> 131.96
    #     # 全程单调，不会像 MANUS spread 一样在 P3 后反向。
    #     thumb_azimuth_deg = math.degrees(
    #         math.atan2(thumb_local_y, thumb_local_x)
    #     )
    #
    #     aero_abd = self._map_clamped(
    #         value=thumb_azimuth_deg,
    #         input_start=self.config.thumb_abd_geom_open_deg,
    #         input_end=self.config.thumb_abd_geom_closed_deg,
    #         output_start=self.config.thumb_abd_output_open_deg,
    #         output_end=self.config.thumb_abd_output_closed_deg,
    #     )
    #
    #     # MANUS flex 从张开到对掌是递减的：
    #     # 52.5 -> 37.4 -> 2.6 -> -7.1
    #     #
    #     # 负数不是错误，只是 MANUS 的角度零点定义。
    #     aero_flex = self._map_clamped(
    #         value=manus_thumb_flex,
    #         input_start=self.config.thumb_flex_manus_open_deg,
    #         input_end=self.config.thumb_flex_manus_closed_deg,
    #         output_start=self.config.thumb_flex_output_open_deg,
    #         output_end=self.config.thumb_flex_output_closed_deg,
    #     )
    #
    #     print(
    #         "[thumb CMC] "
    #         f"azimuth={thumb_azimuth_deg:7.2f}, "
    #         f"local=({thumb_local_x:6.3f}, "
    #         f"{thumb_local_y:6.3f}, "
    #         f"{thumb_local_z:6.3f}), "
    #         f"manus_flex={manus_thumb_flex:7.2f}, "
    #         f"aero_abd={aero_abd:7.2f}, "
    #         f"aero_flex={aero_flex:7.2f}"
    #     )
    #
    #     return aero_abd, aero_flex, thumb_azimuth_deg
    #
    # def _get_skeleton_by_side(
    #         self,
    #         frame: Dict[str, Any],
    #         side: str,
    # ) -> Optional[Dict[str, Any]]:
    #     side = side.lower()
    #
    #     for skeleton in frame.get("skeletons", []):
    #         skeleton_side = str(
    #             skeleton.get("side", "")
    #         ).strip().lower()
    #
    #         if skeleton_side == side:
    #             return skeleton
    #
    #     return None
    #
    # def _get_nodes_by_side(
    #         self,
    #         frame: Dict[str, Any],
    #         side: str,
    # ) -> List[Dict[str, Any]]:
    #     skeleton = self._get_skeleton_by_side(frame, side)
    #
    #     if skeleton is None:
    #         return []
    #
    #     return skeleton.get("nodes", [])
    # @staticmethod
    # def _unit_vector(v: np.ndarray, eps: float = 1e-9) -> Optional[np.ndarray]:
    #     v = np.asarray(v, dtype=float)
    #     norm = float(np.linalg.norm(v))
    #
    #     if norm < eps:
    #         return None
    #
    #     return v / norm
    #
    # @staticmethod
    # def _map_clamped(
    #         value: float,
    #         input_start: float,
    #         input_end: float,
    #         output_start: float,
    #         output_end: float,
    # ) -> float:
    #     """
    #     支持输入范围递增或递减的线性映射。
    #     """
    #     denominator = input_end - input_start
    #
    #     if abs(denominator) < 1e-9:
    #         return float(output_start)
    #
    #     t = (float(value) - input_start) / denominator
    #     t = float(np.clip(t, 0.0, 1.0))
    #
    #     return float(output_start + t * (output_end - output_start))

    # def _has_ergonomics(self, frame: Dict[str, Any]) -> bool:
    #     erg = self._get_ergonomics(frame)
    #     return erg is not None and len(erg) >= 20
    #
    # def _get_side(self, frame: Dict[str, Any]) -> str:
    #     # 期望 C++ 端后续发送 "side": "Right" 或 "Left"。
    #     # 如果没有，默认右手。
    #     side=''
    #     skeletons = frame.get("skeletons", [])
    #     if skeletons:
    #         side = skeletons[0].get("side")
    #
    #     if side not in ("right", "left"):
    #         side = "right"
    #
    #     return side
    #
    # def _get_nodes(self, frame: Dict[str, Any]) -> List[Dict[str, Any]]:
    #     skeletons = frame.get("skeletons", [])
    #     if not skeletons:
    #         return []

        return skeletons[0].get("nodes", [])

    def _get_ergonomics_by_side(self, frame: Dict[str, Any], side: str) -> Optional[List[float]]:
        """
        从 MANUS ZMQ frame 中提取有效 ergonomics。

        当前你的 MANUS 数据格式：
        ergonomics length = 40
        0-19   左手
        20-39  右手

        返回值统一为 20 维：
        [
            thumb_cmc_spread,
            thumb_cmc_flex,
            thumb_pip_flex,
            thumb_dip_flex,

            index_mcp_spread,
            index_mcp_flex,
            index_pip_flex,
            index_dip_flex,

            middle_mcp_spread,
            middle_mcp_flex,
            middle_pip_flex,
            middle_dip_flex,

            ring_mcp_spread,
            ring_mcp_flex,
            ring_pip_flex,
            ring_dip_flex,

            pinky_mcp_spread,
            pinky_mcp_flex,
            pinky_pip_flex,
            pinky_dip_flex,
        ]
        """
        erg = None
        skeletons = frame.get("skeletons", [])
        for skeleton in skeletons:
            skeleton_side = skeleton.get("side")
            if skeleton_side == side:
                erg = skeleton.get("ergonomics")
        # if skeletons:
        #     erg = skeletons[0].get("ergonomics")

        if erg is None:
            return None

        erg = [float(x) for x in erg]

        if side.lower() == "right":
            return erg[20:40]
        elif side.lower() == "left":
            return erg[0:20]

        return None

    def _retarget_from_ergonomics_two_hands(self, frame: Dict[str, Any], side: str) -> Optional[np.ndarray]:
        joint_values = self._get_ergonomics_by_side(frame,side)

        if joint_values is None or len(joint_values) < 20:
            return None

        q = np.array(joint_values[:20], dtype=float)

        if not self.config.ergonomics_in_degrees:
            q = np.rad2deg(q)

        # 创建 Aero 16维输出
        q16 = np.zeros(16, dtype=float)

        # CMC 两个关节由独立标定类负责。
        if side.lower() == "left":
            q16[0], q16[1] = self.left_thumb_calibrator.map_frame(frame)
        elif side.lower() == "right":
            q16[0], q16[1] = self.right_thumb_calibrator.map_frame(frame)
        else:
            raise ValueError(f"Invalid side: {side}")

        # 拇指 MCP 和 IP
        q16[2] = np.clip(q[2], 0, 90)  # thumb_mcp
        q16[3] = np.clip(q[3], 0, 90)  # thumb_ip

        # === 食指映射（3个关节）===
        q16[4] = np.clip(q[5], 0, 90)  # index_mcp
        q16[5] = np.clip(q[6], 0, 90)  # index_pip
        q16[6] = np.clip(q[7], 0, 90)  # index_dip

        # === 中指映射（3个关节）===
        q16[7] = np.clip(q[9], 0, 90)  # middle_mcp
        q16[8] = np.clip(q[10], 0, 90)  # middle_pip
        q16[9] = np.clip(q[11], 0, 90)  # middle_dip

        # === 无名指映射（3个关节）===
        q16[10] = np.clip(q[13], 0, 90)  # ring_mcp
        q16[11] = np.clip(q[14], 0, 90)  # ring_pip
        q16[12] = np.clip(q[15], 0, 90)  # ring_dip

        # === 小指映射（3个关节）===
        q16[13] = np.clip(q[17], 0, 90)  # pinky_mcp
        q16[14] = np.clip(q[18], 0, 90)  # pinky_pip
        q16[15] = np.clip(q[19], 0, 90)  # pinky_dip

        # 输出调试信息
        # print(f"[thumb] abd={q16[0]:.1f}, flex={q16[1]:.1f}, mcp={q16[2]:.1f}, ip={q16[3]:.1f}")

        return q16

    @staticmethod
    def to_compact_7dof(q16: Sequence[float]) -> np.ndarray:
        """
        Aero Hand compact 7 维：
        0 thumb_cmc_abd
        1 thumb_cmc_flex
        2 thumb_mcp & thumb_ip
        3 index_mcp/pip/dip
        4 middle_mcp/pip/dip
        5 ring_mcp/pip/dip
        6 pinky_mcp/pip/dip
        """
        q = np.asarray(q16, dtype=float)

        return np.array(
            [
                q[0],
                q[1],
                0.5 * (q[2] + q[3]),
                np.mean(q[4:7]),
                np.mean(q[7:10]),
                np.mean(q[10:13]),
                np.mean(q[13:16]),
            ],
            dtype=float,
        )

    @staticmethod
    def compact_7dof_to_16(q7: Sequence[float]) -> np.ndarray:
        """
        Aero Hand 文档里的 7→16 compact 展开逻辑。
        """
        q = np.asarray(q7, dtype=float)

        return np.array(
            [
                q[0],
                q[1],
                q[2],
                q[2],
                q[3],
                q[3],
                q[3],
                q[4],
                q[4],
                q[4],
                q[5],
                q[5],
                q[5],
                q[6],
                q[6],
                q[6],
            ],
            dtype=float,
        )

def main():
    left_thumb_calibrator = ThumbCMCCalibrator(
        operator_id="operator_01",
        side="left",
        calibration_file="thumb_calibration.json",
        debug=False,
    )
    right_thumb_calibrator = ThumbCMCCalibrator(
        operator_id="operator_01",
        side="right",
        calibration_file="thumb_calibration.json",
        debug=False,
    )
    left_thumb_calibrator.load_profile()
    right_thumb_calibrator.load_profile()

    # 改这里选择 16 维或 7 维输出。
    cfg = RetargetingConfig(
        compact_7dof=True,  # False 输出 16 维；True 输出 7 维
        output_degrees=True,
        enable_filter=True,
        filter_alpha=0.8,
    )

    reader = ManusZmqReader(
        address="tcp://127.0.0.1:9000",
        topic="manus",
    )

    retargeter = ManusToAeroRetargeter(cfg, left_calibrator=left_thumb_calibrator, right_calibrator=right_thumb_calibrator)

    # 配置aero hand
    use_real_hand = True
    aero_hand_left = None
    aero_hand_right = None

    if use_real_hand:
        from aero_open_sdk.aero_hand import AeroHand
        aero_hand_left = AeroHand(port="COM18")  # 左手端口
        aero_hand_right = AeroHand(port="COM23")  # 右手端口,
        #+
        aero_hand_left.send_homing()
        aero_hand_right.send_homing()
        time.sleep(5)

    try:
        while True:

            frame = reader.read_latest(timeout_ms=1000)

            if frame is None:
                print("No MANUS ZMQ frame.")
                continue

            # q = retargeter.retarget(frame)
            q = retargeter.retarget_two_hands(frame)

            if q is None:
                print("No usable MANUS data. Need ergonomics or valid raw nodes.")
                continue

            if aero_hand_left is not None:
            #     print("Sending joint positions to Aero Hand:", q)
                aero_hand_left.set_joint_positions(q["left"])

            if aero_hand_right is not None:
                # print("Sending joint positions to Aero Hand:", q)
                aero_hand_right.set_joint_positions(q["right"])
                # aero_hand_right.set_joint_positions(q)

            # time.sleep(0.001)

    except KeyboardInterrupt:
        # aero_hand_right.send_homing()
        print("Stopped by user.")


    finally:
        reader.close()

def calibrate():
    CALIBRATION_MODE = True
    OPERATOR_ID = "operator_01"
    SIDE = "right"

    reader = ManusZmqReader(
        address="tcp://127.0.0.1:9000",
        topic="manus",
    )

    thumb_calibrator = ThumbCMCCalibrator(
        operator_id=OPERATOR_ID,
        side=SIDE,
        calibration_file="thumb_calibration.json",
        debug=True,
    )

    if CALIBRATION_MODE:
        try:
            thumb_calibrator.calibrate(
                reader=reader,
                seconds_per_pose=2.0,
            )
        finally:
            reader.close()

        return



if __name__ == "__main__":
    # calibrate()
    main()
