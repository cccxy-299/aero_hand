import numpy as np
# Aero Hand 16 维关节顺序，来自 Aero Hand SDK 文档：
# 0 thumb_cmc_abd
# 1 thumb_cmc_flex
# 2 thumb_mcp
# 3 thumb_ip
# 4 index_mcp
# 5 index_pip
# 6 index_dip
# 7 middle_mcp
# 8 middle_pip
# 9 middle_dip
# 10 ring_mcp
# 11 ring_pip
# 12 ring_dip
# 13 pinky_mcp
# 14 pinky_pip
# 15 pinky_dip
AERO_JOINT_NAMES_16 = [
    "thumb_cmc_abd",
    "thumb_cmc_flex",
    "thumb_mcp",
    "thumb_ip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
]

AERO_JOINT_LOWER_DEG_16 = np.array(
    [0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0,
     0.0, 0.0, 0.0,
     0.0, 0.0, 0.0,
     0.0, 0.0, 0.0],
    dtype=float,
)

AERO_JOINT_UPPER_DEG_16 = np.array(
    [100.0, 55.0, 90.0, 90.0,
     90.0, 90.0, 90.0,
     90.0, 90.0, 90.0,
     90.0, 90.0, 90.0,
     90.0, 90.0, 90.0],
    dtype=float,
)