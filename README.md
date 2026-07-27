# LeRobot v3 双机遥操作采集 Demo

这是一个“仿真先行、硬件渐进接入”的双侧参考实现。设备 A 同步采集左右
MANUS/VIVE，并在一个带序列号的数据包中发送左右目标；设备 B 分别执行左右侧
重定向和安全控制，同时采集双 Piper、双 Aerohand、左腕相机、右腕相机及全景相机，
最终完成时间对齐和 LeRobot v3 写入。

## 快速验证

```powershell
python -m pip install -r teleop_collect/requirements.txt
python -m teleop_collect.simulate_main --config teleop_collect/config/demo.yaml --seconds 5
python -m unittest discover -s teleop_collect/tests -v
```

`simulate` 默认使用 `debug_jsonl` 写入器，便于在没有 LeRobot/FFmpeg 的开发机上
验证调度与对齐。真机正式采集必须把 `dataset.writer` 改为 `lerobot_v3`。

## 双机运行

设备 B 先启动：

```powershell
python -m teleop_collect.robot_main --config teleop_collect/config/robot.yaml
```

设备 A 再启动：

```powershell
python -m teleop_collect.operator_main --config teleop_collect/config/operator.yaml
```

按 `Ctrl+C` 安全停止。首次真机调试建议依次启用：网络 → 相机 → 手 → 机械臂，
并保持 `robot.enabled: false`，直到时间同步、急停和工作空间检查通过。

## 线程与数据流

```text
设备 A: 左右MANUS/VIVE采集 -> 双侧合法性检查 -> 同一seq/mono_ns -> UDP
设备 B: UDP接收 -> 时钟映射 -> teleop有界缓冲
                      |
                100 Hz 控制线程 -> 左右独立安全门 -> 双Piper/双Aerohand
                      |
 全景/左腕/右腕相机线程 + 双侧状态线程 -> 各自有界时间缓冲
                      |
                30 Hz 帧组装线程 -> 有界写入队列
                                           |
                                  LeRobot写入线程/视频编码
```

控制线程从不调用图像编码、磁盘写入或预览。队列溢出采用丢弃旧帧并计数，不反压
控制线程。原始源时间、映射时间、选择样本的 lag/valid、网络序列号均进入诊断特征。

## 真机适配点

- `adapters.py` 中的操作者适配器：分别读取左右 MANUS 和左右 VIVE，把两侧数据
  归一成同一个 `BimanualTeleopCommand`。
- `RobotHardware`：分别封装左右
  `lerobot_demo/device/piper_arm.py` 与 `device/hand.py`，一次提交双侧命令快照。
- `CameraHardware`：全景、左腕、右腕相机分别独占采集线程，只返回 RGB ndarray
  和真实采集时刻。
- `retarget.py`：替换或包装现有 `ManusToAeroRetargeter`，IK 应返回关节目标及
  成功标志，失败时控制器保持上一安全命令。

正式采集前需分别完成左右 VIVE→机器人基座外参、左右 MANUS 手型、两台 Piper
TCP 和左右工作空间标定。

## LeRobot 双侧字段

- `observation.state`：左臂/左手在前，右臂/右手在后；16 DOF 手时共 56 维。
- `action`：与 state 使用完全相同的左右顺序，共 56 维。
- `observation.images.scene`：共享全景相机。
- `observation.images.wrist_left`：左腕相机。
- `observation.images.wrist_right`：右腕相机。
- `diagnostics.safety_flags`：两维，依次为左、右安全标志。
