# LeRobot v3 双机遥操作采集 Demo

这是一个“仿真先行、硬件渐进接入”的双侧参考实现。设备 A 同步采集左右
MANUS/VIVE，在设备 A 将 MANUS 重定向成左右各7维灵巧手控制指令，再与左右 VIVE
原始位姿放入同一个序列号数据包；设备 B 映射 VIVE 并执行左右侧安全控制，同时采集
双 Piper、双 Aerohand、左腕相机、右腕相机及全景相机，
最终完成时间对齐和 LeRobot v3 写入。

## 快速验证

```powershell
cd teleop_collect
python -m pip install -r requirements.txt
python simulate_main.py --config cfg/demo.yaml --seconds 5
```

`simulate` 默认使用 `debug_jsonl` 写入器，便于在没有 LeRobot/FFmpeg 的开发机上
验证调度与对齐。真机正式采集必须把 `dataset.writer` 改为 `lerobot_v3`。

## 双机运行

设备 B 先启动：

```powershell
cd teleop_collect
python robot_main.py --config cfg/robot.yaml
```

设备 A 再启动：

```powershell
cd teleop_collect
python operator_main.py --config cfg/operator.yaml
```

设备 A 可选择开启或关闭 VIVE 三维可视化：

```powershell
# 命令行显式开启
python operator_main.py --config cfg/operator.yaml --visualize

# 显式关闭，适用于无桌面环境
python operator_main.py --config cfg/operator.yaml --no-visualize
```

未传参数时使用 `operator.yaml` 中的 `visualization.enabled`。开启可视化需要额外安装
`pyqtgraph`、`PyOpenGL` 和 `PySide6`（或 PyQt5）。GUI 只读取 VIVE 快照，不参与
MANUS 重定向、网络发送或机器人控制。

```powershell
python -m pip install -r requirements-visualization.txt
```

当前入口使用本目录内的模块引用，因此请先进入 `teleop_collect` 目录，再运行设备 A、
设备 B 或仿真入口。

按 `Ctrl+C` 安全停止；若 episode 正在录制，默认先保存该 episode 再退出。
`cfg/robot.yaml` 和 `cfg/operator.yaml` 默认选择真机适配器。
首次调试建议复制配置并暂时设置 `robot.enabled: false`、`operator.hardware_enabled:
false` 运行仿真，按照网络 → 相机 → 手 → 机械臂的顺序逐项使能。

## Episode 人工控制

设备 B 启动完成后，三路相机和机器人控制保持运行，但默认不写数据。在
`robot_main` 所在终端输入：

```text
start
start Pick up the red block
stop
discard
status
quit
```

- `start [可选任务描述]`：开始一个新 episode。
- `stop`：停止当前 episode，等待三路视频、Parquet 和元数据保存完成。
- `discard`：丢弃当前 episode，不增加 episode 编号。
- `status`：输出当前状态、帧数和已有 episode 数。
- `quit`：保存正在录制的 episode（由 `save_on_shutdown` 控制）并退出。

保存期间状态为 `saving`，此时新的 `start` 会被拒绝，防止上一段视频编码尚未完成
就混入下一段。`episode.min_frames` 可阻止误触产生过短 episode。

视觉 feature 的 dtype 已设置为 `video`，最终数据位于 LeRobot v3 的 `videos/`
目录并使用 MP4。支持 `streaming_encoding` 的新版 LeRobot 会在采集时直接编码；
LeRobot 0.4 会先暂存 PNG，在 `stop` 时编码为 MP4，最终训练数据仍是视频格式。

再次启动时，程序以 `meta/info.json` 判断是否为有效数据集。新版调用 `resume()`；
LeRobot 0.4 使用其构造器续写。已有 fps 或 feature schema 与当前配置不一致时会拒绝
追加，避免静默破坏数据集。相对 `dataset.root` 固定解析到 `teleop_collect` 目录，
不会因启动时工作目录不同而写入另一个位置。

## 多进程与数据流

```text
设备 A: 左右MANUS/VIVE采集 -> MANUS双手7维重定向 -> 同一seq/mono_ns -> UDP
设备 B 父进程: 信号处理、子进程监督、统一退出
        |
        +-- 控制进程: UDP/时钟映射 -> 100 Hz重定向/安全门 -> 双Piper/双Aerohand
        |                                |
        |                        有界IPC（仅小型状态/动作）
        |                                |
        +-- 采集进程: 三路相机 -> 时间缓冲 -> 30 Hz组帧 -> LeRobot编码/写盘
```

设备 B 固定使用 `spawn`，控制进程不会加载相机或 LeRobot 编码依赖。原始图像不跨
进程传输；IPC 只传递遥操作、机器人状态和最终动作。队列溢出时丢弃旧样本并计数，
不反压控制循环。原始源时间、映射时间、选择样本的 lag/valid、网络序列号均进入诊断特征。

## 已实现的真机适配器

- `operator_hardware.py`：通过 MANUS ZMQ 读取双手 frame，并复用参考代码的
  `_retarget_from_ergonomics_two_hands`、限位/滤波和 `to_compact_7dof`，在设备 A
  生成左右各7维 Aerohand 指令；同时通过 `pysurvive` 读取两只 VIVE Tracker。
- `retarget.py`：设备 B 直接使用 A 发来的7维手指令，只把 VIVE 相对位移映射到
  左右 Piper 基座坐标系。
- `hardware_adapters.py`：通过 `pyAgxArm` 控制双 Piper，通过 `aero_open_sdk`
  控制双 Aerohand；左右腕相机复用 TechNexion 接口，全景相机使用 Intel
  RealSense 的 RGB8 彩色流（不启用深度流）。
- CAN 和串口下发各有容量为 1 的最新值工作队列；硬件 SDK 卡顿不会阻塞控制线程。

正式采集前需分别完成左右 VIVE→机器人基座外参、左右 MANUS 手型、两台 Piper
TCP 和左右工作空间标定。

设备 A 启动时强制加载 `manus.calibration_file` 中该操作者的左右拇指 CMC 标定；
缺失或不完整时拒绝启动。VIVE 当前默认只跟随位置并使用固定末端姿态。

## LeRobot 双侧字段

- `observation.state`：左臂/左手在前，右臂/右手在后；单手 7 DOF，共 38 维。
- `action`：与 state 使用完全相同的左右顺序，共 38 维。
- `observation.images.scene`：共享全景相机。
- `observation.images.wrist_left`：左腕相机。
- `observation.images.wrist_right`：右腕相机。
- `diagnostics.safety_flags`：两维，依次为左、右安全标志。

## Intel RealSense 全景相机

设备 B 的 `cameras.scene` 使用 `driver: realsense`。单台 RealSense 时可将
`serial` 留空；连接多台时必须填写目标相机序列号。当前只配置 `rs.stream.color`
和 `rs.format.rgb8`，不会采集或写入深度图。安装方式：

```powershell
python -m pip install pyrealsense2
```

彩色帧在到达设备 B 后立即使用 `time.perf_counter_ns()` 打时间戳，以便与腕部相机、
机器人状态和控制动作在同一单调时间轴上对齐。
