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

设备 B 启动完成后处于 `idle`：机器人、CAN/串口和三路相机均未连接，控制循环与
相机进程也未启动；仅保留父进程、UDP/时钟同步监听和阻塞式命令队列。在
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

`start` 会先启动并确认三路独立相机进程，再连接双 Piper/双 Aerohand 并启动状态、
控制和组帧循环。`stop`/`discard` 会立即停止机器人控制、关闭相机进程并释放设备；
`stop` 随后保存 episode，`discard` 则清理临时帧。下一次 `start` 会建立全新的硬件
会话和时间缓冲，旧 episode 的图像、state 或 action 不会被复用。

保存期间状态为 `saving`，此时新的 `start` 会被拒绝，防止上一段视频编码尚未完成
就混入下一段。`episode.min_frames` 可阻止误触产生过短 episode。

视觉 feature 的 dtype 已设置为 `video`，最终数据位于 LeRobot v3 的 `videos/`
目录并使用 MP4。支持 `streaming_encoding` 的新版 LeRobot 会在采集时直接编码；
LeRobot 0.4 会先暂存 PNG，在 `stop` 时编码为 MP4，最终训练数据仍是视频格式。
三路 SVT-AV1 编码器默认各限制为2个线程，避免编码器按CPU核心数自动扩张后与
100 Hz控制进程争抢调度；编码仍在记录路径中，不会反向阻塞控制下发。

再次启动时，程序以 `meta/info.json` 判断是否为有效数据集。新版调用 `resume()`；
LeRobot 0.4 使用其构造器续写。已有 fps 或 feature schema 与当前配置不一致时会拒绝
追加，避免静默破坏数据集。相对 `dataset.root` 固定解析到 `teleop_collect` 目录，
不会因启动时工作目录不同而写入另一个位置。

每次 `episode_saved` 日志都会输出完整帧链路诊断：

- `camera_scene/wrist_left/wrist_right`：每路相机实际进入时间缓冲的唯一帧数。
- `frame_attempts` 与 `incomplete_frames`：30 Hz 组帧尝试数和输入尚未齐全的次数。
- `grouped_frames`、`writer_frames`、`dataset_frame_delta`：组帧、Writer 缓冲及
  LeRobot 元数据实际增加的帧数；三者不一致时程序直接报错并停止。
- `writer_drops`：Writer 队列拥塞导致的丢帧数。
- `source_fps_*`：相机真正产生唯一画面的速率，不是配置中的轮询频率。
- `unique_ratio_*`：写入 frame 中实际使用的新相机画面比例；其余为维持真实时间轴而
  重复使用的上一帧。
- `camera_frame_errors_*` 与 `camera_error_ratio_*`：SDK失败、空MJPEG包及损坏帧
  的累计数量和占总取帧结果的比例；`camera_<reason>_*` 会进一步区分错误原因。

默认质量门限为每路相机至少 24 FPS、唯一画面比例至少 0.70、坏帧比例不超过
0.10。任一路不达标时输出
`episode_quality_failed` 并丢弃该 episode，避免把“30 FPS 容器中只有几张画面”的
卡顿视频混入训练集。

## 多进程与数据流

```text
设备 A: 左右MANUS/VIVE采集 -> MANUS双手7维重定向 -> 同一seq/mono_ns -> UDP
设备 B 父进程: 信号处理、子进程监督、统一退出
        |
        +-- 控制进程: UDP/时钟映射 -> 100 Hz重定向/安全门 -> 双Piper/双Aerohand
        |                                |
        |                        有界IPC（仅小型状态/动作）
        |                                |
        +-- 记录进程: 共享内存最新帧 -> 时间缓冲 -> 30 Hz组帧 -> LeRobot编码/写盘
        +-- scene相机进程      -> 一帧共享内存
        +-- wrist_left相机进程 -> 一帧共享内存
        +-- wrist_right相机进程-> 一帧共享内存
```

设备 B 固定使用 `spawn`，控制进程不会加载相机或 LeRobot 编码依赖。三路相机 SDK
和 MJPEG 解码各自在独立 OS 进程中运行，即使某个 C 扩展持有 Python GIL，也不会
阻塞其他相机或 30 Hz 组帧循环。图像不经过 multiprocessing Queue/pickle，而是每路
只发布一帧共享内存，记录进程按周期复制最新帧。原始图像不做跨进程序列化传输；
Queue IPC 只传递遥操作、机器人状态和最终动作。队列溢出时丢弃旧样本并计数，
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

## TechNexion 腕部相机

腕部相机默认使用 `format_idx: null`，连接时从设备报告的 MJPG 模式中自动选择目标
分辨率且最接近 `camera_hz` 的格式。`strict_fps: true` 会在设备只能提供低帧率模式时
拒绝开始 episode，而不是以 30 Hz 反复读取同一张画面。若必须手动指定
`format_idx`，应先查看启动日志中的完整格式列表，并确认选中模式为 30 FPS。

VizionSDK 的单次失败返回码、空 MJPEG buffer 或损坏 JPEG 只会丢弃当前帧，上一张
有效共享内存画面保持可用；恢复后继续当前 episode。只有连续
`camera_failure_timeout_ms`（默认 1500 ms）没有任何有效图像才会停止系统。这样既
不会因一个 USB 瞬时坏包误停机，也不会在相机真正离线后继续控制和保存坏数据。

### 双相机并发诊断

`camera_concurrency_test.py` 完全不启动机器人、LeRobot Writer、RealSense或视频
编码器，只测试两台 TechNexion 的 VizionSDK 取流。真机排查时依次执行：

```bash
# 1. 各自单测30秒
python camera_concurrency_test.py --mode single-left --duration-s 30
python camera_concurrency_test.py --mode single-right --duration-s 30

# 2. 同一进程、两个线程（接近旧Demo结构）
python camera_concurrency_test.py --mode dual-thread --duration-s 30

# 3. 两个独立进程（与正式采集结构一致）
python camera_concurrency_test.py --mode dual-process --duration-s 30

# 4. 模拟正式程序的30Hz轮询；默认 poll-hz=0 会尽快持续排空SDK
python camera_concurrency_test.py --mode dual-process --poll-hz 30 --duration-s 30

# 5. 扫描左/右相机报告的每个MJPG format
python camera_concurrency_test.py --mode sweep-left --sweep-duration-s 10
python camera_concurrency_test.py --mode sweep-right --sweep-duration-s 10
```

若已从启动日志确认稳定的格式索引，可显式比较，例如：

```bash
python camera_concurrency_test.py \
  --mode dual-process \
  --left-format-idx 1 \
  --right-format-idx 1 \
  --no-strict-fps \
  --timeout-ms 1000 \
  --duration-s 60
```

每秒输出一条JSON `progress`，结束输出 `summary` 和 `verdict`。重点查看
`sdk_timeout`、`source_fps`、`error_ratio`、`max_gap_ms` 和
`max_consecutive_errors`。单测均正常但 `dual-thread` 失败，优先检查SDK双设备或USB；
`dual-thread` 正常而 `dual-process` 失败，优先检查VizionSDK跨进程兼容性；
默认持续取流正常但 `--poll-hz 30` 失败，说明SDK需要持续排空，正式相机进程不应额外
限频；只有部分 format 稳定时，应在 `robot.yaml` 固定各自已验证的索引。

## Intel RealSense 全景相机

设备 B 的 `cameras.scene` 使用 `driver: realsense`。单台 RealSense 时可将
`serial` 留空；连接多台时必须填写目标相机序列号。当前只配置 `rs.stream.color`
和 `rs.format.rgb8`，不会采集或写入深度图。安装方式：

```powershell
python -m pip install pyrealsense2
```

彩色帧在到达设备 B 后立即使用 `time.perf_counter_ns()` 打时间戳，以便与腕部相机、
机器人状态和控制动作在同一单调时间轴上对齐。
