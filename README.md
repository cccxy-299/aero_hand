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

设备 B 启动时，control 子进程会创建并连接双 Piper/双 Aerohand，完成双侧通信和
状态检查后立即确保两台机械臂均为 `disable`。CAN/串口会跨 episode 保持连接；
三路相机仍保持关闭，控制循环也不会启动。在
`robot_main` 所在终端输入：

```text
home
start
start Pick up the red block
stop
discard
status
quit
```

- `home`：不启动相机和 episode；依次低速回零左右 Piper、回零双 Aerohand，
  校验成功后再次 disable 双臂。
- `start [可选任务描述]`：开始一个新 episode。
- `stop`：停止当前 episode，等待三路视频、Parquet 和元数据保存完成。
- `discard`：丢弃当前 episode，不增加 episode 编号。
- `status`：输出当前状态、帧数和已有 episode 数。
- `quit`：保存正在录制的 episode（由 `save_on_shutdown` 控制）并退出。

左右 `robot.<side>.home_pose` 必须由操作者按真实安装位置标定为6维关节角。
收到 `start` 后，系统先以 `home_speed_percent` 依次执行右臂、左臂
`move_j(home_pose)`，并使用真实关节反馈校验到位；任一侧失败都会 disable 双臂并
拒绝启动。两侧完成后保持 enable，再启动三路相机，所以回 home 的轨迹不会写进
episode，也不会在遥操作开始前重复 disable/enable。相机启动期间若收到
`stop`/`discard` 或启动失败，系统会立即 disable 双臂。

相机全部就绪后，系统在保持 enable 的状态下再次检查硬件，并直接启动状态、控制和
组帧循环。每个 episode 都以此时读取的真实法兰位姿重新建立 VIVE 参考点和安全
限速历史，而不是直接把关节 `home_pose` 当作笛卡尔控制目标。`stop`/`discard`
会立即停止命令工作线程并 disable 双臂，
灵巧手保持最后位置；相机随之关闭。CAN/串口和硬件 SDK 实例只在 `robot_main`
退出时最终释放。下一次 `start` 复用硬件会话，但不会复用旧 episode 的命令、图像、
state、action 或 VIVE 参考。

`control_hz` 是安全计算频率，不等于真实硬件命令频率。双臂 `move_p` 默认通过
`robot.arm_command_hz: 30` 限频，等待期间只保留最新目标；机械臂反馈通过
`robot.arm_feedback_hz: 30` 限制真实 SDK 读取频率，并采用命令优先的缓存读取，
避免100Hz状态线程与 `move_p` 争抢同侧 SDK；双手默认通过
`robot.hand_command_hz: 60` 限频。控制线程不会等待硬件 SDK。运行日志中的
`hardware.workers` 会输出 `effective_hz`、`dropped` 和 `coalesced`，用于判断
命令是否过密或 SDK 调用是否变慢。

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
- `camera_frame_errors_*` 与 `camera_error_ratio_*`：V4L2/RealSense 取帧失败、
  空帧及异常帧的累计数量和占总取帧结果的比例；`camera_<reason>_*` 会进一步
  区分错误原因。

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

设备 B 固定使用 `spawn`，控制进程不会加载相机或 LeRobot 编码依赖。recorder、
control 和三路相机服务均由设备 B 主进程直接创建，彼此为同级进程，不再由 recorder
嵌套创建相机孙进程。idle 时相机服务只等待 session 事件，不打开相机；start 后按
配置的 `startup_delay_ms` 错峰连接。三路相机后端
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
  控制双 Aerohand；左右腕相机通过 OpenCV/V4L2 采集，全景相机使用 Intel
  RealSense 的 RGB8 彩色流（不启用深度流）。
- CAN 和串口下发各有容量为 1 的最新值工作队列；硬件 SDK 卡顿不会阻塞控制线程。

正式采集前需分别完成左右 VIVE→机器人基座外参、左右 MANUS 手型、两台 Piper
TCP 和左右工作空间标定。

设备 A 启动时强制加载 `manus.calibration_file` 中该操作者的左右拇指 CMC 标定；
缺失或不完整时拒绝启动。VIVE 默认只跟随位置；每个 episode 在 `start` 时读取并
锁定左右机械臂各自的真实法兰姿态（`orientation_mode: current_on_start`），首帧
不会跳转到配置中的另一套姿态。只有经过可达性验证后，才应显式改为
`orientation_mode: configured_fixed` 并使用 `fixed_orientation`。

相对位置控制使用可标定的3x3轴交换/翻转矩阵：

```text
Piper目标位置 = start时真实法兰位置 + VIVE相对位移映射
robot_delta = vive_to_robot_matrix @ vive_delta * vive_scale
```

旧工程使用 OpenVR，硬编码为 `VIVE Y → Piper Z`；当前设备 A 使用 pysurvive，
真机观察已证明该假设不成立。当前配置交换了旧映射的 Piper Y/Z 输出，使可视化
Y 位移对应 Piper Y。首次真机验证应把 `vive_scale` 临时调小，并分别只沿可视化
X、Y、Z 单轴移动；控制日志的 `teleop_mapping` 会同时显示输入和输出增量。

左右 VIVE 参考必须由同一个有效网络包原子建立。任一侧 VIVE、MANUS 或 `valid`
无效时不会建立/修改参考，也不会向双臂和双手发送新目标；数据恢复后继续使用原参考。

## LeRobot 双侧字段

- `observation.state`：左臂/左手在前，右臂/右手在后；单手 7 DOF，共 38 维。
- `action`：与 state 使用完全相同的左右顺序，共 38 维。
- `observation.images.scene`：共享全景相机。
- `observation.images.wrist_left`：左腕相机。
- `observation.images.wrist_right`：右腕相机。
- `diagnostics.safety_flags`：两维，依次为左、右安全标志。

## OpenCV/V4L2 双腕相机

`opencv_camera.py` 是不依赖 `pyvizionsdk` 的同步相机类，已经通过
`OpenCVWristCamera` 接入主 pipeline。正式采集为左右腕相机各一个 spawn 子进程，
每个进程直接调用 `VideoCapture.read()`，在进程内完成 MJPG 解码与 BGR→RGB
转换，再通过各自的一帧共享内存发布给 recorder。

先在 Linux 上确认设备节点和相机支持的格式：

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video6 --list-formats-ext
v4l2-ctl -d /dev/video8 --list-formats-ext
ls -l /dev/v4l/by-id/
```

当前真机配置使用 `/dev/video6` 和 `/dev/video8`，以
`640x480 MJPG 30 FPS` 打开：

```bash
# 两只相机分别运行60秒
python opencv_camera_concurrency_test.py --mode single-left \
  --left-device /dev/video6 --duration-s 60
python opencv_camera_concurrency_test.py --mode single-right \
  --right-device /dev/video8 --duration-s 60

# 单进程、每只相机一个采集线程
python opencv_camera_concurrency_test.py --mode dual-thread \
  --left-device /dev/video6 --right-device /dev/video8 --duration-s 60

# 每只相机一个独立 spawn 子进程
python opencv_camera_concurrency_test.py --mode dual-process \
  --left-device /dev/video6 --right-device /dev/video8 --duration-s 60
```

设备节点不同时显式传入；正式配置建议使用 `/dev/v4l/by-id/...` 稳定路径：

```bash
python opencv_camera_concurrency_test.py \
  --mode dual-process \
  --left-device /dev/video6 \
  --right-device /dev/video8 \
  --width 640 \
  --height 480 \
  --fps 30 \
  --fourcc MJPG \
  --duration-s 120
```

每秒会输出 JSON `progress`，结束时输出 `summary` 和 `verdict`。重点检查
`actual_properties` 中实际生效的 FourCC、分辨率和 FPS，以及 `source_fps`、
`error_ratio`、`max_gap_ms`、`last_frame_age_ms` 和 `unique_ratio`。如果线程模式
正常而进程模式异常，才考虑单进程内双线程；当前真机两种模式都通过，因此正式系统
采用双进程隔离。若实际分辨率不符合要求，程序默认直接报错；调试其他模式时可临时加
`--no-strict-resolution`。

部分 OpenCV V4L2 backend 不支持 `CAP_PROP_READ_TIMEOUT_MSEC`。正式 recorder
会独立监控共享内存中的最新帧时间戳；超过 `failure_timeout_ms` 未更新时，即使相机
子进程仍存活，也会判定 `VideoCapture.read()` 可能卡死并终止当前采集会话。stop
阶段若子进程不能自行退出，会依次执行 `terminate` 和 `kill`，不会拖住控制进程。
recorder 的每秒 `metrics.cameras` 还会输出各路 `seq`、`last_frame_age_ms` 和
`phase`，用于区分阻塞发生在 `read`、共享内存发布还是设备释放阶段。

## Intel RealSense 全景相机

设备 B 的 `cameras.scene` 使用 `driver: realsense`。单台 RealSense 时可将
`serial` 留空；连接多台时必须填写目标相机序列号。当前只配置 `rs.stream.color`
和 `rs.format.rgb8`，不会采集或写入深度图。安装方式：

```powershell
python -m pip install pyrealsense2
```

彩色帧在到达设备 B 后立即使用 `time.perf_counter_ns()` 打时间戳，以便与腕部相机、
机器人状态和控制动作在同一单调时间轴上对齐。
