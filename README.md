# CS2 Vision C++ 运行时

本仓库包含 CS2 视觉项目的 C++ 运行时。Python 项目负责训练和导出 YOLO 模型；本运行时
加载导出的模型，从视频文件或 DXGI 桌面复制读取画面，检测目标，融合身体/头部检测结果，
跟踪选中的目标，对瞄准点进行滤波和预测，最后通过 RP2350 HID 桥接器 SDK 规划有边界的
相对鼠标移动。

主要产物是 `vision_runtime.dll`，它导出
`include/vision_analyzer/vision_runtime_c_api.h` 中声明的稳定 C API。
`vision_analyzer.exe` 是可选的诊断 CLI，用于验证模型、视频、DXGI、标定和端到端运行时。
项目不构建图形界面，也不依赖图形界面。

构建产物：

```text
vision_runtime.dll  主要运行时库和 C API 实现
vision_runtime.lib  供原生调用方使用的 MSVC 导入库
vision_analyzer.exe 可选诊断 CLI
```

## 环境要求

Windows 构建环境：

```text
Visual Studio 2022 Build Tools with MSVC
xmake
CMake 3.20+（使用 CMake 构建时）
Git
```

运行时和模型要求：

```text
OpenCV DNN，由 xmake 解析依赖
导出的 YOLO ONNX 模型
实时 HID 模式需要匹配的 *.schema.json 文件
真实 HID 输出需要 RP2350 HID Bridge C++ SDK
```

真实 HID 模式要求板卡固件和 C++ SDK 都使用协议 v2。运行时打开 COM 口后会依次执行
`ping()`、`info()` 和 `caps()` 三项只读健康检查，并要求固件报告鼠标、可靠重试、
安全租约和取消能力；检查失败时不会进入标定或物理输出。协议 v2 SDK 每 500 ms 发送
一次心跳，固件在心跳、DTR 或 USB 连接中断后使用两秒安全租约释放保持中的输入。

GTX 1080 Ti 生产加速环境：

```text
ONNX Runtime GPU 1.17.x
TensorRT 8.6.x
CUDA 11.8
cuDNN 8.9.x
GeForce GTX 1080 Ti / SM 6.1
FP32
```

默认后端为 `ort-tensorrt`。ONNX Runtime、CUDA 或 TensorRT 不可用时，仍可明确指定
`opencv-onnx` 作为 CPU 兜底后端。

## 使用 xmake 构建

在本仓库中执行：

```powershell
xmake f -m release
xmake
xmake run vision_analyzer_tests
xmake run vision_runtime_c_api_tests
```

在父仓库中执行：

```powershell
cd tools\cpp_analyzer
xmake f -m release
xmake
xmake run vision_analyzer_tests
xmake run vision_runtime_c_api_tests
```

默认自动支持父仓库的目录布局。如果 SDK 位于其他位置，请显式传入路径：

```powershell
xmake f -m release --hid_sdk_root=D:\project\cs2-vision-trainer\tools\rp2350_hid_bridge_cpp
xmake
```

启用 ONNX Runtime 后端：

```powershell
$env:ONNXRUNTIME_ROOT = "D:\SDK\onnxruntime-win-x64-gpu"
xmake f -m release --onnxruntime_root=$env:ONNXRUNTIME_ROOT --hid_sdk_root=..\rp2350_hid_bridge_cpp
xmake
```

## GTX 1080 Ti 生产运行时

同一进程内必须使用相互匹配的版本。不要把生产 DLL 目录与 CUDA 12、cuDNN 9、
TensorRT 11 或更新的 ONNX Runtime Provider DLL 混用。

一种可用的目录布局如下：

```text
D:\runtime\sm61\onnxruntime-win-x64-gpu-1.17.3
D:\runtime\sm61\TensorRT-8.6.1.6\lib
D:\runtime\sm61\cudnn-8.9\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
```

只为构建或运行生产包的当前进程配置环境：

```powershell
$env:ONNXRUNTIME_ROOT='D:\runtime\sm61\onnxruntime-win-x64-gpu-1.17.3'
$env:PATH='D:\runtime\sm61\TensorRT-8.6.1.6\lib;D:\runtime\sm61\cudnn-8.9\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin;' + $env:PATH
xmake f -c -m release --onnxruntime_root=$env:ONNXRUNTIME_ROOT
xmake
```

第一次打开模型时，会在进程工作目录的 `ort-trt-cache-sm61-fp32` 中构建 FP32
引擎，后续打开会复用该引擎。更换模型、GPU、ONNX Runtime、TensorRT、CUDA 或
cuDNN 后，需要清空该目录。

TensorRT 8.6 不支持 RTX 5060（SM 12.0），因此最终的 Provider 和 FPS 验证必须在
GTX 1080 Ti 主机上运行。RTX 5060 开发机仍可编译源码并运行与硬件无关的测试。

主要 DLL 产物：

```text
build\windows\x64\release\vision_runtime.dll
build\windows\x64\release\vision_runtime.lib
```

可选诊断程序生成在：

```text
build\windows\x64\release\vision_analyzer.exe
```

只构建可复用 DLL 及其 API 测试：

```powershell
xmake build vision_runtime
xmake run vision_runtime_c_api_tests
```

尽可能使用 `xmake run`。它会把配置的 ONNX Runtime `lib` 目录加入进程的 DLL
搜索路径。CUDA、cuDNN 和 TensorRT 仍从上面为当前进程设置的匹配 `PATH` 中加载。

## 使用 CMake 构建

CMake 会生成相同的核心库、DLL、CLI 和测试目标，并且不需要任何 UI 框架：

```powershell
cmake -S . -B build-cmake -A x64
cmake --build build-cmake --config Release
ctest --test-dir build-cmake -C Release --output-on-failure
```

CMake 产物生成在：

```text
build-cmake\Release\vision_runtime.dll
build-cmake\Release\vision_runtime.lib
build-cmake\Release\vision_analyzer.exe
```

如果这些可选组件没有安装在默认的同级目录中，请在配置 CMake 前设置
`ONNXRUNTIME_ROOT` 或 `RP2350_HID_BRIDGE_SDK`。

## 模型契约

运行时要求使用 Python 项目导出的 YOLO ONNX 模型。类别顺序必须为：

```text
0 ct_body
1 ct_head
2 t_body
3 t_head
```

实时 HID 输出要求 ONNX 文件旁存在生成的模型结构说明 JSON：

```text
best.onnx
best.onnx.schema.json
```

为了快速测试输入，试运行可以在没有模型结构说明文件时继续。实时模式会把文件缺失或
不匹配视为错误。

## 验证输入

使用 `xmake run` 时请传入绝对路径，因为 xmake 可能从构建目录启动程序。

验证视频文件：

```powershell
xmake run vision_analyzer --video D:\project\cs2-vision-trainer\videos\02.mp4 --verify-input
```

预期输出中的宽度、高度和 RGB 平均值均不为零。

列出并探测 DXGI 输出：

```powershell
xmake run vision_analyzer --list-dxgi-outputs
xmake run vision_analyzer --probe-dxgi-outputs
```

验证一个 DXGI 输出：

```powershell
xmake run vision_analyzer --input dxgi --dxgi-adapter 0 --dxgi-output 0 --verify-input --dxgi-debug
```

请选择 `duplicate_output=0x0` 的适配器/输出。在混合 GPU 系统上，有效输出通常属于
实际连接显示器的适配器，不一定是负责 3D 渲染的高性能 GPU。

如有需要，可在推理前裁剪实时输入：

```powershell
--dxgi-roi X Y W H
```

ROI 坐标相对于选中的 DXGI 输出，目标偏移量以 ROI 中心为基准。配置 ROI 后，程序会在
CPU 回读前将其复制到与 ROI 同尺寸的 D3D11 暂存纹理中，而不是先转换整个桌面再裁剪。

## 离线试运行

试运行会加载模型并执行检测和规划，但不会发送 HID 命令：

```powershell
xmake run vision_analyzer `
  --backend opencv-onnx `
  --model D:\project\cs2-vision-trainer\runs\detect\train\weights\best.onnx `
  --video D:\project\cs2-vision-trainer\videos\02.mp4 `
  --player-side unknown `
  --dry-run `
  --preview `
  --status-every 30 `
  --action-log actions.txt
```

动作日志列：

```text
frame timestamp_ms target dx dy click lock distance offset_x offset_y
```

字段说明：

```text
target=1  已选中目标
dx/dy     规划的相对鼠标移动
click=1   正常模式会发送左键；试运行只记录日志
lock=1    目标锁定足够稳定，可以评估是否满足开火候选条件
```

身体兜底检测可以引导移动，但只有头部检测可以成为左键候选。

## 实时 HID 模式

先在不加载模型的情况下验证板卡：

```powershell
xmake run vision_analyzer --hid-port COM3 --test-hid-move 300 0
```

然后运行实时 DXGI 移动，但不点击：

```powershell
xmake run vision_analyzer `
  --backend ort-tensorrt `
  --tensorrt-cache-path ort-trt-cache-sm61-fp32 `
  --model D:\project\cs2-vision-trainer\runs\detect\train\weights\best.onnx `
  --schema D:\project\cs2-vision-trainer\runs\detect\train\weights\best.onnx.schema.json `
  --input dxgi `
  --dxgi-output 0 `
  --dxgi-roi 640 220 640 640 `
  --player-side ct `
  --hid-port COM3 `
  --hid-gain 1.0 `
  --hid-max-step 120 `
  --output-enabled `
  --preview
```

只有完成移动标定后才启用左键输出：

```powershell
xmake run vision_analyzer `
  --backend ort-tensorrt `
  --tensorrt-cache-path ort-trt-cache-sm61-fp32 `
  --model D:\project\cs2-vision-trainer\runs\detect\train\weights\best.onnx `
  --schema D:\project\cs2-vision-trainer\runs\detect\train\weights\best.onnx.schema.json `
  --input dxgi `
  --dxgi-output 0 `
  --dxgi-roi 640 220 640 640 `
  --player-side ct `
  --hid-port COM3 `
  --hid-click `
  --hid-click-cooldown 6 `
  --output-enabled
```

输出默认关闭。DLL 宿主只在自身热键或武装条件生效时调用
`va_set_output_enabled(runtime, 1)`，并通过 `va_set_output_enabled(runtime, 0)`
立即停止输出。输出关闭期间，检测和动作返回值仍会继续更新。

实时 HID 模式要求使用：

```text
--player-side ct
```

或者：

```text
--player-side t
```

试运行允许使用 `unknown`，实时 HID 输出则不允许。

## HID 标定

标定会通过板卡发送受控的相对鼠标移动，通过 DXGI 观察画面位移，并写出拟合后的配置片段：

启动标定会先为 X、Y 两个方向分别寻找可测量的探测量；低灵敏度账号在这个阶段可能使用
最高 2048 counts。2048 只用于启动标定，标定通过后的正常输出仍固定以
`max_step=120` 为上限。每次探测后都会发送数值完全相反的移动让视角归位；如果达到
2048 counts 仍无法得到可靠画面位移，标定会拒绝该配置，而不会带着无效结果继续运行。

```powershell
xmake run vision_analyzer `
  --calibrate-hid `
  --hid-port COM3 `
  --dxgi-output 0 `
  --calibration-step 40 `
  --calibration-noise-samples 2 `
  --calibration-output hid-calibration.txt `
  --calibration-config-output hid-tuned.cfg
```

检查生成的 `hid-tuned.cfg`，然后在 CLI 覆盖参数之前传入该文件：

```powershell
xmake run vision_analyzer --config hid-tuned.cfg --backend ort-tensorrt --model D:\project\cs2-vision-trainer\runs\detect\train\weights\best.onnx --input dxgi --dxgi-output 0 --player-side ct --hid-port COM3 --output-enabled
```

## 后端

```text
opencv-onnx   通过 OpenCV DNN 显式使用 CPU ONNX 兜底后端。
opencv-cuda   要求 OpenCV 构建时启用 CUDA DNN 支持。
ort-cuda      ONNX Runtime CUDA Execution Provider。
ort-tensorrt  默认的 GTX 1080 Ti TensorRT EP，CUDA 子图作为兜底。
tensorrt      预留给原生 TensorRT C++ 构建。
```

如果没有配置 ONNX Runtime，ORT 后端会在运行时报告不可用，OpenCV 后端仍可使用。

## 算法说明

- 按类别执行的 NMS 可避免相互重叠的头部和身体候选框彼此抑制。
- 跟踪前会先关联同一阵营的身体/头部检测结果。
- 优先使用已匹配的头部检测；未匹配的身体检测仍可在身体框顶部附近作为兜底锚点。
- 跟踪 ID 使用 IoU 和锚点距离进行匹配。
- 目标选择优先考虑稳定、距离近且置信度高的目标，并通过切换惩罚减少抖动。
- 目标点使用二维卡尔曼状态并进行延迟补偿预测。
- `--player-side ct` 以 `t_body` 和 `t_head` 为目标。
- `--player-side t` 以 `ct_body` 和 `ct_head` 为目标。
- 只有头部类别可以触发 `--hid-click`。

## Windows 指针设置

RP2350 固件生成标准的相对 USB HID 鼠标报告，不应用指针曲线。标定通过
`SystemParametersInfo` 读取并输出 Windows 指针阈值、加速状态和指针速度，但不会修改
这些设置。

如果目标应用使用普通 Windows 指针移动，指针速度和“提高指针精确度”会影响移动；如果
应用使用 Raw Input，移动通常主要由 HID 计数和应用内灵敏度决定。

请在实际目标机器上调整以下参数：

```text
--hid-gain
--hid-max-step
--hid-deadzone
```

## CLI 帮助

```powershell
xmake run vision_analyzer --help
```

## C API DLL

`vision_runtime.dll` 导出稳定的 C ABI，其声明位于：

```text
include\vision_analyzer\vision_runtime_c_api.h
```

该 API 使用不透明的 `VaRuntime*` 句柄和普通 C 结构体：

```c
VaRuntime* runtime = va_create();
va_set_model(runtime, "best.onnx");
va_set_schema(runtime, "best.onnx.schema.json");
va_set_backend(runtime, "ort-tensorrt");
va_set_tensorrt_cache_path(runtime, "ort-trt-cache-sm61-fp32");
va_set_dxgi_roi(runtime, 640, 220, 640, 640);
va_set_player_side(runtime, "ct");
va_set_hid_port(runtime, "COM3");
VaHidCalibrationProfile calibration;
if (va_calibrate_hid(runtime, 0, 0, &calibration) != 0) {
    fprintf(stderr, "%s\n", va_last_error(runtime));
    va_destroy(runtime);
    return 1;
}
va_set_fire_policy(runtime, 1, 0.35f, 0.45f, 3);
if (va_open_dxgi(runtime, 0, 0, 0) != 0) {
    fprintf(stderr, "%s\n", va_last_error(runtime));
    va_destroy(runtime);
    return 1;
}
va_set_output_enabled(runtime, 1);
va_set_fire_enabled(runtime, 1);

VaRuntimeAction action;
while (va_process_next(runtime, &action) == 1) {
    printf("%d %d %d\n", action.frame_index, action.dx, action.dy);
}

va_set_fire_enabled(runtime, 0);
va_set_output_enabled(runtime, 0);
va_stop_all(runtime);
va_destroy(runtime);
```

返回码：

```text
0   配置/打开/关闭调用成功
1   va_process_next 已处理一帧
0   va_process_next 已到达流末尾
-1  发生错误；读取 va_last_error(runtime)
```

父 Python 仓库通过 `cs2_vision_runtime.VisionRuntime` 封装此 DLL。任何需要实时 HID
输出的包装层都必须绑定并调用 `va_set_output_enabled`；旧包装层即使继续通过
`va_process_next` 获取规划动作，也会安全地保持未武装状态。

## GTX 1080 Ti 便携包

构建固定版本的 ORT 1.17.3 / TensorRT 8.6.1.6 / CUDA 11.8 便携包时，需要显式指定
与之匹配的外层 Python 工作区：

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass `
  -File packaging\sm61\build-portable-package.ps1 `
  -PythonProjectRoot 'C:\path\to\cs2-vision-trainer' `
  -TensorRtArchive 'C:\path\to\TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8.zip'
```

压缩包包含只依赖 Python 标准库的包装层和实时示例，但不包含 Python 解释器。所有一键
诊断均保持为试运行，绝不会调用输出武装 API。
