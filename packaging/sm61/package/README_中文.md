# GTX 1080 Ti 便携运行环境

本目录固定用于一套环境：GTX 1080 Ti、ONNX Runtime GPU 1.17.3、CUDA
11.8 运行库、cuDNN 8.9.7、TensorRT 8.6.1.6、FP32。

## 为什么不用安装完整 CUDA Toolkit

程序已经编译完成，生产机不需要 `nvcc`、头文件、静态库、Nsight、Visual
Studio 插件或 CUDA 示例。本包把程序真正需要的 CUDA/cuDNN/TensorRT DLL
放在自身 `runtime` 目录，通过子进程 PATH 加载。它不会修改系统 PATH、注册表
或 Windows 系统目录，也不需要管理员权限。

`nvidia-smi` 显示的 `CUDA Version` 是驱动最高兼容级别，不代表已安装同版本
CUDA Toolkit。546.33 驱动足以加载本包的 CUDA 11.8 私有运行库。

## 第一次测试

1. 完整解压 ZIP，不要直接在压缩软件预览窗口里运行。
2. 双击 `一键检查并测试.cmd`。
3. 第一次打开模型时，TensorRT 会在 GTX 1080 Ti 上生成 FP32 引擎，可能需要
   几分钟；窗口没有立即输出不代表卡死。
4. 脚本随后第二次打开模型，用于确认缓存可以复用。
5. 看到 `PASS` 后，把 `logs` 中的结果保留用于 FPS 对比。

一键测试始终使用试运行模式，配置中的 `output_enabled=false` 也不会被脚本覆盖，
因此不会向 RP2350 发送移动或点击。

## RP2350 协议 v2

真实 HID 输出要求板卡已刷入与包内 DLL 匹配的协议 v2 固件。打开 COM 口时运行时会先
完成 `ping/info/caps` 三项只读健康检查，确认协议版本、鼠标、可靠重试、安全租约和
取消能力；检查失败时不会开始灵敏度标定，也不会产生物理输入。

连接保持期间，SDK 每 500 ms 发送一次心跳。进程停止、串口断开、DTR 断开或心跳
停止后，固件的两秒安全租约会释放保持中的键和鼠标按钮。包清单和组包流程都会拒绝
旧协议 DLL。`vision_runtime.dll` 和 `rp2350_hid_bridge.dll` 是一组协调版本，
因此不要只替换其中一个 DLL 而保留旧清单。

## 单独测试当前屏幕

先完成视频测试，再在本目录打开 PowerShell：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\test-dxgi.ps1 -Adapter 0 -Output 0
```

脚本会先列出并探测显示输出。如果输出 0 不支持 Desktop Duplication，请根据
日志显示的适配器/输出修改参数。DXGI 测试同样不会向 RP2350 输出。

## 日志和缓存

- `logs/`：清单检查、GPU、驱动、视频/DXGI 输出和诊断报告。
- `cache/ort-trt-sm61-fp32/`：GTX 1080 Ti 生成的 TensorRT 引擎缓存。

更换模型、GPU、ORT、CUDA、cuDNN 或 TensorRT 后必须删除旧缓存再测试。本包
不携带开发机生成的 `.engine` 文件。

## 失败时

运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\collect-diagnostics.ps1
```

把生成的 `logs\diagnostics-*.log` 发回来。报告只收集本包文件、GPU/驱动、
缓存和本包日志，不收集密码、浏览器数据或无关进程。

## 接入 DLL

两套头文件、导入库和 DLL 位于 `app`。Python 包装分别位于
`python\cs2_vision_runtime` 与 `python\rp2350_hid_bridge`；前者依赖后者，
本压缩包不捆绑 Python 解释器。

把 Python SDK 冻结进调用端 EXE、生成同级 `vision_runtime.dll`、
`rp2350_hid_bridge.dll` 和
`resources\vision-runtime` 的正式流程见
[`docs\PYTHON_RUNTIME_SDK_INTEGRATION.md`](docs/PYTHON_RUNTIME_SDK_INTEGRATION.md)。本包内
命令主要用于环境诊断和硬件验收。

宿主必须按以下顺序调用：

1. 调用端创建一个 `HidSession` 打开 RP2350 串口，并通过
   `va_attach_hid_session` 把同一个原生会话附加给视觉运行时。
2. 调用 `set_hid_calibration_path("hid-calibration.json")` 选择一个本地文件。
3. 调用 `get_hid_calibration()`；只有返回的 `valid` 为假时，才在已进入对局且
   画面稳定的场景调用 `calibrate_hid()`。
4. 设置头/身体开火策略，再打开 DXGI。
5. 分别调用 `set_output_enabled(True)` 和 `set_fire_enabled(True)`。
6. 视觉暂停时只关闭开火和移动输出，不释放调用端保持的键；只有整个控制会话
   结束或需要紧急全局释放时才调用 `hid.stop_all()`。

对应 C API 是 `va_set_hid_calibration_path`、`va_get_hid_calibration` 和现有的
`va_calibrate_hid`。第一次成功标定会先完整校验候选 profile，再原子写入文件，最后
才安装到内存。后续进程设置同一路径时会直接加载，不打开 DXGI 标定流程，也不会移动
鼠标。文件损坏、保存失败或重新标定失败都不会覆盖已经有效的旧文件和旧内存 profile。

DLL 只保存调用端指定的一个文件，不识别账号，也不决定何时重标定。调用端在自己的
设置发生变化时显式再次调用 `calibrate_hid()`；Python 示例对应使用 `--recalibrate`。

两个开关相互独立。新建运行时默认都关闭；只有 Python 明确开启后才会产生物理
移动或点击。头部优先，身体只在躯干有效区域内作为低优先级兜底。

## 运行 Python 示例

先进入 CS2 对局并让画面稳定，在本目录打开 PowerShell：

```powershell
$env:PYTHONPATH=(Resolve-Path '.\python').Path
$env:CS2_VISION_RUNTIME_DLL=(Resolve-Path '.\app\vision_runtime.dll').Path
python .\examples\runtime_live_move.py --hid-port COM4 --player-side ct --calibration-path .\hid-calibration.json --enable-live-output --click
```

把 `COM4` 改为 RP2350 的实际串口。去掉 `--click` 时只自动瞄准、不自动开火；
不加 `--enable-live-output` 时程序会在标定前退出，完全不触碰硬件。包装层会自动
加载包内 CUDA 11.8、cuDNN 8.9、TensorRT 8.6.1.6 和 MSVC 私有运行库，不需要
修改系统 PATH。按 `Ctrl+C` 会通过 `finally` 撤销开火和移动输出。

第一次运行且本地文件不存在时，DLL 会在中心 ROI 内用光流测量 X、Y 两个方向的视角
位移。标定探测和正式采样都限制在最多 120 counts，每个移出动作都会立即发送精确反向
移动归位；标定通过后的正常瞄准同样受 `max_step=120` 限制。中心 ROI 光流会避开下方
武器/HUD并使用多区域一致性过滤坏画面。正常首次完成时会看到：

```text
probe_levels axis=x counts=...
probe_levels axis=y counts=...
fit valid=1
标定完成 quality=...
DXGI 已打开
```

关闭进程后用同一条命令再次运行，应先看到“已加载本地标定，不移动鼠标”，随后直接打开
DXGI。只有调用端明确增加 `--recalibrate` 才会再次执行受控标定移动。如果 120 counts
范围内仍无法得到一致的中心场景位移，DLL 会报告 `HID calibration input not ready` 并
保留旧 profile；此时应检查游戏是否接收输入、画面是否稳定，而不是继续扩大运行输出。

本次更新不需要重新刷 RP2350 固件，也不改变现有 ORT 1.17.3、TensorRT 8.6.1.6、
CUDA 11.8、cuDNN 8.9.7 环境。

`probe_levels` 是初始计划，`sample_levels` 是经过有界向下降档后真正用于拟合的三个
档位。正式 `sample` counts 只来自当次尝试的档位及其相反数；每个 repeat 会分别输出
`leg=outward` 和 `leg=return`，表示真实的移出腿和反向归位腿。

`一键检查并测试.cmd` 始终只做安全试运行，与上述真实 Python 会话是两条独立
路径，不会自动开始标定或武装输出。
