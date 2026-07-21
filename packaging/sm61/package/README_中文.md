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

一键测试始终使用 dry-run，配置中的 `output_enabled=false` 也不会被脚本覆盖，
因此不会向 RP2350 发送移动或点击。

## 单独测试当前屏幕

先完成视频测试，再在本目录打开 PowerShell：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\test-dxgi.ps1 -Adapter 0 -Output 0
```

脚本会先列出并探测显示输出。如果输出 0 不支持 Desktop Duplication，请根据
日志显示的 adapter/output 改参数。DXGI 测试同样不会向 RP2350 输出。

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

头文件和导入库位于 `app`。Python 包装位于 `python\cs2_vision_runtime`，无第三方
Python 依赖；本压缩包不捆绑 Python 解释器。

宿主必须按以下顺序调用：

1. 设置模型、RP2350 串口和玩家阵营。
2. 进入对局并保持画面稳定，然后调用一次 `calibrate_hid()`。
3. 设置头/身体开火策略，再打开 DXGI。
4. 分别调用 `set_output_enabled(True)` 和 `set_fire_enabled(True)`。
5. 停止或异常退出时，依次关闭开火、关闭移动输出并调用 `stop_all()`。

两个开关相互独立。新建运行时默认都关闭；只有 Python 明确开启后才会产生物理
移动或点击。头部优先，身体只在躯干有效区域内作为低优先级兜底。

## 运行 Python 示例

先进入 CS2 对局并让画面稳定，在本目录打开 PowerShell：

```powershell
$env:PYTHONPATH=(Resolve-Path '.\python').Path
$env:CS2_VISION_RUNTIME_DLL=(Resolve-Path '.\app\vision_runtime.dll').Path
python .\examples\runtime_live_move.py --hid-port COM3 --player-side ct --enable-live-output --click
```

把 `COM3` 改为 RP2350 的实际串口。去掉 `--click` 时只自动瞄准、不自动开火；
不加 `--enable-live-output` 时程序会在标定前退出，完全不触碰硬件。包装层会自动
加载包内 CUDA 11.8、cuDNN 8.9、TensorRT 8.6.1.6 和 MSVC 私有运行库，不需要
修改系统 PATH。按 `Ctrl+C` 会通过 `finally` 撤销开火和移动输出。

`一键检查并测试.cmd` 始终只做安全 dry-run，与上述真实 Python 会话是两条独立
路径，不会自动开始标定或武装输出。
