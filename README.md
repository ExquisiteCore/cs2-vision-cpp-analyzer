# CS2 Vision C++ Runtime Controller

C++ runtime for exported YOLO models.

The Python project trains and exports the model. This C++ tool consumes that
model at runtime: it reads frames, runs visual detection, tracks enemy head
candidates, filters and predicts the target point, converts the result into
bounded relative mouse movement, and sends commands through the RP2350 HID
bridge SDK.

## Build

```powershell
cd tools\cpp_analyzer
xmake f
xmake
xmake run vision_analyzer_tests
```

可选的 EUI 桌面控制台：

```powershell
cd tools\cpp_analyzer
cmake -S . -B build-ui -DEUI_NEO_ROOT=D:/project/EUI-NEO
cmake --build build-ui --config Release --target vision_analyzer_ui
.\build-ui\Release\vision_analyzer_ui.exe
```

这个 UI 是现有 `vision_analyzer.exe` 的启动控制台：可以编辑模型路径、
DXGI/HID 参数和调参值，执行输入验证、HID 探针、标定、干跑预览和实机
运行，并把子进程输出流式显示在日志面板里。底层分析器二进制仍然由
`xmake` 构建。

The RP2350 HID bridge SDK is optional at build time. Provide it with
`xmake f --hid_sdk_root=...` or the `RP2350_HID_BRIDGE_SDK` environment
variable when live HID output is needed. In the parent repository the default
SDK path is `tools\rp2350_hid_bridge_cpp`. ONNX Runtime is also optional and
can be provided with `xmake f --onnxruntime_root=...` or `ONNXRUNTIME_ROOT`.

## Runtime Modes

Use absolute paths with `xmake run`; xmake may launch the binary from the build
directory, so short relative paths can point at the wrong place.

Dry-run prints status without sending SDK commands:

```powershell
xmake run vision_analyzer --config runtime.example.cfg --backend opencv-onnx --model D:\project\cs2-vision-trainer\runs\detect\train\weights\best.onnx --video D:\project\cs2-vision-trainer\videos\02.mp4 --dry-run --preview --action-log actions.txt
```

DXGI Desktop Duplication input captures a live Windows monitor:

```powershell
xmake run vision_analyzer --list-dxgi-outputs
xmake run vision_analyzer --probe-dxgi-outputs
xmake run vision_analyzer --input dxgi --dxgi-output 0 --verify-input
xmake run vision_analyzer --backend opencv-onnx --model D:\project\cs2-vision-trainer\runs\detect\train\weights\best.onnx --input dxgi --dxgi-output 0 --dry-run --preview
```

Use `--dxgi-roi X Y W H` to crop the selected output before inference. ROI
coordinates are relative to the DXGI output. The target offset is then measured
from the ROI center, so a centered ROI is the normal choice for live tuning.

On hybrid GPU laptops, Desktop Duplication must run on the adapter that owns the
actual display. If `--probe-dxgi-outputs` shows `duplicate_output=0x887A0004`
for NVIDIA while dxdiag reports the internal panel on Intel, set the built
binary to Windows Graphics "Power saving" / integrated GPU and start the binary
again:

```powershell
$exe = (Resolve-Path .\build\windows\x64\release\vision_analyzer.exe).Path
New-Item -Path 'HKCU:\Software\Microsoft\DirectX\UserGpuPreferences' -Force | Out-Null
New-ItemProperty -Path 'HKCU:\Software\Microsoft\DirectX\UserGpuPreferences' -Name $exe -Value 'GpuPreference=1;' -PropertyType String -Force
.\build\windows\x64\release\vision_analyzer.exe --probe-dxgi-outputs
.\build\windows\x64\release\vision_analyzer.exe --input dxgi --verify-input
```

`--dxgi-debug` prints the first DXGI frame metadata and texture byte statistics.
The runtime skips initial pointer-only frames (`AccumulatedFrames=0`) because
some hybrid systems return an all-black surface before the first real desktop
present.

Live SDK movement:

```powershell
xmake run vision_analyzer --backend opencv-onnx --model D:\project\cs2-vision-trainer\runs\detect\train\weights\best.onnx --input dxgi --dxgi-output 0 --player-side ct --hid-port COM3 --hid-gain 1.0 --hid-max-step 120 --preview
```

Live SDK movement requires a matching exported schema JSON. Either keep
`best.onnx.schema.json` next to `best.onnx`, or pass it explicitly with
`--schema`. Dry-run still allows the schema to be missing so video and DXGI
tuning can continue while a model export is being prepared.

Quickly verify the RP2350 HID bridge without loading a model or using DXGI:

```powershell
xmake run vision_analyzer --hid-port COM4 --test-hid-move 300 0
```

The command sends one relative mouse move, then prints the Windows cursor
position before and after the move. It does not click.

Run controlled HID calibration without loading a model:

```powershell
xmake run vision_analyzer --calibrate-hid --hid-port COM3 --dxgi-output 0 --calibration-step 40 --calibration-noise-samples 2 --calibration-output hid-calibration.txt --calibration-config-output hid-tuned.cfg
```

Calibration records no-op noise samples, controlled HID movement samples, prints
the fitted `hid_gain`, `hid_deadzone_px`, and `hid_max_step`, and writes a
runtime config fragment such as `hid-tuned.cfg`. Merge or pass that config after
checking it in the actual target environment.

Enable left-click candidates only after movement is calibrated:

```powershell
xmake run vision_analyzer --backend opencv-onnx --model D:\project\cs2-vision-trainer\runs\detect\train\weights\best.onnx --input dxgi --player-side ct --hid-port COM3 --hid-click --hid-click-cooldown 6
```

Skip an intro/loading segment and process a short slice:

```powershell
xmake run vision_analyzer --backend opencv-onnx --model D:\project\cs2-vision-trainer\runs\detect\train\weights\best.onnx --video D:\project\cs2-vision-trainer\videos\01.mp4 --dry-run --start-time 160 --max-frames 300
```

## Backends

```text
--backend opencv-onnx  Stable CPU ONNX path.
--backend opencv-cuda  Requires an OpenCV build with CUDA DNN support.
--backend ort-cuda     ONNX Runtime CUDA Execution Provider.
--backend ort-tensorrt ONNX Runtime TensorRT Execution Provider, with CUDA fallback.
--backend tensorrt     Reserved for native TensorRT C++ builds.
```

The xmake file uses the `onnxruntime_root` config value or `ONNXRUNTIME_ROOT`.
If ONNX Runtime is not found, the OpenCV backend still builds and the ORT
backends report unavailable at runtime.

Use `xmake run` unless you have manually added the ONNX Runtime, CUDA/cuDNN, and
TensorRT DLL directories to `PATH`. The xmake target sets the runtime paths for
the current project environment.

The first ORT CUDA frame includes provider initialization and graph setup. For
performance checks, ignore the first few frames and compare warmed
`inference_ms` values.

## Algorithm Notes

- YOLO handles per-frame object detection.
- Class-aware NMS keeps overlapping head and body candidates from suppressing
  each other.
- Body/head detections from the same faction are associated before tracking.
  When a head is matched to its body, the head target is kept and the body
  duplicate is suppressed; unmatched bodies remain as fallback targets.
- Track IDs are assigned with IoU plus anchor-distance matching.
- Target selection favors stable, close, high-confidence targets, with head
  preference and a switch penalty to reduce jitter.
- The target point uses a 2D Kalman state (`x`, `y`, `vx`, `vy`) plus
  latency-compensated prediction.
- Head detections use the head box center. Body detections are only a fallback
  anchor and aim near the top of the body box via `--body-head-anchor-ratio`;
  body center is not used as the movement target.
- `--player-side ct` targets `t_body` and `t_head`; `--player-side t` targets
  `ct_body` and `ct_head`; `unknown` keeps all classes.
- Live SDK output requires `--player-side ct` or `--player-side t`.
- Only head classes can become left-click fire candidates; body classes can
  help movement tracking but never trigger `--hid-click`.
- `--hid-gain` scales the target offset before sending `mouse_move(dx, dy)`.
- `--hid-max-step` clamps each movement axis per frame.
- `--hid-deadzone` suppresses tiny per-axis movement.
- `--hid-click` enables left-click output when the planner reports a fire
  candidate.
- `--schema` validates the exported model class schema JSON. If omitted, the
  runtime tries `best.onnx.schema.json` next to the model and warns when it is
  absent during dry-run. Live SDK output treats a missing schema as an error.

## Windows Mouse Acceleration

The RP2350 firmware sends standard relative USB HID mouse reports. It does not
apply a pointer curve. The calibration mode reads and prints Windows pointer
thresholds, acceleration state, and pointer speed through `SystemParametersInfo`
but does not modify them. If the target program consumes normal Windows pointer
movement, those settings can affect movement. If the target program consumes Raw
Input, OS pointer acceleration is typically bypassed and the result is dominated
by HID counts and in-application sensitivity.

Tune `--hid-gain`, `--hid-max-step`, and `--hid-deadzone` for the actual
environment. Use `--calibrate-hid` to capture text samples that pair HID counts
with observed DXGI frame shift and to generate a fitted config file.

## Hardware Calibration Checklist

1. Confirm the board appears as a COM port.
2. Run `--dry-run --preview` on a representative video.
3. Run live movement without `--hid-click`.
4. Start with a low gain such as `--hid-gain 0.25`.
5. Run `--calibrate-hid` and inspect the generated `hid-tuned.cfg`.
6. Increase or reduce `hid_gain` until the filtered target point converges
   without oscillation.
7. Reduce `hid_max_step` if movement jumps too far per frame.
8. Enable `--hid-click` only after movement is stable.
9. If movement differs between desktop and the target application, treat that as
   a Raw Input or pointer-acceleration difference and calibrate gain for the
   target application.
