# CS2 Vision C++ Runtime

This repository contains the C++ runtime side of the CS2 vision project. The
Python project trains and exports YOLO models. This runtime loads the exported
model, reads frames from a video file or DXGI Desktop Duplication, detects
targets, fuses body/head detections, tracks the selected target, filters and
predicts the aim point, then plans bounded relative mouse movement through the
RP2350 HID bridge SDK.

The primary product is `vision_runtime.dll`, which exposes the stable C API in
`include/vision_analyzer/vision_runtime_c_api.h`. `vision_analyzer.exe` is an
optional diagnostic CLI for model, video, DXGI, calibration, and end-to-end
runtime validation. No graphical UI is built or required.

Build artifacts:

```text
vision_runtime.dll  primary runtime library and C API implementation
vision_runtime.lib  MSVC import library for native consumers
vision_analyzer.exe optional diagnostic CLI
```

## Requirements

Windows build requirements:

```text
Visual Studio 2022 Build Tools with MSVC
xmake
CMake 3.14+ (when using the CMake build)
Git
```

Runtime/model requirements:

```text
OpenCV DNN, provided by xmake package resolution
Exported YOLO ONNX model
Matching *.schema.json file for live HID mode
RP2350 HID Bridge C++ SDK for real HID output
```

GTX 1080 Ti production acceleration target:

```text
ONNX Runtime GPU 1.17.x
TensorRT 8.6.x
CUDA 11.8
cuDNN 8.9.x
GeForce GTX 1080 Ti / SM 6.1
FP32
```

The default backend is `ort-tensorrt`. `opencv-onnx` remains available as an
explicit CPU fallback when ONNX Runtime, CUDA, or TensorRT is unavailable.

## Build with xmake

From this repository:

```powershell
xmake f -m release
xmake
xmake run vision_analyzer_tests
xmake run vision_runtime_c_api_tests
```

From the parent repository:

```powershell
cd tools\cpp_analyzer
xmake f -m release
xmake
xmake run vision_analyzer_tests
xmake run vision_runtime_c_api_tests
```

The parent repository layout is automatically supported. If the SDK is elsewhere,
pass it explicitly:

```powershell
xmake f -m release --hid_sdk_root=D:\project\cs2-vision-trainer\tools\rp2350_hid_bridge_cpp
xmake
```

Enable ONNX Runtime backends:

```powershell
$env:ONNXRUNTIME_ROOT = "D:\SDK\onnxruntime-win-x64-gpu"
xmake f -m release --onnxruntime_root=$env:ONNXRUNTIME_ROOT --hid_sdk_root=..\rp2350_hid_bridge_cpp
xmake
```

## GTX 1080 Ti Production Runtime

Use matching versions in one process. Do not mix the production DLL directory
with CUDA 12, cuDNN 9, TensorRT 11, or a newer ONNX Runtime provider DLL.

One concrete directory layout is:

```text
D:\runtime\sm61\onnxruntime-win-x64-gpu-1.17.3
D:\runtime\sm61\TensorRT-8.6.1.6\lib
D:\runtime\sm61\cudnn-8.9\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
```

Configure only the process that builds/runs the production package:

```powershell
$env:ONNXRUNTIME_ROOT='D:\runtime\sm61\onnxruntime-win-x64-gpu-1.17.3'
$env:PATH='D:\runtime\sm61\TensorRT-8.6.1.6\lib;D:\runtime\sm61\cudnn-8.9\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin;' + $env:PATH
xmake f -c -m release --onnxruntime_root=$env:ONNXRUNTIME_ROOT
xmake
```

The first model open builds an FP32 engine in
`ort-trt-cache-sm61-fp32` under the process working directory. Later opens
reuse it. Clear that directory after changing the model, GPU, ONNX Runtime,
TensorRT, CUDA, or cuDNN.

TensorRT 8.6 does not support the RTX 5060 (SM 12.0), so final provider and FPS
validation must run on the GTX 1080 Ti host. The RTX 5060 development machine
can still compile the source and run hardware-independent tests.

Primary DLL outputs:

```text
build\windows\x64\release\vision_runtime.dll
build\windows\x64\release\vision_runtime.lib
```

The optional diagnostic executable is generated under:

```text
build\windows\x64\release\vision_analyzer.exe
```

To build only the reusable DLL and its API test:

```powershell
xmake build vision_runtime
xmake run vision_runtime_c_api_tests
```

Use `xmake run` when possible. It adds the configured ONNX Runtime `lib`
directory to the process DLL search path. CUDA, cuDNN, and TensorRT still come
from the matching process-specific `PATH` shown above.

## Build with CMake

The CMake build produces the same core library, DLL, CLI, and test targets and
does not require any UI framework:

```powershell
cmake -S . -B build-cmake -A x64
cmake --build build-cmake --config Release
ctest --test-dir build-cmake -C Release --output-on-failure
```

CMake outputs are generated under:

```text
build-cmake\Release\vision_runtime.dll
build-cmake\Release\vision_runtime.lib
build-cmake\Release\vision_analyzer.exe
```

Set `ONNXRUNTIME_ROOT` or `RP2350_HID_BRIDGE_SDK` before configuring CMake when
those optional integrations are installed outside the default sibling layout.

## Model Contract

The runtime expects a YOLO ONNX model exported by the Python project. The class
order must be:

```text
0 ct_body
1 ct_head
2 t_body
3 t_head
```

Live HID output requires a schema JSON generated next to the ONNX file:

```text
best.onnx
best.onnx.schema.json
```

Dry-run can continue without schema for quick input testing. Live mode treats a
missing or mismatched schema as an error.

## Verify Inputs

Use absolute paths with `xmake run`; xmake may launch the binary from the build
directory.

Verify a video file:

```powershell
xmake run vision_analyzer --video D:\project\cs2-vision-trainer\videos\02.mp4 --verify-input
```

Expected output contains non-zero width, height, and RGB mean values.

List and probe DXGI outputs:

```powershell
xmake run vision_analyzer --list-dxgi-outputs
xmake run vision_analyzer --probe-dxgi-outputs
```

Verify one DXGI output:

```powershell
xmake run vision_analyzer --input dxgi --dxgi-adapter 0 --dxgi-output 0 --verify-input --dxgi-debug
```

Choose the adapter/output where `duplicate_output=0x0`. On hybrid GPU systems,
the valid output is usually the adapter that owns the physical monitor, not
necessarily the high-performance GPU doing 3D rendering.

If needed, crop the live input before inference:

```powershell
--dxgi-roi X Y W H
```

ROI coordinates are relative to the selected DXGI output. The target offset is
measured from the ROI center. A configured ROI is copied into an ROI-sized
D3D11 staging texture before CPU readback; the full desktop is not converted
and cropped afterward.

## Offline Dry-Run

Dry-run loads the model, runs detection and planning, but does not send HID
commands:

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

Action log columns:

```text
frame timestamp_ms target dx dy click lock distance offset_x offset_y
```

Interpretation:

```text
target=1  target selected
dx/dy     planned relative mouse movement
click=1   left click would be emitted; dry-run only logs it
lock=1    target lock is stable enough for fire-candidate evaluation
```

Body fallback detections can guide movement, but only head detections can become
left-click candidates.

## Live HID Mode

First verify the board without loading a model:

```powershell
xmake run vision_analyzer --hid-port COM3 --test-hid-move 300 0
```

Then run live DXGI movement without clicking:

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

Enable left-click output only after movement is calibrated:

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

Output is disabled by default. A DLL host calls
`va_set_output_enabled(runtime, 1)` when its own hotkey/arming condition is
active and calls `va_set_output_enabled(runtime, 0)` to stop output immediately.
Detection and returned action values continue while output is disabled.

Live HID mode requires:

```text
--player-side ct
```

or:

```text
--player-side t
```

`unknown` is allowed for dry-run, but not for live HID output.

## HID Calibration

Calibration sends controlled relative mouse moves through the board, observes
the visual shift through DXGI, and writes a fitted config fragment:

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

Review the generated `hid-tuned.cfg`, then pass it before CLI overrides:

```powershell
xmake run vision_analyzer --config hid-tuned.cfg --backend ort-tensorrt --model D:\project\cs2-vision-trainer\runs\detect\train\weights\best.onnx --input dxgi --dxgi-output 0 --player-side ct --hid-port COM3 --output-enabled
```

## Backends

```text
opencv-onnx   Explicit CPU ONNX fallback through OpenCV DNN.
opencv-cuda   Requires OpenCV built with CUDA DNN support.
ort-cuda      ONNX Runtime CUDA Execution Provider.
ort-tensorrt  Default GTX 1080 Ti TensorRT EP, with CUDA subgraph fallback.
tensorrt      Reserved for native TensorRT C++ builds.
```

If ONNX Runtime is not configured, ORT backends report unavailable at runtime
and the OpenCV backend remains usable.

## Algorithm Notes

- Class-aware NMS keeps overlapping head and body candidates from suppressing
  each other.
- Body/head detections from the same faction are associated before tracking.
- Matched head detections are preferred; unmatched body detections remain as
  fallback anchors near the top of the body box.
- Track IDs use IoU and anchor-distance matching.
- Target selection favors stable, close, high-confidence targets and applies a
  switch penalty to reduce jitter.
- The target point uses a 2D Kalman state with latency-compensated prediction.
- `--player-side ct` targets `t_body` and `t_head`.
- `--player-side t` targets `ct_body` and `ct_head`.
- Only head classes can trigger `--hid-click`.

## Windows Pointer Settings

The RP2350 firmware emits standard relative USB HID mouse reports. It does not
apply a pointer curve. Calibration reads and prints Windows pointer thresholds,
acceleration state, and pointer speed through `SystemParametersInfo`, but it
does not modify those settings.

If the target application consumes normal Windows pointer movement, pointer
speed and Enhance Pointer Precision can affect motion. If it consumes Raw Input,
movement is usually dominated by HID counts and in-application sensitivity.

Tune these values on the actual target machine:

```text
--hid-gain
--hid-max-step
--hid-deadzone
```

## CLI Help

```powershell
xmake run vision_analyzer --help
```

## C API DLL

`vision_runtime.dll` exports a stable C ABI declared in:

```text
include\vision_analyzer\vision_runtime_c_api.h
```

The API uses an opaque `VaRuntime*` handle and plain C structs:

```c
VaRuntime* runtime = va_create();
va_set_model(runtime, "best.onnx");
va_set_schema(runtime, "best.onnx.schema.json");
va_set_backend(runtime, "ort-tensorrt");
va_set_tensorrt_cache_path(runtime, "ort-trt-cache-sm61-fp32");
va_set_dxgi_roi(runtime, 640, 220, 640, 640);
va_set_player_side(runtime, "ct");
va_set_hid_port(runtime, "COM3");
if (va_open_dxgi(runtime, 0, 0, 0) != 0) {
    fprintf(stderr, "%s\n", va_last_error(runtime));
    va_destroy(runtime);
    return 1;
}
va_set_output_enabled(runtime, 1);

VaRuntimeAction action;
while (va_process_next(runtime, &action) == 1) {
    printf("%d %d %d\n", action.frame_index, action.dx, action.dy);
}

va_set_output_enabled(runtime, 0);
va_destroy(runtime);
```

Return codes:

```text
0   success for configuration/open/close calls
1   frame processed for va_process_next
0   end-of-stream for va_process_next
-1  error; read va_last_error(runtime)
```

The main Python repository wraps this DLL with `cs2_vision_runtime.VisionRuntime`.
Any wrapper that needs live HID output must bind and call
`va_set_output_enabled`; an older wrapper remains safely disarmed even though
`va_process_next` continues returning planned actions.
