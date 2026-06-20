# CS2 Vision C++ Runtime Controller

This repository contains the C++ runtime side of the CS2 vision project. The
Python project trains and exports YOLO models. This runtime loads the exported
model, reads frames from a video file or DXGI Desktop Duplication, detects
targets, fuses body/head detections, tracks the selected target, filters and
predicts the aim point, then plans bounded relative mouse movement through the
RP2350 HID bridge SDK.

The maintained entry point is `vision_analyzer.exe`. The old UI integration is
kept as archived code only; normal validation and usage should go through the
CLI.

## Requirements

Windows build requirements:

```text
Visual Studio 2022 Build Tools with MSVC
xmake
Git
```

Runtime/model requirements:

```text
OpenCV DNN, provided by xmake package resolution
Exported YOLO ONNX model
Matching *.schema.json file for live HID mode
RP2350 HID Bridge C++ SDK for real HID output
```

Optional acceleration:

```text
ONNX Runtime GPU
CUDA/cuDNN
TensorRT through ONNX Runtime TensorRT Execution Provider
```

The stable default backend is `opencv-onnx`. It runs without ONNX Runtime,
CUDA, or TensorRT.

## Build

From this repository:

```powershell
xmake f -m release
xmake
xmake run vision_analyzer_tests
```

From the parent repository:

```powershell
cd tools\cpp_analyzer
xmake f -m release
xmake
xmake run vision_analyzer_tests
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

The executable is generated under:

```text
build\windows\x64\release\vision_analyzer.exe
```

Build the reusable DLL:

```powershell
xmake build vision_runtime
xmake run vision_runtime_c_api_tests
```

DLL outputs:

```text
build\windows\x64\release\vision_runtime.dll
build\windows\x64\release\vision_runtime.lib
```

Use `xmake run` when possible. It sets runtime DLL search paths for dependencies
resolved by this project.

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
measured from the ROI center.

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
  --backend opencv-onnx `
  --model D:\project\cs2-vision-trainer\runs\detect\train\weights\best.onnx `
  --input dxgi `
  --dxgi-output 0 `
  --player-side ct `
  --hid-port COM3 `
  --hid-gain 1.0 `
  --hid-max-step 120 `
  --preview
```

Enable left-click output only after movement is calibrated:

```powershell
xmake run vision_analyzer `
  --backend opencv-onnx `
  --model D:\project\cs2-vision-trainer\runs\detect\train\weights\best.onnx `
  --input dxgi `
  --dxgi-output 0 `
  --player-side ct `
  --hid-port COM3 `
  --hid-click `
  --hid-click-cooldown 6
```

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
xmake run vision_analyzer --config hid-tuned.cfg --backend opencv-onnx --model D:\project\cs2-vision-trainer\runs\detect\train\weights\best.onnx --input dxgi --dxgi-output 0 --player-side ct --hid-port COM3
```

## Backends

```text
opencv-onnx   Stable CPU ONNX path through OpenCV DNN.
opencv-cuda   Requires OpenCV built with CUDA DNN support.
ort-cuda      ONNX Runtime CUDA Execution Provider.
ort-tensorrt  ONNX Runtime TensorRT Execution Provider, with CUDA fallback.
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
va_set_backend(runtime, "opencv-onnx");
va_open_video(runtime, "videos/02.mp4", 1);

VaRuntimeAction action;
while (va_process_next(runtime, &action) == 1) {
    printf("%d %d %d\n", action.frame_index, action.dx, action.dy);
}

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
