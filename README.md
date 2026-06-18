# CS2 Vision C++ Analyzer

Read-only C++ video analyzer for exported YOLO models.

It does not control mouse, keyboard, clicks, memory, process injection, or game
state. It reads a video file, runs visual detection, draws boxes, tracks targets,
filters the virtual analysis point, and writes CSV analysis logs.

## Build

```powershell
cd tools\cpp_analyzer
xmake
xmake run vision_analyzer_tests
```

## Run

Use absolute paths with `xmake run`; xmake may launch the binary from the build
directory, so short relative paths can point at the wrong place.

```powershell
xmake run vision_analyzer --backend opencv-onnx --model D:\project\cs2-vision-trainer\runs\detect\train\weights\best.onnx --video D:\project\cs2-vision-trainer\videos\02.mp4 --output D:\project\cs2-vision-trainer\runs\analysis\02_analysis.csv --preview
```

Write an additional human-readable TXT action log:

```powershell
xmake run vision_analyzer --backend opencv-onnx --model D:\project\cs2-vision-trainer\runs\detect\train\weights\best.onnx --video D:\project\cs2-vision-trainer\videos\02.mp4 --output D:\project\cs2-vision-trainer\runs\analysis\02_analysis.csv --actions-output D:\project\cs2-vision-trainer\runs\analysis\02_actions.txt --preview
```

Example action line:

```text
frame=42 time_ms=1400.000 action=move target_id=9 target_class=ct_head confidence=0.910000 move_to=(976.000,541.000) aim_to=(978.000,542.000) move_delta=(18.000,2.000) distance=18.110 velocity=(120.000,6.000) acceleration=(10.000,1.000) stability=0.820 switched=0 lock_state=locked left_button=press_candidate
```

`move_to` is the filtered offline analysis point. `move_delta` is the offset
from the frame center to the predicted target point. `left_button` is only a
candidate state written to text.

Send relative mouse actions through the RP2350 HID bridge SDK:

```powershell
xmake run vision_analyzer --backend opencv-onnx --model D:\project\cs2-vision-trainer\runs\detect\train\weights\best.onnx --video D:\project\cs2-vision-trainer\videos\02.mp4 --output D:\project\cs2-vision-trainer\runs\analysis\02_analysis.csv --hid-port COM3 --hid-gain 1.0 --hid-max-step 120
```

Enable left-click candidates explicitly:

```powershell
xmake run vision_analyzer --backend opencv-onnx --model D:\project\cs2-vision-trainer\runs\detect\train\weights\best.onnx --video D:\project\cs2-vision-trainer\videos\02.mp4 --output D:\project\cs2-vision-trainer\runs\analysis\02_analysis.csv --hid-port COM3 --hid-click --hid-click-cooldown 6
```

The analyzer expects the SDK at `D:\project\pi\test\sdk\cpp` by default. To use
another checkout, set `RP2350_HID_BRIDGE_SDK` to the SDK root before building.
`--hid-gain` scales the target offset before sending `mouse_move(dx, dy)`;
`--hid-max-step` clamps each axis per frame. Without `--hid-click`, the SDK
backend only sends relative mouse movement and `stop_all` on shutdown.

Skip an intro/loading segment and only process a short slice:

```powershell
xmake run vision_analyzer --backend opencv-onnx --model D:\project\cs2-vision-trainer\runs\detect\train\weights\best.onnx --video D:\project\cs2-vision-trainer\videos\01.mp4 --output D:\project\cs2-vision-trainer\runs\analysis\01_analysis.csv --start-time 160 --max-frames 300
```

Run the CUDA backend:

```powershell
xmake run vision_analyzer --backend ort-cuda --model D:\project\cs2-vision-trainer\runs\detect\train\weights\best.onnx --video D:\project\cs2-vision-trainer\videos\01.mp4 --output D:\project\cs2-vision-trainer\runs\analysis\01_ort_cuda.csv --start-time 160 --max-frames 300
```

Set `--warmup-frames 0` only when you deliberately want to include provider
startup time in the CSV. The default is `3`.

## Backends

```text
--backend opencv-onnx  Stable CPU ONNX path.
--backend opencv-cuda  Requires an OpenCV build with CUDA DNN support.
--backend ort-cuda     ONNX Runtime CUDA Execution Provider.
--backend ort-tensorrt ONNX Runtime TensorRT Execution Provider, with CUDA fallback.
--backend tensorrt     Reserved for native TensorRT C++ builds.
```

The xmake file uses `ONNXRUNTIME_ROOT` when it is set. Otherwise it falls back to
`D:\Tool\onnxruntime-win-x64-gpu-1.22.1`.

Use `xmake run` unless you have manually added the ONNX Runtime, CUDA/cuDNN, and
TensorRT DLL directories to `PATH`. The xmake target sets the runtime paths for
the current project environment.

The first ORT CUDA frame includes provider initialization and graph setup. For
performance checks, ignore the first few frames and compare the warmed
`inference_ms` / `latency_ms` values.

The current Python environment contains TensorRT Python bindings and DLLs, but
not the native TensorRT C++ SDK headers/import libraries (`NvInfer.h`,
`nvinfer*.lib`) required for the separate `--backend tensorrt` path. Use
`--backend ort-cuda` first.

`--backend ort-tensorrt` requires the TensorRT DLL major version expected by the
ONNX Runtime build. If it reports a missing `nvinfer_10.dll`, install a matching
TensorRT 10 runtime or use `ort-cuda`.

## Algorithm Notes

- YOLO handles per-frame object detection only.
- Class-aware NMS keeps overlapping `head` and `body` candidates from suppressing
  each other.
- Track IDs are assigned with IoU plus center-distance matching.
- Target selection favors stable, close, high-confidence targets, with head
  preference and a switch penalty to reduce jitter.
- The virtual analysis point uses a One Euro filter and latency-compensated
  predicted center.

## CSV Fields

```text
frame_index,timestamp_ms,fps,latency_ms,detection_count,
preprocess_ms,inference_ms,postprocess_ms,
target_id,target_class,target_confidence,
target_x1,target_y1,target_x2,target_y2,target_cx,target_cy,
offset_x,offset_y,distance,switched,analysis_x,analysis_y,
predicted_x,predicted_y,velocity_x,velocity_y,accel_x,accel_y,
stability,tracking_error,lock_state,fire_candidate
```

`analysis_x`, `analysis_y`, `predicted_x`, `predicted_y`, and `fire_candidate`
are offline analysis fields. They are not mouse commands, click commands, or SDK
inputs.
