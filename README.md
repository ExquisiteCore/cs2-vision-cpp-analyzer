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
xmake f --hid_sdk_root=D:\project\pi\test\sdk\cpp
xmake
xmake run vision_analyzer_tests
```

The RP2350 HID bridge SDK is optional at build time. Provide it with
`xmake f --hid_sdk_root=...` or the `RP2350_HID_BRIDGE_SDK` environment
variable when live HID output is needed. ONNX Runtime is also optional and can
be provided with `xmake f --onnxruntime_root=...` or `ONNXRUNTIME_ROOT`.

## Runtime Modes

Use absolute paths with `xmake run`; xmake may launch the binary from the build
directory, so short relative paths can point at the wrong place.

Dry-run prints status without sending SDK commands:

```powershell
xmake run vision_analyzer --backend opencv-onnx --model D:\project\cs2-vision-trainer\runs\detect\train\weights\best.onnx --video D:\project\cs2-vision-trainer\videos\02.mp4 --dry-run --preview
```

DXGI Desktop Duplication input captures a live Windows monitor:

```powershell
xmake run vision_analyzer --backend opencv-onnx --model D:\project\cs2-vision-trainer\runs\detect\train\weights\best.onnx --input dxgi --dxgi-output 0 --dry-run --preview
```

Live SDK movement:

```powershell
xmake run vision_analyzer --backend opencv-onnx --model D:\project\cs2-vision-trainer\runs\detect\train\weights\best.onnx --input dxgi --dxgi-output 0 --player-side ct --hid-port COM3 --hid-gain 1.0 --hid-max-step 120 --preview
```

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
- Track IDs are assigned with IoU plus center-distance matching.
- Target selection favors stable, close, high-confidence targets, with head
  preference and a switch penalty to reduce jitter.
- The target point uses a 2D Kalman state (`x`, `y`, `vx`, `vy`) plus
  latency-compensated prediction.
- `--player-side ct` targets `t_body` and `t_head`; `--player-side t` targets
  `ct_body` and `ct_head`; `unknown` keeps all classes.
- Live SDK output requires `--player-side ct` or `--player-side t`.
- Only head classes can become left-click fire candidates; body classes can
  help movement tracking but never trigger `--hid-click`.
- `--hid-gain` scales the target offset before sending `mouse_move(dx, dy)`.
- `--hid-max-step` clamps each movement axis per frame.
- `--hid-click` enables left-click output when the planner reports a fire
  candidate.

## Windows Mouse Acceleration

The RP2350 firmware sends standard relative USB HID mouse reports. It does not
apply a pointer curve. If the target program consumes normal Windows pointer
movement, Windows pointer speed and Enhance Pointer Precision can affect the
effective movement. If the target program consumes Raw Input, OS pointer
acceleration is typically bypassed and the result is dominated by HID counts and
in-application sensitivity.

Tune `--hid-gain` and `--hid-max-step` for the actual environment. The runtime
does not read or modify Windows pointer settings.

## Hardware Calibration Checklist

1. Confirm the board appears as a COM port.
2. Run `--dry-run --preview` on a representative video.
3. Run live movement without `--hid-click`.
4. Start with a low gain such as `--hid-gain 0.25`.
5. Increase `--hid-gain` until the filtered target point converges without
   oscillation.
6. Reduce `--hid-max-step` if movement jumps too far per frame.
7. Enable `--hid-click` only after movement is stable.
8. If movement differs between desktop and the target application, treat that as
   a Raw Input or pointer-acceleration difference and calibrate gain for the
   target application.
