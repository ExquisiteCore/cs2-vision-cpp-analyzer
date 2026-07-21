# GTX 1080 Ti ORT TensorRT Runtime Design

## Goal

Optimize the DLL-first runtime for one production target: NVIDIA GeForce GTX 1080 Ti. Make ONNX Runtime TensorRT Execution Provider the default inference path, reduce DXGI capture transfer work, and require an explicit C API arm before the RP2350 can emit mouse actions.

The existing FP32 ONNX model remains the model contract. It has a static NCHW input of `[1, 3, 640, 640]` and a raw YOLO output of `[1, 8, 8400]`.

## Production compatibility baseline

The production runtime is fixed to this compatible stack:

- GPU: GeForce GTX 1080 Ti, CUDA compute capability 6.1.
- ONNX Runtime: 1.17.x GPU build.
- TensorRT: 8.6.x.
- CUDA: 11.8.
- cuDNN: 8.9.x.
- TensorRT precision: FP32.

This is a production dependency contract, not a request to maintain two runtime profiles. The current RTX 5060 development machine may compile and run hardware-independent tests, but it cannot validate a TensorRT 8.6 engine because TensorRT 8.6 does not support SM 12.0. Final provider and performance validation must run on the GTX 1080 Ti machine.

The compatibility decision is based on the official matrices:

- NVIDIA lists the GTX 1080 Ti as compute capability 6.1: <https://developer.nvidia.com/cuda-legacy-gpus>
- TensorRT 8.6.1 supports compute capability 6.1 on Windows, CUDA 11.8, and cuDNN 8.9.0: <https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/support-matrix/index.html>
- ONNX Runtime maps release 1.17 to TensorRT 8.6 and CUDA 11.8: <https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html>

## Scope

The implementation includes:

1. Make `ort-tensorrt` the default backend.
2. Configure TensorRT EP for GPU 0, FP32, a 2 GiB build workspace, engine caching, and CUDA EP subgraph fallback.
3. Replace the hard-coded ORT input allocation with validated model input-shape discovery.
4. Crop a configured DXGI ROI on the GPU before mapping the staging texture to CPU memory.
5. Add `va_set_output_enabled(runtime, enabled)` to arm or disarm RP2350 output at runtime.
6. Update the example config and deployment documentation for the fixed GTX 1080 Ti stack.

The implementation does not include:

- IbInputSimulator or any output backend other than RP2350.
- Native TensorRT C++ inference.
- FP16, INT8, calibration, or precision-selection APIs.
- Asynchronous capture/inference queues.
- End-to-end `[N, 6]` YOLO output decoding.
- UI or DLL-owned hotkey handling.
- A second build or runtime profile for the RTX 5060 machine.

## Runtime data flow

```text
DXGI desktop texture
    -> D3D11 GPU ROI copy
    -> ROI-sized staging texture
    -> BGRA-to-BGR conversion
    -> letterbox using the ONNX input dimensions
    -> ONNX Runtime TensorRT EP (FP32)
    -> CUDA EP for unsupported TensorRT subgraphs
    -> existing raw-YOLO decode, tracking, and aim planning
    -> C API action result
    -> RP2350 only when output is armed
```

The loop remains synchronous. Avoiding a queued pipeline keeps the newest-frame latency predictable even when inference is slower than capture.

## TensorRT provider behavior

`ort-tensorrt` registers TensorRT EP first and CUDA EP second. CUDA EP handles model nodes that TensorRT cannot claim; it is not a silent replacement for a missing or incompatible TensorRT installation.

The provider settings are fixed for the production target:

- `device_id=0`
- `trt_fp16_enable=0`
- `trt_engine_cache_enable=1`
- `trt_max_workspace_size=2147483648`
- `trt_min_subgraph_size=1`

The cache directory defaults to `ort-trt-cache-sm61-fp32` under the process working directory. A config key and C API setter allow changing only this path, because the host may need a known writable location. The runtime creates the directory during session setup. Cache contents must be cleared after replacing the model or changing ONNX Runtime, TensorRT, CUDA, cuDNN, or the GPU.

TensorRT provider registration or session creation failure must return a clear error through `va_last_error` that names the expected ORT 1.17.x / TensorRT 8.6.x / CUDA 11.8 / cuDNN 8.9.x stack. The runtime must not report `ort-tensorrt` while secretly constructing a CUDA-only session.

## Model input discovery

After creating the ORT session, the detector reads the first input tensor metadata and accepts only this model contract:

- one image input used for inference;
- rank 4 NCHW layout;
- batch size 1;
- three channels;
- positive, static height and width;
- FP32 tensor input.

The discovered height and width size the input buffer, input tensor, letterbox operation, and warm-up frame. Invalid or dynamic shapes fail during `va_open_*` with the actual shape in the error message. Dynamic-shape profiles are outside this change.

The existing raw YOLO output decoder remains unchanged and continues to validate the four-class output schema.

## GPU-side DXGI ROI

When ROI width and height are nonzero, the frame source intersects the requested rectangle with the desktop texture bounds before copying. An empty intersection is an error. The staging texture is created at the clipped ROI dimensions, and `CopySubresourceRegion` copies only that source box.

The mapped `cv::Mat` therefore already represents the ROI; no full-desktop BGRA-to-BGR conversion or later CPU clone is performed. With ROI disabled, the same code selects the entire desktop and preserves existing full-frame behavior.

Texture recreation occurs when copied width, height, or pixel format changes. DXGI access-loss recovery rebuilds the duplication resources as it does now.

## RP2350 output arming

The new C function is:

```c
int32_t va_set_output_enabled(VaRuntime* runtime, int32_t enabled);
```

Output starts disabled for every new runtime. The function may be called before or after `va_open_*` and copies nonzero values to an atomic armed state. Disabling an open runtime immediately calls `stop_all` on the RP2350 sender.

`va_process_next` always performs capture, inference, tracking, and aim planning. It always returns the planned `dx`, `dy`, and click candidate in `VaRuntimeAction`; it sends that command to RP2350 only while armed. The host application owns hotkey handling and calls this function to change the state.

Dry-run behavior is unchanged and never creates an RP2350 client.

## Configuration and C API surface

The existing C ABI remains source-compatible. Two additive functions are introduced:

```c
int32_t va_set_tensorrt_cache_path(VaRuntime* runtime, const char* path);
int32_t va_set_output_enabled(VaRuntime* runtime, int32_t enabled);
```

The configuration file gains:

```text
backend=ort-tensorrt
tensorrt_cache_path=ort-trt-cache-sm61-fp32
output_enabled=false
```

No provider-options struct, device selector, or precision selector is exposed. This keeps the public API aligned with the single GTX 1080 Ti deployment target.

## Packaging behavior

The build continues to resolve ONNX Runtime through `ONNXRUNTIME_ROOT` and copy that SDK's DLLs next to `vision_runtime.dll`. The xmake runtime environment must stop automatically adding the project's Python `torch/lib` and `tensorrt_libs` directories, because those directories contain CUDA 12, cuDNN 9, and TensorRT 11 DLLs that are incompatible with the production stack.

The GTX 1080 Ti host must expose the matching CUDA 11.8, cuDNN 8.9.x, and TensorRT 8.6.x DLLs through the application directory or a process-specific `PATH`. New and old NVIDIA DLL sets must not be mixed in one process.

## Performance expectations

The change targets lower per-frame latency and less CPU/memory traffic, not an unconditional FPS number. DXGI processing FPS cannot exceed the display's presentation rate, even if TensorRT inference is faster. `VaRuntimeAction.inference_ms` is the primary TensorRT performance measurement; `VaRuntimeAction.fps` remains the end-to-end new-frame rate.

The first open may take substantially longer while TensorRT builds an engine. Later opens should reuse the cache. FP16 is deliberately disabled because the GTX 1080 Ti has no Tensor Cores and FP16 is not assumed to outperform FP32 on this GPU.

## Testing and acceptance

Implementation follows test-first development. Hardware-independent acceptance requires:

1. Model-shape tests accept `[1, 3, 640, 640]` and reject dynamic, non-NCHW, non-FP32, or invalid dimensions.
2. ROI-region tests cover full-frame mode, exact ROI, clipped ROI, and empty intersection.
3. Runtime configuration tests cover the TensorRT cache path, default TensorRT backend, and disabled output default.
4. C API tests cover both additive setters and their invalid arguments.
5. HID sender/session tests prove that a planned action is returned while disarmed but is emitted only while armed, and that disarming invokes `stop_all`.
6. Existing algorithm and C API tests continue to pass.
7. Clean xmake and CMake builds produce the DLL and diagnostic CLI without adding Python's TensorRT 11 directory to the runtime path.

Final hardware acceptance on the GTX 1080 Ti requires:

1. `va_open_dxgi` successfully creates an ORT TensorRT session with the fixed production stack.
2. The first open creates an engine cache and a later open reuses it.
3. Provider incompatibility produces a clear C API error rather than silent CUDA-only execution.
4. DXGI ROI output dimensions match the configured clipped region.
5. RP2350 emits no movement before `va_set_output_enabled(runtime, 1)` and stops after disabling.
6. A timed run records end-to-end FPS plus preprocessing, inference, and postprocessing milliseconds so performance can be compared with the existing baseline.
