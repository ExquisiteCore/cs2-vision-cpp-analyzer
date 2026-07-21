# GTX 1080 Ti Portable Runtime Package Design

## Goal

Produce one Windows x64 ZIP that can be copied to the GTX 1080 Ti production
machine, extracted, and tested by double-clicking one command file. The target
machine is assumed to have only a working NVIDIA display driver. It is not
assumed to have CUDA Toolkit, cuDNN, TensorRT, ONNX Runtime, Visual Studio, CMake,
xmake, Python, or administrator access.

The package is a diagnostic delivery for the existing DLL-first runtime. It
contains the built DLL and CLI, the fixed GTX 1080 Ti inference environment, the
model contract, a short test input, configuration, and scripts. It does not
contain a UI and does not enable RP2350 output during setup or smoke testing.

## Standard production dependency model

The production machine keeps the NVIDIA display driver as a system component.
All application-specific inference DLLs are private, application-local
dependencies:

- ONNX Runtime GPU 1.17.3
- CUDA 11.8 redistributable runtime libraries
- cuDNN 8.9.x runtime libraries for CUDA 11.x
- TensorRT 8.6.1.6 runtime libraries for CUDA 11.8
- Microsoft Visual C++ x64 runtime libraries required by the compiled binaries

The full CUDA Toolkit is intentionally not installed. A compiled inference
application does not need `nvcc`, CUDA headers, static development libraries,
Nsight, Visual Studio integration, or CUDA samples. Installing those development
components would make the package larger, require administrator access, alter
machine-wide state, and create more opportunities for CUDA 11/12 DLL conflicts.
Using official redistributable runtime components beside the application is the
standard Windows private-deployment model for this type of program.

The package launcher prepends only the package's runtime directories to the
child process `PATH`. It must not write the registry, modify the system or user
`PATH`, set a system-wide CUDA installation, or copy DLLs into Windows system
directories. Closing the launched process removes the temporary environment.

## Fixed compatibility profile

The package has one named profile, `sm61-ort1173-trt861-fp32`:

```text
GPU              NVIDIA GeForce GTX 1080 Ti, compute capability 6.1
Driver           546.33 shown on the production host (sufficient for CUDA 11.8)
ONNX Runtime     1.17.3 GPU, Windows x64
TensorRT         8.6.1.6, Windows x64, CUDA 11.8 build
CUDA runtime     11.8
cuDNN runtime    8.9.x for CUDA 11.x
Precision        FP32
Model input      NCHW [1,3,640,640], FP32
Runtime output   disabled until the DLL host explicitly arms it
```

The `CUDA Version: 12.3` value reported by `nvidia-smi` is the driver's maximum
CUDA compatibility level. It does not prove that CUDA Toolkit or any CUDA
runtime DLL is installed, and it does not prevent an application from loading
private CUDA 11.8 DLLs.

## Package layout

```text
cs2-vision-runtime-sm61/
├── app/
│   ├── vision_runtime.dll
│   ├── vision_runtime.lib
│   ├── vision_analyzer.exe
│   ├── vision_runtime_c_api.h
│   └── onnxruntime*.dll
├── model/
│   ├── best.onnx
│   └── best.onnx.schema.json
├── licenses/
│   ├── onnxruntime/
│   ├── cuda/
│   ├── cudnn/
│   ├── tensorrt/
│   └── msvc/
├── config/
│   └── runtime-sm61.cfg
├── runtime/
│   ├── cuda-11.8/
│   ├── cudnn-8.9/
│   ├── tensorrt-8.6.1.6/
│   └── msvc-x64/
├── samples/
│   └── smoke-test.mp4
├── scripts/
│   ├── common.ps1
│   ├── setup-and-test.ps1
│   ├── verify-runtime.ps1
│   ├── test-video.ps1
│   ├── test-dxgi.ps1
│   └── collect-diagnostics.ps1
├── logs/
├── cache/
│   └── ort-trt-sm61-fp32/
├── runtime-manifest.json
├── 一键检查并测试.cmd
└── README_中文.md
```

`app` holds only this project's binaries and ONNX Runtime. NVIDIA and Microsoft
runtime DLLs remain grouped by component so their origin and version stay
auditable. The launcher constructs a process-local search path from these fixed
directories before starting the CLI.

## Package assembly

A repository-side PowerShell builder creates the staging directory and ZIP. It
performs these operations in order:

1. Verify the release DLL, import library, CLI, C API header, model, schema, and
   sample input.
2. Copy the known ONNX Runtime 1.17.3 DLL set used by the successful build.
3. Acquire missing CUDA 11.8 runtime components from NVIDIA's official CUDA
   redistributable archives rather than installing CUDA Toolkit.
4. Acquire the matching official cuDNN 8.9 archive and extract only runtime
   files.
5. Acquire TensorRT 8.6.1.6 from NVIDIA's official Windows archive. If NVIDIA
   requires an authenticated/EULA download, accept a previously downloaded
   official archive from `packaging/sm61/installers/`. The builder must stop
   with the exact required filename and official download page when that archive
   is unavailable; it must not create a package labelled complete.
6. Add the Microsoft x64 runtime required by the binaries using Microsoft's
   supported redistributable source or private deployment files.
7. Preserve the license and notice files shipped with every redistributed
   component under `licenses/`.
8. Write a deterministic manifest containing profile name, component version,
   original source URL or archive name, packaged filenames, file sizes, and
   SHA256 values.
9. Reject CUDA 12, cuDNN 9, TensorRT 10/11, or ORT versions other than 1.17.3 in
   the staging tree.
10. Run package-level static checks and safe smoke tests, then create the ZIP.

Downloads are cached outside the final staging directory so repeated builds do
not redownload multi-gigabyte archives. Partial downloads use a temporary name
and are never treated as valid. A recorded SHA256 mismatch is a hard failure.

## One-click setup and test flow

`一键检查并测试.cmd` invokes PowerShell with an execution-policy override scoped
only to that process. The PowerShell scripts do not require elevated privileges.
The flow is:

1. Resolve the package root from the script location, allowing extraction to
   any directory whose path may contain spaces or Chinese characters.
2. Create writable `logs` and `cache/ort-trt-sm61-fp32` directories.
3. Query `nvidia-smi`, require a visible GTX 1080 Ti, record the driver version
   and available VRAM, and explain that its CUDA column describes driver
   compatibility rather than an installed Toolkit.
4. Validate every file against `runtime-manifest.json` and report missing or
   changed files before loading the runtime.
5. Build a child-process-only `PATH` containing the package runtime directories.
6. Check that required DLLs can be found and collect a diagnostic report without
   permanently loading an incompatible NVIDIA stack into the launcher.
7. Run a safe video TensorRT smoke test with `output_enabled=false`, `--dry-run`,
   and the packaged model/schema/sample. The first run may spend several minutes
   building the FP32 engine cache.
8. Run a short second open to exercise reuse of the engine cache.
9. Print a Chinese PASS/FAIL summary and keep the console open when launched by
   double-click. Detailed stdout, stderr, environment, GPU information, and file
   validation results are written under `logs`.

The default one-click test never sends mouse movement or clicks. DXGI testing is
separate because monitor ownership and adapter/output indexes vary between
machines. `test-dxgi.ps1` first lists and probes outputs, then tests the selected
output without arming RP2350.

## Configuration

`config/runtime-sm61.cfg` is based on `runtime.example.cfg` and fixes the
deployment defaults:

```text
backend=ort-tensorrt
tensorrt_cache_path=cache/ort-trt-sm61-fp32
output_enabled=false
input=dxgi
dxgi_timeout_ms=16
```

The ROI remains configurable because the correct crop depends on the production
display resolution. The package does not guess or persist an RP2350 port and
does not turn output on. A DLL host must call
`va_set_output_enabled(runtime, 1)` explicitly after its own hotkey/safety logic
is active.

## Errors and diagnostics

Every script returns a nonzero exit code on failure and presents one primary
remediation in Chinese. Failure classes are kept distinct:

- unsupported or missing GPU/driver;
- package corruption or incomplete extraction;
- wrong CUDA, cuDNN, TensorRT, or ONNX Runtime DLL generation;
- Windows loader failure with the missing DLL name when detectable;
- TensorRT provider registration/session creation failure;
- model/schema contract failure;
- engine cache directory not writable;
- DXGI adapter/output selection failure.

The diagnostics collector records versions and paths but excludes user secrets,
unrelated environment variables, and full process lists. It produces one text
report that can be returned for troubleshooting.

## Verification boundary

The current development machine can verify package construction, manifest
hashes, clean extraction, the CLI's non-TensorRT paths, and hardware-independent
tests. It cannot validate a TensorRT 8.6 engine because its RTX 5060 is outside
TensorRT 8.6's supported compute capabilities.

Final acceptance occurs on the GTX 1080 Ti machine and requires:

1. The ZIP extracts and the one-click script runs without administrator access
   or a preinstalled CUDA Toolkit.
2. The runtime validation reports the exact fixed profile and no DLL from a
   machine-wide CUDA 12/cuDNN 9/TensorRT 11 environment is selected.
3. The first video test creates files in `cache/ort-trt-sm61-fp32`.
4. The second test reuses the cache and processes frames successfully.
5. A separate DXGI dry-run identifies a valid output and processes a frame.
6. RP2350 receives no action during any setup or smoke test.
7. The diagnostics report contains end-to-end FPS plus preprocessing,
   inference, and postprocessing timings for production performance review.
