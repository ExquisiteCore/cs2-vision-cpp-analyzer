# GTX 1080 Ti Portable Runtime Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a version-locked Windows x64 ZIP that runs the existing ORT TensorRT DLL/CLI on a GTX 1080 Ti machine that has only a compatible NVIDIA driver.

**Architecture:** A repository-side PowerShell builder downloads and verifies official redistributable archives, stages private application-local DLL directories, copies immutable templates, generates a SHA256 manifest, and creates a ZIP. Package-side PowerShell scripts construct a child-process-only `PATH`, validate the package and GPU, then run safe video and DXGI diagnostics without arming HID output.

**Tech Stack:** PowerShell 5.1-compatible scripts, Windows CMD launcher, ONNX Runtime GPU 1.17.3, CUDA 11.8 redistributables, cuDNN 8.9.7.29, TensorRT 8.6.1.6, MSVC v14-compatible CRT, native `tar.exe`, existing C++ CLI/DLL.

---

## File map

- `packaging/sm61/PackageTools.psm1`: pure hashing, manifest, archive-layout, runtime-conflict, and path-list helpers shared by the builder tests.
- `packaging/sm61/dependencies.lock.json`: fixed public URLs, versions, archive names, sizes, and SHA256 values; TensorRT is explicitly an authenticated/manual official archive.
- `packaging/sm61/build-portable-package.ps1`: acquire, verify, extract, stage, validate, and archive the package.
- `packaging/sm61/tests/run-tests.ps1`: dependency-free PowerShell unit and template safety tests.
- `packaging/sm61/package/config/runtime-sm61.cfg`: fixed safe runtime defaults.
- `packaging/sm61/package/scripts/common.ps1`: package root, logging, manifest validation, and child runtime environment helpers.
- `packaging/sm61/package/scripts/verify-runtime.ps1`: manifest, required DLL, `nvidia-smi`, GPU, driver, and writable-cache checks.
- `packaging/sm61/package/scripts/test-video.ps1`: two safe TensorRT video passes and timing/cache evidence.
- `packaging/sm61/package/scripts/test-dxgi.ps1`: list/probe/select DXGI output and run a dry test.
- `packaging/sm61/package/scripts/collect-diagnostics.ps1`: bounded troubleshooting report without unrelated environment data.
- `packaging/sm61/package/scripts/setup-and-test.ps1`: one-click orchestration and Chinese PASS/FAIL summary.
- `packaging/sm61/package/一键检查并测试.cmd`: double-click entry point.
- `packaging/sm61/package/README_中文.md`: production-machine instructions and expected first-run behavior.
- `include/vision_analyzer/runtime_status.hpp` and `src/runtime_status.cpp`: stable CLI status formatting with FPS and stage timings consumed by production diagnostics.
- `src/main.cpp` and `tests/test_algorithms.cpp`: emit and verify parseable per-frame performance fields.
- `.gitignore`: exclude local archive cache and generated staging/output.

### Task 1: Add tested package primitives

**Files:**
- Create: `packaging/sm61/tests/run-tests.ps1`
- Create: `packaging/sm61/PackageTools.psm1`

- [ ] **Step 1: Write the failing unit test harness**

Create `run-tests.ps1` with explicit assertions for SHA256 output, immutable-file manifest round-trip, tamper detection, incompatible DLL rejection, required TensorRT archive layout, and stable runtime path ordering. The core fixture is:

```powershell
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot '..\PackageTools.psm1') -Force

function Assert-True([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw "ASSERT: $Message" }
}

$root = Join-Path ([System.IO.Path]::GetTempPath()) ('sm61-package-test-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path (Join-Path $root 'app') -Force | Out-Null
try {
    [IO.File]::WriteAllText((Join-Path $root 'app\probe.txt'), 'known-content')
    $sha = Get-FileSha256 -LiteralPath (Join-Path $root 'app\probe.txt')
    Assert-True ($sha -eq 'C651CCB96B0C0E490DE4CC12B9B46D643E6DBA87840FAB27E2C8D4D5CC2037FA') 'SHA256 must be uppercase and stable'

    Write-PackageManifest -PackageRoot $root -Profile 'sm61-test' -Components @()
    Assert-True ((Test-PackageManifest -PackageRoot $root).Valid) 'fresh manifest must validate'
    [IO.File]::AppendAllText((Join-Path $root 'app\probe.txt'), 'tampered')
    Assert-True (-not (Test-PackageManifest -PackageRoot $root).Valid) 'tampering must fail validation'
} finally {
    if (Test-Path -LiteralPath $root) { Remove-Item -LiteralPath $root -Recurse -Force }
}
```

The same harness creates fake `runtime/cuda-11.8`, `runtime/cudnn-8.9`, and `runtime/tensorrt-8.6.1.6` files to assert that `cudart64_12.dll`, `cudnn64_9.dll`, and `nvinfer_10.dll` are rejected while `cudart64_110.dll`, `cudnn64_8.dll`, and `nvinfer.dll` are accepted.

- [ ] **Step 2: Run the harness and confirm it fails before the module exists**

Run:

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File packaging\sm61\tests\run-tests.ps1
```

Expected: nonzero exit because `PackageTools.psm1` or its exported functions do not exist.

- [ ] **Step 3: Implement the minimal package primitives**

Implement and export these functions in `PackageTools.psm1`:

```powershell
function Get-FileSha256 {
    param([Parameter(Mandatory)][string]$LiteralPath)
    (Get-FileHash -LiteralPath $LiteralPath -Algorithm SHA256).Hash.ToUpperInvariant()
}

function Get-ImmutablePackageFiles {
    param([Parameter(Mandatory)][string]$PackageRoot)
    Get-ChildItem -LiteralPath $PackageRoot -File -Recurse |
        Where-Object {
            $relative = $_.FullName.Substring($PackageRoot.TrimEnd('\').Length + 1).Replace('\', '/')
            $relative -ne 'runtime-manifest.json' -and
            $relative -notlike 'logs/*' -and
            $relative -notlike 'cache/*'
        } |
        Sort-Object FullName
}
```

`Write-PackageManifest` serializes `schemaVersion`, `profile`, `createdUtc`, `components`, and sorted `files` entries (`path`, `size`, `sha256`). `Test-PackageManifest` reports missing, changed, and unexpected immutable files. `Assert-CompatibleRuntimeFiles` rejects CUDA 12 generation names such as `cublas64_12.dll`, `cublasLt64_12.dll`, `cudart64_12.dll`, and `nvrtc64_12x_0.dll`; it also rejects `cudnn*_9.dll`, `nvinfer*_10.dll`, `nvinfer*_11.dll`, or ORT provider DLLs outside the app directory. `Test-TensorRtArchiveLayout` requires `NvInferVersion.h` macros `8`, `6`, `1`, `6` and the four DLLs `nvinfer.dll`, `nvinfer_plugin.dll`, `nvonnxparser.dll`, and `nvinfer_builder_resource.dll`. `Get-RuntimePathEntries` returns app, TensorRT, cuDNN, CUDA, then MSVC directories in that exact order.

- [ ] **Step 4: Run the harness and confirm all primitive tests pass**

Run the same command from Step 2.

Expected: `PASS package tool tests` and exit code 0.

- [ ] **Step 5: Commit the primitives**

```powershell
git add packaging/sm61/PackageTools.psm1 packaging/sm61/tests/run-tests.ps1
git commit -m "test: add portable package primitives"
```

### Task 2: Lock exact redistributable inputs and test the lock

**Files:**
- Create: `packaging/sm61/dependencies.lock.json`
- Modify: `packaging/sm61/tests/run-tests.ps1`

- [ ] **Step 1: Add failing lockfile assertions**

Extend `run-tests.ps1` to require the profile `sm61-ort1173-trt861-fp32`, unique component IDs, HTTPS sources on `github.com`, `developer.download.nvidia.com`, or `developer.nvidia.com`, 64-character SHA256 values for every public archive, and `sourceMode=authenticated-manual` only for TensorRT. Assert these component IDs exactly:

```text
onnxruntime-gpu
cuda-cudart
cuda-cublas
cuda-cufft
cuda-nvrtc
cudnn
tensorrt
msvc-crt
```

- [ ] **Step 2: Run tests and confirm failure because the lockfile is absent**

Run the Task 1 test command.

Expected: nonzero exit with a missing `dependencies.lock.json` error.

- [ ] **Step 3: Add the fixed lockfile**

Record these already verified official public inputs:

```json
{
  "schemaVersion": 1,
  "profile": "sm61-ort1173-trt861-fp32",
  "components": [
    {"id":"onnxruntime-gpu","version":"1.17.3","sourceMode":"public","archive":"onnxruntime-win-x64-gpu-1.17.3.zip","url":"https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-win-x64-gpu-1.17.3.zip","sha256":"82D545BBCDE0976ED3A1BFE44F4DF27721975DCD806AB312C37F05A16C52ACED"},
    {"id":"cuda-cudart","version":"11.8.89","sourceMode":"public","archive":"cuda_cudart-windows-x86_64-11.8.89-archive.zip","url":"https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-11.8.89-archive.zip","sha256":"988CC9E7D3785D4B1975521F312C57C6814CBF15E73A2B7941D961835F2A945E"},
    {"id":"cuda-cublas","version":"11.11.3.6","sourceMode":"public","archive":"libcublas-windows-x86_64-11.11.3.6-archive.zip","url":"https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-11.11.3.6-archive.zip","sha256":"67B0934A6359E4EE26FFF823C356021589D392C4FD49CA12624F570EDC08E2B9"},
    {"id":"cuda-cufft","version":"10.9.0.58","sourceMode":"public","archive":"libcufft-windows-x86_64-10.9.0.58-archive.zip","url":"https://developer.download.nvidia.com/compute/cuda/redist/libcufft/windows-x86_64/libcufft-windows-x86_64-10.9.0.58-archive.zip","sha256":"A4071A85E3983BF42EA7A2E9BEBE3B0B3C9AC258668580ADC32EE1C385F7556F"},
    {"id":"cuda-nvrtc","version":"11.8.89","sourceMode":"public","archive":"cuda_nvrtc-windows-x86_64-11.8.89-archive.zip","url":"https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvrtc/windows-x86_64/cuda_nvrtc-windows-x86_64-11.8.89-archive.zip","sha256":"E5D571247E71E0B0922A929516175844EFA9E7AC424ED3C1B764BFFB4899D3C9"},
    {"id":"cudnn","version":"8.9.7.29","sourceMode":"public","archive":"cudnn-windows-x86_64-8.9.7.29_cuda11-archive.zip","url":"https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-8.9.7.29_cuda11-archive.zip","sha256":"5E45478EFE71A96329E6C0D2A3A2F79C747C15B2A51FEAD4B84C89B02CBF1671"},
    {"id":"tensorrt","version":"8.6.1.6","sourceMode":"authenticated-manual","archive":"TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8.zip","url":"https://developer.nvidia.com/nvidia-tensorrt-8x-download","validation":"version-header-and-required-dlls"},
    {"id":"msvc-crt","version":"14.x-v143","sourceMode":"local-visual-studio-redist","validation":"required-dlls-and-manifest-hashes"}
  ]
}
```

TensorRT has no invented public checksum: NVIDIA redirects anonymous archive access to an authenticated/EULA page. The builder validates the official archive's exact name, embedded version header, required DLL layout, and then records both the archive and packaged DLL hashes in the generated manifest.

- [ ] **Step 4: Run tests and confirm the lock passes**

Expected: all Task 1 and lockfile assertions pass.

- [ ] **Step 5: Commit the lock**

```powershell
git add packaging/sm61/dependencies.lock.json packaging/sm61/tests/run-tests.ps1
git commit -m "build: lock GTX 1080 Ti runtime inputs"
```

### Task 3: Add safe package templates and template tests

**Files:**
- Create: `packaging/sm61/package/config/runtime-sm61.cfg`
- Create: `packaging/sm61/package/scripts/common.ps1`
- Create: `packaging/sm61/package/scripts/verify-runtime.ps1`
- Create: `packaging/sm61/package/scripts/test-video.ps1`
- Create: `packaging/sm61/package/scripts/test-dxgi.ps1`
- Create: `packaging/sm61/package/scripts/collect-diagnostics.ps1`
- Create: `packaging/sm61/package/scripts/setup-and-test.ps1`
- Create: `packaging/sm61/package/一键检查并测试.cmd`
- Create: `packaging/sm61/package/README_中文.md`
- Modify: `packaging/sm61/tests/run-tests.ps1`

- [ ] **Step 1: Add failing safety and interface tests**

Extend the test harness to require every template file, assert
`backend=ort-tensorrt`, `output_enabled=false`, and the relative cache path in
the config, and scan `.cmd`/`.ps1` templates to ensure they never contain
`--output-enabled`, `setx`, or writes to registry/system directories. Require
the video command to contain `--dry-run`, `--max-frames`, `--model`, `--schema`,
and `--tensorrt-cache-path`.

- [ ] **Step 2: Run tests and confirm the missing-template failure**

Expected: nonzero exit listing the first absent package template.

- [ ] **Step 3: Implement `common.ps1` and `verify-runtime.ps1`**

`common.ps1` resolves the package root from `$PSScriptRoot`, imports no external
modules, creates `logs`/`cache`, validates `runtime-manifest.json`, and runs a
script block after temporarily assigning this order to `$env:PATH`:

```powershell
@(
    (Join-Path $PackageRoot 'app'),
    (Join-Path $PackageRoot 'runtime\tensorrt-8.6.1.6'),
    (Join-Path $PackageRoot 'runtime\cudnn-8.9'),
    (Join-Path $PackageRoot 'runtime\cuda-11.8'),
    (Join-Path $PackageRoot 'runtime\msvc-x64')
) -join ';'
```

`verify-runtime.ps1` requires all direct ORT provider imports plus TensorRT's
builder resource, verifies a writable cache, calls `nvidia-smi` with the query
`name,driver_version,memory.total`, requires an exact GTX 1080 Ti unless
`-AllowUnsupportedGpu` is supplied for developer diagnostics, and emits one
structured result object plus a text log. `-StaticOnly` performs manifest, DLL,
path, model/schema, cache-write, and CLI-help checks but deliberately skips
loading ORT providers or running inference on the unsupported development GPU.

- [ ] **Step 4: Implement the safe video, DXGI, diagnostics, and orchestration scripts**

`test-video.ps1` runs this CLI shape twice and requires both exit codes to be 0:

```powershell
& $exe --config $config --backend ort-tensorrt `
    --tensorrt-cache-path $cache --model $model --schema $schema `
    --video $sample --player-side unknown --dry-run --warmup-frames 1 `
    --max-frames 3 --status-every 1 --action-log $actions
```

It parses status output for FPS and preprocessing/inference/postprocessing
milliseconds, confirms the first run leaves at least one nonempty cache file,
and records the second-open result. `test-dxgi.ps1` always runs list and probe
first, accepts explicit adapter/output parameters, and uses `--verify-input`
plus a three-frame `--dry-run` without any HID port or output flag.

`setup-and-test.ps1` calls validation then video testing, writes a Chinese
summary, and never calls DXGI automatically. `collect-diagnostics.ps1` records
the manifest profile, package paths, required file presence, GPU query, CLI
version/help result, cache listing, and existing package logs only. The CMD
launcher uses `%~dp0`, process-scoped execution policy, preserves the exit code,
and pauses when double-clicked.

- [ ] **Step 5: Add the Chinese guide and fixed config**

Document extraction, double-click testing, expected multi-minute first engine
build, later cache reuse, separate DXGI invocation, the meaning of
`nvidia-smi`'s CUDA column, log collection, and the explicit rule that a host
application alone may arm `va_set_output_enabled`.

- [ ] **Step 6: Run template tests and commit**

Expected: `PASS package tool tests` with all safety assertions passing.

```powershell
git add packaging/sm61/package packaging/sm61/tests/run-tests.ps1
git commit -m "feat: add safe one-click production diagnostics"
```

### Task 3A: Expose parseable CLI performance metrics

**Files:**
- Create: `include/vision_analyzer/runtime_status.hpp`
- Create: `src/runtime_status.cpp`
- Modify: `src/main.cpp`
- Modify: `tests/test_algorithms.cpp`
- Modify: `xmake.lua`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add a failing formatter test**

Construct a `RuntimeStepResult` with known FPS and preprocessing, inference,
postprocessing, total, detection, target, action, and lock values. Require one
stable line containing `fps=123.46`, `preprocess_ms=1.25`,
`inference_ms=4.50`, `postprocess_ms=0.75`, and `total_ms=6.50`.

- [ ] **Step 2: Run the algorithm tests and confirm the formatter is missing**

Run `xmake build -P . vision_analyzer_tests`. Expected: compilation fails
because `runtime_status.hpp` or `format_runtime_status` does not exist.

- [ ] **Step 3: Implement the formatter and use it in the CLI**

Add the formatter to both core source lists and replace the ad-hoc status output
in `main.cpp`. Preserve existing frame/detection/action fields while adding
fixed two-decimal `fps`, preprocessing, inference, postprocessing, and total
milliseconds in the token order consumed by `test-video.ps1`.

- [ ] **Step 4: Run algorithm, C API, and packaging tests**

Expected: all three suites pass and the formatter test proves the CLI/package
contract.

- [ ] **Step 5: Commit the CLI metric contract**

```powershell
git add include/vision_analyzer/runtime_status.hpp src/runtime_status.cpp src/main.cpp tests/test_algorithms.cpp xmake.lua CMakeLists.txt docs/superpowers/plans/2026-07-21-sm61-portable-runtime-package.md
git commit -m "feat: expose parseable runtime timing metrics"
```

### Task 4: Implement and test the package builder

**Files:**
- Create: `packaging/sm61/build-portable-package.ps1`
- Modify: `packaging/sm61/PackageTools.psm1`
- Modify: `packaging/sm61/tests/run-tests.ps1`
- Modify: `.gitignore`

- [ ] **Step 1: Add failing builder-helper fixture tests**

Test that archive acquisition accepts an already cached file only after SHA256
validation, rejects a mismatched cache entry, flattens only `bin/*.dll` from CUDA
and cuDNN archives, copies only `lib/*.dll` from TensorRT, preserves component
license/notice files, and refuses to label a staging tree complete when the
authenticated TensorRT archive is absent.

- [ ] **Step 2: Run tests and confirm builder helpers are missing**

Expected: nonzero exit naming the first missing acquisition/extraction helper.

- [ ] **Step 3: Implement resumable, verified acquisition and extraction**

Use `curl.exe --location --fail --retry 3 --continue-at - --output` for public
archives, write to `.partial`, rename only after the locked hash matches, and
use `Expand-Archive` into a component-specific temporary directory. A cached
archive with a wrong hash is quarantined with a `.bad-<timestamp>` suffix rather
than reused.

The TensorRT gate searches the explicit `-TensorRtArchive` parameter and then
`packaging/sm61/installers/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8.zip`.
When absent it exits before ZIP creation and prints the exact official page and
filename. When present it runs `Test-TensorRtArchiveLayout` and records the
computed archive SHA256.

- [ ] **Step 4: Implement deterministic staging**

The builder parameters default from repository layout but allow overrides for
release output, ORT root, model/schema, sample video, archive cache, staging,
output ZIP, MSVC redistributable root, and TensorRT archive. It:

1. clears only its resolved staging directory after verifying that path is under
   the supplied output root;
2. copies the package templates;
3. copies `vision_runtime.dll`, `.lib`, `vision_analyzer.exe`, and the C header;
4. copies ORT 1.17.3 provider DLLs and license;
5. extracts and flattens the four CUDA components, cuDNN, and TensorRT runtime
   DLLs into their versioned directories;
6. locates the newest complete desktop x64 `Microsoft.VC14*.CRT` and copies its DLLs/license;
7. trims five seconds from `videos/02.mp4` with installed `ffmpeg`, failing
   clearly if ffmpeg is absent;
8. copies `best.onnx` and its exact schema;
9. rejects incompatible DLL names and verifies required imports;
10. writes `runtime-manifest.json`, runs template/unit tests, and creates a ZIP
    with `tar.exe -a -cf` so Zip64-sized output is supported.

- [ ] **Step 5: Ignore local generated state**

Add these entries to `.gitignore`:

```gitignore
.packaging-cache/
packaging/sm61/installers/*.zip
packaging/sm61/out/
```

- [ ] **Step 6: Run unit tests and a missing-TensorRT dry assembly**

Run:

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File packaging\sm61\tests\run-tests.ps1
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File packaging\sm61\build-portable-package.ps1 -DownloadPublicDependencies
```

Expected: unit tests pass; the builder downloads/verifies public dependencies
and then stops with the exact official TensorRT archive requirement without
creating a final ZIP.

- [ ] **Step 7: Commit the builder**

```powershell
git add .gitignore packaging/sm61
git commit -m "build: assemble version-locked SM61 runtime package"
```

### Task 5: Acquire dependencies and assemble the real package

**Files:**
- Local ignored input: `packaging/sm61/installers/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8.zip`
- Generated ignored directory: `D:\project\cs2-vision-trainer\dist\cs2-vision-runtime-sm61`
- Generated artifact: `D:\project\cs2-vision-trainer\dist\cs2-vision-runtime-sm61.zip`

- [ ] **Step 1: Download all anonymous official archives**

Run the builder with public dependency acquisition. Expected verified downloads
are ORT 1.17.3, CUDA cudart/cublas/cufft/nvrtc 11.8 components, and cuDNN
8.9.7.29 CUDA 11.

- [ ] **Step 2: Resolve the authenticated TensorRT archive gate**

If the official archive is not already local, have the user log into NVIDIA in
Chrome, accept the TensorRT 8.x download terms, and download exactly
`TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8.zip`. Do not substitute TensorRT
10/11 or an unverified third-party binary mirror.

- [ ] **Step 3: Build the complete staging tree and ZIP**

Run:

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass `
  -File packaging\sm61\build-portable-package.ps1 `
  -DownloadPublicDependencies `
  -TensorRtArchive packaging\sm61\installers\TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8.zip
```

Expected: manifest validation passes and both the staging directory and final
ZIP are produced.

### Task 6: Verify the final archive before handoff

**Files:**
- Temporary extraction: `D:\project\cs2-vision-trainer\dist\verify-sm61-package`
- Generated report: package `logs/` directory

- [ ] **Step 1: Run all project and packaging tests**

Run:

```powershell
xmake run vision_analyzer_tests
xmake run vision_runtime_c_api_tests
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File packaging\sm61\tests\run-tests.ps1
```

Expected: 37 algorithm tests, 5 C API tests, and all package tests pass.

- [ ] **Step 2: Extract the final ZIP into a clean path**

Resolve the absolute verification path, ensure it stays under the project
`dist` directory, remove only that verified path, and extract with `tar.exe -xf`.

- [ ] **Step 3: Run static validation from the clean extraction**

Run `scripts/verify-runtime.ps1 -AllowUnsupportedGpu -StaticOnly` from the
extracted package. Expected: manifest, DLL generation, writable path, CLI help,
model/schema, and runtime directory checks pass. The RTX 5060 mismatch is
reported as an allowed development-only condition.

- [ ] **Step 4: Run a non-TensorRT local smoke check**

Run the packaged CLI with `--backend opencv-onnx`, the packaged sample/model,
`--dry-run`, and three frames. Expected: three frames process successfully with
no HID output. Do not create or ship a TensorRT 8.6 engine cache from the RTX
5060 machine.

- [ ] **Step 5: Audit the archive and handoff facts**

Record ZIP size/SHA256, profile, component versions, file count, and commands
the user should run. Confirm the archive contains no `.engine` files, no CUDA
12/cuDNN 9/TensorRT 10/11 DLLs, and no script contains `--output-enabled`.

- [ ] **Step 6: Run final repository checks**

```powershell
git diff --check
git status --short --branch
```

Expected: tracked implementation commits are clean; only ignored downloaded
archives and generated `dist` artifacts remain outside Git status.
