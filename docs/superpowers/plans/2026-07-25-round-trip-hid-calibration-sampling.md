# Round-Trip HID Calibration Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make final HID calibration collect robust signed samples from exact round trips without ever rescaling an ordinary account to 2048 after a bad phase-correlation estimate.

**Architecture:** Keep the existing calibration-only discovery ladder and runtime clamp. Add pure, unit-tested round-trip command planning, three-frame estimation, and signed-sample construction helpers; the live HID loop only executes those plans and logs the two measured legs. Ordinary target shifts become `8/24/48` while a 60-counts-per-pixel account still reaches `480/1024/2048`.

**Tech Stack:** C++17, OpenCV phase correlation, xmake, MSVC 2022, Windows PowerShell 5.1, RP2350 protocol v2 SDK, ONNX Runtime GPU 1.17.3, TensorRT 8.6.1.6, CUDA 11.8, cuDNN 8.9.7.

---

## File map

- `include/vision_analyzer/calibration.hpp`: round-trip measurement and command value types plus pure helper declarations.
- `src/calibration.cpp`: smaller target-level derivation, three-frame estimation, command planning, signed sample construction, and live round-trip execution.
- `tests/test_algorithms.cpp`: RED/GREEN coverage for target levels, alternating commands, real outward/return measurements, signed sample mapping, and fitter recovery from one rejected path.
- `README.md`: developer explanation of final round trips and the discovery-only 2048 ceiling.
- `packaging/sm61/package/README_中文.md`: production behavior, expected logs, and no-final-escalation rule.
- Generated `D:/project/cs2-vision-trainer/dist/cs2-vision-runtime-sm61-round-trip/`: fully validated second full package.
- Generated `C:/Users/xiaol/Downloads/cs2-vision-runtime-sm61-round-trip-hotfix.zip`: overlay relative to the already deployed first hotfix.

### Task 1: Reduce ordinary visual target shifts without reducing the low-sensitivity range

**Files:**
- Modify: `tests/test_algorithms.cpp:738-749`
- Modify: `src/calibration.cpp:162-180`

- [ ] **Step 1: Change the ordinary-level expectation first**

Replace the normal part of `test_calibration_level_plan_compresses_low_sensitivity_range` with:

```cpp
const CalibrationLevelPlan normal = derive_calibration_level_plan(2.0, 1.5);
require(normal.counts == std::array<int, 3>{16, 48, 96},
        "normal sensitivity should use 8/24/48-pixel targets");
require(normal.target_shift_px == std::array<double, 3>{8.0, 24.0, 48.0},
        "normal target shifts should remain in the reliable phase-correlation range");
```

Keep the existing low-sensitivity assertion:

```cpp
const CalibrationLevelPlan low = derive_calibration_level_plan(60.0, 1.5);
require(low.counts == std::array<int, 3>{480, 1024, 2048},
        "low sensitivity should use the complete calibration range");
```

- [ ] **Step 2: Run the RED test**

Run:

```powershell
xmake build -P . vision_analyzer_tests
xmake run -P . vision_analyzer_tests
```

Expected: the build succeeds and the test run fails with
`normal sensitivity should use 8/24/48-pixel targets` because the implementation still returns
`16,64,160`.

- [ ] **Step 3: Implement the smaller ordinary targets**

In `derive_calibration_level_plan` replace the three target calculations with:

```cpp
const double high = std::min(48.0, static_cast<double>(maximum_counts) / counts_per_pixel);
const double low = std::min(8.0, high / 4.0);
const double middle = std::min(24.0, high / 2.0);
```

Do not change the 2048 maximum, minimum measurable-shift rejection, or strictly increasing integer clamping.

- [ ] **Step 4: Run GREEN tests**

Run:

```powershell
xmake build -P . vision_analyzer_tests
xmake run -P . vision_analyzer_tests
```

Expected: `algorithm tests passed`.

- [ ] **Step 5: Commit**

```powershell
git add src/calibration.cpp tests/test_algorithms.cpp
git commit -m "fix: keep ordinary calibration shifts measurable"
```

### Task 2: Add tested round-trip planning, estimation, and signed sample construction

**Files:**
- Modify: `tests/test_algorithms.cpp:1-15,727-850,1250-1270`
- Modify: `include/vision_analyzer/calibration.hpp:32-125`
- Modify: `src/calibration.cpp:114-330`

- [ ] **Step 1: Add failing round-trip helper tests**

Add `#include <opencv2/imgproc.hpp>` to `tests/test_algorithms.cpp`, then add:

```cpp
void test_round_trip_estimation_measures_outward_and_inverse_frames() {
    cv::Mat baseline(64, 64, CV_32F);
    cv::randu(baseline, 0.0F, 255.0F);
    cv::Mat moved;
    const cv::Mat transform = (cv::Mat_<double>(2, 3) <<
        1.0, 0.0, 5.0,
        0.0, 1.0, -3.0);
    cv::warpAffine(
        baseline,
        moved,
        transform,
        baseline.size(),
        cv::INTER_LINEAR,
        cv::BORDER_WRAP
    );

    const CalibrationRoundTripMeasurement measurement =
        estimate_calibration_round_trip(baseline, moved, baseline);
    require_near(static_cast<float>(measurement.outward.shift.x), 5.0F, 0.1F,
                 "outward leg should use baseline-to-moved frames");
    require_near(static_cast<float>(measurement.outward.shift.y), -3.0F, 0.1F,
                 "outward Y shift should be preserved");
    require_near(static_cast<float>(measurement.inverse.shift.x), -5.0F, 0.1F,
                 "inverse leg should use moved-to-returned frames");
    require_near(static_cast<float>(measurement.inverse.shift.y), 3.0F, 0.1F,
                 "inverse Y shift should be measured independently");
    require(measurement.outward.response > 0.90 && measurement.inverse.response > 0.90,
            "both synthetic round-trip legs should have strong responses");
}

void test_round_trip_command_plan_alternates_without_final_escalation() {
    std::array<CalibrationAxisDiscovery, 2> discoveries;
    discoveries[0].levels.counts = {11, 32, 65};
    discoveries[1].levels.counts = {12, 33, 66};

    const auto commands = plan_calibration_round_trip_commands(discoveries, 2);
    std::vector<int> actual;
    for (const auto& command : commands) {
        actual.push_back(command.outward_counts);
    }
    require(actual == std::vector<int>({
                11, -11, 32, -32, 65, -65,
                12, -12, 33, -33, 66, -66,
            }),
            "round-trip plans should alternate signs and keep discovered counts");
}

void test_round_trip_samples_assign_real_signed_legs() {
    CalibrationRoundTripMeasurement measurement;
    measurement.outward = {{-12.0, 0.25}, 0.70};
    measurement.inverse = {{11.5, -0.20}, 0.65};

    const auto x_samples = make_calibration_round_trip_samples(
        0, 1, 40, measurement
    );
    require(x_samples[0].counts_dx == 40 && x_samples[0].counts_dy == 0,
            "X outward sample should keep its signed command");
    require(x_samples[1].counts_dx == -40 && x_samples[1].counts_dy == 0,
            "X return sample should use the exact inverse command");
    require_near(static_cast<float>(x_samples[0].visual_shift.x), -12.0F, 0.001F,
                 "outward visual shift must not be synthesized");
    require_near(static_cast<float>(x_samples[1].visual_shift.x), 11.5F, 0.001F,
                 "return visual shift must be preserved independently");
    require_near(static_cast<float>(x_samples[1].phase_response), 0.65F, 0.001F,
                 "return response must be preserved independently");

    const auto y_samples = make_calibration_round_trip_samples(
        1, 2, -50, measurement
    );
    require(y_samples[0].counts_dx == 0 && y_samples[0].counts_dy == -50,
            "Y outward sample should keep a negative command");
    require(y_samples[1].counts_dx == 0 && y_samples[1].counts_dy == 50,
            "Y return sample should invert the command");
}

void test_round_trip_sampling_can_recover_from_one_bad_path() {
    std::vector<CalibrationSample> samples = {
        {0, 0, {0.01, -0.01}, 0.99, -1},
    };
    const std::array<int, 3> counts = {16, 48, 96};
    const std::array<double, 3> shifts = {8.0, 24.0, 48.0};
    for (std::size_t axis = 0; axis < 2; ++axis) {
        for (std::size_t level = 0; level < counts.size(); ++level) {
            CalibrationRoundTripMeasurement bad;
            bad.outward = {{0.0, 0.0}, 0.05};
            bad.inverse = {{0.0, 0.0}, 0.05};
            const auto rejected = make_calibration_round_trip_samples(
                axis, static_cast<int>(level), counts[level], bad
            );
            samples.insert(samples.end(), rejected.begin(), rejected.end());

            CalibrationRoundTripMeasurement good;
            if (axis == 0) {
                good.outward = {{shifts[level], 0.1}, 0.90};
                good.inverse = {{-shifts[level], -0.1}, 0.90};
            } else {
                good.outward = {{0.1, shifts[level]}, 0.90};
                good.inverse = {{-0.1, -shifts[level]}, 0.90};
            }
            const auto accepted = make_calibration_round_trip_samples(
                axis, static_cast<int>(level), -counts[level], good
            );
            samples.insert(samples.end(), accepted.begin(), accepted.end());
        }
    }

    const auto profile = fit_adaptive_hid_calibration(
        samples, {1920, 1080}, kCalibrationRuntimeMaxStep
    );
    require(profile.valid,
            "one reliable opposite round trip should fill both signed fitter buckets");
    require(profile.accepted_samples == 12,
            "only the twelve reliable signed samples should be accepted");
    require(profile.max_step == 120, "round-trip fitting must retain the runtime clamp");
}
```

Register all four tests in `main()` immediately after the existing discovery tests.

- [ ] **Step 2: Run the RED build**

Run:

```powershell
xmake build -P . vision_analyzer_tests
```

Expected: compilation fails because `CalibrationRoundTripMeasurement`,
`CalibrationRoundTripCommand`, `estimate_calibration_round_trip`,
`plan_calibration_round_trip_commands`, and
`make_calibration_round_trip_samples` do not exist.

- [ ] **Step 3: Declare the round-trip units**

After `VisualShiftEstimate` in `calibration.hpp` add:

```cpp
struct CalibrationRoundTripMeasurement {
    VisualShiftEstimate outward;
    VisualShiftEstimate inverse;
};
```

After `CalibrationAxisDiscovery` add:

```cpp
struct CalibrationRoundTripCommand {
    std::size_t axis = 0;
    int level = 0;
    int repeat = 0;
    int outward_counts = 0;
};
```

Add these declarations after `estimate_visual_shift`:

```cpp
[[nodiscard]] CalibrationRoundTripMeasurement estimate_calibration_round_trip(
    const cv::Mat& baseline,
    const cv::Mat& moved,
    const cv::Mat& returned
);
[[nodiscard]] std::vector<CalibrationRoundTripCommand>
plan_calibration_round_trip_commands(
    const std::array<CalibrationAxisDiscovery, 2>& discoveries,
    int repeats
);
[[nodiscard]] std::array<CalibrationSample, 2>
make_calibration_round_trip_samples(
    std::size_t axis,
    int level,
    int outward_counts,
    const CalibrationRoundTripMeasurement& measurement
);
```

- [ ] **Step 4: Implement three-frame estimation**

Immediately after `estimate_visual_shift` in `calibration.cpp` add:

```cpp
CalibrationRoundTripMeasurement estimate_calibration_round_trip(
    const cv::Mat& baseline,
    const cv::Mat& moved,
    const cv::Mat& returned
) {
    return {
        estimate_visual_shift_with_response(baseline, moved),
        estimate_visual_shift_with_response(moved, returned),
    };
}
```

- [ ] **Step 5: Implement command planning**

After `discover_calibration_axis` add:

```cpp
std::vector<CalibrationRoundTripCommand> plan_calibration_round_trip_commands(
    const std::array<CalibrationAxisDiscovery, 2>& discoveries,
    int repeats
) {
    if (repeats < 2) {
        throw std::invalid_argument("HID calibration round-trip repeats must be at least two");
    }
    std::vector<CalibrationRoundTripCommand> commands;
    commands.reserve(
        discoveries.size() * kHidCalibrationLevels * static_cast<std::size_t>(repeats)
    );
    for (std::size_t axis = 0; axis < discoveries.size(); ++axis) {
        for (std::size_t level = 0; level < kHidCalibrationLevels; ++level) {
            const int counts = discoveries[axis].levels.counts[level];
            if (counts < kCalibrationProbeMinimumCounts ||
                counts > kCalibrationProbeMaximumCounts) {
                throw std::invalid_argument(
                    "HID calibration round-trip count is outside the discovered range"
                );
            }
            for (int repeat = 0; repeat < repeats; ++repeat) {
                commands.push_back({
                    axis,
                    static_cast<int>(level),
                    repeat,
                    repeat % 2 == 0 ? counts : -counts,
                });
            }
        }
    }
    return commands;
}
```

- [ ] **Step 6: Implement signed sample construction**

Immediately after the command planner add:

```cpp
std::array<CalibrationSample, 2> make_calibration_round_trip_samples(
    std::size_t axis,
    int level,
    int outward_counts,
    const CalibrationRoundTripMeasurement& measurement
) {
    if (axis > 1 || level < 0 || level >= static_cast<int>(kHidCalibrationLevels) ||
        outward_counts == 0 ||
        outward_counts < -kCalibrationProbeMaximumCounts ||
        outward_counts > kCalibrationProbeMaximumCounts) {
        throw std::invalid_argument("invalid HID calibration round-trip sample values");
    }
    const auto make_sample = [&](int signed_counts, const VisualShiftEstimate& estimate) {
        return CalibrationSample{
            axis == 0 ? signed_counts : 0,
            axis == 1 ? signed_counts : 0,
            estimate.shift,
            estimate.response,
            level,
        };
    };
    return {
        make_sample(outward_counts, measurement.outward),
        make_sample(-outward_counts, measurement.inverse),
    };
}
```

- [ ] **Step 7: Run GREEN tests**

Run:

```powershell
xmake build -P . vision_analyzer_tests
xmake run -P . vision_analyzer_tests
```

Expected: `algorithm tests passed`.

- [ ] **Step 8: Commit**

```powershell
git add include/vision_analyzer/calibration.hpp src/calibration.cpp tests/test_algorithms.cpp
git commit -m "feat: plan signed HID calibration round trips"
```

### Task 3: Execute final live sampling only through tested round trips

**Files:**
- Modify: `include/vision_analyzer/calibration.hpp:118-126`
- Modify: `src/calibration.cpp:329-590`
- Modify: `tests/test_algorithms.cpp:824-836,1264-1268`

- [ ] **Step 1: Remove the obsolete final-rescale contract**

Delete both retry-selector tests:

```text
test_calibration_sample_retry_does_not_rescale_low_confidence_shift
test_calibration_sample_retry_can_exceed_runtime_step
```

Delete their calls from `main()`. Delete the
`select_calibration_sample_retry_count` declaration from `calibration.hpp` and its implementation
from `calibration.cpp`. Discovery continues to use `adjust_calibration_probe_count` and is not
changed.

- [ ] **Step 2: Make the balanced helper return both real legs**

Replace `measure_balanced_probe` with:

```cpp
const auto measure_balanced_probe = [&](std::size_t axis, int signed_counts) {
    const int dx = axis == 0 ? signed_counts : 0;
    const int dy = axis == 1 ? signed_counts : 0;
    send_move(dx, dy);
    bool inverse_required = true;
    try {
        wait_for_settle();
        CapturedFrame moved = capture(
            "failed to capture moved DXGI frame for HID calibration"
        );
        send_move(-dx, -dy);
        inverse_required = false;
        wait_for_settle();
        CapturedFrame returned = capture(
            "failed to capture returned DXGI frame after HID calibration probe"
        );
        const CalibrationRoundTripMeasurement measurement =
            estimate_calibration_round_trip(
                baseline.image,
                moved.image,
                returned.image
            );
        baseline = std::move(returned);
        return measurement;
    } catch (...) {
        if (inverse_required) {
            try {
                send_move(-dx, -dy);
            } catch (...) {
            }
        }
        throw;
    }
};
```

The inverse send occurs before phase estimation, so any estimator exception happens only after
the physical return command.

- [ ] **Step 3: Keep discovery on the outward leg**

In the discovery callback replace the single estimate assignment with:

```cpp
const CalibrationRoundTripMeasurement measurement =
    measure_balanced_probe(axis, counts);
const VisualShiftEstimate& estimate = measurement.outward;
```

Keep all existing `probe_discovery` and `probe_levels` fields.

- [ ] **Step 4: Replace the sign/retry loop with planned round trips**

Delete the nested final loop and its `probe_retry` branch. Insert:

```cpp
const auto round_trip_commands = plan_calibration_round_trip_commands(
    discoveries,
    options.calibration_repeats
);
for (const auto& command : round_trip_commands) {
    const CalibrationRoundTripMeasurement measurement =
        measure_balanced_probe(command.axis, command.outward_counts);
    const auto round_trip_samples = make_calibration_round_trip_samples(
        command.axis,
        command.level,
        command.outward_counts,
        measurement
    );
    for (std::size_t leg = 0; leg < round_trip_samples.size(); ++leg) {
        const CalibrationSample& sample = round_trip_samples[leg];
        samples.push_back(sample);
        output << "sample"
               << " type=move"
               << " leg=" << (leg == 0 ? "outward" : "return")
               << " axis=" << (command.axis == 0 ? 'x' : 'y')
               << " level=" << command.level
               << " repeat=" << command.repeat
               << " counts_dx=" << sample.counts_dx
               << " counts_dy=" << sample.counts_dy
               << " visual_shift_x=" << sample.visual_shift.x
               << " visual_shift_y=" << sample.visual_shift.y
               << " response=" << sample.phase_response
               << '\n';
    }
}
```

Do not change the call that passes `kCalibrationRuntimeMaxStep` to the fitter.

- [ ] **Step 5: Build and run all native tests**

Run:

```powershell
xmake build -P . vision_analyzer_tests
xmake build -P . vision_runtime_c_api_tests
xmake run -P . vision_analyzer_tests
xmake run -P . vision_runtime_c_api_tests
```

Expected:

```text
algorithm tests passed
C API tests passed
```

- [ ] **Step 6: Confirm the live source has no final retry path**

Run:

```powershell
rg -n "probe_retry|select_calibration_sample_retry_count" include src tests
```

Expected: no matches. `adjust_calibration_probe_count` remains because discovery uses it.

- [ ] **Step 7: Commit**

```powershell
git add include/vision_analyzer/calibration.hpp src/calibration.cpp tests/test_algorithms.cpp
git commit -m "fix: sample HID calibration as signed round trips"
```

### Task 4: Document the observable production behavior

**Files:**
- Modify: `README.md:324-345`
- Modify: `packaging/sm61/package/README_中文.md:99-118`

- [ ] **Step 1: Update developer documentation**

Extend `README.md` HID calibration section with:

```text
Final sampling uses exact outward/return pairs and measures both legs. Two repeats start in
opposite directions. Ordinary target shifts are 8/24/48 pixels; low-sensitivity discovery may
still use 2048 counts. Final sampling never rescales a measured level, and accepted runtime
output remains limited to 120.
```

- [ ] **Step 2: Update production instructions**

Add this operator rule to `README_中文.md`:

```text
probe_levels 之后的 sample counts 应当只来自该轴列出的三个档位及其相反数。除非
probe_levels 本身包含 2048，否则正式 sample 不应突然出现 2048。每个 repeat 会看到
leg=outward 和 leg=return。无需重刷固件或重装运行环境。
```

- [ ] **Step 3: Check and commit**

Run:

```powershell
git diff --check
git add README.md packaging/sm61/package/README_中文.md
git commit -m "docs: explain round-trip HID calibration"
```

### Task 5: Rebuild with fixed production dependencies and run safety tests

**Files:**
- Verify only; no expected source changes.

- [ ] **Step 1: Cleanly configure the isolated worktree**

Run:

```powershell
xmake f -c -m release --onnxruntime_root=D:\Tool\onnxruntime-win-x64-gpu-1.17.3 --hid_sdk_root=D:\project\cs2-vision-trainer\tools\rp2350_hid_bridge_cpp -P .
xmake -P .
```

Expected: release builds of `vision_runtime.dll`, `vision_runtime.lib`, and
`vision_analyzer.exe` succeed.

- [ ] **Step 2: Run native and package tests**

```powershell
xmake run -P . vision_analyzer_tests
xmake run -P . vision_runtime_c_api_tests
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File .\packaging\sm61\tests\run-tests.ps1 -PythonProjectRoot D:\project\cs2-vision-trainer
```

Expected: algorithm tests, C API tests, and all 26 package tests pass.

- [ ] **Step 3: Audit**

```powershell
git diff --check main...HEAD
git status --short
git log --oneline main..HEAD
```

Expected: no whitespace errors, a clean worktree, and focused calibration commits.

### Task 6: Build and verify the second incremental package

**Files:**
- Generated: `D:/project/cs2-vision-trainer/dist/cs2-vision-runtime-sm61-round-trip/`
- Generated: `D:/project/cs2-vision-trainer/dist/cs2-vision-runtime-sm61-round-trip.zip`
- Generated: `D:/project/cs2-vision-trainer/dist/hotfix-round-trip-stage/cs2-vision-runtime-sm61/`
- Generated: `C:/Users/xiaol/Downloads/cs2-vision-runtime-sm61-round-trip-hotfix.zip`

- [ ] **Step 1: Build the second full package**

Run:

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File .\packaging\sm61\build-portable-package.ps1 -ReleaseRoot .\build\windows\x64\release -OrtRoot D:\Tool\onnxruntime-win-x64-gpu-1.17.3 -ModelPath D:\project\cs2-vision-trainer\runs\detect\train-2\weights\best.onnx -SchemaPath D:\project\cs2-vision-trainer\runs\detect\train-2\weights\best.onnx.schema.json -SampleVideoPath D:\project\cs2-vision-trainer\videos\02.mp4 -DependencyCache D:\project\cs2-vision-trainer\dist\.sm61-cache -OutputRoot D:\project\cs2-vision-trainer\dist\cs2-vision-runtime-sm61-round-trip -OutputZip D:\project\cs2-vision-trainer\dist\cs2-vision-runtime-sm61-round-trip.zip -TensorRtArchive C:\Users\xiaol\Downloads\TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8.zip -PythonProjectRoot D:\project\cs2-vision-trainer
```

Require profile `sm61-ort1173-trt861-fp32`, component
`rp2350-hid-sdk=protocol-v2`, and a valid manifest.

- [ ] **Step 2: Generate an overlay relative to the deployed first hotfix**

Run:

```powershell
$dist = [IO.Path]::GetFullPath('D:\project\cs2-vision-trainer\dist').TrimEnd('\')
$oldRoot = Join-Path $dist 'cs2-vision-runtime-sm61-low-sensitivity'
$newRoot = Join-Path $dist 'cs2-vision-runtime-sm61-round-trip'
$stageParent = Join-Path $dist 'hotfix-round-trip-stage'
$stageRoot = Join-Path $stageParent 'cs2-vision-runtime-sm61'
$zip = 'C:\Users\xiaol\Downloads\cs2-vision-runtime-sm61-round-trip-hotfix.zip'

$resolvedStageParent = [IO.Path]::GetFullPath($stageParent).TrimEnd('\')
if (-not $resolvedStageParent.StartsWith($dist + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to recreate stage outside dist: $resolvedStageParent"
}
if (Test-Path -LiteralPath $resolvedStageParent) {
    Remove-Item -LiteralPath $resolvedStageParent -Recurse -Force
}
New-Item -ItemType Directory -Path $stageRoot -Force | Out-Null

$oldManifest = Get-Content -Raw -LiteralPath (Join-Path $oldRoot 'runtime-manifest.json') |
    ConvertFrom-Json
$newManifest = Get-Content -Raw -LiteralPath (Join-Path $newRoot 'runtime-manifest.json') |
    ConvertFrom-Json
$oldByPath = @{}
foreach ($file in @($oldManifest.files)) {
    $oldByPath[$file.path.Replace('\', '/')] = $file.sha256
}
$changed = @($newManifest.files | Where-Object {
    $path = $_.path.Replace('\', '/')
    -not $oldByPath.ContainsKey($path) -or $oldByPath[$path] -ne $_.sha256
})
foreach ($file in $changed) {
    $relative = $file.path.Replace('/', '\')
    $source = Join-Path $newRoot $relative
    $destination = Join-Path $stageRoot $relative
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination -Force
}
Copy-Item -LiteralPath (Join-Path $newRoot 'runtime-manifest.json') -Destination (Join-Path $stageRoot 'runtime-manifest.json') -Force

$forbidden = @($changed | Where-Object {
    $_.path -match '^(model|runtime|cache)/' -or
    $_.path -match '^app/onnxruntime' -or
    $_.path -match '^app/(cublas|cudart|cudnn|nvinfer|nvonnxparser)'
})
if ($forbidden.Count -ne 0) {
    throw "Unexpected dependency/model files in overlay: $($forbidden.path -join ', ')"
}
if (Test-Path -LiteralPath $zip) {
    Remove-Item -LiteralPath $zip -Force
}
Compress-Archive -Path $stageRoot -DestinationPath $zip -Force
$changed.path
```

Expected native changes are `app/vision_runtime.dll` and `app/vision_analyzer.exe`.
`vision_runtime.lib` may remain byte-identical because the C exports do not change. Documentation
and `runtime-manifest.json` must be included. No dependency, model, cache, firmware, or Python
files may appear.

- [ ] **Step 3: Simulate production overlay and validate the full manifest**

```powershell
$verifyRoot = Join-Path $dist 'verify-sm61-round-trip'
$resolvedVerify = [IO.Path]::GetFullPath($verifyRoot).TrimEnd('\')
if (-not $resolvedVerify.StartsWith($dist + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to recreate verification root outside dist: $resolvedVerify"
}
if (Test-Path -LiteralPath $resolvedVerify) {
    Remove-Item -LiteralPath $resolvedVerify -Recurse -Force
}
New-Item -ItemType Directory -Path $resolvedVerify -Force | Out-Null
Copy-Item -Path (Join-Path $oldRoot '*') -Destination $resolvedVerify -Recurse -Force
Copy-Item -Path (Join-Path $stageRoot '*') -Destination $resolvedVerify -Recurse -Force
Import-Module .\packaging\sm61\PackageTools.psm1 -Force
$result = Test-PackageManifest -PackageRoot $verifyRoot
if (-not $result.Valid) {
    throw 'Round-trip overlay manifest validation failed.'
}
$result | Format-List *
```

Expected: `Valid=True` with empty missing, changed, and unexpected lists.

- [ ] **Step 4: Verify the ZIP members and record delivery evidence**

Extract the ZIP under `dist/hotfix-round-trip-zip-extract`, compare every extracted relative path
and SHA256 to the verified stage, and fail if either side has an extra member:

```powershell
$dist = [IO.Path]::GetFullPath('D:\project\cs2-vision-trainer\dist').TrimEnd('\')
$stageRoot = Join-Path $dist 'hotfix-round-trip-stage\cs2-vision-runtime-sm61'
$extract = Join-Path $dist 'hotfix-round-trip-zip-extract'
$resolvedExtract = [IO.Path]::GetFullPath($extract).TrimEnd('\')
if (-not $resolvedExtract.StartsWith($dist + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to recreate extraction root outside dist: $resolvedExtract"
}
if (Test-Path -LiteralPath $resolvedExtract) {
    Remove-Item -LiteralPath $resolvedExtract -Recurse -Force
}
New-Item -ItemType Directory -Path $resolvedExtract -Force | Out-Null

$zip = 'C:\Users\xiaol\Downloads\cs2-vision-runtime-sm61-round-trip-hotfix.zip'
Expand-Archive -LiteralPath $zip -DestinationPath $extract -Force
$unpacked = Join-Path $extract 'cs2-vision-runtime-sm61'
$stageFiles = @(Get-ChildItem -LiteralPath $stageRoot -Recurse -File)
$unpackedFiles = @(Get-ChildItem -LiteralPath $unpacked -Recurse -File)
$stageMap = @{}
foreach ($file in $stageFiles) {
    $relative = $file.FullName.Substring($stageRoot.Length).TrimStart('\').Replace('\', '/')
    $stageMap[$relative] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
}
$changed = New-Object Collections.Generic.List[string]
foreach ($file in $unpackedFiles) {
    $relative = $file.FullName.Substring($unpacked.Length).TrimStart('\').Replace('\', '/')
    $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
    if (-not $stageMap.ContainsKey($relative) -or $stageMap[$relative] -ne $hash) {
        $changed.Add($relative)
    }
    $stageMap.Remove($relative)
}
if ($stageMap.Count -ne 0 -or $changed.Count -ne 0) {
    throw "ZIP verification failed: missing=$($stageMap.Keys -join ',') changed=$($changed -join ',')"
}

Get-Item -LiteralPath $zip | Select-Object FullName,Length,LastWriteTime
Get-FileHash -Algorithm SHA256 -LiteralPath $zip
$unpackedFiles | ForEach-Object {
    $_.FullName.Substring($extract.Length).TrimStart('\').Replace('\', '/')
} | Sort-Object
```

Record every archive member. The operator closes Python, overlays the included
`cs2-vision-runtime-sm61` folder onto the existing production folder, approves replacement, and
runs:

```powershell
Start-Sleep -Seconds 5; python .\examples\runtime_live_move.py --hid-port COM4 --player-side ct --enable-live-output --show-every 1
```

No firmware flash or environment reinstall is part of this delivery.
