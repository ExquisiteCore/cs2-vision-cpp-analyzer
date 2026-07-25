# Persistent Robust HID Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task in the current session. Steps use checkbox (`- [ ]`) syntax for tracking. Do not dispatch subagents for this work.

**Goal:** Make the DLL load and atomically persist one caller-selected HID calibration profile, while making calibration tolerate isolated bad visual measurements and unusable high levels without ever fabricating a profile or raising the normal runtime movement limit above 120.

**Architecture:** Add a focused JSON profile store beside the existing profile math, then make the C API own only the selected path and installed profile. Split robust tile-based visual measurement and pure calibration selection rules from the hardware orchestration so synthetic tests can prove the recovery behavior without opening DXGI or COM. The Python wrapper and example expose caller-controlled cache/recalibration flow; no account or UI policy enters the DLL.

**Tech Stack:** C++17, OpenCV core/imgproc `FileStorage` and `phaseCorrelate`, Win32 write-through rename, stable C ABI, Python 3.11 `ctypes`, pytest, xmake/CMake, PowerShell SM61 packaging with pinned ONNX Runtime 1.17.3 / TensorRT 8.6.1.6 / CUDA 11.8 / cuDNN 8.9.7.

---

## File map

- `include/vision_analyzer/hid_calibration_profile.hpp`: profile-wide schema constants and strict validation declaration.
- `src/hid_calibration_profile.cpp`: curve and complete-profile validation shared by fitting, loading, and the C API.
- `include/vision_analyzer/hid_calibration_store.hpp`: narrow load/save interface for one profile file.
- `src/hid_calibration_store.cpp`: versioned OpenCV JSON parsing and temporary-sibling atomic replacement.
- `include/vision_analyzer/calibration.hpp`: robust-estimator validity flag, discovery callback, and bounded level-selection declarations.
- `src/calibration.cpp`: multi-tile estimation plus real DXGI/HID orchestration using the tested discovery and fallback rules.
- `src/calibration_fit.cpp`: retain the existing signed fitter and use the shared minimum-quality constant.
- `include/vision_analyzer/vision_runtime_c_api.h`: two ABI-safe function declarations; `VaHidCalibrationProfile` layout remains 84 bytes.
- `src/vision_runtime_c_api.cpp`: selected path state, transactional load/get, and save-before-install calibration commit.
- `tests/test_algorithms.cpp`: pure persistence, robust-estimator, discovery, downward fallback, and runtime-clamp tests.
- `tests/test_c_api.cpp`: null/missing/valid/corrupt path and getter transaction tests.
- `xmake.lua`, `CMakeLists.txt`: compile the new store source.
- `../../src/cs2_vision_runtime/runtime.py`: bind and expose the two new C functions.
- `../../tests/test_vision_runtime_sdk.py`: Python forwarding, conversion, and error propagation tests.
- `../../examples/runtime_live_move.py`: caller-controlled cached-profile flow and explicit `--recalibrate` switch.
- `packaging/sm61/package/README_中文.md`, `../../README.md`: API order, persistence, failure safety, and restart instructions.
- `packaging/sm61/tests/run-tests.ps1`: assert that the package includes updated API/example/docs but no changed inference or firmware payloads.

### Task 1: Strict profile validation and versioned atomic JSON store

**Files:**
- Modify: `include/vision_analyzer/hid_calibration_profile.hpp`
- Modify: `src/hid_calibration_profile.cpp`
- Create: `include/vision_analyzer/hid_calibration_store.hpp`
- Create: `src/hid_calibration_store.cpp`
- Modify: `tests/test_algorithms.cpp`
- Modify: `xmake.lua`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing round-trip and rejection tests**

Add `<filesystem>` and the store header to `tests/test_algorithms.cpp`. Add a helper that returns the already proven profile shape and tests that compare every persisted field:

```cpp
HidCalibrationProfile make_persistable_hid_profile() {
    HidCalibrationProfile profile;
    profile.valid = true;
    profile.frame_width = 1920;
    profile.frame_height = 1080;
    profile.x.shift_px = {8.05F, 23.53F, 46.58F};
    profile.x.counts_per_pixel = {1.37F, 1.40F, 1.42F};
    profile.y.shift_px = {7.93F, 24.20F, 47.15F};
    profile.y.counts_per_pixel = {1.39F, 1.41F, 1.42F};
    profile.deadzone_px = 1.0F;
    profile.max_step = 120;
    profile.noise_px = 0.009F;
    profile.quality = 0.678F;
    profile.accepted_samples = 24;
    return profile;
}

void test_hid_calibration_store_round_trips_every_field() {
    const auto path = std::filesystem::temp_directory_path() /
                      "vision-analyzer-hid-calibration.json";
    std::filesystem::remove(path);
    const auto expected = make_persistable_hid_profile();
    save_hid_calibration_profile_atomic(path, expected);
    const auto actual = load_hid_calibration_profile(path);
    require(actual.valid && actual.frame_width == 1920 && actual.frame_height == 1080,
            "loaded profile should retain validity and dimensions");
    require(actual.x.shift_px == expected.x.shift_px &&
            actual.x.counts_per_pixel == expected.x.counts_per_pixel &&
            actual.y.shift_px == expected.y.shift_px &&
            actual.y.counts_per_pixel == expected.y.counts_per_pixel,
            "loaded profile should retain every curve value");
    require_near(actual.deadzone_px, expected.deadzone_px, 0.0001F,
                 "loaded deadzone should match");
    require(actual.max_step == 120 && actual.accepted_samples == 24,
            "loaded safety and sample fields should match");
    std::filesystem::remove(path);
}
```

Write literal JSON fixtures for these invalid cases and require `load_hid_calibration_profile()` to throw while naming the invalid field: `schema_version=2`, truncated JSON, `nan` gain, non-monotonic shifts, mixed gain signs on one axis, `quality=0.54`, `accepted_samples=11`, and `max_step=121`.

Add an atomic-preservation test that first saves a valid destination, calls save with an invalid candidate, and confirms reloading the destination still yields the old profile.

- [ ] **Step 2: Run the algorithm target and confirm RED**

Run:

```powershell
xmake build vision_analyzer_tests
```

Expected: compilation fails because `hid_calibration_store.hpp`, `load_hid_calibration_profile`, `save_hid_calibration_profile_atomic`, and complete-profile validation do not exist.

- [ ] **Step 3: Add the strict public contract**

Extend `hid_calibration_profile.hpp` with:

```cpp
constexpr int kHidCalibrationSchemaVersion = 1;
constexpr float kHidCalibrationMinimumQuality = 0.55F;
constexpr int kHidCalibrationMinimumAcceptedSamples = 12;

[[nodiscard]] bool valid_hid_calibration_profile(
    const HidCalibrationProfile& profile
);
```

Make `valid_hid_calibration_curve()` also reject a sign change between gain knots. Implement `valid_hid_calibration_profile()` so it requires `valid=true`, positive dimensions, both valid curves, finite `deadzone_px` in `[0,8]`, `max_step` in `[1,120]`, finite non-negative noise, finite quality in `[0.55,1]`, and at least 12 accepted samples.

Replace the literal `0.55F` in `fit_adaptive_hid_calibration()` with `kHidCalibrationMinimumQuality`.

- [ ] **Step 4: Implement JSON load and save-before-replace**

Create `hid_calibration_store.hpp`:

```cpp
#pragma once

#include <filesystem>

#include "vision_analyzer/hid_calibration_profile.hpp"

namespace vision_analyzer {

[[nodiscard]] HidCalibrationProfile load_hid_calibration_profile(
    const std::filesystem::path& path
);
void save_hid_calibration_profile_atomic(
    const std::filesystem::path& path,
    const HidCalibrationProfile& profile
);

}  // namespace vision_analyzer
```

In `hid_calibration_store.cpp`, read each scalar and fixed three-element sequence explicitly with OpenCV `FileStorage`; reject a missing key, wrong sequence length, schema mismatch, or any profile that fails `valid_hid_calibration_profile()`. Write JSON to a unique sibling such as `<name>.tmp.<pid>.<counter>`, close it, reload it through the same strict loader, then replace the destination. On Windows use:

```cpp
if (!MoveFileExW(
        temporary.c_str(),
        destination.c_str(),
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    throw std::runtime_error("failed to atomically replace HID calibration profile");
}
```

Use `std::filesystem::rename` on non-Windows builds. A scope guard removes only the temporary sibling after any error. Validate the candidate before opening the temporary file, so invalid data cannot touch the destination.

Add `src/hid_calibration_store.cpp` to `runtime_core_files` in `xmake.lua` and `VISION_ANALYZER_CORE_SOURCES` in `CMakeLists.txt`.

- [ ] **Step 5: Run tests and confirm GREEN**

Run:

```powershell
xmake build vision_analyzer_tests
xmake run vision_analyzer_tests
```

Expected: `Algorithm tests passed`, including JSON rejection and old-destination preservation.

- [ ] **Step 6: Commit**

```powershell
git add include/vision_analyzer/hid_calibration_profile.hpp include/vision_analyzer/hid_calibration_store.hpp src/hid_calibration_profile.cpp src/hid_calibration_store.cpp src/calibration_fit.cpp tests/test_algorithms.cpp xmake.lua CMakeLists.txt
git commit -m "feat: persist validated HID calibration profiles"
```

### Task 2: Transactional calibration path and getter in the C API

**Files:**
- Modify: `include/vision_analyzer/vision_runtime_c_api.h`
- Modify: `src/vision_runtime_c_api.cpp`
- Modify: `tests/test_c_api.cpp`

- [ ] **Step 1: Write failing ABI and transaction tests**

Keep `static_assert(sizeof(VaHidCalibrationProfile) == 84)`. Add tests for:

```cpp
void test_calibration_path_load_and_get_are_transactional() {
    VaRuntime* runtime = va_create();
    require(runtime != nullptr, "runtime should be created");
    VaHidCalibrationProfile profile{};

    require(va_get_hid_calibration(runtime, &profile) == 0 && profile.valid == 0,
            "a new runtime should report no installed calibration");

    const auto valid_path = write_valid_calibration_json("valid-hid-profile.json");
    require(va_set_hid_calibration_path(runtime, valid_path.string().c_str()) == 0,
            "valid profile path should load");
    require(va_get_hid_calibration(runtime, &profile) == 0 && profile.valid == 1,
            "loaded profile should be installed");
    require(profile.max_step == 120 && profile.accepted_samples == 24,
            "getter should expose loaded safety fields");

    const auto corrupt_path = write_text_file("corrupt-hid-profile.json", "{broken");
    require(va_set_hid_calibration_path(runtime, corrupt_path.string().c_str()) == -1,
            "corrupt profile path should fail");
    require(va_get_hid_calibration(runtime, &profile) == 0 && profile.valid == 1,
            "corrupt selection must retain the old installed profile");

    const auto missing_path = unique_missing_path("new-hid-profile.json");
    require(va_set_hid_calibration_path(runtime, missing_path.string().c_str()) == 0,
            "missing destination should be accepted");
    require(va_get_hid_calibration(runtime, &profile) == 0 && profile.valid == 0,
            "selecting a new empty destination should clear the previous profile");
    va_destroy(runtime);
}
```

Also require `-1` for null runtime, null/empty path, and null getter output; verify the getter zero-initializes a nonzero-filled output even when no profile is installed.

- [ ] **Step 2: Run the C API target and confirm RED**

Run:

```powershell
xmake build vision_runtime_c_api_tests
```

Expected: compilation fails because the two exports are undeclared.

- [ ] **Step 3: Declare ABI-safe exports and runtime-owned path state**

Add these declarations without modifying `VaHidCalibrationProfile`:

```c
VA_API int32_t va_set_hid_calibration_path(
    VaRuntime* runtime,
    const char* calibration_path
);
VA_API int32_t va_get_hid_calibration(
    VaRuntime* runtime,
    VaHidCalibrationProfile* profile
);
```

Add `std::filesystem::path hid_calibration_path;` to the private `VaRuntime` definition. Implement `va_set_hid_calibration_path` with local candidate state:

```cpp
const std::filesystem::path candidate = required_string(
    calibration_path, "HID calibration path"
);
if (std::filesystem::exists(candidate)) {
    const auto loaded = load_hid_calibration_profile(candidate);
    runtime->hid_calibration_path = candidate;
    runtime->options.hid_calibration = loaded;
} else {
    runtime->hid_calibration_path = candidate;
    runtime->options.hid_calibration.reset();
}
```

Parsing happens before either assignment. Implement the getter by zeroing the output first, then filling it only when `runtime->options.hid_calibration` exists.

- [ ] **Step 4: Make recalibration save before installation**

In `va_calibrate_hid`, keep the fitted profile in a local. Require `valid_hid_calibration_profile(fitted)`. If `hid_calibration_path` is non-empty, call `save_hid_calibration_profile_atomic()` before assigning `runtime->options.hid_calibration`. Fill the caller output only after save and assignment succeed:

```cpp
const HidCalibrationProfile candidate = run_hid_calibration(calibration_options);
if (!valid_hid_calibration_profile(candidate)) {
    throw std::runtime_error("HID calibration returned an invalid profile");
}
if (!runtime->hid_calibration_path.empty()) {
    save_hid_calibration_profile_atomic(runtime->hid_calibration_path, candidate);
}
runtime->options.hid_calibration = candidate;
fill_calibration_profile(candidate, profile);
```

Because all mutation follows the measurement and save, capture, HID, fit, or save errors retain the previously installed profile and file. Calibration without a selected path remains memory-only.

- [ ] **Step 5: Run C API and algorithm tests**

Run:

```powershell
xmake build vision_runtime_c_api_tests vision_analyzer_tests
xmake run vision_runtime_c_api_tests
xmake run vision_analyzer_tests
```

Expected: `C API tests passed` and `Algorithm tests passed`; the ABI size remains 84.

- [ ] **Step 6: Commit**

```powershell
git add include/vision_analyzer/vision_runtime_c_api.h src/vision_runtime_c_api.cpp tests/test_c_api.cpp
git commit -m "feat: expose persistent HID calibration API"
```

### Task 3: Python wrapper and caller-controlled cache flow

**Files:**
- Modify: `../../src/cs2_vision_runtime/runtime.py`
- Modify: `../../tests/test_vision_runtime_sdk.py`
- Modify: `../../examples/runtime_live_move.py`

- [ ] **Step 1: Write failing wrapper tests**

Extend `FakeApi` with `set_hid_calibration_path()` and `get_hid_calibration()` methods. Add:

```python
def test_runtime_loads_and_reads_persistent_calibration(tmp_path):
    api = FakeApi()
    runtime = VisionRuntime(_api=api)
    path = tmp_path / "hid-calibration.json"

    runtime.set_hid_calibration_path(path)
    profile = runtime.get_hid_calibration()

    assert ("set_hid_calibration_path", os.fsencode(path)) in api.calls
    assert profile.valid is True
    assert profile.max_step == 120


def test_runtime_calibration_path_failure_uses_last_error(tmp_path):
    api = FakeApi()
    api.error = "corrupt HID calibration profile"
    api.set_hid_calibration_path = lambda handle, path: -1
    runtime = VisionRuntime(_api=api)
    with pytest.raises(RuntimeError, match="corrupt"):
        runtime.set_hid_calibration_path(tmp_path / "bad.json")
```

Add a parser-level example test proving that an already valid cached profile skips `calibrate_hid()`, while `--recalibrate` calls it explicitly.

- [ ] **Step 2: Run pytest and confirm RED**

Run from `D:\project\cs2-vision-trainer`:

```powershell
uv run pytest tests/test_vision_runtime_sdk.py -q
```

Expected: failures report missing wrapper and example methods.

- [ ] **Step 3: Bind and expose the new C functions**

In `_RuntimeApi._configure()` add:

```python
dll.va_set_hid_calibration_path.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
dll.va_set_hid_calibration_path.restype = ctypes.c_int32
dll.va_get_hid_calibration.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(_CCalibrationProfile),
]
dll.va_get_hid_calibration.restype = ctypes.c_int32
```

Add forwarding methods to `_RuntimeApi`, then public methods:

```python
def set_hid_calibration_path(self, path: str | os.PathLike[str]) -> None:
    self._check(
        self._api.set_hid_calibration_path(
            self._require_handle(), _encode_path(path)
        )
    )

def get_hid_calibration(self) -> HidCalibrationProfile:
    profile = _CCalibrationProfile()
    self._check(
        self._api.get_hid_calibration(self._require_handle(), profile)
    )
    return HidCalibrationProfile.from_c(profile)
```

- [ ] **Step 4: Make the example caller decide load versus recalibrate**

Add `--calibration-path` with default `ROOT / "hid-calibration.json"` and `--recalibrate`. Extract a testable helper:

```python
def load_or_calibrate(
    runtime: VisionRuntime,
    calibration_path: Path,
    *,
    recalibrate: bool,
    adapter: int,
    output: int,
):
    runtime.set_hid_calibration_path(calibration_path)
    cached = runtime.get_hid_calibration()
    if cached.valid and not recalibrate:
        print(f"已加载本地标定: {calibration_path}")
        return cached
    print("开始调用 DLL 标定；成功后会原子保存到本地……")
    return runtime.calibrate_hid(adapter=adapter, output=output)
```

Call this helper before `open_dxgi()`. The example contains no account name, account lookup, focus manipulation, or automatic invalidation policy.

- [ ] **Step 5: Run Python tests and confirm GREEN**

Run:

```powershell
uv run pytest tests/test_vision_runtime_sdk.py -q
```

Expected: all SDK tests pass, including the pinned-header export check and cache-skip behavior.

- [ ] **Step 6: Commit parent-repository wrapper changes**

Run from `D:\project\cs2-vision-trainer`:

```powershell
git add src/cs2_vision_runtime/runtime.py tests/test_vision_runtime_sdk.py examples/runtime_live_move.py
git commit -m "feat: expose persistent HID calibration in Python"
```

### Task 4: Robust multi-tile visual shift estimation

**Files:**
- Modify: `include/vision_analyzer/calibration.hpp`
- Modify: `src/calibration.cpp`
- Modify: `tests/test_algorithms.cpp`

- [ ] **Step 1: Write synthetic estimator tests**

Add a deterministic textured 960×540 synthetic scene. Shift the background by `(18,-7)`, keep a large central overlay identical between frames, blank one upper tile, and require the robust estimate to recover the background shift within 1.5 pixels with at least two agreeing tiles:

```cpp
void test_robust_visual_shift_ignores_static_overlay_and_blank_tile() {
    cv::Mat before = make_textured_calibration_scene({960, 540});
    cv::Mat after = translated_wrap(before, 18.0, -7.0);
    paint_static_crosshair_overlay(before, after);
    blank_one_upper_tile(before, after);

    const auto estimate = estimate_robust_visual_shift(before, after);
    require(estimate.coherent, "agreeing textured tiles should be coherent");
    require_near(static_cast<float>(estimate.shift.x), 18.0F, 1.5F,
                 "robust estimator should recover X translation");
    require_near(static_cast<float>(estimate.shift.y), -7.0F, 1.5F,
                 "robust estimator should recover Y translation");
}
```

Add a second test where only one tile has texture and require `coherent=false`. Add a third test with two unrelated tile motions and require the estimator to reject the disagreement rather than return a median zero.

- [ ] **Step 2: Run the algorithm target and confirm RED**

Run:

```powershell
xmake build vision_analyzer_tests
```

Expected: compilation fails because `coherent` and `estimate_robust_visual_shift()` do not exist.

- [ ] **Step 3: Add explicit estimator validity without breaking simple callers**

Extend the existing struct by appending a defaulted field so existing `{shift, response}` initializers stay valid:

```cpp
struct VisualShiftEstimate {
    cv::Point2d shift;
    double response = 0.0;
    bool coherent = true;
};

[[nodiscard]] VisualShiftEstimate estimate_robust_visual_shift(
    const cv::Mat& before,
    const cv::Mat& after
);
[[nodiscard]] CalibrationRoundTripMeasurement
estimate_robust_calibration_round_trip(
    const cv::Mat& baseline,
    const cv::Mat& moved,
    const cv::Mat& returned
);
```

- [ ] **Step 4: Implement tiled phase correlation and agreement clustering**

The implementation must:

1. Convert full frames to `CV_32F` grayscale and require equal size.
2. Generate overlapping tiles only inside the upper 70%.
3. Skip any tile whose intersection with a centered crosshair exclusion rectangle exceeds 20%.
4. Compute Sobel gradient energy and reject tiles below a fixed normalized texture floor.
5. Cache a Hanning window per tile size and call `phaseCorrelate` with it.
6. Reject non-finite shifts, response below 0.10, and shifts outside 35% of the smaller tile dimension.
7. Cluster candidates when each coordinate differs by at most `max(1.25, 0.18 * max(1, magnitude))`.
8. Choose the largest cluster, breaking ties by summed response.
9. Require at least two candidates; return component-wise median shift and median response with `coherent=true`. Otherwise return `{{0,0},0,false}`.

Do not remove `estimate_visual_shift_with_response()`; existing basic tests and non-calibration callers retain it. Make only calibration orchestration use the robust variants.

- [ ] **Step 5: Run estimator and full algorithm tests**

Run:

```powershell
xmake build vision_analyzer_tests
xmake run vision_analyzer_tests
```

Expected: all synthetic translations pass; incoherent scenes return `coherent=false`.

- [ ] **Step 6: Commit**

```powershell
git add include/vision_analyzer/calibration.hpp src/calibration.cpp tests/test_algorithms.cpp
git commit -m "feat: estimate HID calibration motion across robust tiles"
```

### Task 5: Discovery retries and second bounded sweep

**Files:**
- Modify: `include/vision_analyzer/calibration.hpp`
- Modify: `src/calibration.cpp`
- Modify: `tests/test_algorithms.cpp`

- [ ] **Step 1: Write failing discovery behavior tests**

Change `CalibrationProbeMeasure` to return a complete balanced round trip. Add tests proving:

- one bad measurement followed by a coherent measurement at the same count accepts that count;
- no count is measured more than three times per sweep;
- the full `16,32,64,128,256,512,1024,2048` ladder is attempted twice before failure;
- the thrown text is exactly prefixed with `HID calibration input not ready: axis=x no coherent visual movement through 2048 counts`;
- the callback never receives a value above 2048 and discovery does not alter `kCalibrationRuntimeMaxStep`.

Use a callback like:

```cpp
int calls = 0;
const auto discovery = discover_calibration_axis(
    0, 4.0, 1.5, 270.0,
    [&](int counts) {
        ++calls;
        if (calls == 1) {
            return incoherent_round_trip();
        }
        return coherent_round_trip({-8.0, 0.1}, {8.1, -0.1}, 0.8);
    }
);
require(discovery.probe_counts == 16 && calls == 2,
        "discovery should retry the same safe count before escalating");
```

- [ ] **Step 2: Run tests and confirm RED**

Run:

```powershell
xmake build vision_analyzer_tests
xmake run vision_analyzer_tests
```

Expected: the new retry/sweep assertions fail against the single-measurement implementation.

- [ ] **Step 3: Implement bounded multi-measurement discovery**

Change the callback alias:

```cpp
using CalibrationProbeMeasure =
    std::function<CalibrationRoundTripMeasurement(int)>;
constexpr int kCalibrationProbeMeasurementsPerCount = 3;
constexpr int kCalibrationDiscoverySweeps = 2;
```

For each sweep and ladder count, call the callback up to three times. A measurement is usable only when its outward leg is coherent, finite, response is at least `kCalibrationMinimumPhaseResponse`, main movement meets the discovery threshold, cross movement is at most 35% of main movement, and main movement is below the reliable ceiling. Accept the first usable measurement and derive levels from it.

If a coherent high-response sub-threshold measurement exists, permit `plan_calibration_probe()` to choose a bounded proportional next count. If all three measurements are incoherent or low-response, advance only through the doubling ladder. After reaching 2048, start one more sweep at 16. Throw the input-not-ready diagnostic only after the second sweep.

- [ ] **Step 4: Use robust balanced measurements in hardware discovery**

In `run_hid_calibration()`, make `measure_balanced_probe()` call `estimate_robust_calibration_round_trip()`. Log `coherent=0/1` for outward and return legs. The exact inverse HID command remains inside the existing exception-safe block, so every completed outward probe is balanced before another count is tried.

- [ ] **Step 5: Run all C++ tests**

Run:

```powershell
xmake build vision_analyzer_tests vision_runtime_c_api_tests
xmake run vision_analyzer_tests
xmake run vision_runtime_c_api_tests
```

Expected: both targets pass; exhaustion takes two bounded sweeps and never fabricates a profile.

- [ ] **Step 6: Commit**

```powershell
git add include/vision_analyzer/calibration.hpp src/calibration.cpp tests/test_algorithms.cpp
git commit -m "fix: retry bounded HID calibration discovery"
```

### Task 6: Strictly downward final-level fallback

**Files:**
- Modify: `include/vision_analyzer/calibration.hpp`
- Modify: `src/calibration.cpp`
- Modify: `tests/test_algorithms.cpp`

- [ ] **Step 1: Write failing pure selection tests**

Declare a result that records the selected count and measured round trips:

```cpp
struct CalibrationLevelSelection {
    bool accepted = false;
    int counts = 0;
    std::vector<int> attempted_counts;
    std::vector<CalibrationRoundTripMeasurement> measurements;
};

using CalibrationLevelMeasure =
    std::function<std::vector<CalibrationRoundTripMeasurement>(int)>;
```

Add tests where planned `{11,33,66}` has an incoherent high level at 66 but a coherent fallback at 49. Require `counts=49`, attempted counts start with `{66,66,49}`, every fallback is `<66` and `>33`, and the runtime max remains 120. Add failures proving low level never shrinks, no path tries more than four distinct downward counts, and no midpoint can equal the previous accepted level.

- [ ] **Step 2: Run tests and confirm RED**

Run:

```powershell
xmake build vision_analyzer_tests
```

Expected: compilation fails because the selection API is absent.

- [ ] **Step 3: Implement measurement usability and bounded midpoint fallback**

Add:

```cpp
[[nodiscard]] bool usable_calibration_round_trip(
    std::size_t axis,
    const CalibrationRoundTripMeasurement& measurement,
    double minimum_shift_px,
    double maximum_shift_px
);

[[nodiscard]] CalibrationLevelSelection select_calibration_level(
    std::size_t axis,
    int level,
    int previous_count,
    int planned_count,
    double minimum_shift_px,
    double maximum_shift_px,
    const CalibrationLevelMeasure& measure
);
```

Require both legs of one round trip to be coherent, finite, above the minimum response and shift, below the maximum shift, and within the 35% cross-axis limit. The original count gets one normal batch and one retry batch. For middle/high levels only, calculate `(previous_count + candidate_count) / 2`, reject duplicates, and try at most four distinct smaller counts. Never return a count `<= previous_count` or `> planned_count`.

Return only coherent round trips in `measurements`, so a single bad frame pair cannot poison the fitter. Throw `HID calibration level unavailable: axis=<x|y> level=<n> planned_counts=<n>` only after the bounded candidates are exhausted.

- [ ] **Step 4: Run algorithm tests and confirm GREEN**

Run:

```powershell
xmake build vision_analyzer_tests
xmake run vision_analyzer_tests
```

Expected: high-level fallback selects 49, low-level failure stays bounded, and all old signed fitter tests pass.

- [ ] **Step 5: Commit**

```powershell
git add include/vision_analyzer/calibration.hpp src/calibration.cpp tests/test_algorithms.cpp
git commit -m "fix: fall back HID calibration levels only downward"
```

### Task 7: Integrate robust selection into real calibration orchestration

**Files:**
- Modify: `src/calibration.cpp`
- Modify: `tests/test_algorithms.cpp`

- [ ] **Step 1: Add an orchestration regression around selected counts**

Extend the pure tests to convert `CalibrationLevelSelection.measurements` through `make_calibration_round_trip_samples()`, fit X and Y curves with one high fallback, and assert:

```cpp
require(profile.valid, "one recovered high level should still produce a valid profile");
require(profile.accepted_samples >= 12,
        "one coherent round trip per axis and level should satisfy the minimum");
require(profile.x.shift_px[0] < profile.x.shift_px[1] &&
        profile.x.shift_px[1] < profile.x.shift_px[2],
        "fallback must still produce three increasing knots");
require(profile.max_step == kCalibrationRuntimeMaxStep,
        "calibration probes must never become runtime output limits");
```

- [ ] **Step 2: Replace the fixed command list in `run_hid_calibration()`**

For each axis and level:

1. Call `select_calibration_level()` with the previous accepted count and planned count.
2. Its measurement callback executes `options.calibration_repeats` balanced round trips, alternating the outward sign exactly as the existing command plan did.
3. Log each rejected same-count retry and each midpoint as:

```text
level_fallback axis=x level=2 from_counts=66 to_counts=49 reason=no_coherent_shift
```

4. Convert only the returned coherent measurements to signed samples.
5. Store the selected count as the previous accepted count.

After all three levels for one axis, log:

```text
sample_levels axis=x counts=11,33,49
```

Keep noise sampling, fit output, inverse-on-exception safety, and `HidClient::stop_all()` cleanup unchanged. Remove the runtime use of `plan_calibration_round_trip_commands()` if it is no longer needed, but retain or remove its pure declaration and tests consistently.

- [ ] **Step 3: Enforce complete profile validation at the calibration boundary**

After fitting, require both `profile.valid` and `valid_hid_calibration_profile(profile)`. Preserve the existing detailed `fit`, `curve`, quality, noise, and sample logs. The error must distinguish:

- discovery: `HID calibration input not ready...`;
- level selection: `HID calibration level unavailable...`;
- final fit: `HID calibration rejected: quality=... noise_px=... accepted_samples=...`.

- [ ] **Step 4: Run both C++ targets repeatedly**

Run:

```powershell
1..3 | ForEach-Object {
    xmake run vision_analyzer_tests
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
xmake run vision_runtime_c_api_tests
```

Expected: all three deterministic algorithm runs and the C API run pass.

- [ ] **Step 5: Commit**

```powershell
git add src/calibration.cpp tests/test_algorithms.cpp
git commit -m "fix: recover isolated HID calibration measurement failures"
```

### Task 8: Documentation and package contract

**Files:**
- Modify: `../../README.md`
- Modify: `packaging/sm61/package/README_中文.md`
- Modify: `packaging/sm61/tests/run-tests.ps1`

- [ ] **Step 1: Write failing package assertions**

In `run-tests.ps1`, require the staged header to contain `va_set_hid_calibration_path` and `va_get_hid_calibration`, the Python wrapper to contain `set_hid_calibration_path` and `get_hid_calibration`, and the example to contain `--calibration-path` plus `--recalibrate`. Keep existing assertions that the package uses pinned dependencies and contains no TensorRT engine cache.

- [ ] **Step 2: Run package tests and confirm RED**

Run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\packaging\sm61\tests\run-tests.ps1
```

Expected: source/package contract assertions fail until staged documentation/example updates are reflected.

- [ ] **Step 3: Document the exact caller flow**

Update both READMEs with this policy:

```python
runtime.set_hid_calibration_path("hid-calibration.json")
profile = runtime.get_hid_calibration()
if not profile.valid:
    profile = runtime.calibrate_hid(adapter=0, output=0)

# The caller explicitly invokes calibrate_hid() again after its own settings change.
runtime.open_dxgi(adapter=0, output=0, dry_run=False)
```

State that a valid file loads without mouse movement, a corrupt file returns an error without replacing the old profile, failed recalibration preserves the previous file, the DLL has no account concept, the caller chooses when to recalibrate, probes may reach 2048 only during calibration, and live movement remains clamped to 120.

- [ ] **Step 4: Run documentation/package and Python tests**

Run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\packaging\sm61\tests\run-tests.ps1
Set-Location ..\..
uv run pytest tests/test_vision_runtime_sdk.py -q
```

Expected: package contract checks and Python SDK tests pass.

- [ ] **Step 5: Commit documentation and package tests**

Commit C++ submodule files first:

```powershell
git add packaging/sm61/package/README_中文.md packaging/sm61/tests/run-tests.ps1
git commit -m "docs: explain persistent HID calibration flow"
```

Then commit the parent README from `D:\project\cs2-vision-trainer`:

```powershell
git add README.md tools/cpp_analyzer
git commit -m "docs: document cached HID calibration API"
```

### Task 9: Pinned production build, full verification, and incremental package

**Files:**
- Verify: all changed source and test files
- Generate outside Git: `C:\Users\xiaol\Downloads\cs2-vision-runtime-sm61-persistent-calibration-hotfix.zip`

- [ ] **Step 1: Load the fixed production dependency environment**

Use the same local dependency roots recorded by `packaging/sm61/dependencies.lock.json`. Do not download or switch ORT/TensorRT/CUDA/cuDNN. Configure release mode with the existing ONNX Runtime 1.17.3 and RP2350 SDK paths:

```powershell
xmake f -m release --onnxruntime_root="$env:ONNXRUNTIME_ROOT" --hid_sdk_root="$env:RP2350_HID_BRIDGE_SDK" -y
xmake build vision_runtime vision_analyzer vision_analyzer_tests vision_runtime_c_api_tests
```

Expected: release DLL, CLI, import library, and tests build against the pinned environment.

- [ ] **Step 2: Run the complete safe test matrix**

Run in the C++ worktree:

```powershell
xmake run vision_analyzer_tests
xmake run vision_runtime_c_api_tests
ctest --test-dir build-cmake --output-on-failure
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\packaging\sm61\tests\run-tests.ps1
```

Run in the parent repository:

```powershell
uv run pytest -q
```

Expected: all C++, CMake, package, and Python tests pass. These commands do not open COM, send HID, flash firmware, or arm output.

- [ ] **Step 3: Stage only incremental runtime-facing files**

Create a clean staging directory containing:

```text
app/vision_runtime.dll
app/vision_analyzer.exe
app/vision_runtime.lib
app/vision_runtime_c_api.h
python/cs2_vision_runtime/
examples/runtime_live_move.py
README_中文.md
HOTFIX_MANIFEST.txt
```

The manifest records the base package name, Git commits, file SHA256 values, and fixed dependency versions. It explicitly states that the ZIP contains no model, ORT, TensorRT, CUDA, cuDNN, cache, or firmware files.

- [ ] **Step 4: Create and inspect the ZIP**

Use `Compress-Archive` to create:

```text
C:\Users\xiaol\Downloads\cs2-vision-runtime-sm61-persistent-calibration-hotfix.zip
```

Expand it to a temporary directory, compare every staged SHA256 with `HOTFIX_MANIFEST.txt`, and search the file list to prove these patterns are absent:

```text
*.onnx
onnxruntime*.dll
nvinfer*.dll
cud*.dll
*.engine
*.uf2
```

- [ ] **Step 5: Simulate overlay on the prior portable package**

Copy the previous portable package to a temporary test directory, overlay the incremental ZIP, then run its safe checks:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\verify-runtime.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\test-video.ps1
```

Expected: manifest verification and TensorRT video dry run pass; no RP2350 output is enabled.

- [ ] **Step 6: Report artifact and manual acceptance command**

Report the ZIP path and SHA256. The user-facing one-line manual command must select a calibration file and use COM4 while preserving the package runtime environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force; . .\scripts\common.ps1; Invoke-WithRuntimeEnvironment { $env:PYTHONPATH=(Resolve-Path '.\python').Path; $env:CS2_VISION_RUNTIME_DLL=(Resolve-Path '.\app\vision_runtime.dll').Path; python .\examples\runtime_live_move.py --hid-port COM4 --player-side ct --calibration-path .\hid-calibration.json --enable-live-output --show-every 1 }
```

First run should calibrate and save. A process restart with the same command should print that the local calibration was loaded and proceed to DXGI without calibration movement. Adding `--recalibrate` is the caller-controlled explicit recalibration path.

---

## Acceptance checklist

- [ ] The DLL exports path/load/get APIs and retains the 84-byte profile ABI.
- [ ] A valid profile reloads without DXGI/HID calibration movement.
- [ ] Missing selected path clears only the previous path-associated in-memory profile.
- [ ] Corrupt selection and failed save leave the old path, memory profile, and file unchanged.
- [ ] Discovery uses no more than 2048 counts, retries each count, and performs only two bounded sweeps.
- [ ] Final-level recovery never increases a count and tries at most four distinct downward candidates.
- [ ] Robust estimation requires at least two agreeing textured upper-frame tiles.
- [ ] No invalid or partial measurement can be installed or persisted.
- [ ] Runtime movement remains clamped to 120.
- [ ] Python and documentation contain no account policy or UI behavior.
- [ ] Pinned ORT 1.17.3 / TensorRT 8.6.1.6 / CUDA 11.8 / cuDNN 8.9.7 remain unchanged.
- [ ] Incremental ZIP contains only changed runtime-facing files and passes overlay verification.
