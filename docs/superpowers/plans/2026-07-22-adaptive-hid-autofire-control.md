# Adaptive HID Calibration and Autofire Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Python-controlled movement/fire gates, signed per-axis three-level startup HID calibration, and aggressive head-priority/body-fallback automatic firing to the DLL and portable package.

**Architecture:** The C++ submodule owns calibration fitting, runtime movement mapping, fire policy, DXGI/HID calibration orchestration, and the stable C ABI. The outer Python repository owns the `ctypes` control plane and live example. A successful calibration installs an in-memory curve before DXGI inference opens; Python then independently enables movement and automatic firing.

**Tech Stack:** C++17, OpenCV/DXGI, RP2350 HID bridge, C ABI DLL, Python 3.11 `ctypes`, pytest through `uv`, xmake, CMake/CTest, PowerShell 5.1 packaging.

---

## Repository map

This feature spans two Git repositories that must remain independently clean:

- C++ submodule worktree: `D:\project\cs2-vision-trainer\tools\cpp_analyzer\.worktrees\sm61-portable-package`
- Outer Python repository: `D:\project\cs2-vision-trainer`
- New outer-repository worktree created during execution: `C:\Users\xiaol\.config\superpowers\worktrees\cs2-vision-trainer\adaptive-hid-host`

All C++ commits are made on `feature/sm61-portable-package`. Python wrapper and
outer-repository submodule-pointer commits are made on
`feature/adaptive-hid-host`. Do not edit either repository's `main` branch.

## File map

### C++ submodule

- Create `include/vision_analyzer/hid_calibration_profile.hpp`: fixed three-knot axis/profile types and pure error-to-HID mapping.
- Create `src/hid_calibration_profile.cpp`: curve validation, interpolation, deadzone, and clamping.
- Modify `include/vision_analyzer/calibration.hpp`: response/level-aware samples, visual-shift result, adaptive profile fitting, and profile-returning runner.
- Modify `src/calibration_fit.cpp`: signed per-axis robust fit and rejection rules while retaining scalar compatibility.
- Modify `src/calibration.cpp`: phase response, bounded adaptive probes, balanced movement pairs, cleanup, and profile logging.
- Modify `include/vision_analyzer/types.hpp`: runtime calibration profile and fire-policy state.
- Modify `include/vision_analyzer/aim_controller.hpp` and `src/aim_controller.cpp`: calibrated mapping, synchronized live fire state/policy, and class hit regions.
- Modify `src/tracking.cpp`: head-priority score multiplier.
- Modify `include/vision_analyzer/runtime_session.hpp` and `src/runtime_session.cpp`: live fire/policy updates and calibrated controller construction.
- Modify `src/runtime_config.cpp`, `src/runtime_options.cpp`, and `src/main.cpp`: new fire-policy configuration and updated calibration result handling.
- Modify `include/vision_analyzer/vision_runtime_c_api.h` and `src/vision_runtime_c_api.cpp`: stable calibration struct and three new API functions.
- Modify `tests/test_algorithms.cpp` and `tests/test_c_api.cpp`: all pure behavior, gate, and ABI tests.
- Modify `xmake.lua` and `CMakeLists.txt`: compile the new profile source.
- Modify `packaging/sm61/build-portable-package.ps1`, package tests, and package documentation: ship the Python wrapper/example without arming one-click scripts.

### Outer Python repository

- Modify `src/cs2_vision_runtime/runtime.py`: `ctypes` profile, Python dataclass, bindings, and public methods.
- Modify `src/cs2_vision_runtime/__init__.py`: export `HidCalibrationProfile`.
- Modify `tests/test_vision_runtime_sdk.py`: fake-API forwarding, profile conversion, error, and cleanup tests.
- Modify `examples/runtime_live_move.py`: calibration-first fully automatic flow with explicit live-output acknowledgement and guaranteed disarm.
- Modify `README.md`: Python call order and production acceptance instructions.
- Modify the `tools/cpp_analyzer` submodule pointer after all C++ commits are final.

### Task 1: Establish clean dual-repository baselines

**Files:**
- Verify only; no tracked file changes.

- [ ] **Step 1: Verify the existing C++ worktree and baseline tests**

Run from the C++ worktree:

```powershell
git status --short --branch
xmake run vision_analyzer_tests
xmake run vision_runtime_c_api_tests
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File packaging\sm61\tests\run-tests.ps1
```

Expected: branch `feature/sm61-portable-package`, a clean status, algorithm and C
API success messages, and `PASS package tool tests (21)`.

- [ ] **Step 2: Create an isolated outer-repository worktree**

Run from `D:\project\cs2-vision-trainer`. The selected location is outside the
repository and therefore does not need a `.gitignore` entry:

```powershell
$hostWorktree='C:\Users\xiaol\.config\superpowers\worktrees\cs2-vision-trainer\adaptive-hid-host'
New-Item -ItemType Directory -Path (Split-Path -Parent $hostWorktree) -Force | Out-Null
git worktree add $hostWorktree -b feature/adaptive-hid-host
```

Expected: a named branch worktree at the exact path in the repository map.

- [ ] **Step 3: Verify the Python baseline in the new worktree**

```powershell
uv run pytest tests\test_vision_runtime_sdk.py -q
uv run pytest -q
git status --short --branch
```

Expected: all Python tests pass and the new worktree is clean.

### Task 2: Add the pure three-knot HID calibration profile

**Files:**
- Create: `include/vision_analyzer/hid_calibration_profile.hpp`
- Create: `src/hid_calibration_profile.cpp`
- Modify: `tests/test_algorithms.cpp`
- Modify: `xmake.lua`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing curve mapping tests**

Add the include and these test cases to `tests/test_algorithms.cpp`:

```cpp
#include "vision_analyzer/hid_calibration_profile.hpp"

void test_calibrated_hid_curve_interpolates_signed_gain() {
    HidCalibrationAxisCurve curve{
        {8.0F, 32.0F, 96.0F},
        {2.0F, 3.0F, 4.0F},
    };
    require(calibrated_hid_step(20.0F, curve, 120, 1.0F) == 50,
            "20 px should interpolate to gain 2.5");
    require(calibrated_hid_step(-20.0F, curve, 120, 1.0F) == -50,
            "signed errors should preserve direction");
}

void test_calibrated_hid_curve_supports_inverted_axis_deadzone_and_clamp() {
    HidCalibrationAxisCurve inverted{
        {8.0F, 32.0F, 96.0F},
        {-2.0F, -3.0F, -4.0F},
    };
    require(calibrated_hid_step(20.0F, inverted, 120, 1.0F) == -50,
            "negative gain should support an inverted axis");
    require(calibrated_hid_step(0.5F, inverted, 120, 1.0F) == 0,
            "deadzone should suppress noise");
    require(calibrated_hid_step(200.0F, inverted, 120, 1.0F) == -120,
            "large movement should clamp to max_step");
}
```

Register both functions in `main()`.

- [ ] **Step 2: Run the test to verify RED**

```powershell
xmake build vision_analyzer_tests
```

Expected: compilation fails because `hid_calibration_profile.hpp` does not exist.

- [ ] **Step 3: Define the fixed internal profile types**

Create `include/vision_analyzer/hid_calibration_profile.hpp`:

```cpp
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace vision_analyzer {

constexpr std::size_t kHidCalibrationLevels = 3;

struct HidCalibrationAxisCurve {
    std::array<float, kHidCalibrationLevels> shift_px{};
    std::array<float, kHidCalibrationLevels> counts_per_pixel{};
};

struct HidCalibrationProfile {
    bool valid = false;
    int frame_width = 0;
    int frame_height = 0;
    HidCalibrationAxisCurve x;
    HidCalibrationAxisCurve y;
    float deadzone_px = 1.5F;
    int max_step = 120;
    float noise_px = 0.0F;
    float quality = 0.0F;
    int accepted_samples = 0;
};

[[nodiscard]] bool valid_hid_calibration_curve(const HidCalibrationAxisCurve& curve);
[[nodiscard]] std::int16_t calibrated_hid_step(
    float error_px,
    const HidCalibrationAxisCurve& curve,
    int max_step,
    float deadzone_px
);

}  // namespace vision_analyzer
```

- [ ] **Step 4: Implement validation and interpolation**

Create `src/hid_calibration_profile.cpp` with these rules:

```cpp
float interpolated_gain(float magnitude, const HidCalibrationAxisCurve& curve) {
    if (magnitude <= curve.shift_px[0]) return curve.counts_per_pixel[0];
    if (magnitude >= curve.shift_px[2]) return curve.counts_per_pixel[2];
    const std::size_t left = magnitude <= curve.shift_px[1] ? 0 : 1;
    const std::size_t right = left + 1;
    const float span = curve.shift_px[right] - curve.shift_px[left];
    const float alpha = (magnitude - curve.shift_px[left]) / span;
    return curve.counts_per_pixel[left] * (1.0F - alpha) +
           curve.counts_per_pixel[right] * alpha;
}
```

`valid_hid_calibration_curve` returns true only for finite, positive, strictly
increasing shift knots and finite, nonzero gains. `calibrated_hid_step` returns
zero inside the deadzone, multiplies the signed error by the interpolated signed
gain, rounds with `std::lround`, and clamps to both `max_step` and `int16_t`.

- [ ] **Step 5: Add the new source to both build systems**

Add `src/hid_calibration_profile.cpp` to `runtime_core_files` in `xmake.lua` and
`VISION_ANALYZER_CORE_SOURCES` in `CMakeLists.txt`.

- [ ] **Step 6: Run GREEN verification**

```powershell
xmake build vision_analyzer_tests
xmake run vision_analyzer_tests
```

Expected: compilation succeeds and all algorithm tests pass.

- [ ] **Step 7: Commit the curve primitive**

```powershell
git add include/vision_analyzer/hid_calibration_profile.hpp src/hid_calibration_profile.cpp tests/test_algorithms.cpp xmake.lua CMakeLists.txt
git commit -m "feat: add signed HID calibration curves"
```

### Task 3: Fit signed per-axis calibration curves

**Files:**
- Modify: `include/vision_analyzer/calibration.hpp`
- Modify: `src/calibration_fit.cpp`
- Modify: `tests/test_algorithms.cpp`

- [ ] **Step 1: Add failing signed-fit and rejection tests**

Extend `CalibrationSample` expectations with `phase_response` and `level`, then
add synthetic samples for all three levels:

```cpp
std::vector<CalibrationSample> make_valid_adaptive_calibration_samples() {
    return {
        {0, 0, {0.2, 0.1}, 0.95, -1},
        {16, 0, {-8.0, 0.2}, 0.90, 0}, {-16, 0, {8.1, -0.1}, 0.91, 0},
        {40, 0, {-16.0, 0.2}, 0.92, 1}, {-40, 0, {16.2, -0.2}, 0.90, 1},
        {80, 0, {-20.0, 0.2}, 0.93, 2}, {-80, 0, {20.2, -0.1}, 0.91, 2},
        {0, 16, {0.1, 8.0}, 0.92, 0}, {0, -16, {-0.1, -8.1}, 0.90, 0},
        {0, 40, {0.2, 16.0}, 0.91, 1}, {0, -40, {-0.2, -16.1}, 0.92, 1},
        {0, 80, {0.1, 20.0}, 0.93, 2}, {0, -80, {-0.1, -20.1}, 0.92, 2},
    };
}

void test_adaptive_calibration_fits_signed_axes_and_inverted_y() {
    const auto samples = make_valid_adaptive_calibration_samples();
    const auto profile = fit_adaptive_hid_calibration(samples, {1920, 1080}, 120);
    require(profile.valid, "consistent samples should produce a profile");
    require(profile.x.counts_per_pixel[0] > 0.0F, "normal X should be positive");
    require(profile.y.counts_per_pixel[0] < 0.0F, "inverted Y should be negative");
    require(profile.x.counts_per_pixel[2] > profile.x.counts_per_pixel[0],
            "nonlinear samples should preserve a large-step gain");
}

void test_adaptive_calibration_rejects_bad_response_and_cross_axis_motion() {
    auto samples = make_valid_adaptive_calibration_samples();
    for (auto& sample : samples) {
        if (sample.level == 1 && sample.counts_dx != 0) {
            sample.phase_response = 0.05;
            sample.visual_shift.y = sample.visual_shift.x;
        }
    }
    const auto profile = fit_adaptive_hid_calibration(samples, {1920, 1080}, 120);
    require(!profile.valid, "missing valid X level must reject the profile");
}
```

Register both tests in `main()`.

- [ ] **Step 2: Run the algorithm test to verify RED**

```powershell
xmake build vision_analyzer_tests
```

Expected: compilation fails because the extended sample fields and
`fit_adaptive_hid_calibration` are missing.

- [ ] **Step 3: Extend calibration data contracts**

In `calibration.hpp`, make the sample backward-compatible by appending fields:

```cpp
struct CalibrationSample {
    int counts_dx = 0;
    int counts_dy = 0;
    cv::Point2d visual_shift;
    double phase_response = 1.0;
    int level = 0;
};

struct VisualShiftEstimate {
    cv::Point2d shift;
    double response = 0.0;
};

[[nodiscard]] HidCalibrationProfile fit_adaptive_hid_calibration(
    const std::vector<CalibrationSample>& samples,
    const cv::Size& frame_size,
    int max_step = 120
);
```

Include `hid_calibration_profile.hpp`. Keep `CalibrationFit` and
`fit_hid_calibration` so existing CLI/config consumers remain source-compatible.

- [ ] **Step 4: Implement the exact acceptance and fit rules**

In `calibration_fit.cpp`:

1. Compute `noise_px` as the median norm of zero-count samples.
2. For a movement sample, identify X when `counts_dx != 0`, otherwise Y.
3. Reject response below `0.15`.
4. Reject main-axis magnitude below `max(1.5, 3 * noise_px)`.
5. Reject cross-axis magnitude above 35% of main-axis magnitude.
6. Calculate signed gain as `-counts / main_axis_shift`.
7. Group accepted gains and absolute shifts by axis, level `0..2`, and command
   sign.
8. Require at least one positive and one negative sample for every group.
9. Reject a level when opposite-direction median gain magnitudes differ by more
   than 40% of their larger magnitude.
10. Store the median signed gain and median absolute shift per level, then sort
    the three knots by shift.
11. Set `deadzone_px=clamp(ceil(3*noise_px),1,8)` and the supplied max step.
12. Define `mean_response` as the average accepted response clamped to `[0,1]`.
    Define each level's consistency as `1 - opposite_gain_difference_ratio` and
    average all six X/Y levels. Define signal-to-noise as
    `clamp(min_accepted_main_shift / (3 * max(noise_px, 0.25)), 0, 1)`. Set
    quality to the product of these three factors and require at least `0.55`.

Return an invalid profile rather than throwing for rejected measurements. Throw
only for an invalid frame size, max step, or out-of-range level.

- [ ] **Step 5: Preserve the scalar fit through the new sample layout**

Update the existing scalar fitter to ignore `phase_response` and `level` and keep
its current output. This protects existing config generation and its current
test.

- [ ] **Step 6: Run GREEN verification**

```powershell
xmake run vision_analyzer_tests
```

Expected: signed normal/inverted curves pass, invalid samples are rejected, and
all previous calibration tests remain green.

- [ ] **Step 7: Commit the fitter**

```powershell
git add include/vision_analyzer/calibration.hpp src/calibration_fit.cpp tests/test_algorithms.cpp
git commit -m "feat: fit adaptive per-axis HID calibration"
```

### Task 4: Add aggressive head/body fire policy and calibrated aim mapping

**Files:**
- Modify: `include/vision_analyzer/types.hpp`
- Modify: `include/vision_analyzer/aim_controller.hpp`
- Modify: `src/aim_controller.cpp`
- Modify: `src/tracking.cpp`
- Modify: `tests/test_algorithms.cpp`

- [ ] **Step 1: Write failing calibrated movement and fire-policy tests**

Add tests that construct `FrameReport` values with the frame center at
`(960,540)`:

```cpp
HidCalibrationProfile make_valid_hid_profile() {
    HidCalibrationProfile profile;
    profile.valid = true;
    profile.frame_width = 1920;
    profile.frame_height = 1080;
    profile.x = {{8.0F, 32.0F, 96.0F}, {2.0F, 3.0F, 4.0F}};
    profile.y = {{8.0F, 32.0F, 96.0F}, {1.0F, 1.5F, 2.0F}};
    profile.deadzone_px = 1.0F;
    profile.max_step = 120;
    return profile;
}

FrameReport make_target_report(Detection detection, cv::Point2f offset) {
    TargetFrame target{};
    target.detection = std::move(detection);
    target.offset = offset;
    target.analysis_point = {960.0F + offset.x, 540.0F + offset.y};
    target.lock_state = LockState::Acquiring;
    return FrameReport{1, 16.0, 60.0, InferenceTiming{}, 1, target};
}

FrameReport make_body_report(cv::Point2f offset, float confidence = 0.80F) {
    return make_target_report(
        Detection{0, "ct_body", confidence, cv::Rect(900, 450, 120, 180)},
        offset
    );
}

void test_aim_controller_uses_calibrated_axes() {
    AimControllerOptions options;
    options.calibration = make_valid_hid_profile();
    AimController controller(options);
    const AimCommand command = controller.plan(make_target_report(
        Detection{1, "ct_head", 0.90F, cv::Rect(970, 520, 40, 40)},
        cv::Point2f{20.0F, -20.0F}
    ));
    require(command.dx != command.dy, "per-axis curves should produce independent steps");
}

void test_head_fires_on_first_frame_inside_box() {
    AimController controller;
    controller.set_fire_policy(FirePolicy{true, 0.35F, 0.45F, 3});
    controller.set_fire_enabled(true);
    const AimCommand command = controller.plan(make_target_report(
        Detection{1, "ct_head", 0.80F, cv::Rect(940, 520, 40, 40)},
        cv::Point2f{0.0F, 0.0F}
    ));
    require(command.click_left, "head box should fire on its first qualifying frame");
}

void test_body_fires_only_in_torso_when_enabled() {
    AimController controller;
    controller.set_fire_policy(FirePolicy{true, 0.35F, 0.45F, 3});
    controller.set_fire_enabled(true);
    require(controller.plan(make_body_report(cv::Point2f{0.0F, 0.0F})).click_left,
            "centered torso should fire");
    controller.set_fire_enabled(false);
    require(!controller.plan(make_body_report(cv::Point2f{0.0F, 0.0F})).click_left,
            "fire gate should suppress body clicks");
}

void test_fire_policy_enforces_body_flag_and_class_confidence() {
    AimController controller;
    controller.set_fire_enabled(true);
    controller.set_fire_policy(FirePolicy{false, 0.35F, 0.45F, 3});
    require(!controller.plan(make_body_report({0.0F, 0.0F})).click_left,
            "body-disabled policy must suppress a centered torso");
    controller.set_fire_policy(FirePolicy{true, 0.35F, 0.85F, 3});
    require(!controller.plan(make_body_report({0.0F, 0.0F}, 0.80F)).click_left,
            "body confidence below policy threshold must not fire");
}

void test_fire_cooldown_and_disable_reset() {
    AimController controller;
    controller.set_fire_policy(FirePolicy{true, 0.35F, 0.45F, 3});
    controller.set_fire_enabled(true);
    const FrameReport report = make_body_report({0.0F, 0.0F});
    require(controller.plan(report).click_left, "first torso frame should fire");
    require(!controller.plan(report).click_left, "cooldown should suppress next frame");
    controller.set_fire_enabled(false);
    controller.set_fire_enabled(true);
    require(controller.plan(report).click_left, "disabling fire should clear cooldown");
}
```

Register every new controller test in `main()`. Update the existing target-selector
fixture so a head and body have equal distance/confidence/stability and assert
that the head wins with the new `0.65` multiplier.

- [ ] **Step 2: Run tests to verify RED**

```powershell
xmake build vision_analyzer_tests
```

Expected: compilation fails because calibration/fire policy setters are absent.

- [ ] **Step 3: Define synchronized fire policy state**

Add `FirePolicy` to `types.hpp` before `Options`, so both runtime options and the
controller can use it without a circular include:

```cpp
struct FirePolicy {
    bool body_enabled = true;
    float head_confidence = 0.35F;
    float body_confidence = 0.45F;
    int cooldown_frames = 3;
};
```

In `aim_controller.hpp`, include `<mutex>` and the calibration profile header,
then replace the controller option fields with:

```cpp

struct AimControllerOptions {
    float move_gain = 1.0F;
    int max_step = 120;
    float deadzone_px = 1.5F;
    bool fire_enabled = false;
    FirePolicy fire_policy;
    std::optional<HidCalibrationProfile> calibration;
};
```

Add `set_fire_enabled(bool)` and `set_fire_policy(FirePolicy)`. Protect policy,
fire state, and cooldown with one `std::mutex`. Keep `plan()` synchronized with
these setters.

- [ ] **Step 4: Implement adaptive movement and hit regions**

When a valid calibration is present, map X and Y through their separate curves;
otherwise use the existing scalar `scaled_step` behavior.

For click qualification use the actual detection box and the frame center:

```cpp
cv::Rect fire_region(const Detection& detection) {
    if (is_head(detection.class_id)) return detection.box;
    return cv::Rect(
        detection.box.x + static_cast<int>(detection.box.width * 0.20F),
        detection.box.y + static_cast<int>(detection.box.height * 0.10F),
        std::max(1, static_cast<int>(detection.box.width * 0.60F)),
        std::max(1, static_cast<int>(detection.box.height * 0.60F))
    );
}
```

Use `TargetFrame.analysis_point - TargetFrame.offset` to recover the frame center
without adding another field. Fire on the first qualifying frame when fire is
enabled, class confidence passes, the center is inside the class region, and
cooldown is zero. Do not require lock state, stability, or `fire_candidate`.

- [ ] **Step 5: Strengthen head priority without making it absolute**

Change the head class multiplier in `TargetSelector::score` from `0.70F` to
`0.65F`. Keep body multiplier `1.00F` and the existing distance, confidence,
stability, and switch factors.

- [ ] **Step 6: Run GREEN verification**

```powershell
xmake run vision_analyzer_tests
```

Expected: all new fire/movement cases and the existing tracking/controller tests
pass.

- [ ] **Step 7: Commit controller behavior**

```powershell
git add include/vision_analyzer/types.hpp include/vision_analyzer/aim_controller.hpp src/aim_controller.cpp src/tracking.cpp tests/test_algorithms.cpp
git commit -m "feat: add adaptive aggressive autofire policy"
```

### Task 5: Implement bounded startup calibration and live runtime updates

**Files:**
- Modify: `include/vision_analyzer/calibration.hpp`
- Modify: `src/calibration.cpp`
- Modify: `include/vision_analyzer/types.hpp`
- Modify: `include/vision_analyzer/runtime_session.hpp`
- Modify: `src/runtime_session.cpp`
- Modify: `src/runtime_config.cpp`
- Modify: `src/runtime_options.cpp`
- Modify: `src/main.cpp`
- Modify: `tests/test_algorithms.cpp`

- [ ] **Step 1: Add failing probe-adjustment and config tests**

Declare
`int adjust_calibration_probe_count(int current_counts, double observed_shift_px,
double target_shift_px)` in `calibration.hpp` and test the bounded retry
calculation:

```cpp
void test_calibration_probe_adjustment_is_bounded() {
    require(adjust_calibration_probe_count(16, 0.5, 8.0) == 120,
            "tiny response should scale up to the upper bound");
    require(adjust_calibration_probe_count(80, 200.0, 80.0) == 32,
            "oversized response should scale down proportionally");
}
```

Extend the runtime config fixture with:

```text
fire_enabled=true
body_fire_enabled=true
head_fire_confidence=0.35
body_fire_confidence=0.45
hid_click_cooldown_frames=3
```

Assert every parsed value.

- [ ] **Step 2: Run tests to verify RED**

```powershell
xmake build vision_analyzer_tests
```

Expected: missing helper and option fields cause compilation failure.

- [ ] **Step 3: Add runtime option state**

In `Options`, replace the old click configuration state with:

```cpp
bool fire_enabled = false;
FirePolicy fire_policy;
std::optional<HidCalibrationProfile> hid_calibration;
```

Include `hid_calibration_profile.hpp` from `types.hpp`. Implement
`adjust_calibration_probe_count` as
`round(current_counts * target_shift_px / max(0.5, observed_shift_px))` clamped
to `8..120`; reject non-positive current counts/target shifts and non-finite
observations.

Continue accepting old `hid_click` configuration by mapping it to
`fire_enabled`. Map `hid_click_cooldown_frames` to
`fire_policy.cooldown_frames`. Add the four explicit fire config keys from the
failing fixture and validate their exact ranges.

- [ ] **Step 4: Return phase response and an adaptive profile from calibration**

Change phase correlation to:

```cpp
VisualShiftEstimate estimate_visual_shift_with_response(
    const cv::Mat& before,
    const cv::Mat& after
) {
    cv::Mat first = comparable_gray(before);
    cv::Mat second = comparable_gray(after);
    double response = 0.0;
    const cv::Point2d shift = cv::phaseCorrelate(first, second, cv::noArray(), &response);
    return {shift, response};
}
```

Keep `estimate_visual_shift` as a compatibility wrapper that returns `.shift`.
Change `run_hid_calibration` to return `HidCalibrationProfile`.

- [ ] **Step 5: Implement bounded paired probes and cleanup**

Use initial counts `{16,40,80}` and target shifts `{8.0,32.0,80.0}`. For each
axis, level, repeat, and sign:

1. send one bounded count;
2. wait `calibration_settle_ms`;
3. capture and measure shift/response;
4. if main shift is below the fit threshold or above 25% of the shorter frame
   dimension, send the exact inverse count, wait, capture a fresh baseline, calculate
   `round(count * target_shift / max(0.5, observed_shift))` clamped to `8..120`,
   and retry once;
5. append the accepted or rejected sample including response and level;
6. send the inverse of the final test count, wait, and capture the returned view
   as the next baseline before moving to the next sample.

Wrap the whole routine so `hid_client->stop_all()` and `frame_source->release()`
run on both success and exception. Fit the complete sample set, throw with
quality/noise/sample details if invalid, print every curve knot, and return the
accepted profile.

- [ ] **Step 6: Add live runtime setters**

Add to `RuntimeSession`:

```cpp
void set_fire_enabled(bool enabled);
void set_fire_policy(FirePolicy policy);
```

Construct `AimControllerOptions` from scalar tuning, fire state/policy, and the
optional calibration profile. Forward both live setters to the open controller;
also update stored options so close/reopen preserves the requested state.

- [ ] **Step 7: Update CLI compatibility**

Keep `--hid-click` and `--hid-click-cooldown`. Add help text for body firing and
confidence config through `--config`. When `--calibrate-hid` is used, capture the
returned profile, print `calibration_quality`, and exit without opening an
inference session.

- [ ] **Step 8: Run GREEN verification**

```powershell
xmake run vision_analyzer_tests
```

Expected: probe adjustment, config parsing, all previous scalar calibration,
and all controller tests pass.

- [ ] **Step 9: Commit runtime calibration orchestration**

```powershell
git add include/vision_analyzer/calibration.hpp src/calibration.cpp include/vision_analyzer/types.hpp include/vision_analyzer/runtime_session.hpp src/runtime_session.cpp src/runtime_config.cpp src/runtime_options.cpp src/main.cpp tests/test_algorithms.cpp
git commit -m "feat: orchestrate adaptive startup calibration"
```

### Task 6: Expose calibration and fire control through the C ABI

**Files:**
- Modify: `include/vision_analyzer/vision_runtime_c_api.h`
- Modify: `src/vision_runtime_c_api.cpp`
- Modify: `tests/test_c_api.cpp`

- [ ] **Step 1: Add failing ABI and setter tests**

Add compile-time layout protection and runtime validation:

```cpp
static_assert(VA_HID_CALIBRATION_LEVELS == 3);
static_assert(sizeof(VaHidCalibrationProfile) == 84);

void test_fire_and_calibration_api_validation() {
    VaRuntime* runtime = va_create();
    require(runtime != nullptr, "runtime should exist");
    require(va_set_fire_enabled(runtime, 1) == 0, "fire enable should succeed");
    require(va_set_fire_policy(runtime, 1, 0.35F, 0.45F, 3) == 0,
            "valid fire policy should succeed");
    require(va_set_fire_policy(runtime, 1, 1.5F, 0.45F, 3) == -1,
            "invalid head confidence should fail");
    require(va_calibrate_hid(runtime, 0, 0, nullptr) == -1,
            "null calibration output should fail before touching hardware");
    va_destroy(runtime);
}
```

Register the test in `main()`.

- [ ] **Step 2: Run the C API build to verify RED**

```powershell
xmake build vision_runtime_c_api_tests
```

Expected: missing struct, constant, and functions fail compilation.

- [ ] **Step 3: Add the fixed C layout and declarations**

Add the exact `VA_HID_CALIBRATION_LEVELS`, `VaHidCalibrationProfile`,
`va_calibrate_hid`, `va_set_fire_enabled`, and `va_set_fire_policy` declarations
from the approved design. Append fields/functions; do not reorder any existing
`VaRuntimeAction` field or remove any symbol.

- [ ] **Step 4: Implement conversion and setters**

Add one converter that zero-initializes the C struct, sets `schema_version=1`,
copies all three X/Y knots, and copies dimensions/deadzone/max-step/noise/quality/
sample count.

`va_set_fire_enabled` updates `runtime->options.fire_enabled` and forwards to an
open session. `va_set_fire_policy` performs explicit finite/range validation,
updates options, and forwards to an open session. Update legacy
`va_set_hid_click` to call the same fire-state/policy path.

- [ ] **Step 5: Implement the blocking calibration API**

`va_calibrate_hid` must:

1. reject a null output pointer;
2. reject an open runtime session;
3. require a nonempty HID port;
4. force requested output and fire state false;
5. set input to DXGI and apply adapter/output;
6. call `run_hid_calibration`;
7. install the returned profile only when valid;
8. convert it into the C output structure;
9. propagate every exception through `call_api`/`va_last_error`.

- [ ] **Step 6: Run C++ and C API tests**

```powershell
xmake run vision_analyzer_tests
xmake run vision_runtime_c_api_tests
```

Expected: both suites pass, including the fixed 84-byte ABI assertion.

- [ ] **Step 7: Commit the C API**

```powershell
git add include/vision_analyzer/vision_runtime_c_api.h src/vision_runtime_c_api.cpp tests/test_c_api.cpp
git commit -m "feat: expose calibration and autofire C APIs"
```

### Task 7: Add the Python control plane in the outer repository

**Files (outer worktree):**
- Modify: `src/cs2_vision_runtime/runtime.py`
- Modify: `src/cs2_vision_runtime/__init__.py`
- Modify: `tests/test_vision_runtime_sdk.py`
- Modify: `examples/runtime_live_move.py`
- Modify: `README.md`

- [ ] **Step 1: Add failing wrapper-forwarding and profile tests**

Extend `FakeApi` with `set_output_enabled`, `set_fire_enabled`,
`set_fire_policy`, and `calibrate_hid`. Have `calibrate_hid` fill an
`_CCalibrationProfile` with three knots and return zero.

Add tests:

Insert these methods inside the existing `FakeApi` class:

```python
def set_output_enabled(self, handle, enabled):
    self.calls.append(("set_output_enabled", bool(enabled)))
    return 0

def set_fire_enabled(self, handle, enabled):
    self.calls.append(("set_fire_enabled", bool(enabled)))
    return 0

def set_fire_policy(self, handle, body, head_conf, body_conf, cooldown):
    self.calls.append(("set_fire_policy", bool(body), head_conf, body_conf, cooldown))
    return 0

def calibrate_hid(self, handle, adapter, output, profile):
    self.calls.append(("calibrate_hid", adapter, output))
    profile.schema_version = 1
    profile.valid = 1
    profile.frame_width = 1920
    profile.frame_height = 1080
    profile.x_shift_px[:] = (8.0, 32.0, 96.0)
    profile.x_counts_per_pixel[:] = (2.0, 3.0, 4.0)
    profile.y_shift_px[:] = (8.0, 32.0, 96.0)
    profile.y_counts_per_pixel[:] = (-2.0, -3.0, -4.0)
    profile.deadzone_px = 1.0
    profile.max_step = 120
    profile.quality = 0.9
    return 0
```

Add these tests:

```python

def test_runtime_forwards_live_control_and_fire_policy():
    api = FakeApi()
    runtime = VisionRuntime(_api=api)
    runtime.set_output_enabled(True)
    runtime.set_fire_enabled(True)
    runtime.set_fire_policy(
        body_enabled=True,
        head_confidence=0.35,
        body_confidence=0.45,
        cooldown_frames=3,
    )
    assert ("set_output_enabled", True) in api.calls
    assert ("set_fire_enabled", True) in api.calls
    assert ("set_fire_policy", True, 0.35, 0.45, 3) in api.calls

def test_runtime_converts_calibration_profile():
    api = FakeApi()
    runtime = VisionRuntime(_api=api)
    profile = runtime.calibrate_hid(adapter=1, output=2)
    assert profile.valid is True
    assert profile.x_shift_px == (8.0, 32.0, 96.0)
    assert profile.y_counts_per_pixel[0] < 0.0
    assert ("calibrate_hid", 1, 2) in api.calls

def test_armed_loop_always_disarms_after_processing_error():
    from examples.runtime_live_move import run_armed_loop

    class ExplodingRuntime:
        def __init__(self):
            self.calls = []
        def set_output_enabled(self, value):
            self.calls.append(("output", value))
        def set_fire_enabled(self, value):
            self.calls.append(("fire", value))
        def process_next(self):
            raise RuntimeError("capture failed")
        def stop_all(self):
            self.calls.append(("stop_all",))

    runtime = ExplodingRuntime()
    with pytest.raises(RuntimeError, match="capture failed"):
        run_armed_loop(runtime, fire_enabled=True, show_every=30)
    assert runtime.calls == [
        ("output", True), ("fire", True),
        ("fire", False), ("output", False), ("stop_all",),
    ]
```

- [ ] **Step 2: Run the focused test to verify RED**

From the outer worktree:

```powershell
uv run pytest tests\test_vision_runtime_sdk.py -q
```

Expected: imports or methods fail because calibration/live control is absent.

- [ ] **Step 3: Add ctypes and dataclass layouts**

Define `_CCalibrationProfile` with exactly the C field order and
`ctypes.c_float * 3` arrays. Add a frozen `HidCalibrationProfile` dataclass whose
`from_c` converts every array to an immutable three-element tuple.

Bind exact signatures:

```python
dll.va_set_output_enabled.argtypes = [ctypes.c_void_p, ctypes.c_int32]
dll.va_set_fire_enabled.argtypes = [ctypes.c_void_p, ctypes.c_int32]
dll.va_set_fire_policy.argtypes = [
    ctypes.c_void_p, ctypes.c_int32, ctypes.c_float, ctypes.c_float, ctypes.c_int32
]
dll.va_calibrate_hid.argtypes = [
    ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
    ctypes.POINTER(_CCalibrationProfile),
]
```

Set every restype to `ctypes.c_int32`.

- [ ] **Step 4: Implement low-level and public methods**

Low-level `_RuntimeApi.calibrate_hid` accepts an output structure and passes
`ctypes.byref(profile)`. Public `VisionRuntime.calibrate_hid` allocates it,
checks the result, and returns the dataclass. The three live control methods call
`_check` exactly like existing setters.

Export `HidCalibrationProfile` from `src/cs2_vision_runtime/__init__.py`.

- [ ] **Step 5: Make the no-UI live example calibration-first and fail-safe**

Add `--enable-live-output`. Without it, exit before calibration with a message
that no physical output was armed. With it:

```python
def process_loop(runtime: VisionRuntime, show_every: int) -> None:
    while True:
        action = runtime.process_next()
        if action is None:
            return
        if action.frame_index % show_every == 0:
            print(
                f"frame={action.frame_index} target={int(action.has_target)} "
                f"dx={action.dx} dy={action.dy} click={int(action.click_left)} "
                f"lock={action.lock_state.name} det={action.detection_count}"
            )

def run_armed_loop(runtime: VisionRuntime, *, fire_enabled: bool, show_every: int) -> None:
    try:
        runtime.set_output_enabled(True)
        runtime.set_fire_enabled(fire_enabled)
        process_loop(runtime, show_every)
    finally:
        runtime.set_fire_enabled(False)
        runtime.set_output_enabled(False)
        runtime.stop_all()

with VisionRuntime() as runtime:
    runtime.set_model(args.model, schema_path=args.schema, backend=args.backend)
    runtime.set_hid_port(args.hid_port)
    profile = runtime.calibrate_hid(adapter=args.adapter, output=args.output)
    runtime.set_fire_policy(body_enabled=True, head_confidence=0.35,
                            body_confidence=0.45, cooldown_frames=3)
    runtime.open_dxgi(adapter=args.adapter, output=args.output,
                      player_side=args.player_side, hid_port=args.hid_port,
                      dry_run=False)
    run_armed_loop(runtime, fire_enabled=args.click, show_every=args.show_every)
```

The test above exercises the arming boundary without hardware.

- [ ] **Step 6: Document the Python-controlled lifecycle**

Update the outer README with the exact call order, explain that calibration must
run in a stable playable scene, and state that `set_output_enabled` and
`set_fire_enabled` are independent.

- [ ] **Step 7: Run Python GREEN verification**

```powershell
uv run pytest tests\test_vision_runtime_sdk.py -q
uv run pytest -q
```

Expected: all focused and repository Python tests pass.

- [ ] **Step 8: Commit the outer wrapper**

```powershell
git add src/cs2_vision_runtime/runtime.py src/cs2_vision_runtime/__init__.py tests/test_vision_runtime_sdk.py examples/runtime_live_move.py README.md
git commit -m "feat: add Python adaptive autofire control"
```

### Task 8: Ship the Python wrapper in the portable package

**Files (C++ submodule):**
- Modify: `packaging/sm61/build-portable-package.ps1`
- Modify: `packaging/sm61/tests/run-tests.ps1`
- Modify: `packaging/sm61/package/README_中文.md`
- Modify: `README.md`

- [ ] **Step 1: Add failing builder-source and safety tests**

Read `build-portable-package.ps1` as text and require a
`PythonProjectRoot` parameter plus these exact source/destination fragments:

```text
src\cs2_vision_runtime\__init__.py
src\cs2_vision_runtime\runtime.py
examples\runtime_live_move.py
python\cs2_vision_runtime
```

Add source assertions that the example contains `calibrate_hid`,
`set_output_enabled`, `set_fire_enabled`, `finally`, and the explicit
`--enable-live-output` gate. Continue asserting that every `.ps1`/`.cmd`
one-click diagnostic contains no output-arming flag.

- [ ] **Step 2: Run package tests to verify RED**

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File packaging\sm61\tests\run-tests.ps1
```

Expected: failure because `PythonProjectRoot` and portable Python copy logic are
absent.

- [ ] **Step 3: Copy the wrapper from the outer project root**

Add `[string]$PythonProjectRoot = ''` to the builder parameters. After resolving
`$projectRoot`, default the new value to `$projectRoot`; otherwise normalize the
explicit value with `[IO.Path]::GetFullPath`. Before staging mutation, require:

```powershell
$pythonPackageRoot = Join-Path $PythonProjectRoot 'src\cs2_vision_runtime'
$pythonExample = Join-Path $PythonProjectRoot 'examples\runtime_live_move.py'
Assert-LeafFile -LiteralPath (Join-Path $pythonPackageRoot '__init__.py') -Description 'Python runtime package initializer'
Assert-LeafFile -LiteralPath (Join-Path $pythonPackageRoot 'runtime.py') -Description 'Python runtime wrapper'
Assert-LeafFile -LiteralPath $pythonExample -Description 'Python live runtime example'
```

Create `python\cs2_vision_runtime` and `examples` in staging, then copy only
`__init__.py`, `runtime.py`, and `runtime_live_move.py`. Do not copy `__pycache__`
or outer-repository development files.

- [ ] **Step 4: Update package documentation**

Document `PYTHONPATH=D:\cs2-vision-runtime-sm61\python`, the exact
calibration/open/arm/disarm order, required `--enable-live-output`, and the
distinction between the safe one-click test and a real Python-controlled
session.

- [ ] **Step 5: Run package GREEN verification**

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File packaging\sm61\tests\run-tests.ps1
```

Expected: all package tests pass with the new wrapper and unchanged dry-run
safety assertions.

- [ ] **Step 6: Commit packaging integration**

```powershell
git add packaging/sm61/build-portable-package.ps1 packaging/sm61/tests/run-tests.ps1 packaging/sm61/package/README_中文.md README.md
git commit -m "feat: package Python adaptive runtime controls"
```

### Task 9: Integrate the final C++ commit into the outer branch

**Files (outer worktree):**
- Modify gitlink: `tools/cpp_analyzer`

- [ ] **Step 1: Record and verify the final C++ commit**

From the C++ worktree:

```powershell
$cppCommit = git rev-parse HEAD
git status --short --branch
git show --stat --oneline $cppCommit
```

Expected: a clean C++ feature branch and a commit containing all C++/package
changes.

- [ ] **Step 2: Initialize the submodule in the outer worktree and select that commit**

From `C:\Users\xiaol\.config\superpowers\worktrees\cs2-vision-trainer\adaptive-hid-host`:

```powershell
git submodule update --init tools/cpp_analyzer
$cppCommit = git -C 'D:\project\cs2-vision-trainer\tools\cpp_analyzer\.worktrees\sm61-portable-package' rev-parse HEAD
git -C tools/cpp_analyzer checkout $cppCommit
git add tools/cpp_analyzer
```

Expected: outer status shows only the intended submodule pointer update.

- [ ] **Step 3: Commit the pointer update**

```powershell
git commit -m "build: update adaptive vision runtime submodule"
```

### Task 10: Rebuild, repackage, and verify the complete product

**Files:**
- Generated C++ build outputs under the C++ worktree.
- Generated ignored package under `D:\project\cs2-vision-trainer\dist`.

- [ ] **Step 1: Run every source test suite**

From the C++ worktree:

```powershell
xmake build vision_analyzer vision_runtime vision_analyzer_tests vision_runtime_c_api_tests
xmake run vision_analyzer_tests
xmake run vision_runtime_c_api_tests
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File packaging\sm61\tests\run-tests.ps1
```

From the outer worktree:

```powershell
uv run pytest -q
```

Expected: every command exits zero with no failing test.

- [ ] **Step 2: Run a clean CMake build and CTest**

From the C++ worktree:

```powershell
$env:ONNXRUNTIME_ROOT='D:\Tool\onnxruntime-win-x64-gpu-1.17.3'
cmake -S . -B build-cmake-adaptive -G 'Visual Studio 17 2022' -A x64 -DONNXRUNTIME_ROOT='D:\Tool\onnxruntime-win-x64-gpu-1.17.3'
cmake --build build-cmake-adaptive --config Release --parallel
ctest --test-dir build-cmake-adaptive -C Release --output-on-failure
```

Expected: both CTest executables pass.

- [ ] **Step 3: Build the new portable ZIP**

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass `
  -File packaging\sm61\build-portable-package.ps1 `
  -PythonProjectRoot 'C:\Users\xiaol\.config\superpowers\worktrees\cs2-vision-trainer\adaptive-hid-host' `
  -TensorRtArchive 'C:\Users\xiaol\Downloads\TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8.zip'
```

Expected: staging and `D:\project\cs2-vision-trainer\dist\cs2-vision-runtime-sm61.zip`
are recreated with a new manifest and SHA256.

- [ ] **Step 4: Extract into a verified clean directory**

Resolve `D:\project\cs2-vision-trainer\dist\verify-sm61-adaptive` and verify it
is a strict child of `dist` before deleting it. Recreate it and run:

```powershell
tar.exe -xf 'D:\project\cs2-vision-trainer\dist\cs2-vision-runtime-sm61.zip' `
  -C 'D:\project\cs2-vision-trainer\dist\verify-sm61-adaptive'
```

- [ ] **Step 5: Run static and three-frame dry verification**

Run `scripts\verify-runtime.ps1 -AllowUnsupportedGpu -StaticOnly` from the clean
extraction, then run the packaged CLI for three frames with `--backend
opencv-onnx`, `--video samples\smoke-test.mp4`, and `--dry-run`.

Expected: manifest/DLL/profile checks pass, `processed_frames=3`, and no HID
output is armed.

- [ ] **Step 6: Load the packaged Python wrapper against the real DLL**

From the extracted package:

```powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'python')
$env:CS2_VISION_RUNTIME_DLL=(Join-Path (Get-Location) 'app\vision_runtime.dll')
uv run --project 'C:\Users\xiaol\.config\superpowers\worktrees\cs2-vision-trainer\adaptive-hid-host' python -c "from cs2_vision_runtime import VisionRuntime; r=VisionRuntime(); r.set_output_enabled(False); r.set_fire_enabled(False); r.close(); print('PASS packaged Python C API')"
```

Expected: `PASS packaged Python C API` and exit zero. Do not invoke calibration
or live output on the development machine.

- [ ] **Step 7: Audit archive safety and versions**

Verify the final archive contains the Python wrapper/example and contains no
`.engine`, `.plan`, CUDA 12, cuDNN 9, TensorRT 10/11, or automatic arming in any
PowerShell/CMD diagnostic. Record ZIP bytes/SHA256 and TensorRT archive SHA256.

- [ ] **Step 8: Run final repository checks**

```powershell
git diff --check
git status --short --branch
```

Run the same two commands in both the C++ and outer worktrees. Expected: both
feature branches are clean. Real adaptive calibration and physical autofire
remain explicitly pending GTX 1080 Ti + RP2350 + stable CS2 production-machine
acceptance.
