# Center-ROI Optical-Flow HID Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace phase-correlation HID calibration measurement with bounded, center-ROI, multi-frame optical flow that reliably measures scene movement around a fixed crosshair and persists the fitted profile.

**Architecture:** A pure center-flow estimator tracks distributed features inside a centered 640-by-480 ROI while masking the central crosshair. Runtime calibration captures short post-command bursts, selects the strongest reliable flow result, validates balanced outward/return motion, and reuses the existing profile fitter and atomic store. Probe commands are hard-capped at 120 counts.

**Tech Stack:** C++17, OpenCV `goodFeaturesToTrack` and pyramidal Lucas-Kanade flow, DXGI Desktop Duplication, RP2350 protocol-v2 SDK, CMake/CTest, Python 3.11 ctypes, PowerShell packaging.

---

## File Structure

- `include/vision_analyzer/calibration.hpp`: public pure-algorithm data structures and function declarations used by tests and calibration orchestration.
- `src/calibration.cpp`: center ROI extraction, optical-flow filtering, burst selection, bounded discovery, and live balanced measurement.
- `tests/test_algorithms.cpp`: synthetic camera/HUD tests, selection tests, bounded-probe tests, and regression coverage.
- `tests/test_c_api.cpp`: unchanged persistence/C ABI behavior; rerun as an integration gate.
- `docs/superpowers/specs/2026-07-25-center-roi-optical-flow-hid-calibration-design.md`: approved requirements.

### Task 1: Pure Center-ROI Flow Estimator

**Files:**
- Modify: `include/vision_analyzer/calibration.hpp`
- Modify: `src/calibration.cpp`
- Test: `tests/test_algorithms.cpp`

- [ ] **Step 1: Write the failing fixed-crosshair camera-motion test**

Add this public result shape and wished-for call to the test:

```cpp
struct CenterFlowEstimate {
    cv::Point2d shift;
    int detected_features = 0;
    int tracked_features = 0;
    int inlier_features = 0;
    int occupied_cells = 0;
    double spread_px = 0.0;
    bool reliable = false;
};

const CenterFlowEstimate estimate = estimate_center_flow(before, after);
require(estimate.reliable, "center scene motion should be measurable");
require_near(static_cast<float>(estimate.shift.x), -12.0F, 1.5F,
             "center flow should recover camera X movement");
require(std::abs(estimate.shift.y) < 1.5,
        "fixed crosshair must not create cross-axis flow");
require(estimate.inlier_features >= 12 && estimate.occupied_cells >= 3,
        "camera movement needs distributed scene support");
```

Build the frame from a textured source, translate/perspective-warp the game-world background, and repaint identical static radar, score, kill-feed, and central crosshair overlays onto both frames.

- [ ] **Step 2: Run the test to verify RED**

Run:

```powershell
cmake --build build-cmake --config Release --target vision_analyzer_tests --parallel
```

Expected: compilation fails because `CenterFlowEstimate` and `estimate_center_flow()` do not exist.

- [ ] **Step 3: Declare the estimator API**

In `calibration.hpp`, add the exact `CenterFlowEstimate` structure above and:

```cpp
[[nodiscard]] CenterFlowEstimate estimate_center_flow(
    const cv::Mat& before,
    const cv::Mat& after
);
```

- [ ] **Step 4: Implement the minimal robust flow estimator**

In `calibration.cpp`:

1. Convert both frames to 8-bit grayscale and reject different sizes.
2. Crop a centered ROI with `width=min(640, frame.cols)` and `height=min(480, frame.rows)`.
3. Create a full-white feature mask and fill a centered 96-by-96 rectangle with zero.
4. Detect at most 240 features with:

```cpp
cv::goodFeaturesToTrack(before_roi, before_points, 240, 0.01, 8.0,
                        feature_mask, 3, false, 0.04);
```

5. Track forward and backward with `calcOpticalFlowPyrLK`, a 31-by-31 window, three pyramid levels, and a 30-iteration/0.01 termination criterion.
6. Keep tracks whose forward and backward statuses are true, forward-backward error is at most 1.5 pixels, and coordinates remain inside the ROI.
7. Compute median flow, median residual, and keep residuals at most `max(1.5, 3*median_residual)`.
8. Recompute median X/Y from inliers and count occupied cells in a 4-by-3 grid using baseline feature locations.
9. Set `reliable=true` only for at least 12 inliers, at least three occupied cells, and finite shift/spread.

Include `<opencv2/video/tracking.hpp>`.

- [ ] **Step 5: Run the algorithm test to verify GREEN**

Run:

```powershell
cmake --build build-cmake --config Release --target vision_analyzer_tests --parallel
ctest --test-dir build-cmake -C Release -R vision_analyzer_tests --output-on-failure
```

Expected: build succeeds and the algorithm test passes.

- [ ] **Step 6: Commit**

```powershell
git add include/vision_analyzer/calibration.hpp src/calibration.cpp tests/test_algorithms.cpp
git commit -m "feat: measure center scene motion with optical flow"
```

### Task 2: Reject Texture and Direction Failures

**Files:**
- Modify: `src/calibration.cpp`
- Test: `tests/test_algorithms.cpp`

- [ ] **Step 1: Write failing estimator rejection tests**

Add three tests:

```cpp
require(!estimate_center_flow(blank_before, blank_after).reliable,
        "blank center must report insufficient texture");
require(!estimate_center_flow(one_corner_before, one_corner_after).reliable,
        "one spatial cell must not determine camera movement");
const CenterFlowEstimate perspective = estimate_center_flow(warped_before, warped_after);
require(perspective.reliable && std::abs(perspective.shift.x) >= 8.0,
        "local perspective motion must survive static HUD");
```

- [ ] **Step 2: Run and confirm RED**

Run the CMake algorithm test. Expected: at least one new assertion fails because coverage/outlier filtering is incomplete.

- [ ] **Step 3: Make spatial coverage and MAD rejection exact**

Ensure only final inliers contribute to occupied cells, crosshair-masked features never enter the candidate set, and an empty or undersupported result returns all diagnostic counts with `reliable=false` rather than fabricating zero movement.

- [ ] **Step 4: Run and confirm GREEN**

Run the CMake algorithm test. Expected: all new rejection/perspective tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/calibration.cpp tests/test_algorithms.cpp
git commit -m "test: reject unsupported center flow estimates"
```

### Task 3: Multi-Frame Burst Selection

**Files:**
- Modify: `include/vision_analyzer/calibration.hpp`
- Modify: `src/calibration.cpp`
- Test: `tests/test_algorithms.cpp`

- [ ] **Step 1: Write a failing burst-selection test**

Declare the wished-for pure selector:

```cpp
const std::size_t selected = select_center_flow_candidate({
    CenterFlowEstimate{{0, 0}, 80, 60, 0, 0, 0.0, false},
    CenterFlowEstimate{{-11.8, 0.2}, 90, 70, 38, 7, 0.6, true},
    CenterFlowEstimate{{-12.1, 0.1}, 90, 72, 29, 6, 0.3, true},
});
require(selected == 1,
        "burst selection should prefer more distributed reliable inliers");
```

Also require the selector to throw when no candidate is reliable.

- [ ] **Step 2: Run and confirm RED**

Expected: compilation fails because `select_center_flow_candidate()` is missing.

- [ ] **Step 3: Implement deterministic selection**

Declare and implement:

```cpp
[[nodiscard]] std::size_t select_center_flow_candidate(
    const std::vector<CenterFlowEstimate>& candidates
);
```

Filter to reliable candidates, then rank by inlier count descending, occupied cells descending, spread ascending, and original index ascending.

- [ ] **Step 4: Run and confirm GREEN**

Run the CMake algorithm test. Expected: selector tests pass.

- [ ] **Step 5: Commit**

```powershell
git add include/vision_analyzer/calibration.hpp src/calibration.cpp tests/test_algorithms.cpp
git commit -m "feat: select reliable center flow bursts"
```

### Task 4: Replace Live Phase-Correlation Measurement

**Files:**
- Modify: `src/calibration.cpp`
- Test: `tests/test_algorithms.cpp`

- [ ] **Step 1: Write a failing probe-bound test**

Change the discovery exhaustion test to expect exactly the bounded ladder:

```cpp
const std::array<int, 4> expected = {16, 32, 64, 120};
require(*std::max_element(attempted.begin(), attempted.end()) <= 120,
        "calibration must never exceed the runtime movement bound");
```

Require the final error to contain `stable textured surface` rather than a phase-correlation/coherence message.

- [ ] **Step 2: Run and confirm RED**

Expected: current discovery still attempts values above 120 and the assertion fails.

- [ ] **Step 3: Hard-cap discovery at 120**

Set:

```cpp
constexpr int kCalibrationProbeMaximumCounts = 120;
constexpr int kCalibrationDiscoveryMaximumAttempts = 4;
```

Make the ladder `16 -> 32 -> 64 -> 120`, never scale above 120, and change exhaustion to:

```text
HID calibration input not ready: axis=x center movement was not measurable through 120 counts; present a stable textured surface near the crosshair
```

- [ ] **Step 4: Replace `measure_balanced_probe` frame measurement**

Keep the existing exception-safe inverse command. Replace robust phase correlation with:

1. capture a stable baseline frame;
2. send outward HID command;
3. capture six frames and call `estimate_center_flow(baseline, candidate)`;
4. choose with `select_center_flow_candidate` and retain that exact moved frame;
5. send the inverse command;
6. capture six return candidates measured from the selected moved frame;
7. select the returned frame and set it as the next baseline;
8. convert flow reliability to `VisualShiftEstimate`, with response equal to `inliers / max(1, tracked_features)`.

Log feature, track, inlier, cell, spread, and burst-index fields for both legs.

- [ ] **Step 5: Run and confirm GREEN**

Run CMake build and CTest. Expected: bounded discovery and all algorithm tests pass.

- [ ] **Step 6: Commit**

```powershell
git add include/vision_analyzer/calibration.hpp src/calibration.cpp tests/test_algorithms.cpp
git commit -m "fix: calibrate HID from bounded center flow bursts"
```

### Task 5: Full Integration and Packaging Gates

**Files:**
- Verify: `tests/test_algorithms.cpp`
- Verify: `tests/test_c_api.cpp`
- Generate outside Git: corrected SM61 incremental ZIP

- [ ] **Step 1: Build only from the current worktree with CMake Release**

```powershell
cmake --build build-cmake --config Release --parallel
```

Expected: `build-cmake/Release/vision_runtime.dll` and `vision_analyzer.exe` are rebuilt after the center-flow commits.

- [ ] **Step 2: Run the complete safe test matrix**

```powershell
ctest --test-dir build-cmake -C Release --output-on-failure
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\packaging\sm61\tests\run-tests.ps1
$env:CS2_VISION_CPP_ANALYZER_ROOT=(Resolve-Path '.').Path
Push-Location 'D:\project\cs2-vision-trainer'; uv run pytest -q; Pop-Location
```

Expected: CTest 2/2, package tests 26/26, and parent Python tests 68/68.

- [ ] **Step 3: Inspect packaged exports before ZIP creation**

Use `dumpbin /exports` on the staged DLL and require all 27 exports, including:

```text
va_set_hid_calibration_path
va_get_hid_calibration
va_calibrate_hid
```

- [ ] **Step 4: Instantiate the exact staged DLL through Python**

Run inside the staged package runtime environment:

```powershell
python -c "from cs2_vision_runtime import VisionRuntime; r=VisionRuntime(); r.close(); print('VisionRuntime_init=ok')"
```

Expected: `VisionRuntime_init=ok`.

- [ ] **Step 5: Create and verify the incremental ZIP**

Include only the DLL, CLI, import library, C header, Python wrapper/example, package README, full overlay runtime manifest, and hotfix manifest. Expand the ZIP, compare every SHA256, reject model/runtime/firmware payloads, and inspect the extracted DLL export table again.

- [ ] **Step 6: Report one acceptance command**

The command must use COM4, `--calibration-path .\hid-calibration.json`, and `--recalibrate` for the first run. A successful run must stop discovery at or before 120 counts, print center-flow diagnostics, save the JSON, and start the runtime.
