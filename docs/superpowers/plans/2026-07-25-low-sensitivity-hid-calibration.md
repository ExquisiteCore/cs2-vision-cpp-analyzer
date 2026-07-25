# Low-Sensitivity HID Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make startup HID calibration automatically measure low-sensitivity CS2 accounts with probes up to 2048 counts while keeping normal runtime movement clamped to 120.

**Architecture:** Add pure, unit-tested probe planning and level derivation next to the existing calibration API. `run_hid_calibration` will use a balanced probe callback to discover one reliable gain per axis, derive three axis-specific levels, and feed the unchanged signed fitter. No C ABI, Python API, firmware, model, or dependency version changes are required.

**Tech Stack:** C++17, OpenCV phase correlation, xmake, MSVC 2022, Windows PowerShell 5.1, RP2350 protocol v2 SDK, ONNX Runtime GPU 1.17.3, TensorRT 8.6.1.6, CUDA 11.8, cuDNN 8.9.7.

---

## File map

- `include/vision_analyzer/calibration.hpp`: calibration-only constants, probe/level value types, and pure planning function declarations.
- `src/calibration.cpp`: pure planning math, axis discovery state machine, balanced HID/DXGI probe helper, dynamic sample levels, diagnostics, and runtime/probe limit separation.
- `tests/test_algorithms.cpp`: RED/GREEN coverage for normal sensitivity, low sensitivity, discovery exhaustion, cross-axis rejection, retry behavior, and the unchanged runtime clamp.
- `README.md`: developer-facing explanation of the 2048 calibration-only limit.
- `packaging/sm61/package/README_中文.md`: production operator instructions and success/failure log markers.
- Generated `D:/project/cs2-vision-trainer/dist/cs2-vision-runtime-sm61-low-sensitivity/`: fully validated package staging tree.
- Generated `C:/Users/xiaol/Downloads/cs2-vision-runtime-sm61-low-sensitivity-hotfix.zip`: manifest-aware overlay for the currently deployed v2 package.

### Task 1: Derive bounded three-level plans independently of runtime max step

**Files:**
- Modify: `tests/test_algorithms.cpp:727-733,1116-1174`
- Modify: `include/vision_analyzer/calibration.hpp:1-70`
- Modify: `src/calibration.cpp:137-153`

- [ ] **Step 1: Write failing level-plan tests**

Replace `test_calibration_probe_adjustment_is_bounded` and add two focused tests:

```cpp
void test_calibration_probe_adjustment_uses_calibration_only_limit() {
    require(adjust_calibration_probe_count(16, 0.5, 8.0, 2048) == 256,
            "tiny movement should scale above the runtime max step");
    require(adjust_calibration_probe_count(512, 0.5, 8.0, 2048) == 2048,
            "probe adjustment must stop at the calibration limit");
    require(adjust_calibration_probe_count(80, 200.0, 80.0, 2048) == 32,
            "oversized movement should scale down proportionally");
}

void test_calibration_level_plan_compresses_low_sensitivity_range() {
    const CalibrationLevelPlan normal = derive_calibration_level_plan(2.0, 1.5);
    require(normal.counts == std::array<int, 3>{16, 64, 160},
            "normal sensitivity should retain 8/32/80-pixel targets");

    const CalibrationLevelPlan low = derive_calibration_level_plan(60.0, 1.5);
    require(low.counts == std::array<int, 3>{480, 1024, 2048},
            "low sensitivity should use the complete calibration range");
    require(low.counts[0] < low.counts[1] && low.counts[1] < low.counts[2],
            "derived counts must be strictly increasing");
}

void test_calibration_level_plan_rejects_unmeasurable_range() {
    bool rejected = false;
    try {
        (void)derive_calibration_level_plan(400.0, 1.5);
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("2048") != std::string::npos;
    }
    require(rejected, "an unmeasurable 2048-count range must be rejected explicitly");
}
```

Register all three functions in `main()` where the old probe-adjustment test is called.

- [ ] **Step 2: Run RED tests**

Run:

```powershell
xmake build vision_analyzer_tests
```

Expected: compilation fails because the four-argument adjustment overload and
`CalibrationLevelPlan`/`derive_calibration_level_plan` do not exist.

- [ ] **Step 3: Declare calibration-only limits and the level plan**

Add `<array>` and `<functional>` to `calibration.hpp`, then add after
`VisualShiftEstimate`:

```cpp
constexpr int kCalibrationProbeMinimumCounts = 8;
constexpr int kCalibrationProbeMaximumCounts = 2048;
constexpr int kCalibrationDiscoveryMaximumAttempts = 8;
constexpr int kCalibrationRuntimeMaxStep = 120;
constexpr double kCalibrationMinimumPhaseResponse = 0.15;

struct CalibrationLevelPlan {
    std::array<int, kHidCalibrationLevels> counts{};
    std::array<double, kHidCalibrationLevels> target_shift_px{};
};
```

Change the adjustment declaration and add the level-plan declaration:

```cpp
[[nodiscard]] int adjust_calibration_probe_count(
    int current_counts,
    double observed_shift_px,
    double target_shift_px,
    int maximum_counts = kCalibrationProbeMaximumCounts
);

[[nodiscard]] CalibrationLevelPlan derive_calibration_level_plan(
    double counts_per_pixel,
    double minimum_measurable_shift_px,
    int maximum_counts = kCalibrationProbeMaximumCounts
);
```

- [ ] **Step 4: Implement bounded adjustment and level compression**

Replace `adjust_calibration_probe_count` with:

```cpp
int adjust_calibration_probe_count(
    int current_counts,
    double observed_shift_px,
    double target_shift_px,
    int maximum_counts
) {
    if (current_counts <= 0 || maximum_counts < kCalibrationProbeMinimumCounts ||
        !std::isfinite(observed_shift_px) || observed_shift_px < 0.0 ||
        !std::isfinite(target_shift_px) || target_shift_px <= 0.0) {
        throw std::invalid_argument("invalid adaptive HID calibration probe values");
    }
    const double scaled = static_cast<double>(current_counts) * target_shift_px /
                          std::max(0.5, observed_shift_px);
    if (!std::isfinite(scaled)) {
        throw std::invalid_argument("adaptive HID calibration probe overflowed");
    }
    const double bounded = std::clamp(
        scaled,
        static_cast<double>(kCalibrationProbeMinimumCounts),
        static_cast<double>(maximum_counts)
    );
    return static_cast<int>(std::lround(bounded));
}
```

Add this implementation immediately after it:

```cpp
CalibrationLevelPlan derive_calibration_level_plan(
    double counts_per_pixel,
    double minimum_measurable_shift_px,
    int maximum_counts
) {
    if (!std::isfinite(counts_per_pixel) || counts_per_pixel <= 0.0 ||
        !std::isfinite(minimum_measurable_shift_px) || minimum_measurable_shift_px <= 0.0 ||
        maximum_counts < kCalibrationProbeMinimumCounts + 2) {
        throw std::invalid_argument("invalid HID calibration level-plan values");
    }

    const double high = std::min(80.0, static_cast<double>(maximum_counts) / counts_per_pixel);
    const double low = std::min(8.0, high / 4.0);
    const double middle = std::min(32.0, high / 2.0);
    if (low < minimum_measurable_shift_px) {
        std::ostringstream message;
        message << "HID calibration range is not measurable within "
                << maximum_counts << " counts";
        throw std::runtime_error(message.str());
    }

    CalibrationLevelPlan plan;
    plan.target_shift_px = {low, middle, high};
    const std::array<long, kHidCalibrationLevels> rounded = {
        std::lround(low * counts_per_pixel),
        std::lround(middle * counts_per_pixel),
        std::lround(high * counts_per_pixel),
    };
    plan.counts[0] = static_cast<int>(std::clamp(
        rounded[0],
        static_cast<long>(kCalibrationProbeMinimumCounts),
        static_cast<long>(maximum_counts - 2)
    ));
    plan.counts[1] = static_cast<int>(std::clamp(
        rounded[1],
        static_cast<long>(plan.counts[0] + 1),
        static_cast<long>(maximum_counts - 1)
    ));
    plan.counts[2] = static_cast<int>(std::clamp(
        rounded[2],
        static_cast<long>(plan.counts[1] + 1),
        static_cast<long>(maximum_counts)
    ));
    return plan;
}
```

- [ ] **Step 5: Run GREEN tests**

Run:

```powershell
xmake build vision_analyzer_tests
xmake run vision_analyzer_tests
```

Expected: `algorithm tests passed`.

- [ ] **Step 6: Commit**

```powershell
git add include/vision_analyzer/calibration.hpp src/calibration.cpp tests/test_algorithms.cpp
git commit -m "feat: plan low-sensitivity calibration levels"
```

### Task 2: Discover a reliable count per axis with a tested state machine

**Files:**
- Modify: `tests/test_algorithms.cpp:727-780,1116-1178`
- Modify: `include/vision_analyzer/calibration.hpp`
- Modify: `src/calibration.cpp`

- [ ] **Step 1: Write failing planner and discovery tests**

Add:

```cpp
void test_calibration_probe_planner_escalates_low_response() {
    const CalibrationProbePlan plan = plan_calibration_probe(
        120, 0, 2.0, 0.1, 0.05, 4.0, 270.0
    );
    require(!plan.accepted && !plan.exhausted,
            "low-response probe should continue discovery");
    require(plan.next_counts == 240,
            "unreliable discovery should double instead of trusting its shift");
}

void test_calibration_probe_planner_accepts_signal_and_rejects_cross_axis() {
    const CalibrationProbePlan accepted = plan_calibration_probe(
        1000, 1, 8.0, 0.2, 0.80, 4.0, 270.0
    );
    require(accepted.accepted && !accepted.exhausted,
            "reliable discovery signal should be accepted");

    const CalibrationProbePlan crossed = plan_calibration_probe(
        1000, 1, 8.0, 4.0, 0.80, 4.0, 270.0
    );
    require(!crossed.accepted && crossed.exhausted,
            "dominant cross-axis movement should reject the scene without escalating");
}

void test_calibration_axis_discovery_derives_low_sensitivity_levels() {
    std::vector<int> attempted;
    const CalibrationAxisDiscovery discovery = discover_calibration_axis(
        0,
        4.0,
        1.5,
        270.0,
        [&](int counts) {
            attempted.push_back(counts);
            return VisualShiftEstimate{{-static_cast<double>(counts) / 60.0, 0.05}, 0.90};
        }
    );
    require(attempted == std::vector<int>({16, 480}),
            "reliable small signal should jump proportionally to an eight-pixel probe");
    require(discovery.probe_counts == 480, "discovery should retain the accepted count");
    require(discovery.levels.counts == std::array<int, 3>{480, 1024, 2048},
            "discovery should derive compressed low-sensitivity levels");
}

void test_calibration_axis_discovery_reports_probe_exhaustion() {
    std::vector<int> attempted;
    bool rejected = false;
    try {
        (void)discover_calibration_axis(
            0,
            4.0,
            1.5,
            270.0,
            [&](int counts) {
                attempted.push_back(counts);
                return VisualShiftEstimate{{0.0, 0.0}, 0.0};
            }
        );
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("axis=x") != std::string::npos &&
                   std::string(error.what()).find("2048") != std::string::npos;
    }
    require(rejected, "exhaustion must report the axis and calibration limit");
    require(!attempted.empty() && attempted.back() == 2048,
            "discovery should test the agreed limit before rejecting");
}
```

Register all four tests in `main()`.

- [ ] **Step 2: Run RED tests**

Run `xmake build vision_analyzer_tests`.

Expected: compilation fails because `CalibrationProbePlan`, `CalibrationAxisDiscovery`,
`plan_calibration_probe`, and `discover_calibration_axis` do not exist.

- [ ] **Step 3: Declare the planner and discovery boundary**

Add after `CalibrationLevelPlan`:

```cpp
struct CalibrationProbePlan {
    bool accepted = false;
    bool exhausted = false;
    int next_counts = 0;
};

struct CalibrationAxisDiscovery {
    int probe_counts = 0;
    double probe_shift_px = 0.0;
    double counts_per_pixel = 0.0;
    CalibrationLevelPlan levels;
};

using CalibrationProbeMeasure = std::function<VisualShiftEstimate(int)>;
```

Declare:

```cpp
[[nodiscard]] CalibrationProbePlan plan_calibration_probe(
    int current_counts,
    int attempt_index,
    double main_shift_px,
    double cross_shift_px,
    double phase_response,
    double minimum_discovery_shift_px,
    double maximum_reliable_shift_px
);

[[nodiscard]] CalibrationAxisDiscovery discover_calibration_axis(
    std::size_t axis,
    double minimum_discovery_shift_px,
    double minimum_measurable_shift_px,
    double maximum_reliable_shift_px,
    const CalibrationProbeMeasure& measure
);
```

Add `<cstddef>` for `std::size_t`.

- [ ] **Step 4: Implement the pure planner**

Implement `plan_calibration_probe` exactly as a bounded state transition:

```cpp
CalibrationProbePlan plan_calibration_probe(
    int current_counts,
    int attempt_index,
    double main_shift_px,
    double cross_shift_px,
    double phase_response,
    double minimum_discovery_shift_px,
    double maximum_reliable_shift_px
) {
    if (current_counts < kCalibrationProbeMinimumCounts ||
        current_counts > kCalibrationProbeMaximumCounts || attempt_index < 0 ||
        !std::isfinite(main_shift_px) || main_shift_px < 0.0 ||
        !std::isfinite(cross_shift_px) || cross_shift_px < 0.0 ||
        !std::isfinite(phase_response) ||
        !std::isfinite(minimum_discovery_shift_px) || minimum_discovery_shift_px <= 0.0 ||
        !std::isfinite(maximum_reliable_shift_px) ||
        maximum_reliable_shift_px <= minimum_discovery_shift_px) {
        throw std::invalid_argument("invalid HID calibration discovery values");
    }

    const bool last_attempt =
        attempt_index + 1 >= kCalibrationDiscoveryMaximumAttempts;
    if (main_shift_px > maximum_reliable_shift_px) {
        const int reduced = adjust_calibration_probe_count(
            current_counts,
            main_shift_px,
            8.0,
            kCalibrationProbeMaximumCounts
        );
        if (last_attempt || reduced >= current_counts) {
            return {false, true, current_counts};
        }
        return {false, false, reduced};
    }

    if (phase_response >= kCalibrationMinimumPhaseResponse &&
        main_shift_px >= minimum_discovery_shift_px) {
        if (cross_shift_px > main_shift_px * 0.35) {
            return {false, true, current_counts};
        }
        return {true, false, current_counts};
    }

    if (last_attempt || current_counts >= kCalibrationProbeMaximumCounts) {
        return {false, true, current_counts};
    }

    int proposed = std::min(
        current_counts * 2,
        kCalibrationProbeMaximumCounts
    );
    if (phase_response >= kCalibrationMinimumPhaseResponse && main_shift_px >= 0.5) {
        proposed = adjust_calibration_probe_count(
            current_counts,
            main_shift_px,
            8.0,
            kCalibrationProbeMaximumCounts
        );
    }
    const int next = std::clamp(
        std::max(current_counts + 1, proposed),
        kCalibrationProbeMinimumCounts,
        kCalibrationProbeMaximumCounts
    );
    return {false, next <= current_counts, next};
}
```

- [ ] **Step 5: Implement axis discovery**

Implement an eight-attempt loop starting at 16. It rejects a repeated count so proportional
shrink/growth cannot cycle:

```cpp
CalibrationAxisDiscovery discover_calibration_axis(
    std::size_t axis,
    double minimum_discovery_shift_px,
    double minimum_measurable_shift_px,
    double maximum_reliable_shift_px,
    const CalibrationProbeMeasure& measure
) {
    if (axis > 1 || !measure) {
        throw std::invalid_argument("invalid HID calibration discovery axis or callback");
    }

    int counts = 16;
    std::array<int, kCalibrationDiscoveryMaximumAttempts> attempted{};
    for (int attempt = 0; attempt < kCalibrationDiscoveryMaximumAttempts; ++attempt) {
        if (std::find(attempted.begin(), attempted.begin() + attempt, counts) !=
            attempted.begin() + attempt) {
            throw std::runtime_error("HID calibration discovery repeated a probe count");
        }
        attempted[attempt] = counts;

        const VisualShiftEstimate estimate = measure(counts);
        const double main_shift_px = std::abs(axis == 0 ? estimate.shift.x : estimate.shift.y);
        const double cross_shift_px = std::abs(axis == 0 ? estimate.shift.y : estimate.shift.x);
        const CalibrationProbePlan plan = plan_calibration_probe(
            counts,
            attempt,
            main_shift_px,
            cross_shift_px,
            estimate.response,
            minimum_discovery_shift_px,
            maximum_reliable_shift_px
        );
        if (plan.accepted) {
            const double counts_per_pixel = static_cast<double>(counts) / main_shift_px;
            return CalibrationAxisDiscovery{
                counts,
                main_shift_px,
                counts_per_pixel,
                derive_calibration_level_plan(
                    counts_per_pixel,
                    minimum_measurable_shift_px
                ),
            };
        }
        if (plan.exhausted) {
            std::ostringstream message;
            message << "HID calibration discovery exhausted: axis="
                    << (axis == 0 ? 'x' : 'y')
                    << " max_counts=" << kCalibrationProbeMaximumCounts
                    << " shift_px=" << main_shift_px
                    << " response=" << estimate.response;
            throw std::runtime_error(message.str());
        }
        counts = plan.next_counts;
    }
    throw std::runtime_error("HID calibration discovery exhausted its attempt budget");
}
```

- [ ] **Step 6: Run GREEN tests and commit**

Run:

```powershell
xmake build vision_analyzer_tests
xmake run vision_analyzer_tests
```

Expected: `algorithm tests passed`.

Commit:

```powershell
git add include/vision_analyzer/calibration.hpp src/calibration.cpp tests/test_algorithms.cpp
git commit -m "feat: discover measurable HID calibration probes"
```

### Task 3: Wire balanced discovery and dynamic levels into live calibration

**Files:**
- Modify: `tests/test_algorithms.cpp`
- Modify: `include/vision_analyzer/calibration.hpp`
- Modify: `src/calibration.cpp:155-365`

- [ ] **Step 1: Write failing final-retry tests**

Add and register:

```cpp
void test_calibration_sample_retry_does_not_rescale_low_confidence_shift() {
    require(select_calibration_sample_retry_count(
                480, 8.0, 0.05, 8.0, 1.5, 270.0
            ) == 480,
            "low-response magnitude must be repeated rather than trusted for rescaling");
}

void test_calibration_sample_retry_can_exceed_runtime_step() {
    require(select_calibration_sample_retry_count(
                120, 1.0, 0.90, 8.0, 1.5, 270.0
            ) == 960,
            "small reliable final sample should use the calibration-only range");
}
```

- [ ] **Step 2: Run RED tests**

Run `xmake build vision_analyzer_tests`.

Expected: compilation fails because `select_calibration_sample_retry_count` is missing.

- [ ] **Step 3: Implement the final-retry selector**

Declare and implement:

```cpp
[[nodiscard]] int select_calibration_sample_retry_count(
    int current_counts,
    double main_shift_px,
    double phase_response,
    double target_shift_px,
    double minimum_measurable_shift_px,
    double maximum_reliable_shift_px
);
```

Use this implementation:

```cpp
int select_calibration_sample_retry_count(
    int current_counts,
    double main_shift_px,
    double phase_response,
    double target_shift_px,
    double minimum_measurable_shift_px,
    double maximum_reliable_shift_px
) {
    if (current_counts < kCalibrationProbeMinimumCounts ||
        current_counts > kCalibrationProbeMaximumCounts ||
        !std::isfinite(main_shift_px) || main_shift_px < 0.0 ||
        !std::isfinite(phase_response) ||
        !std::isfinite(target_shift_px) || target_shift_px <= 0.0 ||
        !std::isfinite(minimum_measurable_shift_px) || minimum_measurable_shift_px <= 0.0 ||
        !std::isfinite(maximum_reliable_shift_px) ||
        maximum_reliable_shift_px <= minimum_measurable_shift_px) {
        throw std::invalid_argument("invalid HID calibration sample retry values");
    }
    if (phase_response < kCalibrationMinimumPhaseResponse &&
        main_shift_px >= minimum_measurable_shift_px &&
        main_shift_px <= maximum_reliable_shift_px) {
        return current_counts;
    }
    return adjust_calibration_probe_count(
        current_counts,
        main_shift_px,
        target_shift_px,
        kCalibrationProbeMaximumCounts
    );
}
```

- [ ] **Step 4: Replace open-coded probe movement with one balanced helper**

Inside `run_hid_calibration`, after `send_move`, add a local callback that always returns before
yielding an estimate:

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
        const VisualShiftEstimate estimate = estimate_visual_shift_with_response(
            baseline.image,
            moved.image
        );
        send_move(-dx, -dy);
        inverse_required = false;
        wait_for_settle();
        baseline = capture(
            "failed to capture returned DXGI frame after HID calibration probe"
        );
        return estimate;
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

This helper is the only path used by discovery and final movement samples. Do not call
`stop_all` as a substitute for inverse movement.

- [ ] **Step 5: Discover and log per-axis levels**

After stationary noise is known, add:

```cpp
const double minimum_discovery_shift = std::max(4.0, 3.0 * noise_px);
std::array<CalibrationAxisDiscovery, 2> discoveries;
for (std::size_t axis = 0; axis < discoveries.size(); ++axis) {
    int attempt = 0;
    discoveries[axis] = discover_calibration_axis(
        axis,
        minimum_discovery_shift,
        minimum_measurable_shift,
        maximum_reliable_shift,
        [&](int counts) {
            const VisualShiftEstimate estimate = measure_balanced_probe(axis, counts);
            const double main = std::abs(axis == 0 ? estimate.shift.x : estimate.shift.y);
            const double cross = std::abs(axis == 0 ? estimate.shift.y : estimate.shift.x);
            output << "probe_discovery axis=" << (axis == 0 ? 'x' : 'y')
                   << " attempt=" << attempt++
                   << " counts=" << counts
                   << " shift_px=" << main
                   << " cross_px=" << cross
                   << " response=" << estimate.response << '\n';
            return estimate;
        }
    );
    const auto& counts = discoveries[axis].levels.counts;
    output << "probe_levels axis=" << (axis == 0 ? 'x' : 'y')
           << " counts=" << counts[0] << ',' << counts[1] << ',' << counts[2]
           << " probe_max=" << kCalibrationProbeMaximumCounts << '\n';
}
```

Update the opening `calibration` line to report
`probe_max_counts=2048 runtime_max_step=120`; remove the fixed `levels=16,40,80` claim.

- [ ] **Step 6: Sample discovered levels and retain one corrective retry**

Replace the existing fixed-count movement loop with:

```cpp
for (std::size_t axis = 0; axis < 2; ++axis) {
    for (std::size_t level = 0; level < kHidCalibrationLevels; ++level) {
        for (int repeat = 0; repeat < options.calibration_repeats; ++repeat) {
            for (int sign : {1, -1}) {
                int absolute_counts = discoveries[axis].levels.counts[level];
                int command_counts = absolute_counts * sign;
                VisualShiftEstimate estimate = measure_balanced_probe(axis, command_counts);
                double main_magnitude = std::abs(
                    axis == 0 ? estimate.shift.x : estimate.shift.y
                );
                const bool finite_measurement =
                    std::isfinite(main_magnitude) && std::isfinite(estimate.response);
                const bool bad_range = finite_measurement &&
                    (main_magnitude < minimum_measurable_shift ||
                     main_magnitude > maximum_reliable_shift);
                const bool bad_response = finite_measurement &&
                    estimate.response < kCalibrationMinimumPhaseResponse;

                if (bad_range || bad_response) {
                    absolute_counts = select_calibration_sample_retry_count(
                        absolute_counts,
                        main_magnitude,
                        estimate.response,
                        discoveries[axis].levels.target_shift_px[level],
                        minimum_measurable_shift,
                        maximum_reliable_shift
                    );
                    command_counts = absolute_counts * sign;
                    output << "probe_retry"
                           << " axis=" << (axis == 0 ? 'x' : 'y')
                           << " level=" << level
                           << " repeat=" << repeat
                           << " counts=" << command_counts
                           << " observed_shift=" << main_magnitude
                           << " response=" << estimate.response
                           << " reason=" << (bad_range ? "range" : "response")
                           << '\n';
                    estimate = measure_balanced_probe(axis, command_counts);
                    main_magnitude = std::abs(
                        axis == 0 ? estimate.shift.x : estimate.shift.y
                    );
                }

                const int dx = axis == 0 ? command_counts : 0;
                const int dy = axis == 1 ? command_counts : 0;
                samples.push_back({
                    dx,
                    dy,
                    estimate.shift,
                    estimate.response,
                    static_cast<int>(level),
                });
                output << "sample"
                       << " type=move"
                       << " axis=" << (axis == 0 ? 'x' : 'y')
                       << " level=" << level
                       << " repeat=" << repeat
                       << " counts_dx=" << dx
                       << " counts_dy=" << dy
                       << " visual_shift_x=" << estimate.shift.x
                       << " visual_shift_y=" << estimate.shift.y
                       << " response=" << estimate.response
                       << '\n';
            }
        }
    }
}
```

Keep the existing `sample type=move` record. Extend `probe_retry` with
`response=<value>` and `reason=range|response`.

Call the fitter with the runtime-only limit:

```cpp
const HidCalibrationProfile profile = fit_adaptive_hid_calibration(
    samples,
    frame_size,
    kCalibrationRuntimeMaxStep
);
```

- [ ] **Step 7: Run all native tests and commit**

Run:

```powershell
xmake build vision_analyzer_tests
xmake build vision_runtime_c_api_tests
xmake run vision_analyzer_tests
xmake run vision_runtime_c_api_tests
```

Expected:

```text
algorithm tests passed
C API tests passed
```

Commit:

```powershell
git add include/vision_analyzer/calibration.hpp src/calibration.cpp tests/test_algorithms.cpp
git commit -m "fix: calibrate low-sensitivity HID accounts"
```

### Task 4: Document the new automatic behavior

**Files:**
- Modify: `README.md:260-320`
- Modify: `packaging/sm61/package/README_中文.md:65-110`

- [ ] **Step 1: Update developer documentation**

Add a calibration subsection stating:

```text
Calibration first discovers a measurable per-axis probe and may use up to 2048 counts.
That limit is startup-calibration-only. Accepted profiles keep max_step=120 for normal output.
Every probe is paired with its exact inverse; exhausted discovery rejects the profile.
```

- [ ] **Step 2: Update production instructions**

Explain in Chinese that low-sensitivity accounts may visibly swing farther during startup and that
the expected successful markers are:

```text
probe_levels axis=x counts=...
probe_levels axis=y counts=...
fit valid=1
标定完成 quality=...
DXGI 已打开
```

State explicitly that this update does not require another firmware flash and does not change the
ORT/TensorRT/CUDA environment.

- [ ] **Step 3: Check and commit documentation**

Run `git diff --check`, then:

```powershell
git add README.md packaging/sm61/package/README_中文.md
git commit -m "docs: explain low-sensitivity HID discovery"
```

### Task 5: Verify clean release builds and package safety tests

**Files:**
- Verify only; no expected source changes.

- [ ] **Step 1: Reconfigure the production build explicitly**

Run from the isolated worktree:

```powershell
xmake f -c -m release `
  --onnxruntime_root=D:\Tool\onnxruntime-win-x64-gpu-1.17.3 `
  --hid_sdk_root=D:\project\cs2-vision-trainer\tools\rp2350_hid_bridge_cpp
xmake
```

Expected: release builds of `vision_runtime.dll`, `vision_runtime.lib`, and
`vision_analyzer.exe` succeed.

- [ ] **Step 2: Run both release test binaries**

```powershell
xmake run vision_analyzer_tests
xmake run vision_runtime_c_api_tests
```

Expected: both pass with zero failures.

- [ ] **Step 3: Run packaging tests under Windows PowerShell 5.1**

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass `
  -File .\packaging\sm61\tests\run-tests.ps1 `
  -PythonProjectRoot D:\project\cs2-vision-trainer
```

Expected: all package, manifest, dependency, and safety tests pass.

- [ ] **Step 4: Audit the branch**

Run:

```powershell
git diff --check main...HEAD
git status --short
git log --oneline main..HEAD
```

Expected: no whitespace errors, clean status, and only the design/plan plus focused calibration
commits.

### Task 6: Build the SM61 package and a manifest-aware incremental overlay

**Files:**
- Generated: `D:/project/cs2-vision-trainer/dist/cs2-vision-runtime-sm61-low-sensitivity/`
- Generated: `D:/project/cs2-vision-trainer/dist/cs2-vision-runtime-sm61-low-sensitivity.zip`
- Generated: `D:/project/cs2-vision-trainer/dist/hotfix-low-sensitivity-stage/cs2-vision-runtime-sm61/`
- Generated: `C:/Users/xiaol/Downloads/cs2-vision-runtime-sm61-low-sensitivity-hotfix.zip`

- [ ] **Step 1: Build a separate full package**

Run `build-portable-package.ps1` with explicit immutable inputs:

```powershell
$wt = 'D:\project\cs2-vision-trainer\tools\cpp_analyzer\.worktrees\adaptive-low-sensitivity-calibration'
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass `
  -File "$wt\packaging\sm61\build-portable-package.ps1" `
  -ReleaseRoot "$wt\build\windows\x64\release" `
  -OrtRoot 'D:\Tool\onnxruntime-win-x64-gpu-1.17.3' `
  -ModelPath 'D:\project\cs2-vision-trainer\runs\detect\train-2\weights\best.onnx' `
  -SchemaPath 'D:\project\cs2-vision-trainer\runs\detect\train-2\weights\best.onnx.schema.json' `
  -SampleVideoPath 'D:\project\cs2-vision-trainer\videos\02.mp4' `
  -DependencyCache 'D:\project\cs2-vision-trainer\dist\.sm61-cache' `
  -OutputRoot 'D:\project\cs2-vision-trainer\dist\cs2-vision-runtime-sm61-low-sensitivity' `
  -OutputZip 'D:\project\cs2-vision-trainer\dist\cs2-vision-runtime-sm61-low-sensitivity.zip' `
  -TensorRtArchive 'C:\Users\xiaol\Downloads\TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8.zip' `
  -PythonProjectRoot 'D:\project\cs2-vision-trainer'
```

Require profile `sm61-ort1173-trt861-fp32`, component
`rp2350-hid-sdk=protocol-v2`, and a valid manifest.

- [ ] **Step 2: Generate the small overlay from manifest differences**

Run this PowerShell block. It compares normalized manifest paths and SHA256 values rather than
timestamps, and verifies every deletion target remains below `dist`:

```powershell
$dist = [IO.Path]::GetFullPath('D:\project\cs2-vision-trainer\dist').TrimEnd('\')
$oldRoot = Join-Path $dist 'cs2-vision-runtime-sm61-v2'
$newRoot = Join-Path $dist 'cs2-vision-runtime-sm61-low-sensitivity'
$stageParent = Join-Path $dist 'hotfix-low-sensitivity-stage'
$stageRoot = Join-Path $stageParent 'cs2-vision-runtime-sm61'
$zip = 'C:\Users\xiaol\Downloads\cs2-vision-runtime-sm61-low-sensitivity-hotfix.zip'

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
Copy-Item -LiteralPath (Join-Path $newRoot 'runtime-manifest.json') `
    -Destination (Join-Path $stageRoot 'runtime-manifest.json') -Force

$forbidden = @($changed | Where-Object {
    $_.path -match '^(model|runtime|cache)/' -or
    $_.path -match '^app/onnxruntime' -or
    $_.path -match '^app/(cublas|cudart|cudnn|nvinfer|nvonnxparser)'
})
if ($forbidden.Count -ne 0) {
    throw "Unexpected dependency/model files in overlay: $($forbidden.path -join ', ')"
}
if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
Compress-Archive -Path $stageRoot -DestinationPath $zip -Force
```

- [ ] **Step 3: Verify overlay correctness**

Continue with this verification copy and full manifest check:

```powershell
$verifyRoot = Join-Path $dist 'verify-sm61-low-sensitivity'
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
Import-Module `
  'D:\project\cs2-vision-trainer\tools\cpp_analyzer\.worktrees\adaptive-low-sensitivity-calibration\packaging\sm61\PackageTools.psm1' `
  -Force
$result = Test-PackageManifest -PackageRoot $verifyRoot
if (-not $result.Valid) { throw 'Low-sensitivity overlay manifest validation failed.' }
```

The `$forbidden` assertion ensures that the overlay contains no model, CUDA, cuDNN, TensorRT, ORT,
or cache files. For this ABI-preserving fix the expected native changes are
`app/vision_runtime.dll`, `app/vision_runtime.lib`, `app/vision_analyzer.exe`, documentation, and
`runtime-manifest.json`.

- [ ] **Step 4: Record delivery evidence**

Print and retain:

```powershell
Get-Item -LiteralPath 'C:\Users\xiaol\Downloads\cs2-vision-runtime-sm61-low-sensitivity-hotfix.zip' |
  Select-Object FullName,Length,LastWriteTime
Get-FileHash -Algorithm SHA256 -LiteralPath `
  'C:\Users\xiaol\Downloads\cs2-vision-runtime-sm61-low-sensitivity-hotfix.zip'
```

List every archive member and provide the operator with one instruction: close the Python process,
extract the folder over the existing `cs2-vision-runtime-sm61` directory, approve replacement, and
rerun the same `runtime_live_move.py` command. No firmware or environment reinstall is part of this
delivery.
