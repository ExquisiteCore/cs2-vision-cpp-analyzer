#pragma once

#include <array>
#include <functional>
#include <iosfwd>
#include <vector>

#include <opencv2/core.hpp>

#include "vision_analyzer/hid_calibration_profile.hpp"
#include "vision_analyzer/types.hpp"

namespace vision_analyzer {

struct PointerSettings {
    bool available = false;
    int threshold_1 = 0;
    int threshold_2 = 0;
    int acceleration = 0;
    int pointer_speed = 0;
};

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

constexpr int kCalibrationProbeMinimumCounts = 8;
constexpr int kCalibrationProbeMaximumCounts = 2048;
constexpr int kCalibrationDiscoveryMaximumAttempts = 8;
constexpr int kCalibrationRuntimeMaxStep = 120;
constexpr double kCalibrationMinimumPhaseResponse = 0.15;

struct CalibrationLevelPlan {
    std::array<int, kHidCalibrationLevels> counts{};
    std::array<double, kHidCalibrationLevels> target_shift_px{};
};

struct CalibrationFit {
    bool valid = false;
    double hid_gain = 1.0;
    double gain_x = 0.0;
    double gain_y = 0.0;
    double deadzone_px = 1.5;
    int max_step = 120;
    int movement_samples = 0;
    int noise_samples = 0;
    double noise_px = 0.0;
};

[[nodiscard]] PointerSettings query_windows_pointer_settings();
void print_pointer_settings(std::ostream& output, const PointerSettings& settings);
[[nodiscard]] VisualShiftEstimate estimate_visual_shift_with_response(
    const cv::Mat& before,
    const cv::Mat& after
);
[[nodiscard]] cv::Point2d estimate_visual_shift(const cv::Mat& before, const cv::Mat& after);
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
[[nodiscard]] CalibrationFit fit_hid_calibration(
    const std::vector<CalibrationSample>& samples,
    int calibration_step_counts
);
[[nodiscard]] HidCalibrationProfile fit_adaptive_hid_calibration(
    const std::vector<CalibrationSample>& samples,
    const cv::Size& frame_size,
    int max_step = 120
);
void write_hid_tuning_config(std::ostream& output, const Options& options, const CalibrationFit& fit);
[[nodiscard]] HidCalibrationProfile run_hid_calibration(const Options& options);

}  // namespace vision_analyzer
