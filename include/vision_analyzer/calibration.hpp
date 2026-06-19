#pragma once

#include <iosfwd>
#include <vector>

#include <opencv2/core.hpp>

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
[[nodiscard]] cv::Point2d estimate_visual_shift(const cv::Mat& before, const cv::Mat& after);
[[nodiscard]] CalibrationFit fit_hid_calibration(
    const std::vector<CalibrationSample>& samples,
    int calibration_step_counts
);
void write_hid_tuning_config(std::ostream& output, const Options& options, const CalibrationFit& fit);
void run_hid_calibration(const Options& options);

}  // namespace vision_analyzer
