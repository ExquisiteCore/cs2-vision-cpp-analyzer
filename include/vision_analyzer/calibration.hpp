#pragma once

#include <iosfwd>

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

[[nodiscard]] PointerSettings query_windows_pointer_settings();
void print_pointer_settings(std::ostream& output, const PointerSettings& settings);
[[nodiscard]] cv::Point2d estimate_visual_shift(const cv::Mat& before, const cv::Mat& after);
void run_hid_calibration(const Options& options);

}  // namespace vision_analyzer
