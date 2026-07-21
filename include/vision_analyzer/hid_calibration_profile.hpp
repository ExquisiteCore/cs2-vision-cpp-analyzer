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
