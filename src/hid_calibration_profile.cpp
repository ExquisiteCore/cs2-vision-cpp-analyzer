#include "vision_analyzer/hid_calibration_profile.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vision_analyzer {
namespace {

[[nodiscard]] float interpolated_gain(
    float magnitude,
    const HidCalibrationAxisCurve& curve
) {
    if (magnitude <= curve.shift_px[0]) {
        return curve.counts_per_pixel[0];
    }
    if (magnitude >= curve.shift_px[2]) {
        return curve.counts_per_pixel[2];
    }

    const std::size_t left = magnitude <= curve.shift_px[1] ? 0 : 1;
    const std::size_t right = left + 1;
    const float span = curve.shift_px[right] - curve.shift_px[left];
    const float alpha = (magnitude - curve.shift_px[left]) / span;
    return curve.counts_per_pixel[left] * (1.0F - alpha) +
           curve.counts_per_pixel[right] * alpha;
}

}  // namespace

bool valid_hid_calibration_curve(const HidCalibrationAxisCurve& curve) {
    for (std::size_t index = 0; index < kHidCalibrationLevels; ++index) {
        if (!std::isfinite(curve.shift_px[index]) || curve.shift_px[index] <= 0.0F ||
            !std::isfinite(curve.counts_per_pixel[index]) ||
            curve.counts_per_pixel[index] == 0.0F) {
            return false;
        }
        if (index > 0 && curve.shift_px[index] <= curve.shift_px[index - 1]) {
            return false;
        }
    }
    return true;
}

std::int16_t calibrated_hid_step(
    float error_px,
    const HidCalibrationAxisCurve& curve,
    int max_step,
    float deadzone_px
) {
    if (!valid_hid_calibration_curve(curve)) {
        throw std::runtime_error("invalid HID calibration curve");
    }
    if (!std::isfinite(error_px) || !std::isfinite(deadzone_px) || deadzone_px < 0.0F) {
        throw std::runtime_error("HID calibration step values must be finite");
    }
    if (max_step < 0) {
        throw std::runtime_error("HID calibration max step must be non-negative");
    }
    if (std::abs(error_px) < deadzone_px) {
        return 0;
    }

    const float gain = interpolated_gain(std::abs(error_px), curve);
    const long rounded = std::lround(error_px * gain);
    const long limited_max = std::min<long>(
        max_step,
        std::numeric_limits<std::int16_t>::max()
    );
    return static_cast<std::int16_t>(std::clamp(rounded, -limited_max, limited_max));
}

}  // namespace vision_analyzer
