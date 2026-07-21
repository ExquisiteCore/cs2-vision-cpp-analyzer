#include "vision_analyzer/calibration.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vision_analyzer {
namespace {

[[nodiscard]] double median_or_zero(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 1) {
        return values[middle];
    }
    return (values[middle - 1] + values[middle]) * 0.5;
}

struct DirectionSamples {
    std::vector<double> gains;
    std::vector<double> shifts;
};

using LevelSamples = std::array<DirectionSamples, 2>;
using AxisSamples = std::array<LevelSamples, kHidCalibrationLevels>;

[[nodiscard]] bool finite_point(const cv::Point2d& point) {
    return std::isfinite(point.x) && std::isfinite(point.y);
}

void assign_sorted_curve(
    HidCalibrationAxisCurve& curve,
    std::array<std::pair<float, float>, kHidCalibrationLevels> knots
) {
    std::sort(knots.begin(), knots.end(), [](const auto& left, const auto& right) {
        return left.first < right.first;
    });
    for (std::size_t index = 0; index < knots.size(); ++index) {
        curve.shift_px[index] = knots[index].first;
        curve.counts_per_pixel[index] = knots[index].second;
    }
}

}  // namespace

CalibrationFit fit_hid_calibration(
    const std::vector<CalibrationSample>& samples,
    int calibration_step_counts
) {
    std::vector<double> x_gains;
    std::vector<double> y_gains;
    std::vector<double> noise_values;
    int movement_samples = 0;
    int noise_samples = 0;

    for (const auto& sample : samples) {
        if (sample.counts_dx == 0 && sample.counts_dy == 0) {
            noise_values.push_back(cv::norm(sample.visual_shift));
            ++noise_samples;
            continue;
        }

        if (sample.counts_dx != 0 && std::abs(sample.visual_shift.x) >= 0.75) {
            x_gains.push_back(std::abs(static_cast<double>(sample.counts_dx) / sample.visual_shift.x));
            ++movement_samples;
        }
        if (sample.counts_dy != 0 && std::abs(sample.visual_shift.y) >= 0.75) {
            y_gains.push_back(std::abs(static_cast<double>(sample.counts_dy) / sample.visual_shift.y));
            ++movement_samples;
        }
    }

    CalibrationFit fit;
    fit.gain_x = median_or_zero(x_gains);
    fit.gain_y = median_or_zero(y_gains);
    fit.noise_px = median_or_zero(noise_values);
    fit.noise_samples = noise_samples;
    fit.movement_samples = movement_samples;

    std::vector<double> axis_gains;
    if (fit.gain_x > 0.0 && std::isfinite(fit.gain_x)) {
        axis_gains.push_back(fit.gain_x);
    }
    if (fit.gain_y > 0.0 && std::isfinite(fit.gain_y)) {
        axis_gains.push_back(fit.gain_y);
    }
    fit.hid_gain = median_or_zero(axis_gains);
    fit.valid = fit.hid_gain > 0.0 && std::isfinite(fit.hid_gain);
    fit.deadzone_px = std::clamp(std::ceil(std::max(0.5, fit.noise_px * 3.0)), 1.0, 8.0);
    fit.max_step = std::clamp(calibration_step_counts * 3, 20, 240);
    return fit;
}

HidCalibrationProfile fit_adaptive_hid_calibration(
    const std::vector<CalibrationSample>& samples,
    const cv::Size& frame_size,
    int max_step
) {
    if (frame_size.width <= 0 || frame_size.height <= 0) {
        throw std::invalid_argument("HID calibration frame size must be positive");
    }
    if (max_step <= 0 || max_step > std::numeric_limits<std::int16_t>::max()) {
        throw std::invalid_argument("HID calibration max step is out of range");
    }

    HidCalibrationProfile profile;
    profile.frame_width = frame_size.width;
    profile.frame_height = frame_size.height;
    profile.max_step = max_step;

    std::vector<double> noise_values;
    for (const auto& sample : samples) {
        if (sample.counts_dx == 0 && sample.counts_dy == 0 && finite_point(sample.visual_shift)) {
            noise_values.push_back(cv::norm(sample.visual_shift));
        }
    }
    profile.noise_px = static_cast<float>(median_or_zero(noise_values));
    profile.deadzone_px = static_cast<float>(std::clamp(
        std::ceil(3.0 * static_cast<double>(profile.noise_px)),
        1.0,
        8.0
    ));

    std::array<AxisSamples, 2> grouped;
    const double minimum_shift = std::max(1.5, 3.0 * static_cast<double>(profile.noise_px));
    double response_sum = 0.0;
    double minimum_accepted_shift = std::numeric_limits<double>::infinity();

    for (const auto& sample : samples) {
        if (sample.counts_dx == 0 && sample.counts_dy == 0) {
            continue;
        }
        if (sample.level < 0 || sample.level >= static_cast<int>(kHidCalibrationLevels)) {
            throw std::invalid_argument("HID calibration sample level is out of range");
        }
        if ((sample.counts_dx != 0 && sample.counts_dy != 0) ||
            !finite_point(sample.visual_shift) || !std::isfinite(sample.phase_response) ||
            sample.phase_response < 0.15) {
            continue;
        }

        const std::size_t axis = sample.counts_dx != 0 ? 0U : 1U;
        const int counts = axis == 0 ? sample.counts_dx : sample.counts_dy;
        const double main_shift = axis == 0 ? sample.visual_shift.x : sample.visual_shift.y;
        const double cross_shift = axis == 0 ? sample.visual_shift.y : sample.visual_shift.x;
        const double main_magnitude = std::abs(main_shift);
        if (main_magnitude < minimum_shift ||
            std::abs(cross_shift) > main_magnitude * 0.35) {
            continue;
        }

        const double gain = -static_cast<double>(counts) / main_shift;
        if (!std::isfinite(gain) || gain == 0.0) {
            continue;
        }

        const std::size_t direction = counts > 0 ? 1U : 0U;
        auto& bucket = grouped[axis][static_cast<std::size_t>(sample.level)][direction];
        bucket.gains.push_back(gain);
        bucket.shifts.push_back(main_magnitude);
        response_sum += sample.phase_response;
        minimum_accepted_shift = std::min(minimum_accepted_shift, main_magnitude);
        ++profile.accepted_samples;
    }

    if (profile.accepted_samples == 0) {
        return profile;
    }

    std::array<std::array<std::pair<float, float>, kHidCalibrationLevels>, 2> knots;
    double consistency_sum = 0.0;
    for (std::size_t axis = 0; axis < grouped.size(); ++axis) {
        for (std::size_t level = 0; level < kHidCalibrationLevels; ++level) {
            const auto& negative = grouped[axis][level][0];
            const auto& positive = grouped[axis][level][1];
            if (negative.gains.empty() || positive.gains.empty()) {
                return profile;
            }

            const double negative_gain = median_or_zero(negative.gains);
            const double positive_gain = median_or_zero(positive.gains);
            if (!std::isfinite(negative_gain) || !std::isfinite(positive_gain) ||
                negative_gain == 0.0 || positive_gain == 0.0 ||
                std::signbit(negative_gain) != std::signbit(positive_gain)) {
                return profile;
            }

            const double larger_gain = std::max(std::abs(negative_gain), std::abs(positive_gain));
            const double difference_ratio =
                std::abs(std::abs(negative_gain) - std::abs(positive_gain)) / larger_gain;
            if (difference_ratio > 0.40) {
                return profile;
            }
            consistency_sum += 1.0 - difference_ratio;

            std::vector<double> combined_gains = negative.gains;
            combined_gains.insert(combined_gains.end(), positive.gains.begin(), positive.gains.end());
            std::vector<double> combined_shifts = negative.shifts;
            combined_shifts.insert(combined_shifts.end(), positive.shifts.begin(), positive.shifts.end());
            knots[axis][level] = {
                static_cast<float>(median_or_zero(combined_shifts)),
                static_cast<float>(median_or_zero(combined_gains)),
            };
        }
    }

    assign_sorted_curve(profile.x, knots[0]);
    assign_sorted_curve(profile.y, knots[1]);

    const double mean_response = std::clamp(
        response_sum / static_cast<double>(profile.accepted_samples),
        0.0,
        1.0
    );
    const double mean_consistency = consistency_sum /
        static_cast<double>(2 * kHidCalibrationLevels);
    const double signal_to_noise = std::clamp(
        minimum_accepted_shift /
            (3.0 * std::max(static_cast<double>(profile.noise_px), 0.25)),
        0.0,
        1.0
    );
    profile.quality = static_cast<float>(mean_response * mean_consistency * signal_to_noise);
    profile.valid = profile.quality >= 0.55F &&
                    valid_hid_calibration_curve(profile.x) &&
                    valid_hid_calibration_curve(profile.y);
    return profile;
}

void write_hid_tuning_config(std::ostream& output, const Options& options, const CalibrationFit& fit) {
    output << "# Generated by vision_analyzer --calibrate-hid\n"
           << "input=" << input_source_name(options.input_source) << '\n'
           << "dxgi_adapter=" << options.dxgi_adapter << '\n'
           << "dxgi_output=" << options.dxgi_output << '\n'
           << "dxgi_timeout_ms=" << options.dxgi_timeout_ms << '\n'
           << "dxgi_gpu_preference=" << dxgi_gpu_preference_name(options.dxgi_gpu_preference) << '\n'
           << "hid_gain=" << fit.hid_gain << '\n'
           << "hid_max_step=" << fit.max_step << '\n'
           << "hid_deadzone_px=" << fit.deadzone_px << '\n'
           << "body_head_anchor_ratio=" << options.tuning.body_head_anchor_ratio << '\n'
           << "kalman_process_noise=" << options.tuning.kalman_process_noise << '\n'
           << "kalman_measurement_noise=" << options.tuning.kalman_measurement_noise << '\n'
           << "kalman_error_covariance=" << options.tuning.kalman_error_covariance << '\n';
}

}  // namespace vision_analyzer
