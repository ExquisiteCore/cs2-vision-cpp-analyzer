#include "vision_analyzer/calibration.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "vision_analyzer/frame_source.hpp"
#include "vision_analyzer/hid_output.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace vision_analyzer {
namespace {

[[nodiscard]] cv::Mat comparable_gray(const cv::Mat& input) {
    if (input.empty()) {
        throw std::runtime_error("empty frame cannot be used for calibration");
    }
    cv::Mat gray;
    if (input.channels() == 1) {
        gray = input;
    } else {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    }

    const int width = std::min(gray.cols, 640);
    const int height = std::min(gray.rows, 640);
    const cv::Rect roi((gray.cols - width) / 2, (gray.rows - height) / 2, width, height);
    cv::Mat cropped = gray(roi);
    cv::Mat as_float;
    cropped.convertTo(as_float, CV_32F);
    return as_float;
}

[[nodiscard]] std::ostream& calibration_output_stream(
    const Options& options,
    std::unique_ptr<std::ofstream>& owned_stream
) {
    if (options.calibration_output_path.empty()) {
        return std::cout;
    }
    owned_stream = std::make_unique<std::ofstream>(options.calibration_output_path);
    if (!*owned_stream) {
        throw std::runtime_error("failed to open calibration output: " + options.calibration_output_path);
    }
    return *owned_stream;
}

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

}  // namespace

PointerSettings query_windows_pointer_settings() {
    PointerSettings settings;
#if defined(_WIN32)
    int mouse_values[3] = {0, 0, 0};
    int speed = 0;
    const BOOL mouse_ok = SystemParametersInfoA(SPI_GETMOUSE, 0, mouse_values, 0);
    const BOOL speed_ok = SystemParametersInfoA(SPI_GETMOUSESPEED, 0, &speed, 0);
    settings.available = mouse_ok != FALSE || speed_ok != FALSE;
    if (mouse_ok != FALSE) {
        settings.threshold_1 = mouse_values[0];
        settings.threshold_2 = mouse_values[1];
        settings.acceleration = mouse_values[2];
    }
    if (speed_ok != FALSE) {
        settings.pointer_speed = speed;
    }
#endif
    return settings;
}

void print_pointer_settings(std::ostream& output, const PointerSettings& settings) {
    if (!settings.available) {
        output << "windows_pointer_settings=unavailable\n";
        return;
    }
    output << "windows_pointer_settings"
           << " threshold1=" << settings.threshold_1
           << " threshold2=" << settings.threshold_2
           << " acceleration=" << settings.acceleration
           << " pointer_speed=" << settings.pointer_speed
           << '\n';
}

VisualShiftEstimate estimate_visual_shift_with_response(
    const cv::Mat& before,
    const cv::Mat& after
) {
    cv::Mat before_gray = comparable_gray(before);
    cv::Mat after_gray = comparable_gray(after);
    if (before_gray.size() != after_gray.size()) {
        throw std::runtime_error("calibration frames have different sizes");
    }
    double response = 0.0;
    const cv::Point2d shift = cv::phaseCorrelate(
        before_gray,
        after_gray,
        cv::noArray(),
        &response
    );
    return {shift, response};
}

cv::Point2d estimate_visual_shift(const cv::Mat& before, const cv::Mat& after) {
    return estimate_visual_shift_with_response(before, after).shift;
}

int adjust_calibration_probe_count(
    int current_counts,
    double observed_shift_px,
    double target_shift_px
) {
    if (current_counts <= 0 || !std::isfinite(observed_shift_px) ||
        observed_shift_px < 0.0 || !std::isfinite(target_shift_px) ||
        target_shift_px <= 0.0) {
        throw std::invalid_argument("invalid adaptive HID calibration probe values");
    }
    const double scaled = static_cast<double>(current_counts) * target_shift_px /
                          std::max(0.5, observed_shift_px);
    if (!std::isfinite(scaled)) {
        throw std::invalid_argument("adaptive HID calibration probe overflowed");
    }
    return static_cast<int>(std::clamp(std::lround(scaled), 8L, 120L));
}

HidCalibrationProfile run_hid_calibration(const Options& options) {
    apply_dxgi_gpu_preference(options.dxgi_gpu_preference);
    std::unique_ptr<std::ofstream> owned_stream;
    std::ostream& output = calibration_output_stream(options, owned_stream);
    print_pointer_settings(output, query_windows_pointer_settings());

    auto frame_source = create_frame_source(options);
    std::unique_ptr<HidClient> hid_client;
    try {
        hid_client = create_rp2350_hid_client(options.hid_port);
        const auto wait_for_settle = [&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(options.calibration_settle_ms));
        };
        const auto capture = [&](const char* error_message) {
            CapturedFrame frame;
            if (!frame_source->read(frame)) {
                throw std::runtime_error(error_message);
            }
            return frame;
        };
        const auto send_move = [&](int dx, int dy) {
            hid_client->move_relative(
                static_cast<std::int16_t>(dx),
                static_cast<std::int16_t>(dy)
            );
        };

        CapturedFrame baseline = capture(
            "failed to capture baseline DXGI frame for HID calibration"
        );
        const cv::Size frame_size = baseline.image.size();
        const std::array<int, kHidCalibrationLevels> initial_counts = {16, 40, 80};
        const std::array<double, kHidCalibrationLevels> target_shifts = {8.0, 32.0, 80.0};

        output << "calibration"
               << " levels=16,40,80"
               << " repeats=" << options.calibration_repeats
               << " noise_samples=" << options.calibration_noise_samples
               << " settle_ms=" << options.calibration_settle_ms
               << " frame_width=" << frame_size.width
               << " frame_height=" << frame_size.height
               << '\n';

        std::vector<CalibrationSample> samples;
        samples.reserve(static_cast<std::size_t>(
            options.calibration_noise_samples +
            options.calibration_repeats * 2 * 2 * static_cast<int>(kHidCalibrationLevels)
        ));
        std::vector<double> noise_values;

        for (int sample_index = 0; sample_index < options.calibration_noise_samples; ++sample_index) {
            wait_for_settle();
            CapturedFrame still = capture(
                "failed to capture still DXGI frame for HID calibration"
            );
            const VisualShiftEstimate estimate = estimate_visual_shift_with_response(
                baseline.image,
                still.image
            );
            samples.push_back({0, 0, estimate.shift, estimate.response, -1});
            if (std::isfinite(estimate.shift.x) && std::isfinite(estimate.shift.y)) {
                noise_values.push_back(cv::norm(estimate.shift));
            }
            output << "sample"
                   << " type=noop"
                   << " index=" << sample_index
                   << " counts_dx=0 counts_dy=0"
                   << " visual_shift_x=" << estimate.shift.x
                   << " visual_shift_y=" << estimate.shift.y
                   << " response=" << estimate.response
                   << '\n';
            baseline = std::move(still);
        }

        const double noise_px = median_or_zero(noise_values);
        const double minimum_measurable_shift = std::max(1.5, 3.0 * noise_px);
        const double maximum_reliable_shift =
            static_cast<double>(std::min(frame_size.width, frame_size.height)) * 0.25;

        for (std::size_t axis = 0; axis < 2; ++axis) {
            for (std::size_t level = 0; level < kHidCalibrationLevels; ++level) {
                for (int repeat = 0; repeat < options.calibration_repeats; ++repeat) {
                    for (int sign : {1, -1}) {
                        int command_counts = initial_counts[level] * sign;
                        int dx = axis == 0 ? command_counts : 0;
                        int dy = axis == 1 ? command_counts : 0;
                        send_move(dx, dy);
                        wait_for_settle();
                        CapturedFrame moved = capture(
                            "failed to capture moved DXGI frame for HID calibration"
                        );
                        VisualShiftEstimate estimate = estimate_visual_shift_with_response(
                            baseline.image,
                            moved.image
                        );
                        double main_magnitude = std::abs(
                            axis == 0 ? estimate.shift.x : estimate.shift.y
                        );

                        if (std::isfinite(main_magnitude) &&
                            (main_magnitude < minimum_measurable_shift ||
                             main_magnitude > maximum_reliable_shift)) {
                            send_move(-dx, -dy);
                            wait_for_settle();
                            baseline = capture(
                                "failed to capture returned DXGI frame before calibration retry"
                            );

                            command_counts = sign * adjust_calibration_probe_count(
                                std::abs(command_counts),
                                main_magnitude,
                                target_shifts[level]
                            );
                            dx = axis == 0 ? command_counts : 0;
                            dy = axis == 1 ? command_counts : 0;
                            output << "probe_retry"
                                   << " axis=" << (axis == 0 ? 'x' : 'y')
                                   << " level=" << level
                                   << " repeat=" << repeat
                                   << " counts=" << command_counts
                                   << " observed_shift=" << main_magnitude
                                   << '\n';

                            send_move(dx, dy);
                            wait_for_settle();
                            moved = capture(
                                "failed to capture retried DXGI frame for HID calibration"
                            );
                            estimate = estimate_visual_shift_with_response(
                                baseline.image,
                                moved.image
                            );
                            main_magnitude = std::abs(
                                axis == 0 ? estimate.shift.x : estimate.shift.y
                            );
                        }

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

                        send_move(-dx, -dy);
                        wait_for_settle();
                        baseline = capture(
                            "failed to capture returned DXGI frame after calibration probe"
                        );
                    }
                }
            }
        }

        const HidCalibrationProfile profile = fit_adaptive_hid_calibration(
            samples,
            frame_size,
            120
        );
        output << "fit"
               << " valid=" << (profile.valid ? 1 : 0)
               << " quality=" << profile.quality
               << " accepted_samples=" << profile.accepted_samples
               << " noise_px=" << profile.noise_px
               << " deadzone_px=" << profile.deadzone_px
               << " max_step=" << profile.max_step
               << '\n';
        for (std::size_t level = 0; level < kHidCalibrationLevels; ++level) {
            output << "curve axis=x level=" << level
                   << " shift_px=" << profile.x.shift_px[level]
                   << " counts_per_pixel=" << profile.x.counts_per_pixel[level]
                   << '\n';
            output << "curve axis=y level=" << level
                   << " shift_px=" << profile.y.shift_px[level]
                   << " counts_per_pixel=" << profile.y.counts_per_pixel[level]
                   << '\n';
        }
        if (!options.calibration_config_output_path.empty()) {
            output << "tuned_config_skipped=adaptive_profile_is_startup_memory_only"
                   << " requested_path=" << options.calibration_config_output_path << '\n';
        }
        if (!profile.valid) {
            std::ostringstream message;
            message << "HID calibration rejected: quality=" << profile.quality
                    << " noise_px=" << profile.noise_px
                    << " accepted_samples=" << profile.accepted_samples;
            throw std::runtime_error(message.str());
        }

        hid_client->stop_all();
        frame_source->release();
        return profile;
    } catch (...) {
        if (hid_client) {
            try {
                hid_client->stop_all();
            } catch (...) {
            }
        }
        try {
            frame_source->release();
        } catch (...) {
        }
        throw;
    }
}

}  // namespace vision_analyzer
