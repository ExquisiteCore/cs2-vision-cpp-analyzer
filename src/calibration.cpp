#include "vision_analyzer/calibration.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <thread>
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

cv::Point2d estimate_visual_shift(const cv::Mat& before, const cv::Mat& after) {
    cv::Mat before_gray = comparable_gray(before);
    cv::Mat after_gray = comparable_gray(after);
    if (before_gray.size() != after_gray.size()) {
        throw std::runtime_error("calibration frames have different sizes");
    }
    return cv::phaseCorrelate(before_gray, after_gray);
}

void run_hid_calibration(const Options& options) {
    std::unique_ptr<std::ofstream> owned_stream;
    std::ostream& output = calibration_output_stream(options, owned_stream);
    print_pointer_settings(output, query_windows_pointer_settings());

    auto frame_source = create_frame_source(options);
    auto hid_client = create_rp2350_hid_client(options.hid_port);
    CapturedFrame baseline;
    if (!frame_source->read(baseline)) {
        throw std::runtime_error("failed to capture baseline DXGI frame for HID calibration");
    }

    const std::vector<std::pair<int, int>> moves = {
        {options.calibration_step_counts, 0},
        {-options.calibration_step_counts, 0},
        {0, options.calibration_step_counts},
        {0, -options.calibration_step_counts},
    };

    output << "calibration step_counts=" << options.calibration_step_counts
           << " repeats=" << options.calibration_repeats
           << " noise_samples=" << options.calibration_noise_samples
           << " settle_ms=" << options.calibration_settle_ms
           << " frame_width=" << baseline.image.cols
           << " frame_height=" << baseline.image.rows
           << '\n';

    std::vector<CalibrationSample> samples;
    samples.reserve(static_cast<std::size_t>(options.calibration_noise_samples + options.calibration_repeats * 4));

    for (int sample_index = 0; sample_index < options.calibration_noise_samples; ++sample_index) {
        std::this_thread::sleep_for(std::chrono::milliseconds(options.calibration_settle_ms));
        CapturedFrame still;
        if (!frame_source->read(still)) {
            throw std::runtime_error("failed to capture still DXGI frame for HID calibration");
        }
        const cv::Point2d shift = estimate_visual_shift(baseline.image, still.image);
        samples.push_back(CalibrationSample{0, 0, shift});
        output << "sample"
               << " type=noop"
               << " index=" << sample_index
               << " counts_dx=0 counts_dy=0"
               << " visual_shift_x=" << shift.x
               << " visual_shift_y=" << shift.y
               << '\n';
        baseline = std::move(still);
    }

    for (int repeat = 0; repeat < options.calibration_repeats; ++repeat) {
        for (const auto& [dx, dy] : moves) {
            hid_client->move_relative(static_cast<std::int16_t>(dx), static_cast<std::int16_t>(dy));
            std::this_thread::sleep_for(std::chrono::milliseconds(options.calibration_settle_ms));

            CapturedFrame moved;
            if (!frame_source->read(moved)) {
                throw std::runtime_error("failed to capture moved DXGI frame for HID calibration");
            }

            const cv::Point2d shift = estimate_visual_shift(baseline.image, moved.image);
            samples.push_back(CalibrationSample{dx, dy, shift});
            output << "sample"
                   << " type=move"
                   << " repeat=" << repeat
                   << " counts_dx=" << dx
                   << " counts_dy=" << dy
                   << " visual_shift_x=" << shift.x
                   << " visual_shift_y=" << shift.y
                   << '\n';
            baseline = std::move(moved);
        }
    }

    const CalibrationFit fit = fit_hid_calibration(samples, options.calibration_step_counts);
    if (!fit.valid) {
        throw std::runtime_error("HID calibration did not produce enough visual movement to fit hid_gain");
    }

    output << "fit"
           << " valid=1"
           << " hid_gain=" << fit.hid_gain
           << " gain_x=" << fit.gain_x
           << " gain_y=" << fit.gain_y
           << " hid_deadzone_px=" << fit.deadzone_px
           << " hid_max_step=" << fit.max_step
           << " movement_samples=" << fit.movement_samples
           << " noise_samples=" << fit.noise_samples
           << " noise_px=" << fit.noise_px
           << '\n';

    if (!options.calibration_config_output_path.empty()) {
        std::ofstream config_output(options.calibration_config_output_path);
        if (!config_output) {
            throw std::runtime_error("failed to open calibration config output: " + options.calibration_config_output_path);
        }
        write_hid_tuning_config(config_output, options, fit);
        output << "tuned_config=" << options.calibration_config_output_path << '\n';
    }

    hid_client->stop_all();
    frame_source->release();
}

}  // namespace vision_analyzer
