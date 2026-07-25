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

CalibrationRoundTripMeasurement estimate_calibration_round_trip(
    const cv::Mat& baseline,
    const cv::Mat& moved,
    const cv::Mat& returned
) {
    return {
        estimate_visual_shift_with_response(baseline, moved),
        estimate_visual_shift_with_response(moved, returned),
    };
}

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

    const double high = std::min(48.0, static_cast<double>(maximum_counts) / counts_per_pixel);
    const double low = std::min(8.0, high / 4.0);
    const double middle = std::min(24.0, high / 2.0);
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

std::vector<CalibrationRoundTripCommand> plan_calibration_round_trip_commands(
    const std::array<CalibrationAxisDiscovery, 2>& discoveries,
    int repeats
) {
    if (repeats < 2) {
        throw std::invalid_argument("HID calibration round-trip repeats must be at least two");
    }
    std::vector<CalibrationRoundTripCommand> commands;
    commands.reserve(
        discoveries.size() * kHidCalibrationLevels * static_cast<std::size_t>(repeats)
    );
    for (std::size_t axis = 0; axis < discoveries.size(); ++axis) {
        for (std::size_t level = 0; level < kHidCalibrationLevels; ++level) {
            const int counts = discoveries[axis].levels.counts[level];
            if (counts < kCalibrationProbeMinimumCounts ||
                counts > kCalibrationProbeMaximumCounts) {
                throw std::invalid_argument(
                    "HID calibration round-trip count is outside the discovered range"
                );
            }
            for (int repeat = 0; repeat < repeats; ++repeat) {
                commands.push_back({
                    axis,
                    static_cast<int>(level),
                    repeat,
                    repeat % 2 == 0 ? counts : -counts,
                });
            }
        }
    }
    return commands;
}

std::array<CalibrationSample, 2> make_calibration_round_trip_samples(
    std::size_t axis,
    int level,
    int outward_counts,
    const CalibrationRoundTripMeasurement& measurement
) {
    if (axis > 1 || level < 0 || level >= static_cast<int>(kHidCalibrationLevels) ||
        outward_counts == 0 ||
        outward_counts < -kCalibrationProbeMaximumCounts ||
        outward_counts > kCalibrationProbeMaximumCounts) {
        throw std::invalid_argument("invalid HID calibration round-trip sample values");
    }
    const auto make_sample = [&](int signed_counts, const VisualShiftEstimate& estimate) {
        return CalibrationSample{
            axis == 0 ? signed_counts : 0,
            axis == 1 ? signed_counts : 0,
            estimate.shift,
            estimate.response,
            level,
        };
    };
    return {
        make_sample(outward_counts, measurement.outward),
        make_sample(-outward_counts, measurement.inverse),
    };
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
                send_move(-dx, -dy);
                inverse_required = false;
                wait_for_settle();
                CapturedFrame returned = capture(
                    "failed to capture returned DXGI frame after HID calibration probe"
                );
                const CalibrationRoundTripMeasurement measurement =
                    estimate_calibration_round_trip(
                        baseline.image,
                        moved.image,
                        returned.image
                    );
                baseline = std::move(returned);
                return measurement;
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

        output << "calibration"
               << " probe_max_counts=" << kCalibrationProbeMaximumCounts
               << " runtime_max_step=" << kCalibrationRuntimeMaxStep
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
                    const CalibrationRoundTripMeasurement measurement =
                        measure_balanced_probe(axis, counts);
                    const VisualShiftEstimate& estimate = measurement.outward;
                    const double main = std::abs(
                        axis == 0 ? estimate.shift.x : estimate.shift.y
                    );
                    const double cross = std::abs(
                        axis == 0 ? estimate.shift.y : estimate.shift.x
                    );
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

        const auto round_trip_commands = plan_calibration_round_trip_commands(
            discoveries,
            options.calibration_repeats
        );
        for (const auto& command : round_trip_commands) {
            const CalibrationRoundTripMeasurement measurement =
                measure_balanced_probe(command.axis, command.outward_counts);
            const auto round_trip_samples = make_calibration_round_trip_samples(
                command.axis,
                command.level,
                command.outward_counts,
                measurement
            );
            for (std::size_t leg = 0; leg < round_trip_samples.size(); ++leg) {
                const CalibrationSample& sample = round_trip_samples[leg];
                samples.push_back(sample);
                output << "sample"
                       << " type=move"
                       << " leg=" << (leg == 0 ? "outward" : "return")
                       << " axis=" << (command.axis == 0 ? 'x' : 'y')
                       << " level=" << command.level
                       << " repeat=" << command.repeat
                       << " counts_dx=" << sample.counts_dx
                       << " counts_dy=" << sample.counts_dy
                       << " visual_shift_x=" << sample.visual_shift.x
                       << " visual_shift_y=" << sample.visual_shift.y
                       << " response=" << sample.phase_response
                       << '\n';
            }
        }

        const HidCalibrationProfile profile = fit_adaptive_hid_calibration(
            samples,
            frame_size,
            kCalibrationRuntimeMaxStep
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
