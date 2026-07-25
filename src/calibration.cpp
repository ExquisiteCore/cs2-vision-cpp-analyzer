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

[[nodiscard]] cv::Mat full_gray_float(const cv::Mat& input) {
    if (input.empty()) {
        throw std::runtime_error("empty frame cannot be used for calibration");
    }
    cv::Mat gray;
    if (input.channels() == 1) {
        gray = input;
    } else {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    }
    cv::Mat as_float;
    gray.convertTo(as_float, CV_32F);
    return as_float;
}

[[nodiscard]] std::vector<int> tile_origins(int extent, int tile_extent) {
    if (extent <= 0 || tile_extent <= 0 || tile_extent > extent) {
        return {};
    }
    const int stride = std::max(1, tile_extent / 2);
    std::vector<int> origins;
    for (int origin = 0; origin + tile_extent <= extent; origin += stride) {
        origins.push_back(origin);
    }
    const int final_origin = extent - tile_extent;
    if (origins.empty() || origins.back() != final_origin) {
        origins.push_back(final_origin);
    }
    return origins;
}

[[nodiscard]] double gradient_energy(const cv::Mat& tile) {
    cv::Mat gradient_x;
    cv::Mat gradient_y;
    cv::Sobel(tile, gradient_x, CV_32F, 1, 0, 3);
    cv::Sobel(tile, gradient_y, CV_32F, 0, 1, 3);
    cv::Mat magnitude;
    cv::magnitude(gradient_x, gradient_y, magnitude);
    return cv::mean(magnitude)[0];
}

struct TileShiftCandidate {
    cv::Point2d shift;
    double response = 0.0;
};

[[nodiscard]] double candidate_tolerance(double left, double right) {
    return std::max(1.25, 0.18 * std::max({1.0, std::abs(left), std::abs(right)}));
}

[[nodiscard]] bool candidates_agree(
    const TileShiftCandidate& left,
    const TileShiftCandidate& right
) {
    return std::abs(left.shift.x - right.shift.x) <=
               candidate_tolerance(left.shift.x, right.shift.x) &&
           std::abs(left.shift.y - right.shift.y) <=
               candidate_tolerance(left.shift.y, right.shift.y);
}

[[nodiscard]] VisualShiftEstimate aggregate_tile_candidates(
    const std::vector<TileShiftCandidate>& candidates
) {
    std::vector<std::size_t> best_group;
    double best_response_sum = 0.0;
    for (std::size_t anchor = 0; anchor < candidates.size(); ++anchor) {
        std::vector<std::size_t> group;
        double response_sum = 0.0;
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            if (candidates_agree(candidates[anchor], candidates[index])) {
                group.push_back(index);
                response_sum += candidates[index].response;
            }
        }
        if (group.size() > best_group.size() ||
            (group.size() == best_group.size() && response_sum > best_response_sum)) {
            best_group = std::move(group);
            best_response_sum = response_sum;
        }
    }
    if (best_group.size() < 2) {
        return {{0.0, 0.0}, 0.0, false};
    }

    std::vector<double> shifts_x;
    std::vector<double> shifts_y;
    std::vector<double> responses;
    shifts_x.reserve(best_group.size());
    shifts_y.reserve(best_group.size());
    responses.reserve(best_group.size());
    for (const std::size_t index : best_group) {
        shifts_x.push_back(candidates[index].shift.x);
        shifts_y.push_back(candidates[index].shift.y);
        responses.push_back(candidates[index].response);
    }
    return {
        {median_or_zero(std::move(shifts_x)), median_or_zero(std::move(shifts_y))},
        median_or_zero(std::move(responses)),
        true,
    };
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

VisualShiftEstimate estimate_robust_visual_shift(
    const cv::Mat& before,
    const cv::Mat& after
) {
    const cv::Mat before_gray = full_gray_float(before);
    const cv::Mat after_gray = full_gray_float(after);
    if (before_gray.size() != after_gray.size()) {
        throw std::runtime_error("calibration frames have different sizes");
    }

    const int upper_height = std::max(1, static_cast<int>(before_gray.rows * 0.70));
    const int tile_width = std::min(
        before_gray.cols,
        std::max(64, std::min(384, before_gray.cols / 3))
    );
    const int tile_height = std::min(
        upper_height,
        std::max(64, std::min(256, upper_height / 2))
    );
    if (tile_width < 16 || tile_height < 16) {
        return {{0.0, 0.0}, 0.0, false};
    }

    const cv::Rect crosshair_exclusion(
        static_cast<int>(before_gray.cols * 0.44),
        static_cast<int>(before_gray.rows * 0.42),
        std::max(1, static_cast<int>(before_gray.cols * 0.12)),
        std::max(1, static_cast<int>(before_gray.rows * 0.16))
    );
    cv::Mat hanning;
    cv::createHanningWindow(hanning, {tile_width, tile_height}, CV_32F);

    std::vector<TileShiftCandidate> candidates;
    for (const int y : tile_origins(upper_height, tile_height)) {
        for (const int x : tile_origins(before_gray.cols, tile_width)) {
            const cv::Rect tile_rect(x, y, tile_width, tile_height);
            const cv::Rect excluded = tile_rect & crosshair_exclusion;
            if (excluded.area() > tile_rect.area() * 0.20) {
                continue;
            }
            const cv::Mat before_tile = before_gray(tile_rect);
            const cv::Mat after_tile = after_gray(tile_rect);
            if (gradient_energy(before_tile) < 5.0 ||
                gradient_energy(after_tile) < 5.0) {
                continue;
            }

            double response = 0.0;
            const cv::Point2d shift = cv::phaseCorrelate(
                before_tile,
                after_tile,
                hanning,
                &response
            );
            const double maximum_shift =
                static_cast<double>(std::min(tile_width, tile_height)) * 0.35;
            if (!std::isfinite(shift.x) || !std::isfinite(shift.y) ||
                !std::isfinite(response) || response < 0.10 ||
                std::abs(shift.x) > maximum_shift ||
                std::abs(shift.y) > maximum_shift) {
                continue;
            }
            candidates.push_back({shift, response});
        }
    }
    return aggregate_tile_candidates(candidates);
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

CalibrationRoundTripMeasurement estimate_robust_calibration_round_trip(
    const cv::Mat& baseline,
    const cv::Mat& moved,
    const cv::Mat& returned
) {
    return {
        estimate_robust_visual_shift(baseline, moved),
        estimate_robust_visual_shift(moved, returned),
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
    const CalibrationProbeMeasure& measure,
    const CalibrationDiscoveryPause& between_sweeps
) {
    if (axis > 1 || !measure) {
        throw std::invalid_argument("invalid HID calibration discovery axis or callback");
    }

    for (int sweep = 0; sweep < kCalibrationDiscoverySweeps; ++sweep) {
        int counts = 16;
        for (int attempt = 0; attempt < kCalibrationDiscoveryMaximumAttempts; ++attempt) {
            bool planning_estimate_available = false;
            VisualShiftEstimate planning_estimate{{0.0, 0.0}, 0.0, false};
            double planning_main_shift = 0.0;

            for (int repeat = 0;
                 repeat < kCalibrationProbeMeasurementsPerCount;
                 ++repeat) {
                const CalibrationRoundTripMeasurement measurement = measure(counts);
                const VisualShiftEstimate& estimate = measurement.outward;
                const double main_shift_px =
                    std::abs(axis == 0 ? estimate.shift.x : estimate.shift.y);
                const double cross_shift_px =
                    std::abs(axis == 0 ? estimate.shift.y : estimate.shift.x);
                const bool finite = std::isfinite(estimate.shift.x) &&
                                    std::isfinite(estimate.shift.y) &&
                                    std::isfinite(estimate.response);
                const bool reliable_direction = finite && estimate.coherent &&
                    estimate.response >= kCalibrationMinimumPhaseResponse &&
                    cross_shift_px <= main_shift_px * 0.35 &&
                    main_shift_px <= maximum_reliable_shift_px;
                if (reliable_direction &&
                    main_shift_px >= minimum_discovery_shift_px) {
                    const double counts_per_pixel =
                        static_cast<double>(counts) / main_shift_px;
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
                if (reliable_direction &&
                    (!planning_estimate_available ||
                     estimate.response > planning_estimate.response)) {
                    planning_estimate = estimate;
                    planning_main_shift = main_shift_px;
                    planning_estimate_available = true;
                }
            }

            if (attempt + 1 >= kCalibrationDiscoveryMaximumAttempts ||
                counts >= kCalibrationProbeMaximumCounts) {
                break;
            }

            int next_counts = std::min(
                counts * 2,
                kCalibrationProbeMaximumCounts
            );
            if (planning_estimate_available && planning_main_shift >= 0.5) {
                const int scaled = adjust_calibration_probe_count(
                    counts,
                    planning_main_shift,
                    8.0,
                    kCalibrationProbeMaximumCounts
                );
                next_counts = std::clamp(
                    std::max(counts + 1, scaled),
                    kCalibrationProbeMinimumCounts,
                    kCalibrationProbeMaximumCounts
                );
            }
            counts = next_counts;
        }
        if (sweep + 1 < kCalibrationDiscoverySweeps && between_sweeps) {
            between_sweeps();
        }
    }

    std::ostringstream message;
    message << "HID calibration input not ready: axis="
            << (axis == 0 ? 'x' : 'y')
            << " no coherent visual movement through "
            << kCalibrationProbeMaximumCounts
            << " counts";
    throw std::runtime_error(message.str());
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
                    estimate_robust_calibration_round_trip(
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
            const VisualShiftEstimate estimate = estimate_robust_visual_shift(
                baseline.image,
                still.image
            );
            samples.push_back({0, 0, estimate.shift, estimate.response, -1});
            if (estimate.coherent &&
                std::isfinite(estimate.shift.x) &&
                std::isfinite(estimate.shift.y)) {
                noise_values.push_back(cv::norm(estimate.shift));
            }
            output << "sample"
                   << " type=noop"
                   << " index=" << sample_index
                   << " counts_dx=0 counts_dy=0"
                   << " visual_shift_x=" << estimate.shift.x
                   << " visual_shift_y=" << estimate.shift.y
                   << " response=" << estimate.response
                   << " coherent=" << (estimate.coherent ? 1 : 0)
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
                           << " response=" << estimate.response
                           << " coherent=" << (estimate.coherent ? 1 : 0)
                           << '\n';
                    return measurement;
                },
                wait_for_settle
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
