#include <chrono>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "vision_analyzer/aim_controller.hpp"
#include "vision_analyzer/calibration.hpp"
#include "vision_analyzer/detector.hpp"
#include "vision_analyzer/frame_source.hpp"
#include "vision_analyzer/hid_output.hpp"
#include "vision_analyzer/model_schema.hpp"
#include "vision_analyzer/runtime.hpp"
#include "vision_analyzer/runtime_config.hpp"
#include "vision_analyzer/runtime_session.hpp"
#include "vision_analyzer/tracking.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace vision_analyzer {

[[nodiscard]] Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--config") {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for --config");
            }
            apply_runtime_config_file(options, argv[++i]);
        }
    }

    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        const auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + name);
            }
            return argv[++i];
        };

        if (key == "--config") {
            options.config_path = require_value(key);
        } else if (key == "--model") {
            options.model_path = require_value(key);
        } else if (key == "--schema") {
            options.model_schema_path = require_value(key);
        } else if (key == "--video") {
            options.video_path = require_value(key);
            options.input_source = InputSource::Video;
        } else if (key == "--input") {
            options.input_source = parse_input_source(require_value(key));
        } else if (key == "--dxgi-adapter") {
            options.dxgi_adapter = std::stoi(require_value(key));
        } else if (key == "--dxgi-output") {
            options.dxgi_output = std::stoi(require_value(key));
        } else if (key == "--dxgi-timeout") {
            options.dxgi_timeout_ms = std::stoi(require_value(key));
        } else if (key == "--dxgi-gpu-preference") {
            options.dxgi_gpu_preference = parse_dxgi_gpu_preference(require_value(key));
        } else if (key == "--dxgi-debug") {
            options.dxgi_debug = true;
        } else if (key == "--dxgi-roi") {
            options.dxgi_roi.x = std::stoi(require_value(key));
            options.dxgi_roi.y = std::stoi(require_value(key));
            options.dxgi_roi.width = std::stoi(require_value(key));
            options.dxgi_roi.height = std::stoi(require_value(key));
        } else if (key == "--list-dxgi-outputs") {
            options.list_dxgi_outputs = true;
            options.input_source = InputSource::Dxgi;
        } else if (key == "--probe-dxgi-outputs") {
            options.probe_dxgi_outputs = true;
            options.input_source = InputSource::Dxgi;
        } else if (key == "--verify-input") {
            options.verify_input = true;
        } else if (key == "--hid-port") {
            options.hid_port = require_value(key);
        } else if (key == "--hid-gain") {
            options.hid_move_gain = std::stof(require_value(key));
        } else if (key == "--hid-max-step") {
            options.hid_max_step = std::stoi(require_value(key));
        } else if (key == "--hid-deadzone") {
            options.hid_deadzone_px = std::stof(require_value(key));
        } else if (key == "--hid-click") {
            options.hid_click_enabled = true;
        } else if (key == "--hid-click-cooldown") {
            options.hid_click_cooldown_frames = std::stoi(require_value(key));
        } else if (key == "--test-hid-move") {
            options.test_hid_move = true;
            options.hid_test_dx = std::stoi(require_value(key));
            options.hid_test_dy = std::stoi(require_value(key));
        } else if (key == "--calibrate-hid") {
            options.calibrate_hid = true;
            options.input_source = InputSource::Dxgi;
        } else if (key == "--calibration-step") {
            options.calibration_step_counts = std::stoi(require_value(key));
        } else if (key == "--calibration-repeats") {
            options.calibration_repeats = std::stoi(require_value(key));
        } else if (key == "--calibration-noise-samples") {
            options.calibration_noise_samples = std::stoi(require_value(key));
        } else if (key == "--calibration-settle") {
            options.calibration_settle_ms = std::stoi(require_value(key));
        } else if (key == "--calibration-output") {
            options.calibration_output_path = require_value(key);
        } else if (key == "--calibration-config-output") {
            options.calibration_config_output_path = require_value(key);
        } else if (key == "--action-log") {
            options.action_log_path = require_value(key);
        } else if (key == "--player-side") {
            options.player_side = parse_player_side(require_value(key));
        } else if (key == "--backend") {
            options.backend = parse_backend(require_value(key));
        } else if (key == "--body-head-anchor-ratio") {
            options.tuning.body_head_anchor_ratio = std::stof(require_value(key));
        } else if (key == "--kalman-process-noise") {
            options.tuning.kalman_process_noise = std::stof(require_value(key));
        } else if (key == "--kalman-measurement-noise") {
            options.tuning.kalman_measurement_noise = std::stof(require_value(key));
        } else if (key == "--kalman-error-covariance") {
            options.tuning.kalman_error_covariance = std::stof(require_value(key));
        } else if (key == "--conf") {
            options.confidence = std::stof(require_value(key));
        } else if (key == "--nms") {
            options.nms_threshold = std::stof(require_value(key));
        } else if (key == "--max-frames") {
            options.max_frames = std::stoi(require_value(key));
        } else if (key == "--start-frame") {
            options.start_frame = std::stoi(require_value(key));
        } else if (key == "--warmup-frames") {
            options.warmup_frames = std::stoi(require_value(key));
        } else if (key == "--start-time") {
            options.start_time_seconds = std::stod(require_value(key));
        } else if (key == "--preview") {
            options.preview = true;
        } else if (key == "--dry-run") {
            options.dry_run = true;
        } else if (key == "--status-every") {
            options.status_every_frames = std::stoi(require_value(key));
        } else if (key == "--help" || key == "-h") {
            std::cout
                << "vision_analyzer --backend opencv-onnx --model best.onnx (--video input.mp4 | --input dxgi) (--hid-port COMx | --dry-run) [--preview]\n"
                << "  --config PATH    load key=value runtime config before CLI overrides\n"
                << "  --backend NAME    opencv-onnx, opencv-cuda, ort-cuda, ort-tensorrt, or tensorrt\n"
                << "  --schema PATH     validate exported model class schema JSON\n"
                << "  --input NAME      video or dxgi; --video also selects video input\n"
                << "  --video PATH      video file for offline tuning\n"
                << "  --list-dxgi-outputs  print DXGI adapters/outputs and exit\n"
                << "  --probe-dxgi-outputs  test Desktop Duplication support for every DXGI output and exit\n"
                << "  --verify-input    capture one frame, print dimensions/mean, and exit\n"
                << "  --dxgi-adapter N  DXGI adapter index for desktop duplication, default 0\n"
                << "  --dxgi-output N   DXGI output/monitor index for desktop duplication, default 0\n"
                << "  --dxgi-timeout N  DXGI frame wait timeout in ms, default 16\n"
                << "  --dxgi-gpu-preference NAME  default, minimum-power, or high-performance\n"
                << "  --dxgi-debug    print first captured DXGI texture format and byte statistics\n"
                << "  --dxgi-roi X Y W H  crop live DXGI input to this ROI before inference\n"
                << "  --hid-port COMx   send relative mouse moves through the RP2350 HID bridge SDK\n"
                << "  --hid-gain 1.0    multiply target offset before sending relative mouse movement\n"
                << "  --hid-max-step N  clamp each relative mouse move axis to +/-N, default 120\n"
                << "  --hid-deadzone PX suppress tiny per-axis movements below this offset, default 1.5\n"
                << "  --hid-click       send left click when fire_candidate is true\n"
                << "  --hid-click-cooldown N  minimum frame cooldown between SDK left clicks, default 6\n"
                << "  --test-hid-move DX DY  send one SDK relative mouse move and exit\n"
                << "  --calibrate-hid   run controlled HID moves, estimate DXGI visual shift, and exit\n"
                << "  --calibration-output PATH  write HID calibration samples to a text file\n"
                << "  --calibration-config-output PATH  write fitted HID tuning config, default hid-tuned.cfg\n"
                << "  --action-log PATH write per-frame planned movement/click commands to a text file\n"
                << "  --player-side SIDE  unknown, ct, or t; ct targets T classes, t targets CT classes\n"
                << "  --dry-run         run detection and planning without SDK output\n"
                << "  --status-every N  print one status line every N processed frames, default 30\n"
                << "  --conf 0.25       confidence threshold\n"
                << "  --nms 0.45        NMS threshold\n"
                << "  --start-time S    seek to S seconds before analysis\n"
                << "  --start-frame N   seek to frame N before analysis\n"
                << "  --warmup-frames N run N unlogged warmup frames before analysis, default 3\n"
                << "  --max-frames N    stop after N frames, 0 means full video\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    return options;
}

void validate_options(const Options& options) {
    if (options.dxgi_adapter < 0 || options.dxgi_output < 0) {
        throw std::runtime_error("--dxgi-adapter and --dxgi-output must be greater than or equal to 0");
    }
    if (options.dxgi_timeout_ms <= 0) {
        throw std::runtime_error("--dxgi-timeout must be greater than 0");
    }
    if (options.dxgi_roi.x < 0 || options.dxgi_roi.y < 0 ||
        options.dxgi_roi.width < 0 || options.dxgi_roi.height < 0) {
        throw std::runtime_error("--dxgi-roi values must be greater than or equal to 0");
    }
    if ((options.dxgi_roi.width == 0) != (options.dxgi_roi.height == 0)) {
        throw std::runtime_error("--dxgi-roi width and height must either both be set or both be 0");
    }
    if (!std::isfinite(options.hid_move_gain)) {
        throw std::runtime_error("--hid-gain must be finite");
    }
    if (options.hid_max_step < 0) {
        throw std::runtime_error("--hid-max-step must be greater than or equal to 0");
    }
    if (!std::isfinite(options.hid_deadzone_px) || options.hid_deadzone_px < 0.0F) {
        throw std::runtime_error("--hid-deadzone must be finite and greater than or equal to 0");
    }
    if (!std::isfinite(options.tuning.body_head_anchor_ratio) ||
        options.tuning.body_head_anchor_ratio <= 0.0F ||
        options.tuning.body_head_anchor_ratio >= 0.5F) {
        throw std::runtime_error("--body-head-anchor-ratio must be finite and between 0 and 0.5");
    }
    if (!std::isfinite(options.tuning.kalman_process_noise) || options.tuning.kalman_process_noise <= 0.0F ||
        !std::isfinite(options.tuning.kalman_measurement_noise) || options.tuning.kalman_measurement_noise <= 0.0F ||
        !std::isfinite(options.tuning.kalman_error_covariance) || options.tuning.kalman_error_covariance <= 0.0F) {
        throw std::runtime_error("Kalman tuning values must be finite and greater than 0");
    }
    if (options.list_dxgi_outputs || options.probe_dxgi_outputs || options.verify_input) {
        return;
    }
    if (options.test_hid_move) {
        if (options.hid_port.empty()) {
            throw std::runtime_error("--test-hid-move requires --hid-port COMx");
        }
        return;
    }
    if (options.calibrate_hid) {
        if (options.input_source != InputSource::Dxgi) {
            throw std::runtime_error("--calibrate-hid requires DXGI input");
        }
        if (options.hid_port.empty()) {
            throw std::runtime_error("--calibrate-hid requires --hid-port COMx");
        }
        if (options.calibration_step_counts <= 0 || options.calibration_repeats <= 0 ||
            options.calibration_noise_samples < 0 || options.calibration_settle_ms < 0) {
            throw std::runtime_error("calibration step/repeats must be greater than 0; noise samples and settle must be non-negative");
        }
        return;
    }
    if (options.hid_port.empty() && !options.dry_run) {
        throw std::runtime_error("use --hid-port COMx for live SDK output or --dry-run for tuning");
    }
    if (!options.dry_run && options.player_side == PlayerSide::Unknown) {
        throw std::runtime_error("live SDK output requires --player-side ct or --player-side t");
    }
    if (options.status_every_frames <= 0) {
        throw std::runtime_error("--status-every must be greater than 0");
    }
}

void verify_input(const Options& options) {
    auto frame_source = create_frame_source(options);
    CapturedFrame frame;
    if (!frame_source->read(frame)) {
        throw std::runtime_error("failed to read input frame");
    }
    const cv::Scalar mean = cv::mean(frame.image);
    std::cout << "input_verify"
              << " source=" << frame_source->name()
              << " width=" << frame.image.cols
              << " height=" << frame.image.rows
              << " mean_b=" << mean[0]
              << " mean_g=" << mean[1]
              << " mean_r=" << mean[2]
              << '\n';
    frame_source->release();
}

[[nodiscard]] cv::Scalar class_color(int class_id) {
    if (class_id == 0) {
        return {255, 190, 40};
    }
    if (class_id == 1) {
        return {255, 230, 120};
    }
    if (class_id == 2) {
        return {60, 220, 80};
    }
    return {120, 255, 160};
}

void draw_overlay(cv::Mat& frame, const std::vector<Detection>& detections, const FrameReport& report, const std::string& backend) {
    const cv::Point frame_center(frame.cols / 2, frame.rows / 2);
    cv::drawMarker(frame, frame_center, cv::Scalar(255, 255, 255), cv::MARKER_CROSS, 18, 1);

    for (const auto& detection : detections) {
        const auto color = class_color(detection.class_id);
        cv::rectangle(frame, detection.box, color, 2);
        std::ostringstream label;
        label << detection.label << " " << std::fixed << std::setprecision(2) << detection.confidence;
        cv::putText(frame, label.str(), {detection.box.x, std::max(18, detection.box.y - 6)}, cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 2);
    }

    if (report.target.has_value()) {
        const auto& target = *report.target;
        cv::circle(frame, target.center, 5, cv::Scalar(0, 0, 255), -1);
        cv::circle(frame, target.predicted_center, 4, cv::Scalar(0, 165, 255), -1);
        cv::circle(frame, target.analysis_point, 5, cv::Scalar(255, 0, 255), -1);
        cv::line(frame, frame_center, target.predicted_center, cv::Scalar(0, 0, 255), 1);

        std::ostringstream state;
        state << "track " << target.id << " " << lock_state_name(target.lock_state)
              << " stable " << std::fixed << std::setprecision(2) << target.stability;
        cv::putText(frame, state.str(), {12, 58}, cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 2);
    }

    std::ostringstream status;
    status << backend << " | FPS " << std::fixed << std::setprecision(1) << report.fps
           << " | latency " << std::setprecision(2) << report.timing.total_ms() << " ms"
           << " | infer " << std::setprecision(2) << report.timing.inference_ms << " ms"
           << " | det " << report.detection_count;
    cv::putText(frame, status.str(), {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 255, 255), 2);
}

void test_hid_move(const Options& options) {
    auto hid_client = create_rp2350_hid_client(options.hid_port);
#if defined(_WIN32)
    POINT before{};
    POINT after{};
    const BOOL before_ok = GetCursorPos(&before);
    hid_client->move_relative(
        static_cast<std::int16_t>(options.hid_test_dx),
        static_cast<std::int16_t>(options.hid_test_dy)
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    const BOOL after_ok = GetCursorPos(&after);
    hid_client->stop_all();
    std::cout << "hid_test_move"
              << " port=" << options.hid_port
              << " dx=" << options.hid_test_dx
              << " dy=" << options.hid_test_dy
              << " cursor_before=" << (before_ok ? 1 : 0) << ':' << before.x << ',' << before.y
              << " cursor_after=" << (after_ok ? 1 : 0) << ':' << after.x << ',' << after.y
              << " cursor_delta=" << (after.x - before.x) << ',' << (after.y - before.y)
              << '\n';
#else
    hid_client->move_relative(
        static_cast<std::int16_t>(options.hid_test_dx),
        static_cast<std::int16_t>(options.hid_test_dy)
    );
    hid_client->stop_all();
    std::cout << "hid_test_move"
              << " port=" << options.hid_port
              << " dx=" << options.hid_test_dx
              << " dy=" << options.hid_test_dy
              << '\n';
#endif
}

void run(const Options& options, const std::atomic_bool* stop_requested) {
    const auto should_stop = [stop_requested] {
        return stop_requested != nullptr && stop_requested->load();
    };

    apply_dxgi_gpu_preference(options.dxgi_gpu_preference);
    if (options.list_dxgi_outputs) {
        print_dxgi_outputs(std::cout);
        return;
    }
    if (options.probe_dxgi_outputs) {
        probe_dxgi_outputs(std::cout);
        return;
    }
    if (options.verify_input) {
        verify_input(options);
        return;
    }
    if (options.test_hid_move) {
        test_hid_move(options);
        return;
    }
    if (options.calibrate_hid) {
        run_hid_calibration(options);
        return;
    }

    RuntimeSession session;
    session.open(options);
    while (!should_stop()) {
        const RuntimeStepResult step = session.process_next();
        if (!step.frame_available) {
            break;
        }

        const int processed_index = session.processed_frames() - 1;
        if (processed_index % options.status_every_frames == 0) {
            std::cout << "frame=" << step.report.frame_index
                      << " det=" << step.report.detection_count
                      << " target=" << (step.command.has_target ? 1 : 0)
                      << " dx=" << step.command.dx
                      << " dy=" << step.command.dy
                      << " click=" << (step.command.click_left ? 1 : 0)
                      << " lock=" << lock_state_name(step.command.lock_state)
                      << '\n';
        }

        if (options.preview) {
            cv::Mat display = step.frame.clone();
            draw_overlay(display, step.detections, step.report, session.detector_name());
            cv::imshow("CS2 Vision C++ Analyzer", display);
            const int key = cv::waitKey(1) & 0xFF;
            if (key == 27 || key == 'q' || key == 'Q') {
                break;
            }
        }
    }

    if (should_stop()) {
        std::cout << "stopped=1\n";
    }
    std::cout << "processed_frames=" << session.processed_frames() << '\n';
    session.close();
}

}  // namespace vision_analyzer

#if !defined(VISION_ANALYZER_NO_CLI_MAIN)
int main(int argc, char** argv) {
    try {
        const auto options = vision_analyzer::parse_args(argc, argv);
        vision_analyzer::validate_options(options);
        vision_analyzer::run(options);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
#endif
