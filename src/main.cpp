#include <chrono>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "vision_analyzer/action_writer.hpp"
#include "vision_analyzer/csv_writer.hpp"
#include "vision_analyzer/detector.hpp"
#include "vision_analyzer/hid_output.hpp"
#include "vision_analyzer/tracking.hpp"

namespace vision_analyzer {
namespace {

[[nodiscard]] Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        const auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + name);
            }
            return argv[++i];
        };

        if (key == "--model") {
            options.model_path = require_value(key);
        } else if (key == "--video") {
            options.video_path = require_value(key);
        } else if (key == "--output") {
            options.output_path = require_value(key);
        } else if (key == "--actions-output") {
            options.actions_output_path = require_value(key);
        } else if (key == "--hid-port") {
            options.hid_port = require_value(key);
        } else if (key == "--hid-gain") {
            options.hid_move_gain = std::stof(require_value(key));
        } else if (key == "--hid-max-step") {
            options.hid_max_step = std::stoi(require_value(key));
        } else if (key == "--hid-click") {
            options.hid_click_enabled = true;
        } else if (key == "--hid-click-cooldown") {
            options.hid_click_cooldown_frames = std::stoi(require_value(key));
        } else if (key == "--backend") {
            options.backend = parse_backend(require_value(key));
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
        } else if (key == "--help" || key == "-h") {
            std::cout
                << "vision_analyzer --backend opencv-onnx --model best.onnx --video input.mp4 --output analysis.csv [--preview]\n"
                << "  --backend NAME    opencv-onnx, opencv-cuda, ort-cuda, ort-tensorrt, or tensorrt\n"
                << "  --actions-output PATH write human-readable offline action suggestions to a txt file\n"
                << "  --hid-port COMx   send relative mouse moves through the RP2350 HID bridge SDK\n"
                << "  --hid-gain 1.0    multiply target offset before sending relative mouse movement\n"
                << "  --hid-max-step N  clamp each relative mouse move axis to +/-N, default 120\n"
                << "  --hid-click       send left click when fire_candidate is true\n"
                << "  --hid-click-cooldown N  minimum frame cooldown between SDK left clicks, default 6\n"
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

void run(const Options& options) {
    cv::VideoCapture capture(options.video_path);
    if (!capture.isOpened()) {
        throw std::runtime_error("failed to open video: " + options.video_path);
    }
    if (options.start_time_seconds > 0.0) {
        capture.set(cv::CAP_PROP_POS_MSEC, options.start_time_seconds * 1000.0);
    } else if (options.start_frame > 0) {
        capture.set(cv::CAP_PROP_POS_FRAMES, options.start_frame);
    }

    auto detector = create_detector(options.backend, options.model_path);
    if (options.warmup_frames > 0) {
        cv::Mat warmup_frame(kInputSize, kInputSize, CV_8UC3, cv::Scalar(0, 0, 0));
        for (int i = 0; i < options.warmup_frames; ++i) {
            (void)detector->detect(warmup_frame, options.confidence, options.nms_threshold);
        }
        std::cout << "warmup_frames=" << options.warmup_frames << '\n';
    }

    CsvWriter writer(options.output_path);
    writer.write_header();
    std::unique_ptr<ActionWriter> action_writer;
    if (!options.actions_output_path.empty()) {
        action_writer = std::make_unique<ActionWriter>(options.actions_output_path);
    }
    std::unique_ptr<HidClient> hid_client;
    std::unique_ptr<HidActionSender> hid_sender;
    if (!options.hid_port.empty()) {
        hid_client = create_rp2350_hid_client(options.hid_port);
        hid_sender = std::make_unique<HidActionSender>(
            *hid_client,
            HidActionOptions{
                options.hid_move_gain,
                options.hid_max_step,
                options.hid_click_enabled,
                options.hid_click_cooldown_frames,
            }
        );
        std::cout << "hid_port=" << options.hid_port
                  << " hid_gain=" << options.hid_move_gain
                  << " hid_max_step=" << options.hid_max_step
                  << " hid_click=" << (options.hid_click_enabled ? 1 : 0) << '\n';
    }

    TrackManager track_manager;
    TargetSelector selector;
    AnalysisState analysis_state;
    cv::Mat frame;
    int processed_index = 0;
    auto last_time = std::chrono::steady_clock::now();
    double fps = 0.0;

    std::cout << "backend=" << detector->name() << " model=" << options.model_path << '\n';

    while (capture.read(frame)) {
        if (options.max_frames > 0 && processed_index >= options.max_frames) {
            break;
        }

        const auto detection_result = detector->detect(frame, options.confidence, options.nms_threshold);
        const auto tracks = track_manager.update(detection_result.detections, frame.size());
        const auto selected = selector.select(tracks, frame.size(), analysis_state.active_track_id());

        const auto now = std::chrono::steady_clock::now();
        const double frame_delta = std::chrono::duration<double>(now - last_time).count();
        if (frame_delta > 0.0) {
            const double instant_fps = 1.0 / frame_delta;
            fps = fps == 0.0 ? instant_fps : fps * 0.90 + instant_fps * 0.10;
        }
        last_time = now;

        FrameReport report{
            static_cast<int>(capture.get(cv::CAP_PROP_POS_FRAMES)) - 1,
            capture.get(cv::CAP_PROP_POS_MSEC),
            fps,
            detection_result.timing,
            static_cast<int>(detection_result.detections.size()),
            std::nullopt,
        };
        if (selected.has_value()) {
            report.target = analysis_state.update(*selected, frame.size(), report.timestamp_ms, report.timing.total_ms());
        } else {
            analysis_state.mark_no_target();
        }
        writer.write_report(report);
        if (action_writer) {
            action_writer->write_report(report);
        }
        if (hid_sender) {
            hid_sender->handle_report(report);
        }

        if (options.preview) {
            cv::Mat display = frame.clone();
            draw_overlay(display, detection_result.detections, report, detector->name());
            cv::imshow("CS2 Vision C++ Analyzer", display);
            const int key = cv::waitKey(1) & 0xFF;
            if (key == 27 || key == 'q' || key == 'Q') {
                break;
            }
        }

        ++processed_index;
    }

    std::cout << "processed_frames=" << processed_index << " output=" << options.output_path << '\n';
    if (hid_sender) {
        hid_sender->stop_all();
    }
}

}  // namespace
}  // namespace vision_analyzer

int main(int argc, char** argv) {
    try {
        vision_analyzer::run(vision_analyzer::parse_args(argc, argv));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
