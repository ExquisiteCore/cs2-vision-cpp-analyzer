#include "vision_analyzer/runtime_session.hpp"

#include <iostream>
#include <stdexcept>

#include <opencv2/core.hpp>

#include "vision_analyzer/model_schema.hpp"

namespace vision_analyzer {

RuntimeSession::~RuntimeSession() {
    close();
}

void RuntimeSession::open(const Options& options) {
    close();

    options_ = options;
    apply_dxgi_gpu_preference(options_.dxgi_gpu_preference);

    frame_source_ = create_frame_source(options_);
    validate_configured_model_schema(options_, !options_.dry_run);
    detector_ = create_detector(options_.backend, options_.model_path);

    if (options_.warmup_frames > 0) {
        cv::Mat warmup_frame(kInputSize, kInputSize, CV_8UC3, cv::Scalar(0, 0, 0));
        for (int i = 0; i < options_.warmup_frames; ++i) {
            (void)detector_->detect(warmup_frame, options_.confidence, options_.nms_threshold);
        }
        std::cout << "warmup_frames=" << options_.warmup_frames << '\n';
    }

    aim_controller_ = std::make_unique<AimController>(AimControllerOptions{
        options_.hid_move_gain,
        options_.hid_max_step,
        options_.hid_deadzone_px,
        options_.hid_click_enabled,
        options_.hid_click_cooldown_frames,
    });

    if (!options_.action_log_path.empty()) {
        action_log_.open(options_.action_log_path);
        if (!action_log_) {
            throw std::runtime_error("failed to open action log: " + options_.action_log_path);
        }
        action_log_ << "frame timestamp_ms target dx dy click lock distance offset_x offset_y\n";
    }

    if (!options_.dry_run) {
        hid_client_ = create_rp2350_hid_client(options_.hid_port);
        hid_sender_ = std::make_unique<HidActionSender>(*hid_client_);
        std::cout << "hid_port=" << options_.hid_port
                  << " hid_gain=" << options_.hid_move_gain
                  << " hid_max_step=" << options_.hid_max_step
                  << " hid_deadzone=" << options_.hid_deadzone_px
                  << " hid_click=" << (options_.hid_click_enabled ? 1 : 0)
                  << " player_side=" << player_side_name(options_.player_side) << '\n';
    } else {
        std::cout << "dry_run=1"
                  << " hid_gain=" << options_.hid_move_gain
                  << " hid_max_step=" << options_.hid_max_step
                  << " hid_deadzone=" << options_.hid_deadzone_px
                  << " hid_click=" << (options_.hid_click_enabled ? 1 : 0)
                  << " player_side=" << player_side_name(options_.player_side) << '\n';
    }

    track_manager_ = std::make_unique<TrackManager>();
    selector_ = std::make_unique<TargetSelector>(options_.tuning);
    analysis_state_ = std::make_unique<AnalysisState>(options_.tuning);
    processed_index_ = 0;
    fps_ = 0.0;
    last_time_ = std::chrono::steady_clock::now();
    open_ = true;

    std::cout << "backend=" << detector_->name()
              << " input=" << frame_source_->name()
              << " model=" << options_.model_path << '\n';
}

RuntimeStepResult RuntimeSession::process_next() {
    if (!open_ || !frame_source_ || !detector_) {
        throw std::runtime_error("runtime session is not open");
    }
    if (options_.max_frames > 0 && processed_index_ >= options_.max_frames) {
        return {};
    }

    CapturedFrame captured_frame;
    if (!frame_source_->read(captured_frame)) {
        return {};
    }

    RuntimeStepResult result;
    result.frame_available = true;
    result.frame = captured_frame.image;

    const auto detection_result = detector_->detect(result.frame, options_.confidence, options_.nms_threshold);
    const auto enemy_detections = filter_enemy_detections(detection_result.detections, options_.player_side);
    result.detections = fuse_head_body_detections(enemy_detections, options_.tuning);
    const auto tracks = track_manager_->update(result.detections, result.frame.size(), options_.tuning);
    const auto selected = selector_->select(tracks, result.frame.size(), analysis_state_->active_track_id());

    const auto now = std::chrono::steady_clock::now();
    const double frame_delta = std::chrono::duration<double>(now - last_time_).count();
    if (frame_delta > 0.0) {
        const double instant_fps = 1.0 / frame_delta;
        fps_ = fps_ == 0.0 ? instant_fps : fps_ * 0.90 + instant_fps * 0.10;
    }
    last_time_ = now;

    result.report = FrameReport{
        captured_frame.index,
        captured_frame.timestamp_ms,
        fps_,
        detection_result.timing,
        static_cast<int>(result.detections.size()),
        std::nullopt,
    };
    if (selected.has_value()) {
        result.report.target = analysis_state_->update(
            *selected,
            result.frame.size(),
            result.report.timestamp_ms,
            result.report.timing.total_ms()
        );
    } else {
        analysis_state_->mark_no_target();
    }

    result.command = aim_controller_->plan(result.report);
    if (hid_sender_) {
        hid_sender_->execute(result.command);
    }

    if (action_log_) {
        action_log_ << result.report.frame_index << ' '
                    << result.report.timestamp_ms << ' '
                    << (result.command.has_target ? 1 : 0) << ' '
                    << result.command.dx << ' '
                    << result.command.dy << ' '
                    << (result.command.click_left ? 1 : 0) << ' '
                    << lock_state_name(result.command.lock_state) << ' ';
        if (result.report.target.has_value()) {
            action_log_ << result.report.target->distance << ' '
                        << result.report.target->offset.x << ' '
                        << result.report.target->offset.y;
        } else {
            action_log_ << "0 0 0";
        }
        action_log_ << '\n';
    }

    ++processed_index_;
    return result;
}

void RuntimeSession::stop_all() {
    if (hid_sender_) {
        hid_sender_->stop_all();
    }
}

void RuntimeSession::close() {
    stop_all();
    if (frame_source_) {
        frame_source_->release();
    }
    action_log_.close();
    analysis_state_.reset();
    selector_.reset();
    track_manager_.reset();
    aim_controller_.reset();
    hid_sender_.reset();
    hid_client_.reset();
    detector_.reset();
    frame_source_.reset();
    open_ = false;
}

bool RuntimeSession::is_open() const {
    return open_;
}

std::string RuntimeSession::detector_name() const {
    return detector_ ? detector_->name() : std::string{};
}

std::string RuntimeSession::input_name() const {
    return frame_source_ ? frame_source_->name() : std::string{};
}

int RuntimeSession::processed_frames() const {
    return processed_index_;
}

}  // namespace vision_analyzer
