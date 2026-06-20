#pragma once

#include <chrono>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "vision_analyzer/aim_controller.hpp"
#include "vision_analyzer/detector.hpp"
#include "vision_analyzer/frame_source.hpp"
#include "vision_analyzer/hid_output.hpp"
#include "vision_analyzer/tracking.hpp"
#include "vision_analyzer/types.hpp"

namespace vision_analyzer {

struct RuntimeStepResult {
    bool frame_available = false;
    cv::Mat frame;
    std::vector<Detection> detections;
    FrameReport report;
    AimCommand command;
};

class RuntimeSession {
public:
    RuntimeSession() = default;
    ~RuntimeSession();

    RuntimeSession(const RuntimeSession&) = delete;
    RuntimeSession& operator=(const RuntimeSession&) = delete;

    void open(const Options& options);
    [[nodiscard]] RuntimeStepResult process_next();
    void stop_all();
    void close();

    [[nodiscard]] bool is_open() const;
    [[nodiscard]] std::string detector_name() const;
    [[nodiscard]] std::string input_name() const;
    [[nodiscard]] int processed_frames() const;

private:
    Options options_;
    std::unique_ptr<FrameSource> frame_source_;
    std::unique_ptr<Detector> detector_;
    std::unique_ptr<HidClient> hid_client_;
    std::unique_ptr<HidActionSender> hid_sender_;
    std::unique_ptr<AimController> aim_controller_;
    std::unique_ptr<TrackManager> track_manager_;
    std::unique_ptr<TargetSelector> selector_;
    std::unique_ptr<AnalysisState> analysis_state_;
    std::ofstream action_log_;
    int processed_index_ = 0;
    std::chrono::steady_clock::time_point last_time_{};
    double fps_ = 0.0;
    bool open_ = false;
};

}  // namespace vision_analyzer
