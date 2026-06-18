#pragma once

#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace vision_analyzer {

constexpr int kInputSize = 640;

enum class Backend {
    OpenCvOnnx,
    OpenCvCuda,
    OrtCuda,
    OrtTensorRt,
    TensorRt,
};

enum class LockState {
    Idle,
    Acquiring,
    Tracking,
    Locked,
    Lost,
};

struct Options {
    std::string model_path = "../../runs/detect/train/weights/best.onnx";
    std::string video_path = "../../videos/02.mp4";
    std::string output_path = "../../runs/analysis/analysis.csv";
    std::string actions_output_path;
    std::string hid_port;
    float hid_move_gain = 1.0F;
    int hid_max_step = 120;
    bool hid_click_enabled = false;
    int hid_click_cooldown_frames = 6;
    Backend backend = Backend::OpenCvOnnx;
    float confidence = 0.25F;
    float nms_threshold = 0.45F;
    int max_frames = 0;
    int start_frame = 0;
    int warmup_frames = 3;
    double start_time_seconds = 0.0;
    bool preview = false;
};

struct LetterboxResult {
    cv::Mat image;
    float scale = 1.0F;
    int pad_x = 0;
    int pad_y = 0;
};

struct Detection {
    int class_id = -1;
    std::string label;
    float confidence = 0.0F;
    cv::Rect box;
};

struct InferenceTiming {
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;

    [[nodiscard]] double total_ms() const {
        return preprocess_ms + inference_ms + postprocess_ms;
    }
};

struct DetectionResult {
    std::vector<Detection> detections;
    InferenceTiming timing;
};

struct TrackedDetection {
    int track_id = -1;
    Detection detection;
    cv::Point2f center;
    cv::Point2f predicted_center;
    cv::Point2f velocity;
    int age = 0;
    int hits = 0;
    int missed = 0;
    float confidence_ema = 0.0F;
    float stability = 0.0F;
};

struct TargetFrame {
    int id = -1;
    Detection detection;
    cv::Point2f center;
    cv::Point2f predicted_center;
    cv::Point2f offset;
    float distance = 0.0F;
    bool switched = false;
    cv::Point2f analysis_point;
    cv::Point2f velocity;
    cv::Point2f acceleration;
    float stability = 0.0F;
    float tracking_error = 0.0F;
    LockState lock_state = LockState::Idle;
    bool fire_candidate = false;
};

struct FrameReport {
    int frame_index = 0;
    double timestamp_ms = 0.0;
    double fps = 0.0;
    InferenceTiming timing;
    int detection_count = 0;
    std::optional<TargetFrame> target;
};

[[nodiscard]] const std::vector<std::string>& class_names();
[[nodiscard]] bool is_head(int class_id);
[[nodiscard]] bool is_body(int class_id);
[[nodiscard]] std::string lock_state_name(LockState state);
[[nodiscard]] std::string backend_name(Backend backend);

}  // namespace vision_analyzer
