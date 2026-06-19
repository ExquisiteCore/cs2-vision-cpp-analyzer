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

enum class PlayerSide {
    Unknown,
    Ct,
    T,
};

enum class InputSource {
    Video,
    Dxgi,
};

enum class DxgiGpuPreference {
    Default,
    MinimumPower,
    HighPerformance,
};

struct RuntimeTuningConfig {
    float body_head_anchor_ratio = 0.18F;
    float kalman_process_noise = 0.08F;
    float kalman_measurement_noise = 6.0F;
    float kalman_error_covariance = 8.0F;
};

struct Options {
    std::string model_path = "../../runs/detect/train/weights/best.onnx";
    std::string model_schema_path;
    std::string video_path = "../../videos/02.mp4";
    std::string config_path;
    InputSource input_source = InputSource::Video;
    int dxgi_adapter = 0;
    int dxgi_output = 0;
    int dxgi_timeout_ms = 16;
    DxgiGpuPreference dxgi_gpu_preference = DxgiGpuPreference::Default;
    cv::Rect dxgi_roi;
    bool dxgi_debug = false;
    bool list_dxgi_outputs = false;
    bool probe_dxgi_outputs = false;
    bool verify_input = false;
    std::string hid_port;
    float hid_move_gain = 1.0F;
    int hid_max_step = 120;
    float hid_deadzone_px = 1.5F;
    bool hid_click_enabled = false;
    int hid_click_cooldown_frames = 6;
    bool calibrate_hid = false;
    int calibration_step_counts = 40;
    int calibration_repeats = 3;
    int calibration_settle_ms = 120;
    std::string calibration_output_path;
    std::string action_log_path;
    PlayerSide player_side = PlayerSide::Unknown;
    bool dry_run = false;
    int status_every_frames = 30;
    Backend backend = Backend::OpenCvOnnx;
    float confidence = 0.25F;
    float nms_threshold = 0.45F;
    RuntimeTuningConfig tuning;
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
[[nodiscard]] bool is_enemy_class(PlayerSide player_side, int class_id);
[[nodiscard]] std::vector<Detection> filter_enemy_detections(const std::vector<Detection>& detections, PlayerSide player_side);
[[nodiscard]] std::string lock_state_name(LockState state);
[[nodiscard]] std::string backend_name(Backend backend);
[[nodiscard]] PlayerSide parse_player_side(const std::string& value);
[[nodiscard]] std::string player_side_name(PlayerSide side);
[[nodiscard]] InputSource parse_input_source(const std::string& value);
[[nodiscard]] std::string input_source_name(InputSource source);
[[nodiscard]] DxgiGpuPreference parse_dxgi_gpu_preference(const std::string& value);
[[nodiscard]] std::string dxgi_gpu_preference_name(DxgiGpuPreference preference);
void validate_model_class_schema(int output_dimensions);

}  // namespace vision_analyzer
