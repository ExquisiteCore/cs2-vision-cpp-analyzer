#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "vision_analyzer/aim_controller.hpp"
#include "vision_analyzer/calibration.hpp"
#include "vision_analyzer/dxgi_roi.hpp"
#include "vision_analyzer/hid_calibration_profile.hpp"
#include "vision_analyzer/hid_calibration_store.hpp"
#include "vision_analyzer/hid_output.hpp"
#include "vision_analyzer/model_input.hpp"
#include "vision_analyzer/model_schema.hpp"
#include "vision_analyzer/postprocess.hpp"
#include "vision_analyzer/runtime.hpp"
#include "vision_analyzer/runtime_config.hpp"
#include "vision_analyzer/runtime_session.hpp"
#include "vision_analyzer/runtime_status.hpp"
#include "vision_analyzer/tensorrt_provider_config.hpp"
#include "vision_analyzer/tracking.hpp"

using namespace vision_analyzer;

static_assert(noexcept(std::declval<RuntimeSession&>().close()));

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_near(float actual, float expected, float tolerance, const char* message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

void test_class_aware_nms_keeps_overlapping_different_classes() {
    const std::vector<cv::Rect> boxes = {
        cv::Rect(10, 10, 100, 100),
        cv::Rect(12, 12, 100, 100),
        cv::Rect(14, 14, 100, 100),
    };
    const std::vector<float> scores = {0.90F, 0.85F, 0.80F};
    const std::vector<int> class_ids = {1, 1, 0};

    const auto keep = class_aware_nms_indices(boxes, scores, class_ids, 0.25F, 0.45F);
    require(keep.size() == 2, "class-aware NMS should keep two boxes");
    require((keep[0] == 0 || keep[1] == 0), "class-aware NMS should keep strongest head");
    require((keep[0] == 2 || keep[1] == 2), "class-aware NMS should keep overlapping body");
}

void test_enemy_filter_keeps_opposing_side_only() {
    const std::vector<Detection> detections = {
        Detection{0, "ct_body", 0.81F, cv::Rect(100, 100, 40, 90)},
        Detection{1, "ct_head", 0.82F, cv::Rect(110, 80, 28, 28)},
        Detection{2, "t_body", 0.83F, cv::Rect(300, 100, 40, 90)},
        Detection{3, "t_head", 0.84F, cv::Rect(310, 80, 28, 28)},
    };

    const auto ct_targets = filter_enemy_detections(detections, PlayerSide::Ct);
    const auto t_targets = filter_enemy_detections(detections, PlayerSide::T);
    const auto unknown_targets = filter_enemy_detections(detections, PlayerSide::Unknown);

    require(ct_targets.size() == 2, "CT player should keep T detections only");
    require(ct_targets[0].class_id == 2 && ct_targets[1].class_id == 3, "CT player should target T body/head");
    require(t_targets.size() == 2, "T player should keep CT detections only");
    require(t_targets[0].class_id == 0 && t_targets[1].class_id == 1, "T player should target CT body/head");
    require(unknown_targets.size() == detections.size(), "unknown player side should keep all detections");
}

void test_model_class_schema_rejects_wrong_output_dimensions() {
    validate_model_class_schema(8);

    bool rejected = false;
    try {
        validate_model_class_schema(84);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "model class schema should reject COCO-style output dimensions");
}

void test_model_input_accepts_static_fp32_nchw() {
    const ModelInputSpec spec = parse_model_input_spec(
        {1, 3, 640, 640},
        ModelElementType::Float32
    );

    require(spec.width == 640 && spec.height == 640, "model input should preserve static size");
    require(
        spec.shape == std::array<std::int64_t, 4>{1, 3, 640, 640},
        "model input should preserve NCHW shape"
    );
}

void test_model_input_rejects_dynamic_or_non_fp32_input() {
    const auto rejection_message = [](const std::vector<std::int64_t>& shape, ModelElementType type) {
        try {
            (void)parse_model_input_spec(shape, type);
        } catch (const std::runtime_error& error) {
            return std::string(error.what());
        }
        return std::string{};
    };

    require(
        rejection_message({1, 3, -1, -1}, ModelElementType::Float32).find("[1, 3, -1, -1]") != std::string::npos,
        "dynamic model input error should include the actual shape"
    );
    require(
        rejection_message({1, 4, 640, 640}, ModelElementType::Float32).find("[1, 4, 640, 640]") != std::string::npos,
        "non-NCHW model input error should include the actual shape"
    );
    require(
        rejection_message({1, 3, 0, 640}, ModelElementType::Float32).find("[1, 3, 0, 640]") != std::string::npos,
        "invalid model input dimensions should include the actual shape"
    );
    require(
        rejection_message({1, 3, 640, 640}, ModelElementType::Unsupported).find("[1, 3, 640, 640]") != std::string::npos,
        "non-FP32 model input error should include the actual shape"
    );
}

void test_sm61_tensorrt_profile_is_fp32_and_cached() {
    const auto options = make_sm61_tensorrt_provider_options("D:\\cache\\sm61");
    const auto find_value = [&](const std::string& key) -> std::string {
        for (const auto& option : options) {
            if (option.key == key) {
                return option.value;
            }
        }
        return {};
    };

    require(find_value("device_id") == "0", "TensorRT should use GPU 0");
    require(find_value("trt_fp16_enable") == "0", "1080 Ti profile should use FP32");
    require(find_value("trt_engine_cache_enable") == "1", "TensorRT cache should be enabled");
    require(find_value("trt_engine_cache_path") == "D:\\cache\\sm61", "cache path should be forwarded");
    require(find_value("trt_max_workspace_size") == "2147483648", "workspace should be 2 GiB");
    require(find_value("trt_min_subgraph_size") == "1", "TensorRT should accept single-node subgraphs");
}

void test_runtime_defaults_to_sm61_tensorrt() {
    const Options options;

    require(options.backend == Backend::OrtTensorRt, "runtime should default to ORT TensorRT");
    require(!options.output_enabled, "RP2350 output should start disabled");
    require(
        options.tensorrt_cache_path == "ort-trt-cache-sm61-fp32",
        "runtime should have a writable relative cache default"
    );
}

void test_letterbox_accepts_rectangular_target() {
    const cv::Mat source(720, 1280, CV_8UC3, cv::Scalar(0, 0, 0));
    const LetterboxResult result = letterbox(source, cv::Size(640, 384));

    require(result.image.size() == cv::Size(640, 384), "letterbox should use discovered width and height");
}

void test_dxgi_copy_region_uses_full_frame_when_roi_is_disabled() {
    require(
        resolve_dxgi_copy_region({1920, 1080}, {}) == cv::Rect(0, 0, 1920, 1080),
        "disabled ROI should copy the full desktop"
    );
}

void test_dxgi_copy_region_preserves_or_clips_requested_roi() {
    require(
        resolve_dxgi_copy_region({1920, 1080}, {640, 220, 640, 640}) == cv::Rect(640, 220, 640, 640),
        "in-bounds ROI should be copied exactly"
    );
    require(
        resolve_dxgi_copy_region({1920, 1080}, {1800, 1000, 640, 640}) == cv::Rect(1800, 1000, 120, 80),
        "ROI should be clipped before GPU copy"
    );
}

void test_dxgi_copy_region_rejects_empty_intersection() {
    bool rejected = false;
    try {
        (void)resolve_dxgi_copy_region({1920, 1080}, {2000, 1200, 100, 100});
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "out-of-frame ROI should be rejected");
}

void test_model_schema_file_validates_class_order() {
    const auto path = std::filesystem::temp_directory_path() / "vision_analyzer_schema_ok.json";
    {
        std::ofstream output(path);
        output << R"({"classes":["ct_body","ct_head","t_body","t_head"]})";
    }

    const ModelSchema schema = load_model_schema(path.string());
    validate_model_schema(schema);
    std::filesystem::remove(path);
}

void test_model_schema_file_rejects_wrong_class_order() {
    const auto path = std::filesystem::temp_directory_path() / "vision_analyzer_schema_bad.json";
    {
        std::ofstream output(path);
        output << R"({"classes":["ct_body","t_head","t_body","ct_head"]})";
    }

    bool rejected = false;
    try {
        validate_model_schema(load_model_schema(path.string()));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    std::filesystem::remove(path);
    require(rejected, "model schema should reject wrong class order");
}

void test_live_schema_validation_requires_schema_file() {
    Options options;
    options.model_path = (std::filesystem::temp_directory_path() / "missing-live-model.onnx").string();

    bool rejected = false;
    try {
        validate_configured_model_schema(options, true);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "live schema validation should reject missing schema");
}

void test_decode_yolo_output_accepts_channels_last_shape() {
    const int sizes[] = {1, 1, 8};
    cv::Mat output(3, sizes, CV_32F);
    float* data = output.ptr<float>();
    data[0] = 100.0F;
    data[1] = 100.0F;
    data[2] = 20.0F;
    data[3] = 20.0F;
    data[4] = 0.10F;
    data[5] = 0.95F;
    data[6] = 0.20F;
    data[7] = 0.30F;

    const auto detections = decode_yolo_output(output, LetterboxResult{cv::Mat{}, 1.0F, 0, 0}, cv::Size(640, 640), 0.25F, 0.45F);

    require(detections.size() == 1, "channels-last YOLO output should decode one detection");
    require(detections[0].class_id == 1, "channels-last YOLO output should keep best class");
    require(detections[0].label == "ct_head", "channels-last YOLO output should use class schema label");
}

void test_input_source_parser_accepts_video_and_dxgi() {
    require(parse_input_source("video") == InputSource::Video, "input parser should accept video");
    require(parse_input_source("dxgi") == InputSource::Dxgi, "input parser should accept dxgi");
    require(input_source_name(InputSource::Dxgi) == "dxgi", "input source name should report dxgi");
    require(parse_dxgi_gpu_preference("minimum-power") == DxgiGpuPreference::MinimumPower,
            "DXGI GPU preference parser should accept minimum-power");
    require(parse_dxgi_gpu_preference("integrated") == DxgiGpuPreference::MinimumPower,
            "DXGI GPU preference parser should accept integrated alias");
    require(parse_dxgi_gpu_preference("high-performance") == DxgiGpuPreference::HighPerformance,
            "DXGI GPU preference parser should accept high-performance");
}

void test_track_manager_keeps_id_for_small_motion() {
    TrackManager manager;
    const cv::Size frame_size(1920, 1080);
    const std::vector<Detection> first = {
        Detection{1, "ct_head", 0.90F, cv::Rect(900, 500, 40, 40)},
    };
    const auto tracks_a = manager.update(first, frame_size);
    require(tracks_a.size() == 1, "first detection should create one track");

    const std::vector<Detection> second = {
        Detection{1, "ct_head", 0.92F, cv::Rect(906, 503, 40, 40)},
    };
    const auto tracks_b = manager.update(second, frame_size);
    require(tracks_b.size() == 1, "second detection should keep one track");
    require(tracks_a[0].track_id == tracks_b[0].track_id, "small motion should keep track id");
    require(tracks_b[0].hits == 2, "track hit count should increase");
    require(tracks_b[0].velocity.x > 0.0F, "track velocity should move right");
}

void test_target_selector_prefers_active_track_when_scores_are_close() {
    TargetSelector selector;
    const cv::Size frame_size(1920, 1080);
    const TrackedDetection active{
        7,
        Detection{1, "ct_head", 0.80F, cv::Rect(935, 520, 38, 38)},
        {954.0F, 539.0F},
        {954.0F, 539.0F},
        {0.0F, 0.0F},
        8,
        8,
        0,
        0.80F,
        0.80F,
    };
    const TrackedDetection challenger{
        8,
        Detection{1, "ct_head", 0.84F, cv::Rect(930, 520, 38, 38)},
        {949.0F, 539.0F},
        {949.0F, 539.0F},
        {0.0F, 0.0F},
        1,
        1,
        0,
        0.84F,
        0.10F,
    };

    const auto selected = selector.select({active, challenger}, frame_size, 7);
    require(selected.has_value(), "selector should return a target");
    require(selected->track_id == 7, "selector should keep active target when scores are close");
}

void test_target_selector_switches_when_challenger_is_clearly_better() {
    TargetSelector selector;
    const cv::Size frame_size(1920, 1080);
    const TrackedDetection active{
        7,
        Detection{1, "ct_head", 0.78F, cv::Rect(1280, 650, 42, 42)},
        {1301.0F, 671.0F},
        {1301.0F, 671.0F},
        {0.0F, 0.0F},
        8,
        8,
        0,
        0.78F,
        0.80F,
    };
    const TrackedDetection challenger{
        8,
        Detection{1, "ct_head", 0.92F, cv::Rect(950, 520, 38, 38)},
        {969.0F, 539.0F},
        {969.0F, 539.0F},
        {0.0F, 0.0F},
        2,
        2,
        0,
        0.92F,
        0.20F,
    };

    const auto selected = selector.select({active, challenger}, frame_size, 7);
    require(selected.has_value(), "selector should return a target");
    require(selected->track_id == 8, "selector should switch to clearly better target");
}

void test_target_selector_prefers_head_over_comparable_body() {
    TargetSelector selector;
    const cv::Size frame_size(1920, 1080);
    const TrackedDetection body{
        20,
        Detection{0, "ct_body", 0.90F, cv::Rect(1008, 520, 40, 40)},
        {1028.0F, 540.0F},
        {1028.0F, 540.0F},
        {0.0F, 0.0F},
        5,
        5,
        0,
        0.90F,
        0.60F,
    };
    const TrackedDetection head{
        21,
        Detection{1, "ct_head", 0.90F, cv::Rect(1040, 520, 40, 40)},
        {1060.0F, 540.0F},
        {1060.0F, 540.0F},
        {0.0F, 0.0F},
        5,
        5,
        0,
        0.90F,
        0.60F,
    };

    const auto selected = selector.select({body, head}, frame_size, std::nullopt);
    require(selected.has_value(), "selector should return a target");
    require(selected->track_id == head.track_id,
            "head priority should beat a slightly closer comparable body");
}

void test_track_manager_smooths_velocity_spikes() {
    TrackManager manager;
    const cv::Size frame_size(1920, 1080);
    const auto first = manager.update({Detection{1, "ct_head", 0.90F, cv::Rect(900, 500, 40, 40)}}, frame_size);
    require(first.size() == 1, "first velocity test detection should create one track");
    auto tracks = manager.update({Detection{1, "ct_head", 0.91F, cv::Rect(910, 500, 40, 40)}}, frame_size);
    require(tracks.size() == 1, "second velocity test detection should keep one track");
    require_near(tracks[0].velocity.x, 10.0F, 0.01F, "first velocity estimate should match observed motion");

    tracks = manager.update({Detection{1, "ct_head", 0.92F, cv::Rect(970, 500, 40, 40)}}, frame_size);
    require(tracks.size() == 1, "third velocity test detection should keep one track");
    require(tracks[0].velocity.x > 10.0F, "smoothed velocity should react to faster motion");
    require(tracks[0].velocity.x < 60.0F, "smoothed velocity should damp sudden spikes");
}

void test_target_anchor_point_uses_body_top_fallback() {
    const Detection body{0, "ct_body", 0.95F, cv::Rect(900, 500, 60, 180)};
    const Detection head{1, "ct_head", 0.95F, cv::Rect(916, 494, 28, 28)};

    const cv::Point2f body_anchor = target_anchor_point(body, 0.20F);
    const cv::Point2f head_anchor = target_anchor_point(head, 0.20F);

    require_near(body_anchor.x, 930.0F, 0.01F, "body anchor should use horizontal center");
    require_near(body_anchor.y, 536.0F, 0.01F, "body anchor should use configured top-body ratio");
    require_near(head_anchor.x, 930.0F, 0.01F, "head anchor should use head center x");
    require_near(head_anchor.y, 508.0F, 0.01F, "head anchor should use head center y");
}

void test_fuse_head_body_detections_suppresses_body_when_head_matches() {
    RuntimeTuningConfig tuning;
    tuning.body_head_anchor_ratio = 0.20F;
    const std::vector<Detection> detections = {
        Detection{0, "ct_body", 0.80F, cv::Rect(900, 500, 60, 180)},
        Detection{1, "ct_head", 0.92F, cv::Rect(916, 522, 28, 28)},
        Detection{2, "t_body", 0.85F, cv::Rect(1200, 500, 60, 180)},
    };

    const auto fused = fuse_head_body_detections(detections, tuning);

    require(fused.size() == 2, "matched body/head should become one head target plus unmatched body");
    require(fused[0].class_id == 1, "matched head should be kept first");
    require(fused[1].class_id == 2, "unmatched body should remain as fallback");
}

void test_track_manager_uses_configured_body_anchor_ratio() {
    RuntimeTuningConfig tuning;
    tuning.body_head_anchor_ratio = 0.25F;
    TrackManager manager;
    const cv::Size frame_size(1920, 1080);

    const auto tracks = manager.update(
        {Detection{0, "ct_body", 0.90F, cv::Rect(900, 400, 120, 200)}},
        frame_size,
        tuning
    );

    require(tracks.size() == 1, "body detection should create one track");
    require_near(tracks[0].center.x, 960.0F, 0.01F, "body track center should use body anchor x");
    require_near(tracks[0].center.y, 450.0F, 0.01F, "body track center should use configured anchor ratio");
}

void test_analysis_state_predicts_latency_in_frame_units() {
    AnalysisState state;
    const cv::Size frame_size(1920, 1080);
    const TrackedDetection selected{
        4,
        Detection{1, "ct_head", 0.90F, cv::Rect(940, 520, 40, 40)},
        {960.0F, 540.0F},
        {960.0F, 540.0F},
        {30.0F, 0.0F},
        5,
        5,
        0,
        0.90F,
        0.80F,
    };

    const auto report = state.update(selected, frame_size, 33.333, 16.667);
    require_near(report.predicted_center.x, 975.0F, 0.75F, "latency prediction should use frame units");
    require_near(report.predicted_center.y, 540.0F, 0.01F, "latency prediction should keep y when y velocity is zero");
}

void test_analysis_state_offsets_from_filtered_analysis_point() {
    AnalysisState state;
    const cv::Size frame_size(1920, 1080);
    const TrackedDetection first{
        4,
        Detection{1, "ct_head", 0.90F, cv::Rect(940, 520, 40, 40)},
        {960.0F, 540.0F},
        {960.0F, 540.0F},
        {0.0F, 0.0F},
        5,
        5,
        0,
        0.90F,
        0.80F,
    };
    const TrackedDetection second{
        4,
        Detection{1, "ct_head", 0.92F, cv::Rect(1040, 520, 40, 40)},
        {1060.0F, 540.0F},
        {1060.0F, 540.0F},
        {0.0F, 0.0F},
        6,
        6,
        0,
        0.91F,
        0.85F,
    };

    (void)state.update(first, frame_size, 0.0, 0.0);
    const auto report = state.update(second, frame_size, 16.0, 0.0);
    const cv::Point2f frame_center(frame_size.width / 2.0F, frame_size.height / 2.0F);

    require_near(
        report.offset.x,
        report.analysis_point.x - frame_center.x,
        0.01F,
        "target offset should use filtered analysis point"
    );
    require(std::abs(report.offset.x - (report.predicted_center.x - frame_center.x)) > 1.0F,
            "filtered offset should differ from raw predicted jump");
}

void test_motion_filter_is_stable_and_moves_toward_measurement() {
    MotionFilter2D filter;
    const cv::Point2f first = filter.update({100.0F, 100.0F}, 0.0);
    const cv::Point2f second = filter.update({200.0F, 100.0F}, 16.0);
    const cv::Point2f third = filter.update({200.0F, 100.0F}, 32.0);

    require_near(first.x, 100.0F, 0.001F, "filter should initialize at first measurement");
    require(second.x > first.x, "filter should move toward new measurement");
    require(third.x >= second.x, "filter should not move backward for repeated measurement");
    require(filter.initialized(), "filter should report initialized after update");
}

void test_motion_filter_predicts_with_kalman_velocity() {
    MotionFilter2D filter;
    (void)filter.update({100.0F, 100.0F}, 0.0);
    (void)filter.update({140.0F, 100.0F}, 16.0);
    const auto prediction = filter.predict(16.0);

    require(prediction.x > 100.0F, "kalman prediction should move in the observed direction");
    require_near(prediction.y, 100.0F, 2.0F, "kalman prediction should keep y stable for horizontal motion");
}

void test_analysis_state_never_fires_for_body_class() {
    AnalysisState state;
    const cv::Size frame_size(1920, 1080);
    const TrackedDetection selected{
        4,
        Detection{0, "ct_body", 0.95F, cv::Rect(930, 500, 60, 100)},
        {960.0F, 518.0F},
        {960.0F, 518.0F},
        {0.0F, 0.0F},
        10,
        10,
        0,
        0.95F,
        0.90F,
    };

    TargetFrame report{};
    report = state.update(selected, frame_size, 0.0, 0.0);
    report = state.update(selected, frame_size, 16.0, 0.0);
    report = state.update(selected, frame_size, 32.0, 0.0);

    require(report.lock_state == LockState::Locked, "body can still reach lock state for movement tracking");
    require(!report.fire_candidate, "body class should never become a fire candidate");
}

void test_analysis_state_aims_body_detection_at_head_anchor() {
    RuntimeTuningConfig tuning;
    tuning.body_head_anchor_ratio = 0.20F;
    AnalysisState state(tuning);
    const cv::Size frame_size(1920, 1080);
    const TrackedDetection selected{
        4,
        Detection{0, "ct_body", 0.95F, cv::Rect(930, 450, 60, 180)},
        {960.0F, 486.0F},
        {960.0F, 486.0F},
        {0.0F, 0.0F},
        10,
        10,
        0,
        0.95F,
        0.90F,
    };

    const TargetFrame report = state.update(selected, frame_size, 0.0, 0.0);

    require_near(report.center.x, 960.0F, 0.01F, "body fallback anchor should keep x center");
    require_near(report.center.y, 486.0F, 0.01F, "body fallback anchor should aim near body top, not body center");
    require_near(report.offset.y, -54.0F, 0.01F, "body fallback offset should use head anchor");
    require(!report.fire_candidate, "body fallback should still never fire");
}

void test_aim_controller_scales_and_clamps_target_offset() {
    AimControllerOptions options;
    options.move_gain = 0.5F;
    options.max_step = 12;
    AimController controller(options);

    FrameReport report{
        42,
        1400.0,
        120.0,
        InferenceTiming{1.0, 2.0, 3.0},
        1,
        TargetFrame{
            9,
            Detection{1, "ct_head", 0.91F, cv::Rect(950, 520, 40, 40)},
            {970.0F, 540.0F},
            {978.0F, 542.0F},
            {30.0F, -10.0F},
            31.62F,
            false,
            {976.0F, 541.0F},
            {120.0F, 6.0F},
            {10.0F, 1.0F},
            0.82F,
            2.24F,
            LockState::Locked,
            true,
        },
    };

    const AimCommand command = controller.plan(report);

    require(command.has_target, "aim command should mark target as present");
    require(command.dx == 12, "aim command should clamp scaled x movement");
    require(command.dy == -5, "aim command should scale y movement");
    require(!command.click_left, "aim command should not click when clicks are disabled");
}

void test_aim_controller_deadzone_suppresses_tiny_steps() {
    AimControllerOptions options;
    options.move_gain = 1.0F;
    options.max_step = 20;
    options.deadzone_px = 3.0F;
    AimController controller(options);

    FrameReport report{
        42,
        1400.0,
        120.0,
        InferenceTiming{},
        1,
        TargetFrame{
            9,
            Detection{1, "ct_head", 0.91F, cv::Rect(950, 520, 40, 40)},
            {970.0F, 540.0F},
            {978.0F, 542.0F},
            {2.9F, -3.1F},
            4.25F,
            false,
            {962.9F, 536.9F},
            {0.0F, 0.0F},
            {0.0F, 0.0F},
            0.82F,
            0.0F,
            LockState::Tracking,
            false,
        },
    };

    const AimCommand command = controller.plan(report);

    require(command.dx == 0, "deadzone should suppress tiny x movement");
    require(command.dy == -3, "deadzone should keep movement outside threshold");
}

void test_hid_calibration_fit_generates_tuning_values() {
    const std::vector<CalibrationSample> samples = {
        CalibrationSample{0, 0, {0.2, 0.1}},
        CalibrationSample{40, 0, {10.0, 0.1}},
        CalibrationSample{-40, 0, {-10.0, -0.1}},
        CalibrationSample{0, 40, {0.1, 8.0}},
        CalibrationSample{0, -40, {-0.1, -8.0}},
    };

    const CalibrationFit fit = fit_hid_calibration(samples, 40);

    require(fit.valid, "calibration fit should be valid with movement samples");
    require_near(static_cast<float>(fit.gain_x), 4.0F, 0.01F, "x gain should use counts per visual pixel");
    require_near(static_cast<float>(fit.gain_y), 5.0F, 0.01F, "y gain should use counts per visual pixel");
    require_near(static_cast<float>(fit.hid_gain), 4.5F, 0.01F, "hid gain should combine axis gains");
    require(fit.deadzone_px >= 1.0 && fit.deadzone_px <= 8.0, "deadzone should be bounded");
    require(fit.max_step == 120, "max step should scale from calibration step");

    Options options;
    std::ostringstream output;
    write_hid_tuning_config(output, options, fit);
    require(output.str().find("hid_gain=4.5") != std::string::npos, "tuned config should include fitted gain");
    require(output.str().find("hid_max_step=120") != std::string::npos, "tuned config should include max step");
}

std::vector<CalibrationSample> make_valid_adaptive_calibration_samples() {
    return {
        {0, 0, {0.2, 0.1}, 0.95, -1},
        {16, 0, {-8.0, 0.2}, 0.90, 0},
        {-16, 0, {8.1, -0.1}, 0.91, 0},
        {40, 0, {-16.0, 0.2}, 0.92, 1},
        {-40, 0, {16.2, -0.2}, 0.90, 1},
        {80, 0, {-20.0, 0.2}, 0.93, 2},
        {-80, 0, {20.2, -0.1}, 0.91, 2},
        {0, 16, {0.1, 8.0}, 0.92, 0},
        {0, -16, {-0.1, -8.1}, 0.90, 0},
        {0, 40, {0.2, 16.0}, 0.91, 1},
        {0, -40, {-0.2, -16.1}, 0.92, 1},
        {0, 80, {0.1, 20.0}, 0.93, 2},
        {0, -80, {-0.1, -20.1}, 0.92, 2},
    };
}

void test_adaptive_calibration_fits_signed_axes_and_inverted_y() {
    const auto samples = make_valid_adaptive_calibration_samples();
    const auto profile = fit_adaptive_hid_calibration(samples, {1920, 1080}, 120);
    require(profile.valid, "consistent samples should produce a profile");
    require(profile.x.counts_per_pixel[0] > 0.0F, "normal X should be positive");
    require(profile.y.counts_per_pixel[0] < 0.0F, "inverted Y should be negative");
    require(profile.x.counts_per_pixel[2] > profile.x.counts_per_pixel[0],
            "nonlinear samples should preserve a large-step gain");
}

void test_adaptive_calibration_ignores_incoherent_noise_measurements() {
    auto samples = make_valid_adaptive_calibration_samples();
    samples.push_back({0, 0, {500.0, -500.0}, 0.0, -1});

    const auto profile = fit_adaptive_hid_calibration(samples, {1920, 1080}, 120);

    require(profile.valid,
            "low-response noise measurement should not reject valid movement curves");
    require(profile.noise_px < 1.0F,
            "low-response noise measurement should not inflate the fitted deadzone");
}

void test_adaptive_calibration_rejects_bad_response_and_cross_axis_motion() {
    auto samples = make_valid_adaptive_calibration_samples();
    for (auto& sample : samples) {
        if (sample.level == 1 && sample.counts_dx != 0) {
            sample.phase_response = 0.05;
            sample.visual_shift.y = sample.visual_shift.x;
        }
    }
    const auto profile = fit_adaptive_hid_calibration(samples, {1920, 1080}, 120);
    require(!profile.valid, "missing valid X level must reject the profile");
}

void test_calibration_probe_adjustment_uses_calibration_only_limit() {
    require(adjust_calibration_probe_count(16, 0.5, 8.0, 2048) == 256,
            "tiny movement should scale above the runtime max step");
    require(adjust_calibration_probe_count(512, 0.5, 8.0, 2048) == 2048,
            "probe adjustment must stop at the calibration limit");
    require(adjust_calibration_probe_count(80, 200.0, 80.0, 2048) == 32,
            "oversized response should scale down proportionally");
}

void test_calibration_level_plan_compresses_low_sensitivity_range() {
    const CalibrationLevelPlan normal = derive_calibration_level_plan(2.0, 1.5);
    require(normal.counts == std::array<int, 3>{16, 48, 96},
            "normal sensitivity should use 8/24/48-pixel targets");
    require(normal.target_shift_px == std::array<double, 3>{8.0, 24.0, 48.0},
            "normal target shifts should remain in the reliable phase-correlation range");

    const CalibrationLevelPlan low = derive_calibration_level_plan(60.0, 1.5);
    require(low.counts == std::array<int, 3>{480, 1024, 2048},
            "low sensitivity should use the complete calibration range");
    require(low.counts[0] < low.counts[1] && low.counts[1] < low.counts[2],
            "derived counts must be strictly increasing");
}

void test_calibration_level_plan_rejects_unmeasurable_range() {
    bool rejected = false;
    try {
        (void)derive_calibration_level_plan(400.0, 1.5);
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("2048") != std::string::npos;
    }
    require(rejected, "an unmeasurable 2048-count range must be rejected explicitly");
}

void test_calibration_probe_planner_escalates_low_response() {
    const CalibrationProbePlan plan = plan_calibration_probe(
        120, 0, 2.0, 0.1, 0.05, 4.0, 270.0
    );
    require(!plan.accepted && !plan.exhausted,
            "low-response probe should continue discovery");
    require(plan.next_counts == 240,
            "unreliable discovery should double instead of trusting its shift");
}

void test_calibration_probe_planner_accepts_signal_and_rejects_cross_axis() {
    const CalibrationProbePlan accepted = plan_calibration_probe(
        1000, 1, 8.0, 0.2, 0.80, 4.0, 270.0
    );
    require(accepted.accepted && !accepted.exhausted,
            "reliable discovery signal should be accepted");

    const CalibrationProbePlan crossed = plan_calibration_probe(
        1000, 1, 8.0, 4.0, 0.80, 4.0, 270.0
    );
    require(!crossed.accepted && crossed.exhausted,
            "dominant cross-axis movement should reject the scene without escalating");
}

void test_calibration_axis_discovery_derives_low_sensitivity_levels() {
    std::vector<int> attempted;
    const CalibrationAxisDiscovery discovery = discover_calibration_axis(
        0,
        4.0,
        1.5,
        270.0,
        [&](int counts) {
            attempted.push_back(counts);
            return CalibrationRoundTripMeasurement{
                {{-static_cast<double>(counts) / 60.0, 0.05}, 0.90, true},
                {{static_cast<double>(counts) / 60.0, -0.05}, 0.90, true},
            };
        }
    );
    require(attempted == std::vector<int>({16, 16, 16, 32, 32, 32, 480}),
            "subpixel signal should be sampled three times before bounded scaling");
    require(discovery.probe_counts == 480, "discovery should retain the accepted count");
    require(discovery.levels.counts == std::array<int, 3>{480, 1024, 2048},
            "discovery should derive compressed low-sensitivity levels");
}

void test_calibration_axis_discovery_retries_same_count_before_escalating() {
    std::vector<int> attempted;
    const CalibrationAxisDiscovery discovery = discover_calibration_axis(
        0,
        4.0,
        1.5,
        270.0,
        [&](int counts) {
            attempted.push_back(counts);
            if (attempted.size() == 1) {
                return CalibrationRoundTripMeasurement{
                    {{0.0, 0.0}, 0.0, false},
                    {{0.0, 0.0}, 0.0, false},
                };
            }
            return CalibrationRoundTripMeasurement{
                {{-8.0, 0.1}, 0.80, true},
                {{8.1, -0.1}, 0.80, true},
            };
        }
    );
    require(discovery.probe_counts == 16,
            "second coherent measurement should accept the original count");
    require(attempted == std::vector<int>({16, 16}),
            "discovery should retry the same balanced count before escalating");
}

void test_calibration_axis_discovery_reports_probe_exhaustion() {
    std::vector<int> attempted;
    bool rejected = false;
    try {
        (void)discover_calibration_axis(
            0,
            4.0,
            1.5,
            270.0,
            [&](int counts) {
                attempted.push_back(counts);
                return CalibrationRoundTripMeasurement{
                    {{0.0, 0.0}, 0.0, false},
                    {{0.0, 0.0}, 0.0, false},
                };
            }
        );
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find(
            "HID calibration input not ready: axis=x "
            "movement could not be measured reliably through 2048 counts"
        ) != std::string::npos;
    }
    require(rejected, "two-sweep exhaustion must report input-not-ready explicitly");
    require(attempted.size() ==
                static_cast<std::size_t>(
                    2 * kCalibrationDiscoveryMaximumAttempts *
                    kCalibrationProbeMeasurementsPerCount
                ),
            "discovery should use three measurements per count for two sweeps");
    const std::array<int, 8> ladder = {16, 32, 64, 128, 256, 512, 1024, 2048};
    for (int sweep = 0; sweep < kCalibrationDiscoverySweeps; ++sweep) {
        for (std::size_t level = 0; level < ladder.size(); ++level) {
            for (int repeat = 0; repeat < kCalibrationProbeMeasurementsPerCount; ++repeat) {
                const std::size_t index = static_cast<std::size_t>(
                    sweep * static_cast<int>(ladder.size()) *
                        kCalibrationProbeMeasurementsPerCount +
                    static_cast<int>(level) * kCalibrationProbeMeasurementsPerCount +
                    repeat
                );
                require(attempted[index] == ladder[level],
                        "input-not-ready discovery should follow the bounded ladder twice");
            }
        }
    }
    require(kCalibrationRuntimeMaxStep == 120,
            "discovery retries must not change the runtime movement clamp");
}

void test_round_trip_estimation_measures_outward_and_inverse_frames() {
    cv::Mat baseline(64, 64, CV_32F);
    cv::randu(baseline, 0.0F, 255.0F);
    cv::Mat moved;
    const cv::Mat transform = (cv::Mat_<double>(2, 3) <<
        1.0, 0.0, 5.0,
        0.0, 1.0, -3.0);
    cv::warpAffine(
        baseline,
        moved,
        transform,
        baseline.size(),
        cv::INTER_LINEAR,
        cv::BORDER_WRAP
    );

    const CalibrationRoundTripMeasurement measurement =
        estimate_calibration_round_trip(baseline, moved, baseline);
    require_near(static_cast<float>(measurement.outward.shift.x), 5.0F, 0.1F,
                 "outward leg should use baseline-to-moved frames");
    require_near(static_cast<float>(measurement.outward.shift.y), -3.0F, 0.1F,
                 "outward Y shift should be preserved");
    require_near(static_cast<float>(measurement.inverse.shift.x), -5.0F, 0.1F,
                 "inverse leg should use moved-to-returned frames");
    require_near(static_cast<float>(measurement.inverse.shift.y), 3.0F, 0.1F,
                 "inverse Y shift should be measured independently");
    require(measurement.outward.response > 0.90 && measurement.inverse.response > 0.90,
            "both synthetic round-trip legs should have strong responses");
}

void test_round_trip_command_plan_alternates_without_final_escalation() {
    std::array<CalibrationAxisDiscovery, 2> discoveries;
    discoveries[0].levels.counts = {11, 32, 65};
    discoveries[1].levels.counts = {12, 33, 66};

    const auto commands = plan_calibration_round_trip_commands(discoveries, 2);
    std::vector<int> actual;
    for (const auto& command : commands) {
        actual.push_back(command.outward_counts);
    }
    require(actual == std::vector<int>({
                11, -11, 32, -32, 65, -65,
                12, -12, 33, -33, 66, -66,
            }),
            "round-trip plans should alternate signs and keep discovered counts");
}

void test_round_trip_samples_assign_real_signed_legs() {
    CalibrationRoundTripMeasurement measurement;
    measurement.outward = {{-12.0, 0.25}, 0.70};
    measurement.inverse = {{11.5, -0.20}, 0.65};

    const auto x_samples = make_calibration_round_trip_samples(
        0, 1, 40, measurement
    );
    require(x_samples[0].counts_dx == 40 && x_samples[0].counts_dy == 0,
            "X outward sample should keep its signed command");
    require(x_samples[1].counts_dx == -40 && x_samples[1].counts_dy == 0,
            "X return sample should use the exact inverse command");
    require_near(static_cast<float>(x_samples[0].visual_shift.x), -12.0F, 0.001F,
                 "outward visual shift must not be synthesized");
    require_near(static_cast<float>(x_samples[1].visual_shift.x), 11.5F, 0.001F,
                 "return visual shift must be preserved independently");
    require_near(static_cast<float>(x_samples[1].phase_response), 0.65F, 0.001F,
                 "return response must be preserved independently");

    const auto y_samples = make_calibration_round_trip_samples(
        1, 2, -50, measurement
    );
    require(y_samples[0].counts_dx == 0 && y_samples[0].counts_dy == -50,
            "Y outward sample should keep a negative command");
    require(y_samples[1].counts_dx == 0 && y_samples[1].counts_dy == 50,
            "Y return sample should invert the command");
}

void test_round_trip_sampling_can_recover_from_one_bad_path() {
    std::vector<CalibrationSample> samples = {
        {0, 0, {0.01, -0.01}, 0.99, -1},
    };
    const std::array<int, 3> counts = {16, 48, 96};
    const std::array<double, 3> shifts = {8.0, 24.0, 48.0};
    for (std::size_t axis = 0; axis < 2; ++axis) {
        for (std::size_t level = 0; level < counts.size(); ++level) {
            CalibrationRoundTripMeasurement bad;
            bad.outward = {{0.0, 0.0}, 0.05};
            bad.inverse = {{0.0, 0.0}, 0.05};
            const auto rejected = make_calibration_round_trip_samples(
                axis, static_cast<int>(level), counts[level], bad
            );
            samples.insert(samples.end(), rejected.begin(), rejected.end());

            CalibrationRoundTripMeasurement good;
            if (axis == 0) {
                good.outward = {{shifts[level], 0.1}, 0.90};
                good.inverse = {{-shifts[level], -0.1}, 0.90};
            } else {
                good.outward = {{0.1, shifts[level]}, 0.90};
                good.inverse = {{-0.1, -shifts[level]}, 0.90};
            }
            const auto accepted = make_calibration_round_trip_samples(
                axis, static_cast<int>(level), -counts[level], good
            );
            samples.insert(samples.end(), accepted.begin(), accepted.end());
        }
    }

    const auto profile = fit_adaptive_hid_calibration(
        samples, {1920, 1080}, kCalibrationRuntimeMaxStep
    );
    require(profile.valid,
            "one reliable opposite round trip should fill both signed fitter buckets");
    require(profile.accepted_samples == 12,
            "only the twelve reliable signed samples should be accepted");
    require(profile.max_step == 120, "round-trip fitting must retain the runtime clamp");
}

CalibrationRoundTripMeasurement coherent_axis_round_trip(
    std::size_t axis,
    double shift
) {
    if (axis == 0) {
        return {
            {{-shift, 0.1}, 0.80, true},
            {{shift, -0.1}, 0.82, true},
        };
    }
    return {
        {{0.1, -shift}, 0.80, true},
        {{-0.1, shift}, 0.82, true},
    };
}

CalibrationRoundTripMeasurement incoherent_axis_round_trip() {
    return {
        {{0.0, 0.0}, 0.0, false},
        {{0.0, 0.0}, 0.0, false},
    };
}

CalibrationRoundTripMeasurement coherent_signed_axis_round_trip(
    std::size_t axis,
    int outward_counts,
    double shift
) {
    CalibrationRoundTripMeasurement measurement =
        coherent_axis_round_trip(axis, shift);
    if (outward_counts < 0) {
        std::swap(measurement.outward, measurement.inverse);
    }
    return measurement;
}

CalibrationLevelMeasurement level_measurement(
    int outward_counts,
    CalibrationRoundTripMeasurement round_trip
) {
    return {outward_counts, std::move(round_trip)};
}

void test_calibration_level_selection_falls_back_strictly_downward() {
    std::vector<int> attempted;
    const CalibrationLevelSelection selection = select_calibration_level(
        0,
        2,
        33,
        66,
        1.5,
        270.0,
        [&](int counts) {
            attempted.push_back(counts);
            if (counts == 49) {
                return std::vector<CalibrationLevelMeasurement>{
                    level_measurement(counts, coherent_axis_round_trip(0, 34.0)),
                    level_measurement(-counts, coherent_axis_round_trip(0, 34.2)),
                };
            }
            return std::vector<CalibrationLevelMeasurement>{
                level_measurement(counts, incoherent_axis_round_trip()),
                level_measurement(-counts, incoherent_axis_round_trip()),
            };
        }
    );

    require(selection.accepted && selection.counts == 49,
            "failed high level should select the first coherent midpoint");
    require(attempted == std::vector<int>({66, 66, 49}),
            "planned high count should retry once before falling downward");
    require(selection.measurements.size() == 2,
            "selection should retain coherent round trips only");
    require(selection.counts > 33 && selection.counts < 66,
            "fallback must remain between the previous and planned levels");
}

void test_calibration_level_selection_never_shrinks_the_low_level() {
    std::vector<int> attempted;
    bool rejected = false;
    try {
        (void)select_calibration_level(
            0,
            0,
            0,
            11,
            1.5,
            270.0,
            [&](int counts) {
                attempted.push_back(counts);
                return std::vector<CalibrationLevelMeasurement>{
                    level_measurement(counts, incoherent_axis_round_trip()),
                };
            }
        );
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("level=0") != std::string::npos;
    }
    require(rejected, "unusable low level should report a bounded level error");
    require(attempted == std::vector<int>({11, 11}),
            "low level should retry only its original measurable count");
}

void test_calibration_level_selection_has_bounded_distinct_midpoints() {
    std::vector<int> attempted;
    bool rejected = false;
    try {
        (void)select_calibration_level(
            1,
            2,
            33,
            96,
            1.5,
            270.0,
            [&](int counts) {
                attempted.push_back(counts);
                return std::vector<CalibrationLevelMeasurement>{
                    level_measurement(counts, incoherent_axis_round_trip()),
                };
            }
        );
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "exhausted downward candidates should reject the level");
    require(attempted.size() <=
                static_cast<std::size_t>(
                    2 + kCalibrationMaximumDownwardLevelCandidates
                ),
            "fallback should have a fixed candidate budget");
    for (const int counts : attempted) {
        require(counts > 33 && counts <= 96,
                "fallback must never rise or collide with the previous level");
    }
    require(kCalibrationRuntimeMaxStep == 120,
            "final-level recovery must not alter the runtime clamp");
}

void test_calibration_round_trip_usability_requires_both_opposite_legs() {
    CalibrationRoundTripMeasurement measurement = coherent_axis_round_trip(0, 24.0);
    require(usable_calibration_round_trip(0, measurement, 1.5, 270.0),
            "coherent opposite legs should be usable");
    measurement.inverse.coherent = false;
    require(!usable_calibration_round_trip(0, measurement, 1.5, 270.0),
            "one incoherent return leg should reject the round trip");
    measurement = coherent_axis_round_trip(0, 24.0);
    measurement.inverse.shift.x = -24.0;
    require(!usable_calibration_round_trip(0, measurement, 1.5, 270.0),
            "same-direction visual legs should not fill opposite signed buckets");
}

void test_recovered_level_measurements_fit_three_increasing_knots() {
    std::vector<CalibrationSample> samples = {
        {0, 0, {0.01, -0.01}, 0.99, -1},
        {0, 0, {-0.01, 0.01}, 0.99, -1},
    };
    const std::array<int, 3> planned = {11, 33, 66};
    for (std::size_t axis = 0; axis < 2; ++axis) {
        int previous_count = 0;
        for (std::size_t level = 0; level < planned.size(); ++level) {
            const CalibrationLevelSelection selection = select_calibration_level(
                axis,
                static_cast<int>(level),
                previous_count,
                planned[level],
                1.5,
                270.0,
                [&](int counts) {
                    const bool rejected_planned_high =
                        axis == 0 && level == 2 && counts == planned[level];
                    if (rejected_planned_high) {
                        return std::vector<CalibrationLevelMeasurement>{
                            level_measurement(counts, incoherent_axis_round_trip()),
                            level_measurement(-counts, incoherent_axis_round_trip()),
                        };
                    }
                    const double shift = static_cast<double>(counts) / 1.4;
                    return std::vector<CalibrationLevelMeasurement>{
                        level_measurement(
                            counts,
                            coherent_signed_axis_round_trip(axis, counts, shift)
                        ),
                        level_measurement(
                            -counts,
                            coherent_signed_axis_round_trip(axis, -counts, shift)
                        ),
                    };
                }
            );
            previous_count = selection.counts;
            for (const CalibrationLevelMeasurement& measured : selection.measurements) {
                const auto signed_samples = make_calibration_round_trip_samples(
                    axis,
                    static_cast<int>(level),
                    measured.outward_counts,
                    measured.round_trip
                );
                samples.insert(samples.end(), signed_samples.begin(), signed_samples.end());
            }
        }
    }

    const HidCalibrationProfile profile = fit_adaptive_hid_calibration(
        samples,
        {1920, 1080},
        kCalibrationRuntimeMaxStep
    );
    require(profile.valid,
            "one recovered high level should still produce a valid profile");
    require(profile.accepted_samples >= 12,
            "one coherent round trip per axis and level should satisfy the minimum");
    require(profile.x.shift_px[0] < profile.x.shift_px[1] &&
            profile.x.shift_px[1] < profile.x.shift_px[2],
            "fallback should still produce three increasing X knots");
    require(profile.max_step == 120,
            "recovered calibration must retain the runtime movement clamp");
}

cv::Mat make_textured_calibration_scene(const cv::Size& size) {
    cv::Mat scene(size, CV_8UC3);
    cv::RNG random(0x2350);
    random.fill(scene, cv::RNG::UNIFORM, 0, 255);
    cv::GaussianBlur(scene, scene, {5, 5}, 0.8);
    for (int y = 20; y < size.height; y += 47) {
        cv::line(scene, {0, y}, {size.width - 1, y + 13}, {20, 210, 90}, 2);
    }
    for (int x = 30; x < size.width; x += 71) {
        cv::circle(scene, {x, (x * 7) % size.height}, 11, {230, 40, 180}, 2);
    }
    return scene;
}

cv::Mat translated_wrap(const cv::Mat& source, double dx, double dy) {
    cv::Mat translated;
    const cv::Mat transform = (cv::Mat_<double>(2, 3) <<
        1.0, 0.0, dx,
        0.0, 1.0, dy);
    cv::warpAffine(
        source,
        translated,
        transform,
        source.size(),
        cv::INTER_LINEAR,
        cv::BORDER_WRAP
    );
    return translated;
}

cv::Mat camera_crop(
    const cv::Mat& panorama,
    const cv::Size& viewport_size,
    int offset_x,
    int offset_y
) {
    return panorama(cv::Rect(offset_x, offset_y, viewport_size.width, viewport_size.height)).clone();
}

cv::Mat yaw_camera_view(const cv::Mat& source, double center_shift_px) {
    cv::Mat map_x(source.size(), CV_32F);
    cv::Mat map_y(source.size(), CV_32F);
    const double center_x = static_cast<double>(source.cols - 1) * 0.5;
    const double focal = static_cast<double>(source.cols) * 0.55;
    for (int y = 0; y < source.rows; ++y) {
        auto* x_row = map_x.ptr<float>(y);
        auto* y_row = map_y.ptr<float>(y);
        for (int x = 0; x < source.cols; ++x) {
            const double normalized = (static_cast<double>(x) - center_x) / focal;
            const double perspective = 1.0 + 3.0 * normalized * normalized;
            x_row[x] = static_cast<float>(x + center_shift_px * perspective);
            y_row[x] = static_cast<float>(y + 0.8 * normalized);
        }
    }
    cv::Mat moved;
    cv::remap(
        source,
        moved,
        map_x,
        map_y,
        cv::INTER_LINEAR,
        cv::BORDER_REFLECT101
    );
    return moved;
}

void paint_static_game_hud(cv::Mat& frame) {
    const cv::Point center(frame.cols / 2, frame.rows / 2);
    cv::rectangle(frame, {18, 18, 330, 300}, {18, 18, 18}, cv::FILLED);
    for (int radius = 35; radius <= 125; radius += 30) {
        cv::circle(frame, {174, 168}, radius, {220, 220, 220}, 3);
    }
    cv::rectangle(
        frame,
        {center.x - 330, 0, 660, 92},
        {24, 24, 24},
        cv::FILLED
    );
    for (int x = center.x - 280; x <= center.x + 280; x += 80) {
        cv::line(frame, {x, 12}, {x + 35, 78}, {245, 245, 245}, 5);
    }
    cv::rectangle(
        frame,
        {frame.cols - 430, 18, 412, 270},
        {22, 22, 22},
        cv::FILLED
    );
    for (int y = 42; y < 250; y += 42) {
        cv::line(
            frame,
            {frame.cols - 400, y},
            {frame.cols - 45, y + 18},
            {235, 235, 235},
            4
        );
    }
    cv::line(frame, {center.x - 28, center.y}, {center.x + 28, center.y},
             {255, 255, 255}, 3);
    cv::line(frame, {center.x, center.y - 28}, {center.x, center.y + 28},
             {255, 255, 255}, 3);
}

void paint_static_calibration_overlay(cv::Mat& before, cv::Mat& after) {
    const cv::Point center(before.cols / 2, before.rows / 2);
    const cv::Rect panel(center.x - 130, center.y - 90, 260, 180);
    for (cv::Mat* frame : {&before, &after}) {
        cv::rectangle(*frame, panel, {12, 12, 12}, cv::FILLED);
        cv::rectangle(*frame, panel, {250, 250, 250}, 4);
        cv::line(*frame, {center.x - 110, center.y},
                 {center.x + 110, center.y}, {255, 255, 255}, 6);
        cv::line(*frame, {center.x, center.y - 75},
                 {center.x, center.y + 75}, {255, 255, 255}, 6);
        cv::rectangle(
            *frame,
            {0, static_cast<int>(frame->rows * 0.78), frame->cols, frame->rows},
            {35, 35, 35},
            cv::FILLED
        );
    }
}

void test_robust_visual_shift_ignores_static_overlay_and_blank_tile() {
    cv::Mat before = make_textured_calibration_scene({960, 540});
    cv::Mat after = translated_wrap(before, 18.0, -7.0);
    paint_static_calibration_overlay(before, after);
    cv::rectangle(before, {0, 0, 220, 170}, {64, 64, 64}, cv::FILLED);
    cv::rectangle(after, {0, 0, 220, 170}, {64, 64, 64}, cv::FILLED);

    const VisualShiftEstimate estimate = estimate_robust_visual_shift(before, after);

    require(estimate.coherent,
            "agreeing textured tiles should produce a coherent estimate");
    require_near(static_cast<float>(estimate.shift.x), 18.0F, 1.5F,
                 "robust estimator should recover background X translation");
    require_near(static_cast<float>(estimate.shift.y), -7.0F, 1.5F,
                 "robust estimator should recover background Y translation");
}

void test_axis_calibration_shift_recovers_from_static_hud_zero_cluster() {
    const cv::Size viewport_size{1920, 1080};
    const cv::Mat panorama = make_textured_calibration_scene({2240, 1240});
    cv::Mat before = camera_crop(panorama, viewport_size, 80, 80);
    cv::Mat after = yaw_camera_view(before, 12.0);
    paint_static_game_hud(before);
    paint_static_game_hud(after);

    const VisualShiftEstimate whole_frame =
        estimate_visual_shift_with_response(before, after);
    const VisualShiftEstimate tiled = estimate_robust_visual_shift(before, after);
    const VisualShiftEstimate recovered =
        estimate_calibration_axis_shift(before, after, 0);

    require(std::abs(whole_frame.shift.x) >= 8.0,
            "whole-frame reference must contain visible camera movement");
    require(tiled.coherent,
            "static HUD zero candidates reproduce a coherent but wrong tile cluster");
    require(std::abs(tiled.shift.x) < 1.0,
            "test scene must reproduce the observed static-HUD zero lock");
    require(recovered.coherent && std::abs(recovered.shift.x) >= 8.0,
            "axis calibration must recover visible camera movement from its reference");
}

void test_center_flow_measures_scene_around_fixed_crosshair() {
    const cv::Size viewport_size{1920, 1080};
    const cv::Mat panorama = make_textured_calibration_scene({2240, 1240});
    cv::Mat before = camera_crop(panorama, viewport_size, 80, 80);
    cv::Mat after = yaw_camera_view(before, 12.0);
    paint_static_game_hud(before);
    paint_static_game_hud(after);

    const CenterFlowEstimate estimate = estimate_center_flow(before, after);

    require(estimate.reliable,
            "center scene motion should be measurable around a fixed crosshair");
    require_near(static_cast<float>(estimate.shift.x), -12.0F, 2.5F,
                 "center flow should recover camera X movement");
    require(std::abs(estimate.shift.y) < 1.5,
            "fixed crosshair must not create cross-axis flow");
    require(estimate.inlier_features >= 12 && estimate.occupied_cells >= 3,
            "camera movement needs distributed center-scene support");
}

void test_center_flow_rejects_blank_center() {
    const cv::Mat before(1080, 1920, CV_8UC3, cv::Scalar(32, 32, 32));
    const cv::Mat after = before.clone();

    const CenterFlowEstimate estimate = estimate_center_flow(before, after);

    require(!estimate.reliable && estimate.detected_features < 12,
            "blank center must report insufficient texture");
}

void test_center_flow_rejects_one_spatial_cell() {
    cv::Mat before(1080, 1920, CV_8UC3, cv::Scalar(32, 32, 32));
    const cv::Mat texture = make_textured_calibration_scene({120, 90});
    texture.copyTo(before(cv::Rect(665, 325, texture.cols, texture.rows)));
    cv::Mat after;
    const cv::Mat transform = (cv::Mat_<double>(2, 3) <<
        1.0, 0.0, -12.0,
        0.0, 1.0, 0.0);
    cv::warpAffine(
        before,
        after,
        transform,
        before.size(),
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar(32, 32, 32)
    );

    const CenterFlowEstimate estimate = estimate_center_flow(before, after);

    require(!estimate.reliable,
            "one spatial cell must not determine camera movement");
    require(estimate.inlier_features >= 12 && estimate.occupied_cells < 3,
            "one-cell rejection should preserve useful support diagnostics");
}

void test_center_flow_burst_selects_most_supported_candidate() {
    const std::size_t selected = select_center_flow_candidate({
        CenterFlowEstimate{{0.0, 0.0}, 80, 60, 0, 0, 0.0, false},
        CenterFlowEstimate{{-11.8, 0.2}, 90, 70, 38, 7, 0.6, true},
        CenterFlowEstimate{{-12.1, 0.1}, 90, 72, 29, 6, 0.3, true},
    });

    require(selected == 1,
            "burst selection should prefer more distributed reliable inliers");
}

void test_center_flow_burst_rejects_without_reliable_candidate() {
    bool rejected = false;
    try {
        (void)select_center_flow_candidate({
            CenterFlowEstimate{{0.0, 0.0}, 0, 0, 0, 0, 0.0, false},
            CenterFlowEstimate{{1.0, 0.0}, 10, 8, 8, 1, 0.2, false},
        });
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("center flow") != std::string::npos;
    }
    require(rejected,
            "burst selection must reject when no center flow is reliable");
}

void test_robust_visual_shift_requires_multiple_textured_regions() {
    cv::Mat before(540, 960, CV_8UC3, cv::Scalar(32, 32, 32));
    cv::Mat after = before.clone();

    const VisualShiftEstimate estimate = estimate_robust_visual_shift(before, after);

    require(!estimate.coherent,
            "uniform frames should not fabricate a coherent visual movement");
}

void test_robust_visual_shift_rejects_one_isolated_textured_region() {
    cv::Mat before(540, 960, CV_8UC3, cv::Scalar(32, 32, 32));
    cv::Mat texture = make_textured_calibration_scene({140, 90});
    texture.copyTo(before(cv::Rect(10, 10, texture.cols, texture.rows)));
    const cv::Mat after = translated_wrap(before, 12.0, -4.0);

    const VisualShiftEstimate estimate = estimate_robust_visual_shift(before, after);

    require(!estimate.coherent,
            "one isolated textured region must not satisfy multi-tile agreement");
}

void test_visual_shift_estimate_preserves_phase_response() {
    cv::Mat image(64, 64, CV_32F);
    cv::randu(image, 0.0F, 255.0F);
    const auto estimate = estimate_visual_shift_with_response(image, image);
    require_near(static_cast<float>(estimate.shift.x), 0.0F, 0.01F,
                 "identical frames should have no x shift");
    require_near(static_cast<float>(estimate.shift.y), 0.0F, 0.01F,
                 "identical frames should have no y shift");
    require(estimate.response > 0.90, "identical frames should have a strong phase response");
}

void test_runtime_config_file_overrides_tuning_and_io() {
    const auto path = std::filesystem::temp_directory_path() / "vision_analyzer_runtime.cfg";
    {
        std::ofstream output(path);
        output << "input=dxgi\n"
               << "dxgi_adapter=1\n"
               << "dxgi_output=2\n"
               << "dxgi_gpu_preference=minimum-power\n"
               << "dxgi_debug=true\n"
               << "dxgi_roi_x=100\n"
               << "dxgi_roi_y=50\n"
               << "dxgi_roi_width=800\n"
               << "dxgi_roi_height=600\n"
               << "tensorrt_cache_path=cache-from-config\n"
               << "output_enabled=true\n"
               << "fire_enabled=true\n"
               << "body_fire_enabled=true\n"
               << "head_fire_confidence=0.35\n"
               << "body_fire_confidence=0.45\n"
               << "hid_click_cooldown_frames=3\n"
               << "hid_gain=0.5\n"
               << "hid_deadzone_px=2.5\n"
               << "body_head_anchor_ratio=0.22\n"
               << "kalman_process_noise=0.11\n"
               << "kalman_measurement_noise=5.5\n"
               << "kalman_error_covariance=7.5\n"
               << "action_log=actions.txt\n";
    }

    Options options;
    apply_runtime_config_file(options, path.string());
    std::filesystem::remove(path);

    require(options.input_source == InputSource::Dxgi, "config should set DXGI input");
    require(options.dxgi_adapter == 1 && options.dxgi_output == 2, "config should set DXGI adapter/output");
    require(options.dxgi_gpu_preference == DxgiGpuPreference::MinimumPower,
            "config should set DXGI GPU preference");
    require(options.dxgi_debug, "config should set DXGI debug flag");
    require(
        options.dxgi_roi.x == 100 && options.dxgi_roi.y == 50 &&
        options.dxgi_roi.width == 800 && options.dxgi_roi.height == 600,
        "config should set DXGI ROI"
    );
    require(options.tensorrt_cache_path == "cache-from-config", "config should set TensorRT cache path");
    require(options.output_enabled, "config should enable RP2350 output");
    require(options.fire_enabled, "config should enable automatic fire planning");
    require(options.fire_policy.body_enabled, "config should enable body fallback fire");
    require_near(options.fire_policy.head_confidence, 0.35F, 0.001F,
                 "config should set head fire confidence");
    require_near(options.fire_policy.body_confidence, 0.45F, 0.001F,
                 "config should set body fire confidence");
    require(options.fire_policy.cooldown_frames == 3, "config should set fire cooldown");
    require_near(options.hid_move_gain, 0.5F, 0.001F, "config should set HID gain");
    require_near(options.hid_deadzone_px, 2.5F, 0.001F, "config should set HID deadzone");
    require_near(options.tuning.body_head_anchor_ratio, 0.22F, 0.001F, "config should set body anchor ratio");
    require_near(options.tuning.kalman_process_noise, 0.11F, 0.001F, "config should set process noise");
    require_near(options.tuning.kalman_measurement_noise, 5.5F, 0.001F, "config should set measurement noise");
    require_near(options.tuning.kalman_error_covariance, 7.5F, 0.001F, "config should set error covariance");
    require(options.action_log_path == "actions.txt", "config should set action log path");
}

void test_runtime_options_reject_invalid_fire_policy() {
    Options options;
    options.dry_run = true;
    options.fire_policy.head_confidence = 1.01F;
    bool rejected = false;
    try {
        validate_options(options);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "runtime options should reject fire confidence above one");
}

void test_aim_controller_holds_when_no_target() {
    AimController controller;
    FrameReport report{};

    const AimCommand command = controller.plan(report);

    require(!command.has_target, "aim command should hold when no target exists");
    require(command.dx == 0, "aim command should not move x without target");
    require(command.dy == 0, "aim command should not move y without target");
    require(!command.click_left, "aim command should not click without target");
}

void test_aim_controller_respects_click_cooldown() {
    AimControllerOptions options;
    options.fire_enabled = true;
    options.fire_policy.cooldown_frames = 2;
    AimController controller(options);

    FrameReport report{
        1,
        33.0,
        120.0,
        InferenceTiming{},
        1,
        TargetFrame{
            3,
            Detection{1, "ct_head", 0.95F, cv::Rect(950, 520, 40, 40)},
            {970.0F, 540.0F},
            {960.0F, 540.0F},
            {0.0F, 0.0F},
            0.0F,
            false,
            {960.0F, 540.0F},
            {0.0F, 0.0F},
            {0.0F, 0.0F},
            0.90F,
            0.0F,
            LockState::Locked,
            true,
        },
    };

    const AimCommand first = controller.plan(report);
    const AimCommand second = controller.plan(report);
    const AimCommand third = controller.plan(report);

    require(first.click_left, "first fire candidate should click");
    require(!second.click_left, "second frame should be blocked by cooldown");
    require(!third.click_left, "third frame should still be blocked while cooldown counts down");
}

HidCalibrationProfile make_valid_hid_profile() {
    HidCalibrationProfile profile;
    profile.valid = true;
    profile.frame_width = 1920;
    profile.frame_height = 1080;
    profile.x = {{8.0F, 32.0F, 96.0F}, {2.0F, 3.0F, 4.0F}};
    profile.y = {{8.0F, 32.0F, 96.0F}, {1.0F, 1.5F, 2.0F}};
    profile.deadzone_px = 1.0F;
    profile.max_step = 120;
    return profile;
}

HidCalibrationProfile make_persistable_hid_profile() {
    HidCalibrationProfile profile;
    profile.valid = true;
    profile.frame_width = 1920;
    profile.frame_height = 1080;
    profile.x.shift_px = {8.05F, 23.53F, 46.58F};
    profile.x.counts_per_pixel = {1.37F, 1.40F, 1.42F};
    profile.y.shift_px = {7.93F, 24.20F, 47.15F};
    profile.y.counts_per_pixel = {1.39F, 1.41F, 1.42F};
    profile.deadzone_px = 1.0F;
    profile.max_step = 120;
    profile.noise_px = 0.009F;
    profile.quality = 0.678F;
    profile.accepted_samples = 24;
    return profile;
}

std::filesystem::path calibration_store_test_directory() {
    return std::filesystem::temp_directory_path() /
           "vision-analyzer-hid-calibration-store-tests";
}

void write_calibration_json(
    const std::filesystem::path& path,
    int schema_version,
    int max_step
) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << "{\n"
           << "  \"schema_version\": " << schema_version << ",\n"
           << "  \"frame_width\": 1920,\n"
           << "  \"frame_height\": 1080,\n"
           << "  \"x_shift_px\": [8.05, 23.53, 46.58],\n"
           << "  \"x_counts_per_pixel\": [1.37, 1.40, 1.42],\n"
           << "  \"y_shift_px\": [7.93, 24.20, 47.15],\n"
           << "  \"y_counts_per_pixel\": [1.39, 1.41, 1.42],\n"
           << "  \"deadzone_px\": 1.0,\n"
           << "  \"max_step\": " << max_step << ",\n"
           << "  \"noise_px\": 0.009,\n"
           << "  \"quality\": 0.678,\n"
           << "  \"accepted_samples\": 24\n"
           << "}\n";
}

void test_hid_calibration_profile_validation_is_strict() {
    HidCalibrationProfile profile = make_persistable_hid_profile();
    require(valid_hid_calibration_profile(profile),
            "complete fitted profile should validate");

    profile.x.shift_px[1] = profile.x.shift_px[0];
    require(!valid_hid_calibration_profile(profile),
            "non-increasing shift knots should fail validation");
    profile = make_persistable_hid_profile();
    profile.y.counts_per_pixel[1] = -profile.y.counts_per_pixel[1];
    require(!valid_hid_calibration_profile(profile),
            "mixed gain signs should fail validation");
    profile = make_persistable_hid_profile();
    profile.max_step = 121;
    require(!valid_hid_calibration_profile(profile),
            "persisted runtime step above 120 should fail validation");
    profile = make_persistable_hid_profile();
    profile.quality = 0.54F;
    require(!valid_hid_calibration_profile(profile),
            "profile below fitter quality threshold should fail validation");
    profile = make_persistable_hid_profile();
    profile.accepted_samples = 11;
    require(!valid_hid_calibration_profile(profile),
            "profile with fewer than twelve movement samples should fail validation");
}

void test_hid_calibration_store_round_trips_every_field() {
    const auto directory = calibration_store_test_directory();
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const auto path = directory / "profile.json";
    const HidCalibrationProfile expected = make_persistable_hid_profile();

    save_hid_calibration_profile_atomic(path, expected);
    const HidCalibrationProfile actual = load_hid_calibration_profile(path);

    require(actual.valid, "loaded profile should remain valid");
    require(actual.frame_width == expected.frame_width &&
            actual.frame_height == expected.frame_height,
            "loaded profile should retain frame dimensions");
    require(actual.x.shift_px == expected.x.shift_px &&
            actual.x.counts_per_pixel == expected.x.counts_per_pixel &&
            actual.y.shift_px == expected.y.shift_px &&
            actual.y.counts_per_pixel == expected.y.counts_per_pixel,
            "loaded profile should retain every curve value");
    require_near(actual.deadzone_px, expected.deadzone_px, 0.0001F,
                 "loaded profile should retain deadzone");
    require_near(actual.noise_px, expected.noise_px, 0.0001F,
                 "loaded profile should retain noise");
    require_near(actual.quality, expected.quality, 0.0001F,
                 "loaded profile should retain quality");
    require(actual.max_step == expected.max_step &&
            actual.accepted_samples == expected.accepted_samples,
            "loaded profile should retain safety and sample fields");
    std::filesystem::remove_all(directory);
}

void test_hid_calibration_store_rejects_corrupt_and_incompatible_json() {
    const auto directory = calibration_store_test_directory();
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);

    const auto corrupt = directory / "corrupt.json";
    {
        std::ofstream output(corrupt, std::ios::binary | std::ios::trunc);
        output << "{broken";
    }
    bool corrupt_rejected = false;
    try {
        (void)load_hid_calibration_profile(corrupt);
    } catch (const std::exception&) {
        corrupt_rejected = true;
    }
    require(corrupt_rejected, "truncated JSON should be rejected");

    const auto incompatible = directory / "incompatible.json";
    write_calibration_json(incompatible, 2, 120);
    bool schema_rejected = false;
    try {
        (void)load_hid_calibration_profile(incompatible);
    } catch (const std::exception& error) {
        schema_rejected = std::string(error.what()).find("schema_version") !=
                          std::string::npos;
    }
    require(schema_rejected, "incompatible schema should name schema_version");

    const auto unsafe = directory / "unsafe.json";
    write_calibration_json(unsafe, 1, 121);
    bool unsafe_rejected = false;
    try {
        (void)load_hid_calibration_profile(unsafe);
    } catch (const std::exception&) {
        unsafe_rejected = true;
    }
    require(unsafe_rejected, "profile with runtime max step above 120 should be rejected");
    std::filesystem::remove_all(directory);
}

void test_hid_calibration_atomic_save_preserves_old_destination_on_failure() {
    const auto directory = calibration_store_test_directory();
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const auto path = directory / "profile.json";
    const HidCalibrationProfile old_profile = make_persistable_hid_profile();
    save_hid_calibration_profile_atomic(path, old_profile);

    HidCalibrationProfile invalid_candidate = old_profile;
    invalid_candidate.max_step = 121;
    bool rejected = false;
    try {
        save_hid_calibration_profile_atomic(path, invalid_candidate);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "invalid save candidate should be rejected");
    const HidCalibrationProfile loaded = load_hid_calibration_profile(path);
    require(loaded.max_step == 120 && loaded.quality == old_profile.quality,
            "failed save should preserve the complete old destination");
    std::filesystem::remove_all(directory);
}

FrameReport make_target_report(Detection detection, cv::Point2f offset) {
    TargetFrame target{};
    target.detection = std::move(detection);
    target.offset = offset;
    target.analysis_point = {960.0F + offset.x, 540.0F + offset.y};
    target.lock_state = LockState::Acquiring;
    return FrameReport{1, 16.0, 60.0, InferenceTiming{}, 1, target};
}

FrameReport make_body_report(cv::Point2f offset, float confidence = 0.80F) {
    return make_target_report(
        Detection{0, "ct_body", confidence, cv::Rect(900, 450, 120, 180)},
        offset
    );
}

void test_aim_controller_uses_calibrated_axes() {
    AimControllerOptions options;
    options.calibration = make_valid_hid_profile();
    AimController controller(options);
    const AimCommand command = controller.plan(make_target_report(
        Detection{1, "ct_head", 0.90F, cv::Rect(970, 520, 40, 40)},
        cv::Point2f{20.0F, -20.0F}
    ));
    require(command.dx != command.dy, "per-axis curves should produce independent steps");
}

void test_head_fires_on_first_frame_inside_box() {
    AimController controller;
    controller.set_fire_policy(FirePolicy{true, 0.35F, 0.45F, 3});
    controller.set_fire_enabled(true);
    const AimCommand command = controller.plan(make_target_report(
        Detection{1, "ct_head", 0.80F, cv::Rect(940, 520, 40, 40)},
        cv::Point2f{0.0F, 0.0F}
    ));
    require(command.click_left, "head box should fire on its first qualifying frame");
}

void test_body_fires_only_in_torso_when_enabled() {
    AimController controller;
    controller.set_fire_policy(FirePolicy{true, 0.35F, 0.45F, 3});
    controller.set_fire_enabled(true);
    require(controller.plan(make_body_report(cv::Point2f{0.0F, 0.0F})).click_left,
            "centered torso should fire");
    controller.set_fire_enabled(false);
    require(!controller.plan(make_body_report(cv::Point2f{0.0F, 0.0F})).click_left,
            "fire gate should suppress body clicks");
}

void test_body_does_not_fire_outside_torso_region() {
    AimController controller;
    controller.set_fire_policy(FirePolicy{true, 0.35F, 0.45F, 3});
    controller.set_fire_enabled(true);
    const auto report = make_target_report(
        Detection{0, "ct_body", 0.80F, cv::Rect(700, 450, 120, 180)},
        cv::Point2f{0.0F, 0.0F}
    );
    require(!controller.plan(report).click_left,
            "body should not fire when the crosshair is outside the torso");
}

void test_fire_policy_enforces_body_flag_and_class_confidence() {
    AimController controller;
    controller.set_fire_enabled(true);
    controller.set_fire_policy(FirePolicy{false, 0.35F, 0.45F, 3});
    require(!controller.plan(make_body_report({0.0F, 0.0F})).click_left,
            "body-disabled policy must suppress a centered torso");
    controller.set_fire_policy(FirePolicy{true, 0.35F, 0.85F, 3});
    require(!controller.plan(make_body_report({0.0F, 0.0F}, 0.80F)).click_left,
            "body confidence below policy threshold must not fire");
}

void test_fire_cooldown_and_disable_reset() {
    AimController controller;
    controller.set_fire_policy(FirePolicy{true, 0.35F, 0.45F, 3});
    controller.set_fire_enabled(true);
    const FrameReport report = make_body_report({0.0F, 0.0F});
    require(controller.plan(report).click_left, "first torso frame should fire");
    require(!controller.plan(report).click_left, "cooldown should suppress next frame");
    controller.set_fire_enabled(false);
    controller.set_fire_enabled(true);
    require(controller.plan(report).click_left, "disabling fire should clear cooldown");
}

void test_rp2350_v2_health_accepts_required_capabilities() {
    const std::vector<std::uint8_t> info{2, 0, 240, 3};
    const std::vector<std::uint8_t> caps{2, 0, 240, 0, 0x7F, 1, 1, 0, 8, 20};
    const HidDeviceHealth health = parse_rp2350_v2_health(info, caps);

    require(health.protocol_version == 2, "health must report protocol v2");
    require(health.capabilities == 0x7F, "health must preserve capability bits");
}

void test_rp2350_v2_health_rejects_legacy_or_incomplete_devices() {
    bool rejected_legacy = false;
    try {
        (void)parse_rp2350_v2_health({1, 0, 240, 3}, {1, 0, 240, 0, 0x0F});
    } catch (const std::runtime_error& error) {
        rejected_legacy = std::string(error.what()).find("protocol v2") != std::string::npos;
    }
    require(rejected_legacy, "legacy firmware must be rejected with a v2 error");

    bool rejected_caps = false;
    try {
        (void)parse_rp2350_v2_health({2, 0, 240, 3}, {2, 0, 240, 0, 0x02});
    } catch (const std::runtime_error&) {
        rejected_caps = true;
    }
    require(
        rejected_caps,
        "firmware without retry, lease, and cancellation must be rejected"
    );
}

class RecordingHidClient final : public HidClient {
public:
    void move_relative(std::int16_t dx, std::int16_t dy) override {
        moves.push_back({dx, dy});
    }

    void click_left() override {
        ++left_clicks;
    }

    void stop_all() override {
        ++stop_calls;
        if (throw_on_stop) {
            throw std::runtime_error("simulated stop failure");
        }
    }

    void close() noexcept override {
        ++close_calls;
    }

    std::vector<std::pair<std::int16_t, std::int16_t>> moves;
    int left_clicks = 0;
    int stop_calls = 0;
    int close_calls = 0;
    bool throw_on_stop = false;
};

void test_hid_close_continues_after_stop_failure() {
    RecordingHidClient client;
    client.throw_on_stop = true;

    close_hid_client_noexcept(&client);

    require(client.stop_calls == 1, "shutdown must attempt STOP_ALL once");
    require(client.close_calls == 1, "shutdown must close after STOP_ALL failure");
}

void test_hid_action_sender_requires_arming_and_stops_when_disarmed() {
    RecordingHidClient client;
    HidActionSender sender(client);
    const AimCommand command{
        true,
        12,
        -5,
        true,
        LockState::Locked,
    };

    sender.execute(command);
    require(client.moves.empty(), "disarmed sender must suppress movement");
    require(client.left_clicks == 0, "disarmed sender must suppress clicks");

    sender.set_enabled(true);
    sender.execute(command);

    require(client.moves.size() == 1, "HID sender should emit one relative move");
    require(client.moves[0].first == 12, "HID sender should forward x movement");
    require(client.moves[0].second == -5, "HID sender should forward y movement");
    require(client.left_clicks == 1, "HID sender should forward click command");

    sender.set_enabled(false);
    require(client.stop_calls == 1, "disarming should immediately stop RP2350 state");
}

void test_runtime_session_starts_closed() {
    RuntimeSession session;

    require(!session.is_open(), "runtime session should start closed");
    require(session.processed_frames() == 0, "closed runtime session should report zero processed frames");
    require(session.detector_name().empty(), "closed runtime session should not report a detector");
    require(session.input_name().empty(), "closed runtime session should not report an input");
}

void test_calibrated_hid_curve_interpolates_signed_gain() {
    HidCalibrationAxisCurve curve{
        {8.0F, 32.0F, 96.0F},
        {2.0F, 3.0F, 4.0F},
    };
    require(calibrated_hid_step(20.0F, curve, 120, 1.0F) == 50,
            "20 px should interpolate to gain 2.5");
    require(calibrated_hid_step(-20.0F, curve, 120, 1.0F) == -50,
            "signed errors should preserve direction");
}

void test_calibrated_hid_curve_supports_inverted_axis_deadzone_and_clamp() {
    HidCalibrationAxisCurve inverted{
        {8.0F, 32.0F, 96.0F},
        {-2.0F, -3.0F, -4.0F},
    };
    require(calibrated_hid_step(20.0F, inverted, 120, 1.0F) == -50,
            "negative gain should support an inverted axis");
    require(calibrated_hid_step(0.5F, inverted, 120, 1.0F) == 0,
            "deadzone should suppress noise");
    require(calibrated_hid_step(200.0F, inverted, 120, 1.0F) == -120,
            "large movement should clamp to max_step");
}

void test_runtime_status_includes_parseable_timing_metrics() {
    RuntimeStepResult step;
    step.frame_available = true;
    step.report.frame_index = 7;
    step.report.fps = 123.456;
    step.report.timing = InferenceTiming{1.25, 4.50, 0.75};
    step.report.detection_count = 3;
    step.command = AimCommand{true, -4, 5, false, LockState::Tracking};

    const std::string status = format_runtime_status(step);
    require(status.find("frame=7") != std::string::npos, "runtime status should include frame index");
    require(status.find("fps=123.46") != std::string::npos, "runtime status should include fixed FPS");
    require(status.find("preprocess_ms=1.25") != std::string::npos, "runtime status should include preprocessing time");
    require(status.find("inference_ms=4.50") != std::string::npos, "runtime status should include inference time");
    require(status.find("postprocess_ms=0.75") != std::string::npos, "runtime status should include postprocessing time");
    require(status.find("total_ms=6.50") != std::string::npos, "runtime status should include total processing time");
    require(status.find("det=3 target=1 dx=-4 dy=5 click=0 lock=tracking") != std::string::npos,
            "runtime status should preserve action fields");
}

}  // namespace

int main() {
    try {
        test_class_aware_nms_keeps_overlapping_different_classes();
        test_enemy_filter_keeps_opposing_side_only();
        test_model_class_schema_rejects_wrong_output_dimensions();
        test_model_input_accepts_static_fp32_nchw();
        test_model_input_rejects_dynamic_or_non_fp32_input();
        test_sm61_tensorrt_profile_is_fp32_and_cached();
        test_runtime_defaults_to_sm61_tensorrt();
        test_letterbox_accepts_rectangular_target();
        test_dxgi_copy_region_uses_full_frame_when_roi_is_disabled();
        test_dxgi_copy_region_preserves_or_clips_requested_roi();
        test_dxgi_copy_region_rejects_empty_intersection();
        test_model_schema_file_validates_class_order();
        test_model_schema_file_rejects_wrong_class_order();
        test_live_schema_validation_requires_schema_file();
        test_decode_yolo_output_accepts_channels_last_shape();
        test_input_source_parser_accepts_video_and_dxgi();
        test_track_manager_keeps_id_for_small_motion();
        test_target_selector_prefers_active_track_when_scores_are_close();
        test_target_selector_switches_when_challenger_is_clearly_better();
        test_target_selector_prefers_head_over_comparable_body();
        test_track_manager_smooths_velocity_spikes();
        test_target_anchor_point_uses_body_top_fallback();
        test_fuse_head_body_detections_suppresses_body_when_head_matches();
        test_track_manager_uses_configured_body_anchor_ratio();
        test_analysis_state_predicts_latency_in_frame_units();
        test_analysis_state_offsets_from_filtered_analysis_point();
        test_motion_filter_is_stable_and_moves_toward_measurement();
        test_motion_filter_predicts_with_kalman_velocity();
        test_analysis_state_never_fires_for_body_class();
        test_analysis_state_aims_body_detection_at_head_anchor();
        test_aim_controller_scales_and_clamps_target_offset();
        test_aim_controller_deadzone_suppresses_tiny_steps();
        test_hid_calibration_fit_generates_tuning_values();
        test_adaptive_calibration_fits_signed_axes_and_inverted_y();
        test_adaptive_calibration_ignores_incoherent_noise_measurements();
        test_adaptive_calibration_rejects_bad_response_and_cross_axis_motion();
        test_calibration_probe_adjustment_uses_calibration_only_limit();
        test_calibration_level_plan_compresses_low_sensitivity_range();
        test_calibration_level_plan_rejects_unmeasurable_range();
        test_calibration_probe_planner_escalates_low_response();
        test_calibration_probe_planner_accepts_signal_and_rejects_cross_axis();
        test_calibration_axis_discovery_derives_low_sensitivity_levels();
        test_calibration_axis_discovery_retries_same_count_before_escalating();
        test_calibration_axis_discovery_reports_probe_exhaustion();
        test_round_trip_estimation_measures_outward_and_inverse_frames();
        test_round_trip_command_plan_alternates_without_final_escalation();
        test_round_trip_samples_assign_real_signed_legs();
        test_round_trip_sampling_can_recover_from_one_bad_path();
        test_calibration_level_selection_falls_back_strictly_downward();
        test_calibration_level_selection_never_shrinks_the_low_level();
        test_calibration_level_selection_has_bounded_distinct_midpoints();
        test_calibration_round_trip_usability_requires_both_opposite_legs();
        test_recovered_level_measurements_fit_three_increasing_knots();
        test_robust_visual_shift_ignores_static_overlay_and_blank_tile();
        test_axis_calibration_shift_recovers_from_static_hud_zero_cluster();
        test_center_flow_measures_scene_around_fixed_crosshair();
        test_center_flow_rejects_blank_center();
        test_center_flow_rejects_one_spatial_cell();
        test_center_flow_burst_selects_most_supported_candidate();
        test_center_flow_burst_rejects_without_reliable_candidate();
        test_robust_visual_shift_requires_multiple_textured_regions();
        test_robust_visual_shift_rejects_one_isolated_textured_region();
        test_visual_shift_estimate_preserves_phase_response();
        test_runtime_config_file_overrides_tuning_and_io();
        test_runtime_options_reject_invalid_fire_policy();
        test_calibrated_hid_curve_interpolates_signed_gain();
        test_calibrated_hid_curve_supports_inverted_axis_deadzone_and_clamp();
        test_aim_controller_holds_when_no_target();
        test_aim_controller_respects_click_cooldown();
        test_hid_calibration_profile_validation_is_strict();
        test_hid_calibration_store_round_trips_every_field();
        test_hid_calibration_store_rejects_corrupt_and_incompatible_json();
        test_hid_calibration_atomic_save_preserves_old_destination_on_failure();
        test_aim_controller_uses_calibrated_axes();
        test_head_fires_on_first_frame_inside_box();
        test_body_fires_only_in_torso_when_enabled();
        test_body_does_not_fire_outside_torso_region();
        test_fire_policy_enforces_body_flag_and_class_confidence();
        test_fire_cooldown_and_disable_reset();
        test_rp2350_v2_health_accepts_required_capabilities();
        test_rp2350_v2_health_rejects_legacy_or_incomplete_devices();
        test_hid_close_continues_after_stop_failure();
        test_hid_action_sender_requires_arming_and_stops_when_disarmed();
        test_runtime_session_starts_closed();
        test_runtime_status_includes_parseable_timing_metrics();
        std::cout << "algorithm tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "algorithm test failed: " << error.what() << '\n';
        return 1;
    }
}
