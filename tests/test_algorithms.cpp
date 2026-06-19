#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vision_analyzer/aim_controller.hpp"
#include "vision_analyzer/calibration.hpp"
#include "vision_analyzer/hid_output.hpp"
#include "vision_analyzer/model_schema.hpp"
#include "vision_analyzer/postprocess.hpp"
#include "vision_analyzer/runtime_config.hpp"
#include "vision_analyzer/tracking.hpp"

using namespace vision_analyzer;

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
    options.click_enabled = false;
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
    require_near(options.hid_move_gain, 0.5F, 0.001F, "config should set HID gain");
    require_near(options.hid_deadzone_px, 2.5F, 0.001F, "config should set HID deadzone");
    require_near(options.tuning.body_head_anchor_ratio, 0.22F, 0.001F, "config should set body anchor ratio");
    require_near(options.tuning.kalman_process_noise, 0.11F, 0.001F, "config should set process noise");
    require_near(options.tuning.kalman_measurement_noise, 5.5F, 0.001F, "config should set measurement noise");
    require_near(options.tuning.kalman_error_covariance, 7.5F, 0.001F, "config should set error covariance");
    require(options.action_log_path == "actions.txt", "config should set action log path");
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
    options.click_enabled = true;
    options.click_cooldown_frames = 2;
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

class RecordingHidClient final : public HidClient {
public:
    void move_relative(std::int16_t dx, std::int16_t dy) override {
        moves.push_back({dx, dy});
    }

    void click_left() override {
        ++left_clicks;
    }

    void stop_all() override {
        stopped = true;
    }

    std::vector<std::pair<std::int16_t, std::int16_t>> moves;
    int left_clicks = 0;
    bool stopped = false;
};

void test_hid_action_sender_executes_aim_command() {
    RecordingHidClient client;
    HidActionSender sender(client);

    sender.execute(AimCommand{
        true,
        12,
        -5,
        true,
        LockState::Locked,
    });

    require(client.moves.size() == 1, "HID sender should emit one relative move");
    require(client.moves[0].first == 12, "HID sender should forward x movement");
    require(client.moves[0].second == -5, "HID sender should forward y movement");
    require(client.left_clicks == 1, "HID sender should forward click command");
}

}  // namespace

int main() {
    try {
        test_class_aware_nms_keeps_overlapping_different_classes();
        test_enemy_filter_keeps_opposing_side_only();
        test_model_class_schema_rejects_wrong_output_dimensions();
        test_model_schema_file_validates_class_order();
        test_model_schema_file_rejects_wrong_class_order();
        test_live_schema_validation_requires_schema_file();
        test_decode_yolo_output_accepts_channels_last_shape();
        test_input_source_parser_accepts_video_and_dxgi();
        test_track_manager_keeps_id_for_small_motion();
        test_target_selector_prefers_active_track_when_scores_are_close();
        test_target_selector_switches_when_challenger_is_clearly_better();
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
        test_runtime_config_file_overrides_tuning_and_io();
        test_aim_controller_holds_when_no_target();
        test_aim_controller_respects_click_cooldown();
        test_hid_action_sender_executes_aim_command();
        std::cout << "algorithm tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "algorithm test failed: " << error.what() << '\n';
        return 1;
    }
}
