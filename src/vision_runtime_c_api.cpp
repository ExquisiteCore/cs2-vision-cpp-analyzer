#include "vision_analyzer/vision_runtime_c_api.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>

#include "vision_analyzer/calibration.hpp"
#include "vision_analyzer/detector.hpp"
#include "vision_analyzer/hid_calibration_store.hpp"
#include "vision_analyzer/runtime.hpp"
#include "vision_analyzer/runtime_config.hpp"
#include "vision_analyzer/runtime_session.hpp"

struct VaRuntime {
    vision_analyzer::Options options;
    vision_analyzer::RuntimeSession session;
    std::filesystem::path hid_calibration_path;
    std::string last_error;
};

namespace {

constexpr const char* kNullRuntimeError = "runtime handle is null";

template <typename Function>
int32_t call_api(VaRuntime* runtime, Function&& function) {
    if (runtime == nullptr) {
        return -1;
    }
    try {
        runtime->last_error.clear();
        function();
        return 0;
    } catch (const std::exception& error) {
        runtime->last_error = error.what();
        return -1;
    } catch (...) {
        runtime->last_error = "unknown runtime error";
        return -1;
    }
}

std::string required_string(const char* value, const char* name) {
    if (value == nullptr || value[0] == '\0') {
        throw std::runtime_error(std::string(name) + " must not be empty");
    }
    return value;
}

int32_t lock_state_to_int(vision_analyzer::LockState state) {
    return static_cast<int32_t>(state);
}

void apply_fire_enabled(VaRuntime* runtime, bool enabled) {
    runtime->options.fire_enabled = enabled;
    if (runtime->session.is_open()) {
        runtime->session.set_fire_enabled(enabled);
    }
}

void validate_fire_policy(const vision_analyzer::FirePolicy& policy) {
    if (!std::isfinite(policy.head_confidence) || policy.head_confidence < 0.0F ||
        policy.head_confidence > 1.0F) {
        throw std::runtime_error("head fire confidence must be finite and in [0, 1]");
    }
    if (!std::isfinite(policy.body_confidence) || policy.body_confidence < 0.0F ||
        policy.body_confidence > 1.0F) {
        throw std::runtime_error("body fire confidence must be finite and in [0, 1]");
    }
    if (policy.cooldown_frames < 0) {
        throw std::runtime_error("fire cooldown frames must be greater than or equal to 0");
    }
}

void apply_fire_policy(VaRuntime* runtime, vision_analyzer::FirePolicy policy) {
    validate_fire_policy(policy);
    runtime->options.fire_policy = policy;
    if (runtime->session.is_open()) {
        runtime->session.set_fire_policy(policy);
    }
}

void fill_calibration_profile(
    const vision_analyzer::HidCalibrationProfile& source,
    VaHidCalibrationProfile* destination
) {
    *destination = VaHidCalibrationProfile{};
    destination->schema_version = 1;
    destination->valid = source.valid ? 1 : 0;
    destination->frame_width = source.frame_width;
    destination->frame_height = source.frame_height;
    for (std::size_t level = 0; level < vision_analyzer::kHidCalibrationLevels; ++level) {
        destination->x_shift_px[level] = source.x.shift_px[level];
        destination->x_counts_per_pixel[level] = source.x.counts_per_pixel[level];
        destination->y_shift_px[level] = source.y.shift_px[level];
        destination->y_counts_per_pixel[level] = source.y.counts_per_pixel[level];
    }
    destination->deadzone_px = source.deadzone_px;
    destination->max_step = source.max_step;
    destination->noise_px = source.noise_px;
    destination->quality = source.quality;
    destination->accepted_samples = source.accepted_samples;
}

void fill_action(const vision_analyzer::RuntimeStepResult& step, VaRuntimeAction* action) {
    if (action == nullptr) {
        throw std::runtime_error("action output pointer is null");
    }

    *action = VaRuntimeAction{};
    action->frame_index = step.report.frame_index;
    action->timestamp_ms = step.report.timestamp_ms;
    action->fps = step.report.fps;
    action->preprocess_ms = step.report.timing.preprocess_ms;
    action->inference_ms = step.report.timing.inference_ms;
    action->postprocess_ms = step.report.timing.postprocess_ms;
    action->total_ms = step.report.timing.total_ms();
    action->detection_count = step.report.detection_count;
    action->has_target = step.command.has_target ? 1 : 0;
    action->dx = step.command.dx;
    action->dy = step.command.dy;
    action->click_left = step.command.click_left ? 1 : 0;
    action->lock_state = lock_state_to_int(step.command.lock_state);
    if (step.report.target.has_value()) {
        const auto& target = *step.report.target;
        action->distance = target.distance;
        action->offset_x = target.offset.x;
        action->offset_y = target.offset.y;
        action->target_x = target.analysis_point.x;
        action->target_y = target.analysis_point.y;
    }
}

void validate_and_open(VaRuntime* runtime) {
    vision_analyzer::validate_options(runtime->options);
    runtime->session.open(runtime->options);
}

}  // namespace

extern "C" {

VaRuntime* va_create(void) {
    try {
        return new VaRuntime{};
    } catch (...) {
        return nullptr;
    }
}

void va_destroy(VaRuntime* runtime) {
    delete runtime;
}

const char* va_last_error(VaRuntime* runtime) {
    if (runtime == nullptr) {
        return kNullRuntimeError;
    }
    return runtime->last_error.c_str();
}

int32_t va_load_config(VaRuntime* runtime, const char* path) {
    return call_api(runtime, [&] {
        vision_analyzer::apply_runtime_config_file(runtime->options, required_string(path, "config path"));
    });
}

int32_t va_set_model(VaRuntime* runtime, const char* model_path) {
    return call_api(runtime, [&] {
        runtime->options.model_path = required_string(model_path, "model path");
    });
}

int32_t va_set_schema(VaRuntime* runtime, const char* schema_path) {
    return call_api(runtime, [&] {
        runtime->options.model_schema_path = schema_path == nullptr ? std::string{} : std::string(schema_path);
    });
}

int32_t va_set_backend(VaRuntime* runtime, const char* backend) {
    return call_api(runtime, [&] {
        runtime->options.backend = vision_analyzer::parse_backend(required_string(backend, "backend"));
    });
}

int32_t va_set_tensorrt_cache_path(VaRuntime* runtime, const char* path) {
    return call_api(runtime, [&] {
        runtime->options.tensorrt_cache_path = required_string(path, "TensorRT cache path");
    });
}

int32_t va_set_player_side(VaRuntime* runtime, const char* side) {
    return call_api(runtime, [&] {
        runtime->options.player_side = vision_analyzer::parse_player_side(required_string(side, "player side"));
    });
}

int32_t va_set_hid_port(VaRuntime* runtime, const char* port) {
    return call_api(runtime, [&] {
        runtime->options.hid_port = port == nullptr ? std::string{} : std::string(port);
    });
}

int32_t va_set_dry_run(VaRuntime* runtime, int32_t dry_run) {
    return call_api(runtime, [&] {
        runtime->options.dry_run = dry_run != 0;
    });
}

int32_t va_set_output_enabled(VaRuntime* runtime, int32_t enabled) {
    return call_api(runtime, [&] {
        const bool output_enabled = enabled != 0;
        runtime->options.output_enabled = output_enabled;
        if (runtime->session.is_open()) {
            runtime->session.set_output_enabled(output_enabled);
        }
    });
}

int32_t va_set_fire_enabled(VaRuntime* runtime, int32_t enabled) {
    return call_api(runtime, [&] {
        apply_fire_enabled(runtime, enabled != 0);
    });
}

int32_t va_set_fire_policy(
    VaRuntime* runtime,
    int32_t body_enabled,
    float head_confidence,
    float body_confidence,
    int32_t cooldown_frames
) {
    return call_api(runtime, [&] {
        apply_fire_policy(runtime, vision_analyzer::FirePolicy{
            body_enabled != 0,
            head_confidence,
            body_confidence,
            cooldown_frames,
        });
    });
}

int32_t va_set_hid_click(VaRuntime* runtime, int32_t enabled, int32_t cooldown_frames) {
    return call_api(runtime, [&] {
        vision_analyzer::FirePolicy policy = runtime->options.fire_policy;
        policy.cooldown_frames = cooldown_frames;
        apply_fire_policy(runtime, policy);
        apply_fire_enabled(runtime, enabled != 0);
    });
}

int32_t va_set_hid_tuning(VaRuntime* runtime, float gain, int32_t max_step, float deadzone_px) {
    return call_api(runtime, [&] {
        runtime->options.hid_move_gain = gain;
        runtime->options.hid_max_step = max_step;
        runtime->options.hid_deadzone_px = deadzone_px;
    });
}

int32_t va_set_hid_calibration_path(
    VaRuntime* runtime,
    const char* calibration_path
) {
    return call_api(runtime, [&] {
        const std::filesystem::path candidate_path = required_string(
            calibration_path,
            "HID calibration path"
        );
        if (std::filesystem::exists(candidate_path)) {
            const vision_analyzer::HidCalibrationProfile candidate_profile =
                vision_analyzer::load_hid_calibration_profile(candidate_path);
            runtime->hid_calibration_path = candidate_path;
            runtime->options.hid_calibration = candidate_profile;
            return;
        }
        runtime->hid_calibration_path = candidate_path;
        runtime->options.hid_calibration.reset();
    });
}

int32_t va_get_hid_calibration(
    VaRuntime* runtime,
    VaHidCalibrationProfile* profile
) {
    if (profile != nullptr) {
        *profile = VaHidCalibrationProfile{};
    }
    return call_api(runtime, [&] {
        if (profile == nullptr) {
            throw std::runtime_error("calibration profile output pointer is null");
        }
        if (runtime->options.hid_calibration.has_value()) {
            fill_calibration_profile(*runtime->options.hid_calibration, profile);
        }
    });
}

int32_t va_set_thresholds(VaRuntime* runtime, float confidence, float nms_threshold) {
    return call_api(runtime, [&] {
        runtime->options.confidence = confidence;
        runtime->options.nms_threshold = nms_threshold;
    });
}

int32_t va_set_dxgi_roi(VaRuntime* runtime, int32_t x, int32_t y, int32_t width, int32_t height) {
    return call_api(runtime, [&] {
        runtime->options.dxgi_roi = cv::Rect(x, y, width, height);
    });
}

int32_t va_set_frame_limits(VaRuntime* runtime, int32_t max_frames, int32_t warmup_frames) {
    return call_api(runtime, [&] {
        if (max_frames < 0 || warmup_frames < 0) {
            throw std::runtime_error("frame limits must be greater than or equal to 0");
        }
        runtime->options.max_frames = max_frames;
        runtime->options.warmup_frames = warmup_frames;
    });
}

int32_t va_open_video(VaRuntime* runtime, const char* video_path, int32_t dry_run) {
    return call_api(runtime, [&] {
        runtime->options.input_source = vision_analyzer::InputSource::Video;
        runtime->options.video_path = required_string(video_path, "video path");
        runtime->options.dry_run = dry_run != 0;
        validate_and_open(runtime);
    });
}

int32_t va_open_dxgi(VaRuntime* runtime, int32_t adapter, int32_t output, int32_t dry_run) {
    return call_api(runtime, [&] {
        runtime->options.input_source = vision_analyzer::InputSource::Dxgi;
        runtime->options.dxgi_adapter = adapter;
        runtime->options.dxgi_output = output;
        runtime->options.dry_run = dry_run != 0;
        validate_and_open(runtime);
    });
}

int32_t va_calibrate_hid(
    VaRuntime* runtime,
    int32_t adapter,
    int32_t output,
    VaHidCalibrationProfile* profile
) {
    return call_api(runtime, [&] {
        if (profile == nullptr) {
            throw std::runtime_error("calibration profile output pointer is null");
        }
        *profile = VaHidCalibrationProfile{};
        if (runtime->session.is_open()) {
            throw std::runtime_error("close the runtime session before HID calibration");
        }
        if (runtime->options.hid_port.empty()) {
            throw std::runtime_error("HID calibration requires a configured HID port");
        }
        if (adapter < 0 || output < 0) {
            throw std::runtime_error("DXGI adapter and output must be greater than or equal to 0");
        }

        runtime->options.output_enabled = false;
        apply_fire_enabled(runtime, false);
        runtime->session.set_output_enabled(false);
        runtime->options.input_source = vision_analyzer::InputSource::Dxgi;
        runtime->options.dxgi_adapter = adapter;
        runtime->options.dxgi_output = output;

        vision_analyzer::Options calibration_options = runtime->options;
        calibration_options.calibrate_hid = true;
        calibration_options.output_enabled = false;
        calibration_options.fire_enabled = false;
        vision_analyzer::validate_options(calibration_options);
        const vision_analyzer::HidCalibrationProfile fitted =
            vision_analyzer::run_hid_calibration(calibration_options);
        if (!vision_analyzer::valid_hid_calibration_profile(fitted)) {
            throw std::runtime_error("HID calibration returned an invalid profile");
        }
        if (!runtime->hid_calibration_path.empty()) {
            vision_analyzer::save_hid_calibration_profile_atomic(
                runtime->hid_calibration_path,
                fitted
            );
        }
        runtime->options.hid_calibration = fitted;
        fill_calibration_profile(fitted, profile);
    });
}

int32_t va_process_next(VaRuntime* runtime, VaRuntimeAction* action) {
    if (runtime == nullptr) {
        return -1;
    }
    try {
        runtime->last_error.clear();
        const vision_analyzer::RuntimeStepResult step = runtime->session.process_next();
        if (!step.frame_available) {
            if (action != nullptr) {
                *action = VaRuntimeAction{};
            }
            return 0;
        }
        fill_action(step, action);
        return 1;
    } catch (const std::exception& error) {
        runtime->last_error = error.what();
        return -1;
    } catch (...) {
        runtime->last_error = "unknown runtime error";
        return -1;
    }
}

int32_t va_stop_all(VaRuntime* runtime) {
    return call_api(runtime, [&] {
        runtime->session.stop_all();
    });
}

int32_t va_close(VaRuntime* runtime) {
    return call_api(runtime, [&] {
        runtime->options.output_enabled = false;
        apply_fire_enabled(runtime, false);
        if (runtime->session.is_open()) {
            runtime->session.set_output_enabled(false);
        }
        runtime->session.close();
    });
}

}  // extern "C"
