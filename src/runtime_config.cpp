#include "vision_analyzer/runtime_config.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>

namespace vision_analyzer {
namespace {

[[nodiscard]] std::string trim(std::string value) {
    const auto is_space = [](unsigned char character) {
        return std::isspace(character) != 0;
    };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [&](char character) {
        return !is_space(static_cast<unsigned char>(character));
    }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [&](char character) {
        return !is_space(static_cast<unsigned char>(character));
    }).base(), value.end());
    return value;
}

[[nodiscard]] bool parse_bool(const std::string& value) {
    if (value == "1" || value == "true" || value == "yes" || value == "on") {
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off") {
        return false;
    }
    throw std::runtime_error("invalid boolean value: " + value);
}

[[nodiscard]] Backend parse_backend_config(const std::string& value) {
    if (value == "opencv-onnx") {
        return Backend::OpenCvOnnx;
    }
    if (value == "opencv-cuda") {
        return Backend::OpenCvCuda;
    }
    if (value == "ort-cuda") {
        return Backend::OrtCuda;
    }
    if (value == "ort-tensorrt") {
        return Backend::OrtTensorRt;
    }
    if (value == "tensorrt") {
        return Backend::TensorRt;
    }
    throw std::runtime_error("unknown backend: " + value);
}

void apply_entry(Options& options, const std::string& key, const std::string& value) {
    if (key == "model") {
        options.model_path = value;
    } else if (key == "model_schema" || key == "schema") {
        options.model_schema_path = value;
    } else if (key == "video") {
        options.video_path = value;
        options.input_source = InputSource::Video;
    } else if (key == "input") {
        options.input_source = parse_input_source(value);
    } else if (key == "dxgi_adapter") {
        options.dxgi_adapter = std::stoi(value);
    } else if (key == "dxgi_output") {
        options.dxgi_output = std::stoi(value);
    } else if (key == "dxgi_timeout_ms" || key == "dxgi_timeout") {
        options.dxgi_timeout_ms = std::stoi(value);
    } else if (key == "dxgi_gpu_preference") {
        options.dxgi_gpu_preference = parse_dxgi_gpu_preference(value);
    } else if (key == "dxgi_debug") {
        options.dxgi_debug = parse_bool(value);
    } else if (key == "dxgi_roi_x") {
        options.dxgi_roi.x = std::stoi(value);
    } else if (key == "dxgi_roi_y") {
        options.dxgi_roi.y = std::stoi(value);
    } else if (key == "dxgi_roi_width") {
        options.dxgi_roi.width = std::stoi(value);
    } else if (key == "dxgi_roi_height") {
        options.dxgi_roi.height = std::stoi(value);
    } else if (key == "hid_port") {
        options.hid_port = value;
    } else if (key == "hid_gain" || key == "hid_move_gain") {
        options.hid_move_gain = std::stof(value);
    } else if (key == "hid_max_step") {
        options.hid_max_step = std::stoi(value);
    } else if (key == "deadzone_px" || key == "hid_deadzone_px") {
        options.hid_deadzone_px = std::stof(value);
    } else if (key == "hid_click") {
        options.hid_click_enabled = parse_bool(value);
    } else if (key == "hid_click_cooldown" || key == "hid_click_cooldown_frames") {
        options.hid_click_cooldown_frames = std::stoi(value);
    } else if (key == "player_side") {
        options.player_side = parse_player_side(value);
    } else if (key == "dry_run") {
        options.dry_run = parse_bool(value);
    } else if (key == "status_every" || key == "status_every_frames") {
        options.status_every_frames = std::stoi(value);
    } else if (key == "backend") {
        options.backend = parse_backend_config(value);
    } else if (key == "conf" || key == "confidence") {
        options.confidence = std::stof(value);
    } else if (key == "nms" || key == "nms_threshold") {
        options.nms_threshold = std::stof(value);
    } else if (key == "max_frames") {
        options.max_frames = std::stoi(value);
    } else if (key == "start_frame") {
        options.start_frame = std::stoi(value);
    } else if (key == "warmup_frames") {
        options.warmup_frames = std::stoi(value);
    } else if (key == "start_time" || key == "start_time_seconds") {
        options.start_time_seconds = std::stod(value);
    } else if (key == "preview") {
        options.preview = parse_bool(value);
    } else if (key == "body_head_anchor_ratio") {
        options.tuning.body_head_anchor_ratio = std::stof(value);
    } else if (key == "kalman_process_noise") {
        options.tuning.kalman_process_noise = std::stof(value);
    } else if (key == "kalman_measurement_noise") {
        options.tuning.kalman_measurement_noise = std::stof(value);
    } else if (key == "kalman_error_covariance") {
        options.tuning.kalman_error_covariance = std::stof(value);
    } else if (key == "calibration_step_counts" || key == "calibration_step") {
        options.calibration_step_counts = std::stoi(value);
    } else if (key == "calibration_repeats") {
        options.calibration_repeats = std::stoi(value);
    } else if (key == "calibration_noise_samples") {
        options.calibration_noise_samples = std::stoi(value);
    } else if (key == "calibration_settle_ms") {
        options.calibration_settle_ms = std::stoi(value);
    } else if (key == "calibration_output") {
        options.calibration_output_path = value;
    } else if (key == "calibration_config_output") {
        options.calibration_config_output_path = value;
    } else if (key == "action_log") {
        options.action_log_path = value;
    } else {
        throw std::runtime_error("unknown config key: " + key);
    }
}

}  // namespace

void apply_runtime_config_file(Options& options, const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open runtime config: " + path);
    }

    options.config_path = path;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const auto comment = line.find('#');
        if (comment != std::string::npos) {
            line.erase(comment);
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }

        const auto separator = line.find('=');
        if (separator == std::string::npos) {
            throw std::runtime_error("invalid config line " + std::to_string(line_number) + ": expected key=value");
        }
        const std::string key = trim(line.substr(0, separator));
        const std::string value = trim(line.substr(separator + 1));
        if (key.empty() || value.empty()) {
            throw std::runtime_error("invalid config line " + std::to_string(line_number) + ": empty key or value");
        }
        apply_entry(options, key, value);
    }
}

}  // namespace vision_analyzer
