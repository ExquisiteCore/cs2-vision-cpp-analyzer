#include "vision_analyzer/types.hpp"

#include <stdexcept>

namespace vision_analyzer {

const std::vector<std::string>& class_names() {
    static const std::vector<std::string> names = {"ct_body", "ct_head", "t_body", "t_head"};
    return names;
}

bool is_head(int class_id) {
    return class_id == 1 || class_id == 3;
}

bool is_body(int class_id) {
    return class_id == 0 || class_id == 2;
}

bool is_enemy_class(PlayerSide player_side, int class_id) {
    switch (player_side) {
    case PlayerSide::Unknown:
        return true;
    case PlayerSide::Ct:
        return class_id == 2 || class_id == 3;
    case PlayerSide::T:
        return class_id == 0 || class_id == 1;
    }
    throw std::runtime_error("unknown player side");
}

std::vector<Detection> filter_enemy_detections(const std::vector<Detection>& detections, PlayerSide player_side) {
    if (player_side == PlayerSide::Unknown) {
        return detections;
    }

    std::vector<Detection> filtered;
    filtered.reserve(detections.size());
    for (const auto& detection : detections) {
        if (is_enemy_class(player_side, detection.class_id)) {
            filtered.push_back(detection);
        }
    }
    return filtered;
}

std::string lock_state_name(LockState state) {
    switch (state) {
    case LockState::Idle:
        return "idle";
    case LockState::Acquiring:
        return "acquiring";
    case LockState::Tracking:
        return "tracking";
    case LockState::Locked:
        return "locked";
    case LockState::Lost:
        return "lost";
    }
    throw std::runtime_error("unknown lock state");
}

PlayerSide parse_player_side(const std::string& value) {
    if (value == "unknown") {
        return PlayerSide::Unknown;
    }
    if (value == "ct") {
        return PlayerSide::Ct;
    }
    if (value == "t") {
        return PlayerSide::T;
    }
    throw std::runtime_error("unknown player side: " + value);
}

std::string player_side_name(PlayerSide side) {
    switch (side) {
    case PlayerSide::Unknown:
        return "unknown";
    case PlayerSide::Ct:
        return "ct";
    case PlayerSide::T:
        return "t";
    }
    throw std::runtime_error("unknown player side");
}

std::string backend_name(Backend backend) {
    switch (backend) {
    case Backend::OpenCvOnnx:
        return "opencv-onnx";
    case Backend::OpenCvCuda:
        return "opencv-cuda";
    case Backend::OrtCuda:
        return "ort-cuda";
    case Backend::OrtTensorRt:
        return "ort-tensorrt";
    case Backend::TensorRt:
        return "tensorrt";
    }
    throw std::runtime_error("unknown backend");
}

}  // namespace vision_analyzer
