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
