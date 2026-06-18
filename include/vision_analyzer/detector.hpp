#pragma once

#include <memory>
#include <string>

#include <opencv2/core.hpp>

#include "vision_analyzer/types.hpp"

namespace vision_analyzer {

class Detector {
public:
    virtual ~Detector() = default;

    [[nodiscard]] virtual DetectionResult detect(const cv::Mat& frame, float confidence, float nms_threshold) = 0;
    [[nodiscard]] virtual std::string name() const = 0;
};

[[nodiscard]] Backend parse_backend(const std::string& value);
[[nodiscard]] std::unique_ptr<Detector> create_detector(Backend backend, const std::string& model_path);

}  // namespace vision_analyzer
