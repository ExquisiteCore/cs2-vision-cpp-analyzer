#pragma once

#include <memory>
#include <string>

#include <opencv2/core.hpp>

#include "vision_analyzer/types.hpp"

namespace vision_analyzer {

struct CapturedFrame {
    cv::Mat image;
    int index = 0;
    double timestamp_ms = 0.0;
};

class FrameSource {
public:
    virtual ~FrameSource() = default;

    [[nodiscard]] virtual bool read(CapturedFrame& frame) = 0;
    virtual void release() = 0;
    [[nodiscard]] virtual std::string name() const = 0;
    [[nodiscard]] virtual bool is_live() const = 0;
};

[[nodiscard]] std::unique_ptr<FrameSource> create_frame_source(const Options& options);

}  // namespace vision_analyzer
