#pragma once

#include <opencv2/core.hpp>

namespace vision_analyzer {

[[nodiscard]] cv::Rect resolve_dxgi_copy_region(
    const cv::Size& source_size,
    const cv::Rect& requested_roi
);

}  // namespace vision_analyzer
