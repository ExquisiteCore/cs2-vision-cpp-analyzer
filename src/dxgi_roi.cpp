#include "vision_analyzer/dxgi_roi.hpp"

#include <stdexcept>

namespace vision_analyzer {

cv::Rect resolve_dxgi_copy_region(const cv::Size& source_size, const cv::Rect& requested_roi) {
    if (source_size.width <= 0 || source_size.height <= 0) {
        throw std::runtime_error("DXGI source dimensions must be positive");
    }

    const cv::Rect full_frame(0, 0, source_size.width, source_size.height);
    if (requested_roi.width == 0 && requested_roi.height == 0) {
        return full_frame;
    }
    if (requested_roi.width <= 0 || requested_roi.height <= 0) {
        throw std::runtime_error("DXGI ROI width and height must both be positive");
    }

    const cv::Rect clipped = requested_roi & full_frame;
    if (clipped.empty()) {
        throw std::runtime_error("DXGI ROI is outside the captured frame");
    }
    return clipped;
}

}  // namespace vision_analyzer
