#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "vision_analyzer/types.hpp"

namespace vision_analyzer {

[[nodiscard]] LetterboxResult letterbox(const cv::Mat& frame, int target_size);

[[nodiscard]] cv::Rect restore_box(
    float cx,
    float cy,
    float w,
    float h,
    const LetterboxResult& letterbox_result,
    const cv::Size& frame_size
);

[[nodiscard]] std::vector<int> class_aware_nms_indices(
    const std::vector<cv::Rect>& boxes,
    const std::vector<float>& scores,
    const std::vector<int>& class_ids,
    float confidence,
    float nms_threshold
);

[[nodiscard]] std::vector<Detection> decode_yolo_output(
    const cv::Mat& output,
    const LetterboxResult& letterbox_result,
    const cv::Size& frame_size,
    float confidence,
    float nms_threshold
);

}  // namespace vision_analyzer
