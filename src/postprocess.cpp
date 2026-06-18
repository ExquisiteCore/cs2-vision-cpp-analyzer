#include "vision_analyzer/postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace vision_analyzer {

LetterboxResult letterbox(const cv::Mat& frame, int target_size) {
    const int width = frame.cols;
    const int height = frame.rows;
    const float scale = std::min(target_size / static_cast<float>(width), target_size / static_cast<float>(height));
    const int resized_width = static_cast<int>(std::round(width * scale));
    const int resized_height = static_cast<int>(std::round(height * scale));
    const int pad_x = (target_size - resized_width) / 2;
    const int pad_y = (target_size - resized_height) / 2;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(resized_width, resized_height));

    cv::Mat output(target_size, target_size, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(output(cv::Rect(pad_x, pad_y, resized_width, resized_height)));
    return {output, scale, pad_x, pad_y};
}

cv::Rect restore_box(float cx, float cy, float w, float h, const LetterboxResult& letterbox_result, const cv::Size& frame_size) {
    const float x1 = (cx - w / 2.0F - letterbox_result.pad_x) / letterbox_result.scale;
    const float y1 = (cy - h / 2.0F - letterbox_result.pad_y) / letterbox_result.scale;
    const float x2 = (cx + w / 2.0F - letterbox_result.pad_x) / letterbox_result.scale;
    const float y2 = (cy + h / 2.0F - letterbox_result.pad_y) / letterbox_result.scale;

    const int left = std::clamp(static_cast<int>(std::round(x1)), 0, frame_size.width - 1);
    const int top = std::clamp(static_cast<int>(std::round(y1)), 0, frame_size.height - 1);
    const int right = std::clamp(static_cast<int>(std::round(x2)), 0, frame_size.width - 1);
    const int bottom = std::clamp(static_cast<int>(std::round(y2)), 0, frame_size.height - 1);
    return cv::Rect(cv::Point(left, top), cv::Point(std::max(left + 1, right), std::max(top + 1, bottom)));
}

std::vector<int> class_aware_nms_indices(
    const std::vector<cv::Rect>& boxes,
    const std::vector<float>& scores,
    const std::vector<int>& class_ids,
    float confidence,
    float nms_threshold
) {
    if (boxes.size() != scores.size() || boxes.size() != class_ids.size()) {
        throw std::runtime_error("class-aware NMS input sizes do not match");
    }

    std::map<int, std::vector<int>> by_class;
    for (int i = 0; i < static_cast<int>(class_ids.size()); ++i) {
        by_class[class_ids[static_cast<std::size_t>(i)]].push_back(i);
    }

    std::vector<int> keep_all;
    for (const auto& [class_id, indices] : by_class) {
        (void)class_id;
        std::vector<cv::Rect> class_boxes;
        std::vector<float> class_scores;
        class_boxes.reserve(indices.size());
        class_scores.reserve(indices.size());
        for (const int index : indices) {
            class_boxes.push_back(boxes[static_cast<std::size_t>(index)]);
            class_scores.push_back(scores[static_cast<std::size_t>(index)]);
        }

        std::vector<int> keep_local;
        cv::dnn::NMSBoxes(class_boxes, class_scores, confidence, nms_threshold, keep_local);
        for (const int local_index : keep_local) {
            keep_all.push_back(indices[static_cast<std::size_t>(local_index)]);
        }
    }

    std::sort(keep_all.begin(), keep_all.end(), [&](int left, int right) {
        return scores[static_cast<std::size_t>(left)] > scores[static_cast<std::size_t>(right)];
    });
    return keep_all;
}

std::vector<Detection> decode_yolo_output(
    const cv::Mat& output,
    const LetterboxResult& letterbox_result,
    const cv::Size& frame_size,
    float confidence,
    float nms_threshold
) {
    const auto& names = class_names();
    if (output.dims != 3) {
        throw std::runtime_error("unexpected YOLO output rank: " + std::to_string(output.dims));
    }

    int dimensions = output.size[1];
    int rows = output.size[2];
    bool channels_first = true;
    if (dimensions < 4 + static_cast<int>(names.size()) && rows >= 4 + static_cast<int>(names.size())) {
        dimensions = output.size[2];
        rows = output.size[1];
        channels_first = false;
    }
    if (dimensions < 4 + static_cast<int>(names.size())) {
        throw std::runtime_error("unexpected YOLO output dimensions");
    }

    const float* data = reinterpret_cast<const float*>(output.data);

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    for (int i = 0; i < rows; ++i) {
        const auto at = [&](int dimension) -> float {
            if (channels_first) {
                return data[dimension * rows + i];
            }
            return data[i * dimensions + dimension];
        };

        float best_score = 0.0F;
        int best_class = -1;
        for (int class_id = 0; class_id < static_cast<int>(names.size()); ++class_id) {
            const float score = at(4 + class_id);
            if (score > best_score) {
                best_score = score;
                best_class = class_id;
            }
        }
        if (best_score < confidence || best_class < 0) {
            continue;
        }

        boxes.push_back(restore_box(at(0), at(1), at(2), at(3), letterbox_result, frame_size));
        scores.push_back(best_score);
        class_ids.push_back(best_class);
    }

    const auto keep = class_aware_nms_indices(boxes, scores, class_ids, confidence, nms_threshold);

    std::vector<Detection> detections;
    detections.reserve(keep.size());
    for (const int index : keep) {
        const int class_id = class_ids[static_cast<std::size_t>(index)];
        detections.push_back(Detection{
            class_id,
            names[static_cast<std::size_t>(class_id)],
            scores[static_cast<std::size_t>(index)],
            boxes[static_cast<std::size_t>(index)],
        });
    }
    return detections;
}

}  // namespace vision_analyzer
