#include "vision_analyzer/aim_controller.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vision_analyzer {
namespace {

[[nodiscard]] std::int16_t scaled_step(float value, float gain, int max_step, float deadzone_px) {
    if (std::abs(value) < deadzone_px) {
        return 0;
    }
    const int limited_max = std::clamp(max_step, 0, static_cast<int>(std::numeric_limits<std::int16_t>::max()));
    const int rounded = static_cast<int>(std::lround(value * gain));
    return static_cast<std::int16_t>(std::clamp(rounded, -limited_max, limited_max));
}

void validate_options(const AimControllerOptions& options) {
    if (!std::isfinite(options.move_gain)) {
        throw std::runtime_error("aim move gain must be finite");
    }
    if (options.max_step < 0) {
        throw std::runtime_error("aim max step must be greater than or equal to 0");
    }
    if (!std::isfinite(options.deadzone_px) || options.deadzone_px < 0.0F) {
        throw std::runtime_error("aim deadzone must be finite and greater than or equal to 0");
    }
    if (options.calibration.has_value() && options.calibration->valid &&
        (!valid_hid_calibration_curve(options.calibration->x) ||
         !valid_hid_calibration_curve(options.calibration->y) ||
         options.calibration->max_step < 0 ||
         !std::isfinite(options.calibration->deadzone_px) ||
         options.calibration->deadzone_px < 0.0F)) {
        throw std::runtime_error("aim HID calibration profile is invalid");
    }
}

void validate_fire_policy(const FirePolicy& policy) {
    if (!std::isfinite(policy.head_confidence) || policy.head_confidence < 0.0F ||
        policy.head_confidence > 1.0F) {
        throw std::runtime_error("head fire confidence must be in [0, 1]");
    }
    if (!std::isfinite(policy.body_confidence) || policy.body_confidence < 0.0F ||
        policy.body_confidence > 1.0F) {
        throw std::runtime_error("body fire confidence must be in [0, 1]");
    }
    if (policy.cooldown_frames < 0) {
        throw std::runtime_error("fire cooldown must be greater than or equal to 0");
    }
}

[[nodiscard]] cv::Rect fire_region(const Detection& detection) {
    if (is_head(detection.class_id)) {
        return detection.box;
    }
    return cv::Rect(
        detection.box.x + static_cast<int>(detection.box.width * 0.20F),
        detection.box.y + static_cast<int>(detection.box.height * 0.10F),
        std::max(1, static_cast<int>(detection.box.width * 0.60F)),
        std::max(1, static_cast<int>(detection.box.height * 0.60F))
    );
}

[[nodiscard]] bool contains_point(const cv::Rect& region, const cv::Point2f& point) {
    return point.x >= static_cast<float>(region.x) &&
           point.x < static_cast<float>(region.x + region.width) &&
           point.y >= static_cast<float>(region.y) &&
           point.y < static_cast<float>(region.y + region.height);
}

}  // namespace

AimController::AimController(AimControllerOptions options)
    : options_(options) {
    validate_options(options_);
    validate_fire_policy(options_.fire_policy);
}

AimCommand AimController::plan(const FrameReport& report) {
    std::lock_guard<std::mutex> lock(mutex_);
    const bool click_available = click_cooldown_remaining_ == 0;

    if (!report.target.has_value()) {
        if (click_cooldown_remaining_ > 0) {
            --click_cooldown_remaining_;
        }
        return {};
    }

    const auto& target = *report.target;
    AimCommand command;
    command.has_target = true;
    if (options_.calibration.has_value() && options_.calibration->valid) {
        const auto& calibration = *options_.calibration;
        command.dx = calibrated_hid_step(
            target.offset.x,
            calibration.x,
            calibration.max_step,
            calibration.deadzone_px
        );
        command.dy = calibrated_hid_step(
            target.offset.y,
            calibration.y,
            calibration.max_step,
            calibration.deadzone_px
        );
    } else {
        command.dx = scaled_step(target.offset.x, options_.move_gain, options_.max_step, options_.deadzone_px);
        command.dy = scaled_step(target.offset.y, options_.move_gain, options_.max_step, options_.deadzone_px);
    }
    command.lock_state = target.lock_state;

    const bool head = is_head(target.detection.class_id);
    const bool body = is_body(target.detection.class_id);
    const float confidence_threshold = head
        ? options_.fire_policy.head_confidence
        : options_.fire_policy.body_confidence;
    const cv::Point2f frame_center = target.analysis_point - target.offset;
    const bool fire_qualified =
        (head || (body && options_.fire_policy.body_enabled)) &&
        target.detection.confidence >= confidence_threshold &&
        contains_point(fire_region(target.detection), frame_center);

    if (options_.fire_enabled && fire_qualified && click_available) {
        command.click_left = true;
        click_cooldown_remaining_ = options_.fire_policy.cooldown_frames;
    } else if (click_cooldown_remaining_ > 0) {
        --click_cooldown_remaining_;
    }

    return command;
}

void AimController::set_fire_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    options_.fire_enabled = enabled;
    if (!enabled) {
        click_cooldown_remaining_ = 0;
    }
}

void AimController::set_fire_policy(FirePolicy policy) {
    validate_fire_policy(policy);
    std::lock_guard<std::mutex> lock(mutex_);
    options_.fire_policy = policy;
}

}  // namespace vision_analyzer
