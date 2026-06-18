#include "vision_analyzer/aim_controller.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vision_analyzer {
namespace {

[[nodiscard]] std::int16_t scaled_step(float value, float gain, int max_step) {
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
    if (options.click_cooldown_frames < 0) {
        throw std::runtime_error("aim click cooldown must be greater than or equal to 0");
    }
}

}  // namespace

AimController::AimController(AimControllerOptions options)
    : options_(options) {
    validate_options(options_);
}

AimCommand AimController::plan(const FrameReport& report) {
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
    command.dx = scaled_step(target.offset.x, options_.move_gain, options_.max_step);
    command.dy = scaled_step(target.offset.y, options_.move_gain, options_.max_step);
    command.lock_state = target.lock_state;

    if (options_.click_enabled && target.fire_candidate && click_available) {
        command.click_left = true;
        click_cooldown_remaining_ = options_.click_cooldown_frames;
    } else if (click_cooldown_remaining_ > 0) {
        --click_cooldown_remaining_;
    }

    return command;
}

}  // namespace vision_analyzer
