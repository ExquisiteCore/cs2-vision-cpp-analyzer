#pragma once

#include <cstdint>

#include "vision_analyzer/types.hpp"

namespace vision_analyzer {

struct AimCommand {
    bool has_target = false;
    std::int16_t dx = 0;
    std::int16_t dy = 0;
    bool click_left = false;
    LockState lock_state = LockState::Idle;
};

struct AimControllerOptions {
    float move_gain = 1.0F;
    int max_step = 120;
    float deadzone_px = 1.5F;
    bool click_enabled = false;
    int click_cooldown_frames = 6;
};

class AimController {
public:
    explicit AimController(AimControllerOptions options = {});

    [[nodiscard]] AimCommand plan(const FrameReport& report);

private:
    AimControllerOptions options_;
    int click_cooldown_remaining_ = 0;
};

}  // namespace vision_analyzer
