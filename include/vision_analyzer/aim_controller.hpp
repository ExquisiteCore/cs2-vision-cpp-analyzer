#pragma once

#include <cstdint>
#include <mutex>
#include <optional>

#include "vision_analyzer/hid_calibration_profile.hpp"
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
    bool fire_enabled = false;
    FirePolicy fire_policy;
    std::optional<HidCalibrationProfile> calibration;
};

class AimController {
public:
    explicit AimController(AimControllerOptions options = {});

    [[nodiscard]] AimCommand plan(const FrameReport& report);
    void set_fire_enabled(bool enabled);
    void set_fire_policy(FirePolicy policy);

private:
    AimControllerOptions options_;
    std::mutex mutex_;
    int click_cooldown_remaining_ = 0;
};

}  // namespace vision_analyzer
