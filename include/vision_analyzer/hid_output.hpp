#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "vision_analyzer/types.hpp"

namespace vision_analyzer {

class HidClient {
public:
    virtual ~HidClient() = default;

    virtual void move_relative(std::int16_t dx, std::int16_t dy) = 0;
    virtual void click_left() = 0;
    virtual void stop_all() = 0;
};

struct HidActionOptions {
    float move_gain = 1.0F;
    int max_step = 120;
    bool click_enabled = false;
    int click_cooldown_frames = 6;
};

class HidActionSender {
public:
    explicit HidActionSender(HidClient& client, HidActionOptions options = {});

    void handle_report(const FrameReport& report);
    void stop_all();

private:
    HidClient& client_;
    HidActionOptions options_;
    int click_cooldown_remaining_ = 0;
};

[[nodiscard]] std::unique_ptr<HidClient> create_rp2350_hid_client(const std::string& port);

}  // namespace vision_analyzer
