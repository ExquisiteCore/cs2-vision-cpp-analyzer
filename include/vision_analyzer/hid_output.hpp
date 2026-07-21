#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "vision_analyzer/aim_controller.hpp"

namespace vision_analyzer {

class HidClient {
public:
    virtual ~HidClient() = default;

    virtual void move_relative(std::int16_t dx, std::int16_t dy) = 0;
    virtual void click_left() = 0;
    virtual void stop_all() = 0;
};

class HidActionSender {
public:
    explicit HidActionSender(HidClient& client);

    void set_enabled(bool enabled);
    void execute(const AimCommand& command);
    void stop_all();

private:
    HidClient& client_;
    std::atomic_bool enabled_{false};
    std::mutex output_mutex_;
};

[[nodiscard]] std::unique_ptr<HidClient> create_rp2350_hid_client(const std::string& port);

}  // namespace vision_analyzer
