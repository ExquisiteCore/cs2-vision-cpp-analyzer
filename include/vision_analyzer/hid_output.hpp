#pragma once

#include <cstdint>
#include <memory>
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

    void execute(const AimCommand& command);
    void stop_all();

private:
    HidClient& client_;
};

[[nodiscard]] std::unique_ptr<HidClient> create_rp2350_hid_client(const std::string& port);

}  // namespace vision_analyzer
