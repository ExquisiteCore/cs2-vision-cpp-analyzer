#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "vision_analyzer/aim_controller.hpp"

namespace vision_analyzer {

struct HidDeviceHealth {
    std::uint8_t protocol_version = 0;
    std::uint16_t capabilities = 0;
};

[[nodiscard]] HidDeviceHealth parse_rp2350_v2_health(
    const std::vector<std::uint8_t>& info,
    const std::vector<std::uint8_t>& caps
);

class HidClient {
public:
    virtual ~HidClient() = default;

    virtual void move_relative(std::int16_t dx, std::int16_t dy) = 0;
    virtual void click_left() = 0;
    virtual void stop_all() = 0;
    virtual void close() noexcept = 0;
};

void close_hid_client_noexcept(HidClient* client) noexcept;

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
