#include "vision_analyzer/hid_output.hpp"

#include <stdexcept>

#if defined(VISION_ANALYZER_WITH_RP2350_HID) && defined(_WIN32)
#include "rp2350_hid_bridge.hpp"
#endif

namespace vision_analyzer {
namespace {

constexpr std::uint8_t kRp2350ProtocolV2 = 2;
constexpr std::uint16_t kRp2350RequiredCapabilities = 0x0072;
constexpr const char* kRp2350ProtocolV2Error =
    "RP2350 protocol v2 capabilities are required";

#if defined(VISION_ANALYZER_WITH_RP2350_HID) && defined(_WIN32)
static_assert(
    rp2350_hid_bridge::PROTOCOL_VERSION == kRp2350ProtocolV2,
    "RP2350 HID SDK protocol v2 is required"
);

rp2350_hid_bridge::HidBridgeOptions make_bridge_options(const std::string& port) {
    rp2350_hid_bridge::HidBridgeOptions options;
    options.port = port;
    return options;
}

class Rp2350HidClient final : public HidClient {
public:
    explicit Rp2350HidClient(const std::string& port)
        : bridge_(make_bridge_options(port)) {
        bridge_.open();
        bridge_.ping();
        (void)parse_rp2350_v2_health(bridge_.info(), bridge_.caps());
    }

    ~Rp2350HidClient() override {
        close();
    }

    void move_relative(std::int16_t dx, std::int16_t dy) override {
        bridge_.mouse_move(dx, dy);
    }

    void click_left() override {
        bridge_.mouse_click("left");
    }

    void stop_all() override {
        bridge_.stop_all();
    }

    void close() noexcept override {
        bridge_.close();
    }

private:
    rp2350_hid_bridge::HidBridge bridge_;
};
#endif

}  // namespace

HidDeviceHealth parse_rp2350_v2_health(
    const std::vector<std::uint8_t>& info,
    const std::vector<std::uint8_t>& caps
) {
    if (info.size() < 4 || caps.size() < 5 ||
        info[0] != kRp2350ProtocolV2 || caps[0] != kRp2350ProtocolV2) {
        throw std::runtime_error(kRp2350ProtocolV2Error);
    }

    const auto capabilities = static_cast<std::uint16_t>(
        (static_cast<std::uint16_t>(caps[3]) << 8) |
        static_cast<std::uint16_t>(caps[4])
    );
    if ((capabilities & kRp2350RequiredCapabilities) != kRp2350RequiredCapabilities) {
        throw std::runtime_error(kRp2350ProtocolV2Error);
    }

    return HidDeviceHealth{kRp2350ProtocolV2, capabilities};
}

void close_hid_client_noexcept(HidClient* client) noexcept {
    if (client == nullptr) {
        return;
    }
    try {
        client->stop_all();
    } catch (...) {
    }
    client->close();
}

HidActionSender::HidActionSender(HidClient& client)
    : client_(client) {}

void HidActionSender::set_enabled(bool enabled) {
    std::scoped_lock lock(output_mutex_);
    enabled_.store(enabled);
    if (!enabled) {
        client_.stop_all();
    }
}

void HidActionSender::execute(const AimCommand& command) {
    std::scoped_lock lock(output_mutex_);
    if (!enabled_.load() || !command.has_target) {
        return;
    }

    if (command.dx != 0 || command.dy != 0) {
        client_.move_relative(command.dx, command.dy);
    }

    if (command.click_left) {
        client_.click_left();
    }
}

void HidActionSender::stop_all() {
    std::scoped_lock lock(output_mutex_);
    client_.stop_all();
}

std::unique_ptr<HidClient> create_rp2350_hid_client(const std::string& port) {
#if defined(VISION_ANALYZER_WITH_RP2350_HID) && defined(_WIN32)
    return std::make_unique<Rp2350HidClient>(port);
#else
    (void)port;
    throw std::runtime_error(
        "RP2350 HID bridge SDK is not available in this build. "
        "Set RP2350_HID_BRIDGE_SDK to the SDK root or build on Windows with the SDK include directory."
    );
#endif
}

}  // namespace vision_analyzer
