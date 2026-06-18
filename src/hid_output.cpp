#include "vision_analyzer/hid_output.hpp"

#include <stdexcept>

#if defined(VISION_ANALYZER_WITH_RP2350_HID) && defined(_WIN32)
#include "rp2350_hid_bridge.hpp"
#endif

namespace vision_analyzer {
namespace {

#if defined(VISION_ANALYZER_WITH_RP2350_HID) && defined(_WIN32)
class Rp2350HidClient final : public HidClient {
public:
    explicit Rp2350HidClient(const std::string& port)
        : bridge_(port) {
        bridge_.open();
        bridge_.ping();
    }

    ~Rp2350HidClient() override {
        try {
            bridge_.stop_all();
        } catch (...) {
        }
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

private:
    rp2350_hid_bridge::HidBridge bridge_;
};
#endif

}  // namespace

HidActionSender::HidActionSender(HidClient& client)
    : client_(client) {}

void HidActionSender::execute(const AimCommand& command) {
    if (!command.has_target) {
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
