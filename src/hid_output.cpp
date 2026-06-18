#include "vision_analyzer/hid_output.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#if defined(VISION_ANALYZER_WITH_RP2350_HID) && defined(_WIN32)
#include "rp2350_hid_bridge.hpp"
#endif

namespace vision_analyzer {
namespace {

[[nodiscard]] std::int16_t scaled_step(float value, float gain, int max_step) {
    const int limited_max = std::clamp(max_step, 0, static_cast<int>(std::numeric_limits<std::int16_t>::max()));
    const int rounded = static_cast<int>(std::lround(value * gain));
    const int clamped = std::clamp(rounded, -limited_max, limited_max);
    return static_cast<std::int16_t>(clamped);
}

void validate_options(const HidActionOptions& options) {
    if (!std::isfinite(options.move_gain)) {
        throw std::runtime_error("hid move gain must be finite");
    }
    if (options.max_step < 0) {
        throw std::runtime_error("hid max step must be greater than or equal to 0");
    }
    if (options.click_cooldown_frames < 0) {
        throw std::runtime_error("hid click cooldown must be greater than or equal to 0");
    }
}

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

HidActionSender::HidActionSender(HidClient& client, HidActionOptions options)
    : client_(client),
      options_(options) {
    validate_options(options_);
}

void HidActionSender::handle_report(const FrameReport& report) {
    if (click_cooldown_remaining_ > 0) {
        --click_cooldown_remaining_;
    }

    if (!report.target.has_value()) {
        return;
    }

    const auto& target = *report.target;
    const std::int16_t dx = scaled_step(target.offset.x, options_.move_gain, options_.max_step);
    const std::int16_t dy = scaled_step(target.offset.y, options_.move_gain, options_.max_step);
    if (dx != 0 || dy != 0) {
        client_.move_relative(dx, dy);
    }

    if (options_.click_enabled && target.fire_candidate && click_cooldown_remaining_ == 0) {
        client_.click_left();
        click_cooldown_remaining_ = options_.click_cooldown_frames;
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
