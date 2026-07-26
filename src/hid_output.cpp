#include "vision_analyzer/hid_output.hpp"

#include <array>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(VISION_ANALYZER_WITH_RP2350_HID) && defined(_WIN32)
#include "rp2350_hid_bridge/c_api.h"
#include "rp2350_hid_bridge/protocol.hpp"
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
    "RP2350 HID SDK protocol v2 is required");

void check_hid_status(std::int32_t status) {
    if (status == RP2350_HID_STATUS_OK) {
        return;
    }
    const char* message = rp2350_hid_last_error();
    throw std::runtime_error(
        message == nullptr || message[0] == '\0'
            ? "RP2350 HID command failed"
            : message);
}

std::vector<std::uint8_t> read_payload(
    Rp2350HidSession* session,
    std::int32_t (*function)(
        Rp2350HidSession*,
        std::uint8_t*,
        std::uint32_t,
        std::uint32_t*)) {
    std::array<std::uint8_t, 256> output{};
    std::uint32_t bytes_written = 0;
    check_hid_status(function(
        session,
        output.data(),
        static_cast<std::uint32_t>(output.size()),
        &bytes_written));
    return {output.begin(), output.begin() + bytes_written};
}

class Rp2350HidClient final : public HidClient {
public:
    explicit Rp2350HidClient(const std::string& port) {
        Rp2350HidOptions options{};
        options.struct_size = sizeof(options);
        options.port = port.c_str();
        options.baud = 115200;
        options.timeout_ms = 1000;
        options.retries = 2;
        options.heartbeat_interval_ms = 500;

        Rp2350HidSession* candidate = nullptr;
        check_hid_status(rp2350_hid_session_create(&options, &candidate));
        try {
            check_hid_status(rp2350_hid_session_open(candidate));
            validate_health(candidate);
        } catch (...) {
            rp2350_hid_session_release(candidate);
            throw;
        }
        session_ = candidate;
    }

    explicit Rp2350HidClient(Rp2350HidSession* session) {
        if (session == nullptr) {
            throw std::invalid_argument("attached HID session is null");
        }
        check_hid_status(rp2350_hid_session_retain(session));
        try {
            validate_health(session);
        } catch (...) {
            rp2350_hid_session_release(session);
            throw;
        }
        session_ = session;
    }

    ~Rp2350HidClient() override {
        close();
    }

    void move_relative(std::int16_t dx, std::int16_t dy) override {
        std::scoped_lock lock(mutex_);
        check_hid_status(
            rp2350_hid_session_mouse_move(require_session(), dx, dy));
    }

    void click_left() override {
        std::scoped_lock lock(mutex_);
        check_hid_status(
            rp2350_hid_session_mouse_click(require_session(), "left"));
    }

    void stop_all() override {
        std::scoped_lock lock(mutex_);
        check_hid_status(rp2350_hid_session_stop_all(require_session()));
    }

    void close() noexcept override {
        std::scoped_lock lock(mutex_);
        Rp2350HidSession* previous = std::exchange(session_, nullptr);
        if (previous != nullptr) {
            rp2350_hid_session_release(previous);
        }
    }

private:
    static void validate_health(Rp2350HidSession* session) {
        check_hid_status(rp2350_hid_session_ping(session));
        (void)parse_rp2350_v2_health(
            read_payload(session, rp2350_hid_session_info),
            read_payload(session, rp2350_hid_session_caps));
    }

    Rp2350HidSession* require_session() const {
        if (session_ == nullptr) {
            throw std::runtime_error("RP2350 HID client is closed");
        }
        return session_;
    }

    std::mutex mutex_;
    Rp2350HidSession* session_ = nullptr;
};
#endif

}  // namespace

HidDeviceHealth parse_rp2350_v2_health(
    const std::vector<std::uint8_t>& info,
    const std::vector<std::uint8_t>& caps) {
    if (info.size() < 4 || caps.size() < 5 ||
        info[0] != kRp2350ProtocolV2 || caps[0] != kRp2350ProtocolV2) {
        throw std::runtime_error(kRp2350ProtocolV2Error);
    }

    const auto capabilities = static_cast<std::uint16_t>(
        (static_cast<std::uint16_t>(caps[3]) << 8) |
        static_cast<std::uint16_t>(caps[4]));
    if ((capabilities & kRp2350RequiredCapabilities) !=
        kRp2350RequiredCapabilities) {
        throw std::runtime_error(kRp2350ProtocolV2Error);
    }

    return HidDeviceHealth{kRp2350ProtocolV2, capabilities};
}

void close_hid_client_noexcept(HidClient* client) noexcept {
    if (client != nullptr) {
        client->close();
    }
}

HidActionSender::HidActionSender(HidClient& client) : client_(client) {}

void HidActionSender::set_enabled(bool enabled) {
    std::scoped_lock lock(output_mutex_);
    enabled_.store(enabled);
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
        "Set RP2350_HID_BRIDGE_SDK to the SDK root or build on Windows.");
#endif
}

std::unique_ptr<HidClient> create_rp2350_hid_client(
    Rp2350HidSession* session) {
#if defined(VISION_ANALYZER_WITH_RP2350_HID) && defined(_WIN32)
    return std::make_unique<Rp2350HidClient>(session);
#else
    (void)session;
    throw std::runtime_error(
        "RP2350 HID bridge SDK is not available in this build. "
        "Set RP2350_HID_BRIDGE_SDK to the SDK root or build on Windows.");
#endif
}

}  // namespace vision_analyzer
