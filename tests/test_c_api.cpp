#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "rp2350_hid_bridge/c_api.h"
#include "vision_analyzer/vision_runtime_c_api.h"

static_assert(VA_RUNTIME_ABI_MAJOR == 2u);
static_assert(VA_RUNTIME_ABI_MINOR == 1u);
static_assert(VA_HID_CALIBRATION_LEVELS == 3);
static_assert(sizeof(VaRuntimeAction) == 120);
static_assert(sizeof(VaHidCalibrationProfile) == 84);
static_assert(sizeof(VaRuntimeAbiInfo) == 32);

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_runtime_abi_info() {
    require(va_get_abi_info(nullptr) == -1, "null ABI output must fail");

    VaRuntimeAbiInfo info{};
    info.struct_size = sizeof(info);
    require(va_get_abi_info(&info) == 0, "ABI query should succeed");
    require(info.abi_major == 2 && info.abi_minor == 1,
            "DLL must expose ABI 2.1");
    require(info.runtime_action_size == sizeof(VaRuntimeAction),
            "action size must match the public header");
    require(info.hid_calibration_profile_size == sizeof(VaHidCalibrationProfile),
            "calibration size must match the public header");
    require((info.feature_flags & VA_RUNTIME_FEATURE_TENSORRT_CACHE) != 0,
            "TensorRT cache feature must be declared");
    require((info.feature_flags & VA_RUNTIME_FEATURE_PERSISTENT_CALIBRATION) != 0,
            "persistent calibration feature must be declared");
    require((info.feature_flags & VA_RUNTIME_FEATURE_OUTPUT_ARMING) != 0,
            "output arming feature must be declared");
    require((info.feature_flags & VA_RUNTIME_FEATURE_FIRE_ARMING) != 0,
            "fire arming feature must be declared");
    require((info.feature_flags & VA_RUNTIME_FEATURE_SHARED_HID_SESSION) != 0,
            "shared HID session feature must be declared");
    require(info.feature_flags == 31,
            "ABI 2.1 must expose the coordinated feature set");
}

void test_shared_hid_session_attachment_and_port_exclusion() {
    Rp2350HidOptions options{};
    options.struct_size = sizeof(options);
    options.port = "COM_TEST";
    options.baud = 115200;
    options.timeout_ms = 1000;
    options.retries = 2;
    options.heartbeat_interval_ms = 500;

    Rp2350HidSession* hid = nullptr;
    require(
        rp2350_hid_session_create(&options, &hid) == 0,
        "unopened HID handle should be creatable without hardware");

    VaRuntime* runtime = va_create();
    require(runtime != nullptr, "runtime should exist");
    require(
        va_attach_hid_session(runtime, hid) == 0,
        "runtime should retain an attached HID handle");
    require(
        va_stop_all(runtime) == -1,
        "shared runtime must reject its legacy global stop API");
    require(
        std::strstr(va_last_error(runtime), "hid.stop_all") != nullptr,
        "shared stop error should direct Python ownership to HidSession");
    require(
        va_set_hid_port(runtime, "COM4") == -1,
        "attached session and private port must be mutually exclusive");
    require(
        std::strstr(va_last_error(runtime), "attached HID session") != nullptr,
        "port conflict should explain the attached session");
    require(
        va_close(runtime) == 0,
        "reset should keep the configured attachment valid");
    require(
        va_attach_hid_session(runtime, nullptr) == 0,
        "READY runtime should detach after reset");
    require(
        va_set_hid_port(runtime, "COM4") == 0,
        "private port should be configurable after detach");
    require(
        va_attach_hid_session(runtime, hid) == -1,
        "private port must reject a shared-session attachment");
    require(
        std::strstr(va_last_error(runtime), "private HID port") != nullptr,
        "attachment conflict should explain the private port");
    require(
        va_set_hid_port(runtime, nullptr) == 0,
        "private port should be clearable");

    va_destroy(runtime);
    rp2350_hid_session_release(hid);
}

std::filesystem::path c_api_calibration_test_directory() {
    return std::filesystem::temp_directory_path() /
           "vision-runtime-c-api-calibration-tests";
}

void write_valid_calibration_json(const std::filesystem::path& path) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << "{\n"
           << "  \"schema_version\": 1,\n"
           << "  \"frame_width\": 1920,\n"
           << "  \"frame_height\": 1080,\n"
           << "  \"x_shift_px\": [8.05, 23.53, 46.58],\n"
           << "  \"x_counts_per_pixel\": [1.37, 1.40, 1.42],\n"
           << "  \"y_shift_px\": [7.93, 24.20, 47.15],\n"
           << "  \"y_counts_per_pixel\": [1.39, 1.41, 1.42],\n"
           << "  \"deadzone_px\": 1.0,\n"
           << "  \"max_step\": 120,\n"
           << "  \"noise_px\": 0.009,\n"
           << "  \"quality\": 0.678,\n"
           << "  \"accepted_samples\": 24\n"
           << "}\n";
}

void test_create_destroy() {
    VaRuntime* runtime = va_create();
    require(runtime != nullptr, "va_create should return a runtime handle");
    require(std::strcmp(va_last_error(runtime), "") == 0, "new runtime should have no error");
    va_destroy(runtime);
}

void test_setters_accept_valid_values() {
    VaRuntime* runtime = va_create();
    require(runtime != nullptr, "va_create should return a runtime handle");

    require(va_set_model(runtime, "model.onnx") == 0, "set model should succeed");
    require(va_set_schema(runtime, "model.onnx.schema.json") == 0, "set schema should succeed");
    require(va_set_backend(runtime, "opencv-onnx") == 0, "set backend should succeed");
    require(
        va_set_tensorrt_cache_path(runtime, "D:\\cache\\sm61") == 0,
        "set TensorRT cache should succeed"
    );
    require(va_set_player_side(runtime, "ct") == 0, "set player side should succeed");
    require(va_set_hid_port(runtime, "COM3") == 0, "set HID port should succeed");
    require(va_set_dry_run(runtime, 1) == 0, "set dry-run should succeed");
    require(va_set_output_enabled(runtime, 1) == 0, "arming output should succeed");
    require(va_set_output_enabled(runtime, 0) == 0, "disarming output should succeed");
    require(va_set_hid_click(runtime, 1, 6) == 0, "set HID click should succeed");
    require(va_set_hid_tuning(runtime, 0.5F, 80, 2.0F) == 0, "set HID tuning should succeed");
    require(va_set_thresholds(runtime, 0.25F, 0.45F) == 0, "set thresholds should succeed");
    require(va_set_dxgi_roi(runtime, 0, 0, 640, 480) == 0, "set DXGI ROI should succeed");
    require(va_set_frame_limits(runtime, 10, 0) == 0, "set frame limits should succeed");

    va_destroy(runtime);
}

void test_tensorrt_cache_setter_rejects_empty_path() {
    VaRuntime* runtime = va_create();
    require(runtime != nullptr, "va_create should return a runtime handle");

    require(va_set_tensorrt_cache_path(runtime, nullptr) == -1, "null TensorRT cache path should fail");
    require(
        std::strstr(va_last_error(runtime), "cache path") != nullptr,
        "TensorRT cache error should explain the invalid path"
    );

    va_destroy(runtime);
}

void test_fire_and_calibration_api_validation() {
    VaRuntime* runtime = va_create();
    require(runtime != nullptr, "runtime should exist");
    require(va_set_fire_enabled(runtime, 1) == 0, "fire enable should succeed");
    require(va_set_fire_policy(runtime, 1, 0.35F, 0.45F, 3) == 0,
            "valid fire policy should succeed");
    require(va_set_fire_policy(runtime, 1, 1.5F, 0.45F, 3) == -1,
            "invalid head confidence should fail");
    require(std::strstr(va_last_error(runtime), "head") != nullptr,
            "invalid fire policy should explain the head threshold");
    require(va_calibrate_hid(runtime, 0, 0, nullptr) == -1,
            "null calibration output should fail before touching hardware");
    require(std::strstr(va_last_error(runtime), "pointer") != nullptr,
            "null calibration output should report a pointer error");
    va_destroy(runtime);
}

void test_calibration_path_api_validates_pointers() {
    VaHidCalibrationProfile profile{};
    require(va_set_hid_calibration_path(nullptr, "profile.json") == -1,
            "null runtime path setter should fail");
    require(va_get_hid_calibration(nullptr, &profile) == -1,
            "null runtime calibration getter should fail");

    VaRuntime* runtime = va_create();
    require(runtime != nullptr, "runtime should exist");
    require(va_set_hid_calibration_path(runtime, nullptr) == -1,
            "null calibration path should fail");
    require(va_set_hid_calibration_path(runtime, "") == -1,
            "empty calibration path should fail");
    require(va_get_hid_calibration(runtime, nullptr) == -1,
            "null calibration output should fail");
    require(std::strstr(va_last_error(runtime), "pointer") != nullptr,
            "null calibration output should explain the pointer error");
    va_destroy(runtime);
}

void test_calibration_getter_zeroes_output_when_no_profile_is_installed() {
    VaRuntime* runtime = va_create();
    require(runtime != nullptr, "runtime should exist");
    VaHidCalibrationProfile profile;
    std::memset(&profile, 0x7F, sizeof(profile));

    require(va_get_hid_calibration(runtime, &profile) == 0,
            "empty calibration getter should succeed");
    require(profile.schema_version == 0 && profile.valid == 0 &&
            profile.frame_width == 0 && profile.max_step == 0,
            "empty calibration getter should zero-initialize the output");
    va_destroy(runtime);
}

void test_calibration_path_load_and_get_are_transactional() {
    const auto directory = c_api_calibration_test_directory();
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const auto valid_path = directory / "valid.json";
    write_valid_calibration_json(valid_path);

    VaRuntime* runtime = va_create();
    require(runtime != nullptr, "runtime should exist");
    require(va_set_hid_calibration_path(runtime, valid_path.string().c_str()) == 0,
            "valid calibration path should load");

    VaHidCalibrationProfile profile{};
    require(va_get_hid_calibration(runtime, &profile) == 0 && profile.valid == 1,
            "loaded calibration should be installed");
    require(profile.schema_version == 1 && profile.frame_width == 1920 &&
            profile.frame_height == 1080 && profile.max_step == 120 &&
            profile.accepted_samples == 24,
            "getter should expose the complete loaded profile");

    const auto corrupt_path = directory / "corrupt.json";
    {
        std::ofstream output(corrupt_path, std::ios::binary | std::ios::trunc);
        output << "{broken";
    }
    require(va_set_hid_calibration_path(runtime, corrupt_path.string().c_str()) == -1,
            "corrupt calibration path should fail");
    std::memset(&profile, 0, sizeof(profile));
    require(va_get_hid_calibration(runtime, &profile) == 0 && profile.valid == 1,
            "corrupt path selection should retain the old installed profile");
    require(profile.frame_width == 1920 && profile.max_step == 120,
            "retained profile should remain complete");

    const auto missing_path = directory / "new-profile.json";
    require(!std::filesystem::exists(missing_path),
            "new calibration destination should begin missing");
    require(va_set_hid_calibration_path(runtime, missing_path.string().c_str()) == 0,
            "missing calibration destination should be accepted");
    std::memset(&profile, 0x7F, sizeof(profile));
    require(va_get_hid_calibration(runtime, &profile) == 0 && profile.valid == 0,
            "selecting a missing destination should clear the old profile");
    require(profile.schema_version == 0 && profile.frame_width == 0,
            "cleared profile output should be fully zeroed");

    va_destroy(runtime);
    std::filesystem::remove_all(directory);
}

void test_process_before_open_reports_error() {
    VaRuntime* runtime = va_create();
    require(runtime != nullptr, "va_create should return a runtime handle");

    VaRuntimeAction action{};
    require(va_process_next(runtime, &action) == -1, "process before open should fail");
    require(std::strstr(va_last_error(runtime), "not open") != nullptr, "last error should explain closed session");

    va_destroy(runtime);
}

void test_invalid_video_open_reports_error() {
    VaRuntime* runtime = va_create();
    require(runtime != nullptr, "va_create should return a runtime handle");

    require(va_set_model(runtime, "missing.onnx") == 0, "set model should succeed before invalid open");
    require(va_set_frame_limits(runtime, 1, 0) == 0, "set frame limits should succeed before invalid open");
    require(va_open_video(runtime, "Z:\\definitely_missing_video.mp4", 1) == -1, "invalid video should fail to open");
    require(std::strcmp(va_last_error(runtime), "") != 0, "invalid open should set last error");

    va_destroy(runtime);
}

}  // namespace

int main() {
    try {
        test_runtime_abi_info();
        test_shared_hid_session_attachment_and_port_exclusion();
        test_create_destroy();
        test_setters_accept_valid_values();
        test_tensorrt_cache_setter_rejects_empty_path();
        test_fire_and_calibration_api_validation();
        test_calibration_path_api_validates_pointers();
        test_calibration_getter_zeroes_output_when_no_profile_is_installed();
        test_calibration_path_load_and_get_are_transactional();
        test_process_before_open_reports_error();
        test_invalid_video_open_reports_error();
        std::cout << "C API tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "C API test failed: " << error.what() << '\n';
        return 1;
    }
}
