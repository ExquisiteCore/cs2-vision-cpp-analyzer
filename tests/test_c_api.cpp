#include <cstring>
#include <iostream>
#include <stdexcept>

#include "vision_analyzer/vision_runtime_c_api.h"

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
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
    require(va_set_player_side(runtime, "ct") == 0, "set player side should succeed");
    require(va_set_hid_port(runtime, "COM3") == 0, "set HID port should succeed");
    require(va_set_dry_run(runtime, 1) == 0, "set dry-run should succeed");
    require(va_set_hid_click(runtime, 1, 6) == 0, "set HID click should succeed");
    require(va_set_hid_tuning(runtime, 0.5F, 80, 2.0F) == 0, "set HID tuning should succeed");
    require(va_set_thresholds(runtime, 0.25F, 0.45F) == 0, "set thresholds should succeed");
    require(va_set_dxgi_roi(runtime, 0, 0, 640, 480) == 0, "set DXGI ROI should succeed");
    require(va_set_frame_limits(runtime, 10, 0) == 0, "set frame limits should succeed");

    va_destroy(runtime);
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
        test_create_destroy();
        test_setters_accept_valid_values();
        test_process_before_open_reports_error();
        test_invalid_video_open_reports_error();
        std::cout << "C API tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "C API test failed: " << error.what() << '\n';
        return 1;
    }
}
