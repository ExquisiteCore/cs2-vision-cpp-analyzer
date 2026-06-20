#pragma once

#include <stdint.h>

#if defined(_WIN32) && defined(VISION_RUNTIME_BUILD_DLL)
#define VA_API __declspec(dllexport)
#else
#define VA_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VaRuntime VaRuntime;

typedef struct VaRuntimeAction {
    int32_t frame_index;
    double timestamp_ms;
    double fps;
    double preprocess_ms;
    double inference_ms;
    double postprocess_ms;
    double total_ms;
    int32_t detection_count;
    int32_t has_target;
    int32_t dx;
    int32_t dy;
    int32_t click_left;
    int32_t lock_state;
    double distance;
    double offset_x;
    double offset_y;
    double target_x;
    double target_y;
} VaRuntimeAction;

VA_API VaRuntime* va_create(void);
VA_API void va_destroy(VaRuntime* runtime);
VA_API const char* va_last_error(VaRuntime* runtime);

VA_API int32_t va_load_config(VaRuntime* runtime, const char* path);
VA_API int32_t va_set_model(VaRuntime* runtime, const char* model_path);
VA_API int32_t va_set_schema(VaRuntime* runtime, const char* schema_path);
VA_API int32_t va_set_backend(VaRuntime* runtime, const char* backend);
VA_API int32_t va_set_player_side(VaRuntime* runtime, const char* side);
VA_API int32_t va_set_hid_port(VaRuntime* runtime, const char* port);
VA_API int32_t va_set_dry_run(VaRuntime* runtime, int32_t dry_run);
VA_API int32_t va_set_hid_click(VaRuntime* runtime, int32_t enabled, int32_t cooldown_frames);
VA_API int32_t va_set_hid_tuning(VaRuntime* runtime, float gain, int32_t max_step, float deadzone_px);
VA_API int32_t va_set_thresholds(VaRuntime* runtime, float confidence, float nms_threshold);
VA_API int32_t va_set_dxgi_roi(VaRuntime* runtime, int32_t x, int32_t y, int32_t width, int32_t height);
VA_API int32_t va_set_frame_limits(VaRuntime* runtime, int32_t max_frames, int32_t warmup_frames);

VA_API int32_t va_open_video(VaRuntime* runtime, const char* video_path, int32_t dry_run);
VA_API int32_t va_open_dxgi(VaRuntime* runtime, int32_t adapter, int32_t output, int32_t dry_run);
VA_API int32_t va_process_next(VaRuntime* runtime, VaRuntimeAction* action);
VA_API int32_t va_stop_all(VaRuntime* runtime);
VA_API int32_t va_close(VaRuntime* runtime);

#ifdef __cplusplus
}
#endif
