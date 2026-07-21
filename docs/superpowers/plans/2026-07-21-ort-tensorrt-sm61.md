# GTX 1080 Ti ORT TensorRT Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the DLL runtime use a GTX 1080 Ti-compatible ORT TensorRT FP32 path, reduce DXGI ROI transfer cost, and require explicit arming before RP2350 output.

**Architecture:** Keep the synchronous DLL-first runtime and isolate new behavior in three testable units: static ONNX input validation, fixed SM 6.1 TensorRT provider options, and DXGI copy-region resolution. Wire those units into the existing detector and frame source, then gate the existing HID sender without changing the returned `VaRuntimeAction` contract.

**Tech Stack:** C++17, ONNX Runtime C++ API/TensorRT EP, CUDA EP fallback, Direct3D 11 Desktop Duplication, OpenCV, RP2350 HID SDK, CMake/CTest, xmake.

---

## File map

New focused units:

- `include/vision_analyzer/model_input.hpp`: hardware-independent static NCHW model-input contract.
- `src/model_input.cpp`: model shape/type validation and diagnostic formatting.
- `include/vision_analyzer/tensorrt_provider_config.hpp`: fixed SM 6.1 TensorRT provider key/value contract.
- `src/tensorrt_provider_config.cpp`: GTX 1080 Ti FP32 provider option construction.
- `include/vision_analyzer/dxgi_roi.hpp`: hardware-independent DXGI source-region API.
- `src/dxgi_roi.cpp`: full-frame, clipped-ROI, and invalid-ROI resolution.

Existing files to modify:

- `include/vision_analyzer/types.hpp`: production defaults, cache path, and output-armed option.
- `include/vision_analyzer/detector.hpp`: report detector input size and construct from complete runtime options.
- `src/detector.cpp`: ORT input discovery and fixed TensorRT/CUDA provider chain.
- `include/vision_analyzer/postprocess.hpp`, `src/postprocess.cpp`: support discovered rectangular input sizes.
- `src/runtime_session.cpp`, `include/vision_analyzer/runtime_session.hpp`: detector-sized warm-up and live output arming.
- `src/frame_source.cpp`: ROI-sized staging texture and `CopySubresourceRegion`.
- `include/vision_analyzer/hid_output.hpp`, `src/hid_output.cpp`: thread-safe disabled-by-default action sender.
- `src/runtime_config.cpp`, `src/main.cpp`, `runtime.example.cfg`: simple configuration/CLI surface.
- `include/vision_analyzer/vision_runtime_c_api.h`, `src/vision_runtime_c_api.cpp`: two additive C functions.
- `tests/test_algorithms.cpp`, `tests/test_c_api.cpp`: behavior tests.
- `CMakeLists.txt`, `xmake.lua`: add new core units and remove incompatible Python GPU runtime injection.
- `README.md`: GTX 1080 Ti deployment and C API usage.

### Task 1: Validate and use the ONNX model input shape

**Files:**

- Create: `include/vision_analyzer/model_input.hpp`
- Create: `src/model_input.cpp`
- Modify: `include/vision_analyzer/postprocess.hpp`
- Modify: `src/postprocess.cpp`
- Modify: `include/vision_analyzer/detector.hpp`
- Modify: `src/detector.cpp`
- Modify: `src/runtime_session.cpp`
- Modify: `tests/test_algorithms.cpp`
- Modify: `CMakeLists.txt`
- Modify: `xmake.lua`

- [ ] **Step 1: Write failing static-input and rectangular-letterbox tests**

Add the new header include and these tests to `tests/test_algorithms.cpp`:

```cpp
#include <array>

#include "vision_analyzer/model_input.hpp"

void test_model_input_accepts_static_fp32_nchw() {
    const ModelInputSpec spec = parse_model_input_spec(
        {1, 3, 640, 640},
        ModelElementType::Float32
    );
    require(spec.width == 640 && spec.height == 640, "model input should preserve static size");
    require(spec.shape == std::array<std::int64_t, 4>{1, 3, 640, 640}, "model input should preserve NCHW shape");
}

void test_model_input_rejects_dynamic_or_non_fp32_input() {
    bool dynamic_rejected = false;
    try {
        (void)parse_model_input_spec({1, 3, -1, -1}, ModelElementType::Float32);
    } catch (const std::runtime_error&) {
        dynamic_rejected = true;
    }
    require(dynamic_rejected, "dynamic model input should be rejected");

    bool type_rejected = false;
    try {
        (void)parse_model_input_spec({1, 3, 640, 640}, ModelElementType::Unsupported);
    } catch (const std::runtime_error&) {
        type_rejected = true;
    }
    require(type_rejected, "non-FP32 model input should be rejected");
}

void test_letterbox_accepts_rectangular_target() {
    const cv::Mat source(720, 1280, CV_8UC3, cv::Scalar(0, 0, 0));
    const LetterboxResult result = letterbox(source, cv::Size(640, 384));
    require(result.image.size() == cv::Size(640, 384), "letterbox should use discovered width and height");
}
```

Register all three calls in `main()`.

- [ ] **Step 2: Run the algorithm target and verify RED**

Run:

```powershell
xmake build vision_analyzer_tests
```

Expected: compilation fails because `model_input.hpp`, `ModelInputSpec`, and the `cv::Size` overload do not exist.

- [ ] **Step 3: Add the static model-input contract**

Create `include/vision_analyzer/model_input.hpp`:

```cpp
#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace vision_analyzer {

enum class ModelElementType {
    Float32,
    Unsupported,
};

struct ModelInputSpec {
    int width = 0;
    int height = 0;
    std::array<std::int64_t, 4> shape{};
};

[[nodiscard]] ModelInputSpec parse_model_input_spec(
    const std::vector<std::int64_t>& shape,
    ModelElementType element_type
);

}  // namespace vision_analyzer
```

Create `src/model_input.cpp` with these validations:

```cpp
#include "vision_analyzer/model_input.hpp"

#include <limits>
#include <sstream>
#include <stdexcept>

namespace vision_analyzer {
namespace {

std::string format_shape(const std::vector<std::int64_t>& shape) {
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) output << ", ";
        output << shape[index];
    }
    output << ']';
    return output.str();
}

}  // namespace

ModelInputSpec parse_model_input_spec(const std::vector<std::int64_t>& shape, ModelElementType element_type) {
    if (element_type != ModelElementType::Float32) {
        throw std::runtime_error("ORT model input must be FP32");
    }
    if (shape.size() != 4 || shape[0] != 1 || shape[1] != 3) {
        throw std::runtime_error("ORT model input must be static NCHW [1, 3, H, W], got " + format_shape(shape));
    }
    if (shape[2] <= 0 || shape[3] <= 0 ||
        shape[2] > std::numeric_limits<int>::max() || shape[3] > std::numeric_limits<int>::max()) {
        throw std::runtime_error("ORT model input height and width must be positive static integers, got " + format_shape(shape));
    }
    return ModelInputSpec{
        static_cast<int>(shape[3]),
        static_cast<int>(shape[2]),
        {shape[0], shape[1], shape[2], shape[3]},
    };
}

}  // namespace vision_analyzer
```

- [ ] **Step 4: Generalize letterbox and detector input size**

Add this overload and keep the existing integer overload as a delegating compatibility wrapper:

```cpp
LetterboxResult letterbox(const cv::Mat& frame, const cv::Size& target_size) {
    const float scale = std::min(
        target_size.width / static_cast<float>(frame.cols),
        target_size.height / static_cast<float>(frame.rows)
    );
    const int resized_width = static_cast<int>(std::round(frame.cols * scale));
    const int resized_height = static_cast<int>(std::round(frame.rows * scale));
    const int pad_x = (target_size.width - resized_width) / 2;
    const int pad_y = (target_size.height - resized_height) / 2;
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(resized_width, resized_height));
    cv::Mat output(target_size, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(output(cv::Rect(pad_x, pad_y, resized_width, resized_height)));
    return {output, scale, pad_x, pad_y};
}

LetterboxResult letterbox(const cv::Mat& frame, int target_size) {
    return letterbox(frame, cv::Size(target_size, target_size));
}
```

Add `virtual cv::Size input_size() const = 0;` to `Detector`. Return `640x640` from OpenCV/unavailable detectors. In `OrtDetector`, after session creation:

```cpp
if (session_.GetInputCount() != 1) {
    throw std::runtime_error(
        "ORT model must have exactly one image input, got " +
        std::to_string(session_.GetInputCount())
    );
}
const auto input_info = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
input_spec_ = parse_model_input_spec(
    input_info.GetShape(),
    input_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        ? ModelElementType::Float32
        : ModelElementType::Unsupported
);
input_data_.resize(static_cast<std::size_t>(3) * input_spec_.height * input_spec_.width);
```

Use `input_spec_.shape`, `input_spec_.width`, and `input_spec_.height` in preprocessing, input tensor construction, and `fill_input`. Return the discovered size from `OrtDetector::input_size()`. Change runtime warm-up to:

```cpp
const cv::Size warmup_size = detector_->input_size();
cv::Mat warmup_frame(warmup_size, CV_8UC3, cv::Scalar(0, 0, 0));
```

Add `src/model_input.cpp` to both build-system core source lists.

- [ ] **Step 5: Run GREEN verification**

Run:

```powershell
xmake build vision_analyzer_tests
xmake run vision_analyzer_tests
```

Expected: build succeeds and all algorithm tests, including the three new tests, pass.

- [ ] **Step 6: Commit the model-input change**

```powershell
git add include/vision_analyzer/model_input.hpp include/vision_analyzer/postprocess.hpp include/vision_analyzer/detector.hpp src/model_input.cpp src/postprocess.cpp src/detector.cpp src/runtime_session.cpp tests/test_algorithms.cpp CMakeLists.txt xmake.lua
git commit -m "feat: derive ORT input dimensions from model"
```

### Task 2: Add the fixed GTX 1080 Ti TensorRT provider profile

**Files:**

- Create: `include/vision_analyzer/tensorrt_provider_config.hpp`
- Create: `src/tensorrt_provider_config.cpp`
- Modify: `include/vision_analyzer/types.hpp`
- Modify: `include/vision_analyzer/detector.hpp`
- Modify: `src/detector.cpp`
- Modify: `src/runtime_config.cpp`
- Modify: `src/main.cpp`
- Modify: `include/vision_analyzer/vision_runtime_c_api.h`
- Modify: `src/vision_runtime_c_api.cpp`
- Modify: `tests/test_algorithms.cpp`
- Modify: `tests/test_c_api.cpp`
- Modify: `CMakeLists.txt`
- Modify: `xmake.lua`

- [ ] **Step 1: Write failing provider/default/config/C API tests**

Add tests that assert:

```cpp
void test_sm61_tensorrt_profile_is_fp32_and_cached() {
    const auto options = make_sm61_tensorrt_provider_options("D:\\cache\\sm61");
    const auto find_value = [&](const std::string& key) -> std::string {
        for (const auto& option : options) if (option.key == key) return option.value;
        return {};
    };
    require(find_value("device_id") == "0", "TensorRT should use GPU 0");
    require(find_value("trt_fp16_enable") == "0", "1080 Ti profile should use FP32");
    require(find_value("trt_engine_cache_enable") == "1", "TensorRT cache should be enabled");
    require(find_value("trt_engine_cache_path") == "D:\\cache\\sm61", "cache path should be forwarded");
    require(find_value("trt_max_workspace_size") == "2147483648", "workspace should be 2 GiB");
}

void test_runtime_defaults_to_sm61_tensorrt() {
    const Options options;
    require(options.backend == Backend::OrtTensorRt, "runtime should default to ORT TensorRT");
    require(options.tensorrt_cache_path == "ort-trt-cache-sm61-fp32", "runtime should have a writable relative cache default");
}
```

Extend the existing config fixture with `tensorrt_cache_path=cache-from-config` and assert the stored value. Extend `test_setters_accept_valid_values()` with:

```cpp
require(va_set_tensorrt_cache_path(runtime, "D:\\cache\\sm61") == 0, "set TensorRT cache should succeed");
```

Add a separate invalid test asserting a null or empty cache path returns `-1` and sets `va_last_error`.

- [ ] **Step 2: Run both test targets and verify RED**

```powershell
xmake build vision_analyzer_tests vision_runtime_c_api_tests
```

Expected: compilation fails because the provider option helper, new option field, and C function do not exist.

- [ ] **Step 3: Implement the provider option builder**

Create `include/vision_analyzer/tensorrt_provider_config.hpp`:

```cpp
#pragma once

#include <string>
#include <vector>

namespace vision_analyzer {

struct TensorRtProviderOption {
    std::string key;
    std::string value;
};

[[nodiscard]] std::vector<TensorRtProviderOption> make_sm61_tensorrt_provider_options(
    const std::string& cache_path
);

}  // namespace vision_analyzer
```

Create `src/tensorrt_provider_config.cpp` with:

```cpp
#include "vision_analyzer/tensorrt_provider_config.hpp"

#include <stdexcept>

namespace vision_analyzer {

std::vector<TensorRtProviderOption> make_sm61_tensorrt_provider_options(const std::string& cache_path) {
    if (cache_path.empty()) {
        throw std::runtime_error("TensorRT cache path must not be empty");
    }
    return {
        {"device_id", "0"},
        {"trt_fp16_enable", "0"},
        {"trt_engine_cache_enable", "1"},
        {"trt_engine_cache_path", cache_path},
        {"trt_max_workspace_size", "2147483648"},
        {"trt_min_subgraph_size", "1"},
    };
}

}  // namespace vision_analyzer
```

Add the new source to CMake and xmake.

- [ ] **Step 4: Wire the fixed profile into ORT session creation**

Set these `Options` defaults:

```cpp
Backend backend = Backend::OrtTensorRt;
std::string tensorrt_cache_path = "ort-trt-cache-sm61-fp32";
```

Change the factory to `create_detector(const Options& options)`. Resolve the configured cache path to an absolute path, create it, convert the returned option strings to stable `const char*` arrays, and call `UpdateTensorRTProviderOptions`. Append TensorRT first and CUDA second.

Wrap TensorRT provider/session construction failures with:

```cpp
throw std::runtime_error(
    "ORT TensorRT initialization failed. GTX 1080 Ti production requires "
    "ONNX Runtime 1.17.x, TensorRT 8.6.x, CUDA 11.8, and cuDNN 8.9.x: " +
    std::string(error.what())
);
```

Do not create a second CUDA-only session after that exception.

- [ ] **Step 5: Add config, CLI, and C API cache setters**

Parse `tensorrt_cache_path` in `runtime_config.cpp`. Add CLI `--tensorrt-cache-path PATH`. Export:

```c
VA_API int32_t va_set_tensorrt_cache_path(VaRuntime* runtime, const char* path);
```

Implement it with `required_string(path, "TensorRT cache path")`, copying the string into `runtime->options.tensorrt_cache_path`.

- [ ] **Step 6: Run GREEN verification**

```powershell
xmake build vision_analyzer_tests vision_runtime_c_api_tests
xmake run vision_analyzer_tests
xmake run vision_runtime_c_api_tests
```

Expected: both targets build and all algorithm/C API tests pass.

- [ ] **Step 7: Commit the provider profile**

```powershell
git add include/vision_analyzer/tensorrt_provider_config.hpp include/vision_analyzer/types.hpp include/vision_analyzer/detector.hpp include/vision_analyzer/vision_runtime_c_api.h src/tensorrt_provider_config.cpp src/detector.cpp src/runtime_config.cpp src/main.cpp src/vision_runtime_c_api.cpp tests/test_algorithms.cpp tests/test_c_api.cpp CMakeLists.txt xmake.lua
git commit -m "feat: add GTX 1080 Ti ORT TensorRT profile"
```

### Task 3: Crop DXGI ROI before CPU readback

**Files:**

- Create: `include/vision_analyzer/dxgi_roi.hpp`
- Create: `src/dxgi_roi.cpp`
- Modify: `src/frame_source.cpp`
- Modify: `tests/test_algorithms.cpp`
- Modify: `CMakeLists.txt`
- Modify: `xmake.lua`

- [ ] **Step 1: Write failing copy-region tests**

```cpp
#include "vision_analyzer/dxgi_roi.hpp"

void test_dxgi_copy_region_uses_full_frame_when_roi_is_disabled() {
    require(resolve_dxgi_copy_region({1920, 1080}, {}) == cv::Rect(0, 0, 1920, 1080),
            "disabled ROI should copy the full desktop");
}

void test_dxgi_copy_region_clips_to_desktop() {
    require(resolve_dxgi_copy_region({1920, 1080}, {1800, 1000, 640, 640}) == cv::Rect(1800, 1000, 120, 80),
            "ROI should be clipped before GPU copy");
}

void test_dxgi_copy_region_rejects_empty_intersection() {
    bool rejected = false;
    try {
        (void)resolve_dxgi_copy_region({1920, 1080}, {2000, 1200, 100, 100});
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "out-of-frame ROI should be rejected");
}
```

Register the tests in `main()`.

- [ ] **Step 2: Run the algorithm target and verify RED**

```powershell
xmake build vision_analyzer_tests
```

Expected: compilation fails because `dxgi_roi.hpp` and `resolve_dxgi_copy_region` do not exist.

- [ ] **Step 3: Implement hardware-independent ROI resolution**

Create `include/vision_analyzer/dxgi_roi.hpp`:

```cpp
#pragma once

#include <opencv2/core.hpp>

namespace vision_analyzer {

[[nodiscard]] cv::Rect resolve_dxgi_copy_region(
    const cv::Size& source_size,
    const cv::Rect& requested_roi
);

}  // namespace vision_analyzer
```

Create `src/dxgi_roi.cpp`:

```cpp
#include "vision_analyzer/dxgi_roi.hpp"

#include <stdexcept>

namespace vision_analyzer {

cv::Rect resolve_dxgi_copy_region(const cv::Size& source_size, const cv::Rect& requested_roi) {
    if (source_size.width <= 0 || source_size.height <= 0) {
        throw std::runtime_error("DXGI source dimensions must be positive");
    }
    const cv::Rect full_frame(0, 0, source_size.width, source_size.height);
    if (requested_roi.width == 0 && requested_roi.height == 0) {
        return full_frame;
    }
    if (requested_roi.width <= 0 || requested_roi.height <= 0) {
        throw std::runtime_error("DXGI ROI width and height must both be positive");
    }
    const cv::Rect clipped = requested_roi & full_frame;
    if (clipped.empty()) {
        throw std::runtime_error("DXGI ROI is outside the captured frame");
    }
    return clipped;
}

}  // namespace vision_analyzer
```

Add the new source to both core source lists.

- [ ] **Step 4: Replace full-resource copying with an ROI-sized staging copy**

In `DxgiFrameSource::copy_texture_to_frame`, resolve the region before creating the staging texture. Change `ensure_staging_texture` to compare/create `region.width` and `region.height`. Copy with:

```cpp
const D3D11_BOX source_box{
    static_cast<UINT>(region.x),
    static_cast<UINT>(region.y),
    0,
    static_cast<UINT>(region.x + region.width),
    static_cast<UINT>(region.y + region.height),
    1,
};
context_->CopySubresourceRegion(staging_texture_.Get(), 0, 0, 0, 0, texture, 0, &source_box);
```

Construct the mapped BGRA view with the staging/region dimensions, convert it once to BGR, and delete the later CPU ROI clone. Keep full-frame behavior by treating the full desktop as a copy region.

- [ ] **Step 5: Run GREEN verification**

```powershell
xmake build vision_analyzer_tests
xmake run vision_analyzer_tests
```

Expected: ROI unit tests and all existing algorithm tests pass.

- [ ] **Step 6: Commit the GPU ROI change**

```powershell
git add include/vision_analyzer/dxgi_roi.hpp src/dxgi_roi.cpp src/frame_source.cpp tests/test_algorithms.cpp CMakeLists.txt xmake.lua
git commit -m "perf: crop DXGI frames before CPU readback"
```

### Task 4: Add disabled-by-default RP2350 output arming

**Files:**

- Modify: `include/vision_analyzer/types.hpp`
- Modify: `include/vision_analyzer/hid_output.hpp`
- Modify: `src/hid_output.cpp`
- Modify: `include/vision_analyzer/runtime_session.hpp`
- Modify: `src/runtime_session.cpp`
- Modify: `src/runtime_config.cpp`
- Modify: `src/main.cpp`
- Modify: `include/vision_analyzer/vision_runtime_c_api.h`
- Modify: `src/vision_runtime_c_api.cpp`
- Modify: `tests/test_algorithms.cpp`
- Modify: `tests/test_c_api.cpp`

- [ ] **Step 1: Write failing sender, default, config, and C API tests**

Replace the sender test with explicit arming behavior:

```cpp
void test_hid_action_sender_requires_arming_and_stops_when_disarmed() {
    RecordingHidClient client;
    HidActionSender sender(client);
    const AimCommand command{true, 12, -5, true, LockState::Locked};

    sender.execute(command);
    require(client.moves.empty() && client.left_clicks == 0, "disarmed sender must suppress output");

    sender.set_enabled(true);
    sender.execute(command);
    require(client.moves.size() == 1 && client.left_clicks == 1, "armed sender should emit the planned command");

    sender.set_enabled(false);
    require(client.stop_calls == 1, "disarming should immediately stop RP2350 state");
}
```

Change `RecordingHidClient::stop_all` to increment `stop_calls`. Assert `Options{}.output_enabled == false`; extend the config fixture with `output_enabled=true`; and extend C API tests with:

```cpp
require(va_set_output_enabled(runtime, 1) == 0, "arming output should succeed");
require(va_set_output_enabled(runtime, 0) == 0, "disarming output should succeed");
```

- [ ] **Step 2: Run both test targets and verify RED**

```powershell
xmake build vision_analyzer_tests vision_runtime_c_api_tests
```

Expected: compilation fails because `output_enabled`, `set_enabled`, and `va_set_output_enabled` do not exist.

- [ ] **Step 3: Implement a thread-safe gated HID sender**

Add these members and declaration to `HidActionSender` in `hid_output.hpp` (plus `<atomic>` and `<mutex>` includes):

```cpp
void set_enabled(bool enabled);

private:
    HidClient& client_;
    std::atomic_bool enabled_{false};
    std::mutex output_mutex_;
```

Serialize `execute`, `set_enabled`, and `stop_all` so the RP2350 client is never called concurrently. Add `<mutex>` to `hid_output.cpp` and use:

```cpp
void HidActionSender::set_enabled(bool enabled) {
    std::scoped_lock lock(output_mutex_);
    enabled_.store(enabled);
    if (!enabled) {
        client_.stop_all();
    }
}

void HidActionSender::execute(const AimCommand& command) {
    std::scoped_lock lock(output_mutex_);
    if (!enabled_.load() || !command.has_target) return;
    if (command.dx != 0 || command.dy != 0) client_.move_relative(command.dx, command.dy);
    if (command.click_left) client_.click_left();
}

void HidActionSender::stop_all() {
    std::scoped_lock lock(output_mutex_);
    client_.stop_all();
}
```

The constructor leaves output disabled.

- [ ] **Step 4: Wire arming through options, session, config, CLI, and C API**

Add `bool output_enabled = false;` to `Options`. Parse `output_enabled` in config and add CLI `--output-enabled`.

After creating `HidActionSender`, call `set_enabled(options_.output_enabled)`. Add:

```cpp
void RuntimeSession::set_output_enabled(bool enabled) {
    options_.output_enabled = enabled;
    if (hid_sender_) hid_sender_->set_enabled(enabled);
}
```

Export and implement:

```c
VA_API int32_t va_set_output_enabled(VaRuntime* runtime, int32_t enabled);
```

The wrapper first stores the state in `runtime->options`; if the session is open, it also calls the session setter. Do not zero or alter the `VaRuntimeAction` returned from `va_process_next` while disarmed.

- [ ] **Step 5: Run GREEN verification**

```powershell
xmake build vision_analyzer_tests vision_runtime_c_api_tests
xmake run vision_analyzer_tests
xmake run vision_runtime_c_api_tests
```

Expected: all sender/default/config/C API tests and existing tests pass.

- [ ] **Step 6: Commit output arming**

```powershell
git add include/vision_analyzer/types.hpp include/vision_analyzer/hid_output.hpp include/vision_analyzer/runtime_session.hpp include/vision_analyzer/vision_runtime_c_api.h src/hid_output.cpp src/runtime_session.cpp src/runtime_config.cpp src/main.cpp src/vision_runtime_c_api.cpp tests/test_algorithms.cpp tests/test_c_api.cpp
git commit -m "feat: require explicit RP2350 output arming"
```

### Task 5: Isolate the production GPU runtime and document deployment

**Files:**

- Modify: `xmake.lua`
- Modify: `runtime.example.cfg`
- Modify: `README.md`

- [ ] **Step 1: Remove incompatible implicit runtime directories**

Delete `torch_lib`, `tensorrt_libs`, and both automatic `add_runenvs` branches from `xmake.lua`. Keep only the configured ONNX Runtime library directory:

```lua
local function add_runtime_runenvs()
    if has_ort then
        add_runenvs("PATH", ort_lib)
    end
end
```

- [ ] **Step 2: Update the example production configuration**

Set and explain:

```text
backend=ort-tensorrt
tensorrt_cache_path=ort-trt-cache-sm61-fp32
output_enabled=false
```

Keep the existing ROI fields and add a comment that a configured ROI activates GPU-side cropping.

- [ ] **Step 3: Add a concise GTX 1080 Ti deployment section**

Document this exact matrix in `README.md`:

```text
ONNX Runtime GPU 1.17.x
TensorRT 8.6.x
CUDA 11.8
cuDNN 8.9.x
GTX 1080 Ti / SM 6.1
FP32
```

Explain that matching DLLs must be in the application directory or process-specific `PATH`, that the first open builds the cache, and that cache contents must be cleared after model/runtime changes. Show the C API sequence:

```c
VaRuntime* runtime = va_create();
va_set_model(runtime, "best.onnx");
va_set_schema(runtime, "best.onnx.schema.json");
va_set_backend(runtime, "ort-tensorrt");
va_set_tensorrt_cache_path(runtime, "ort-trt-cache-sm61-fp32");
va_set_dxgi_roi(runtime, 640, 220, 640, 640);
va_set_player_side(runtime, "ct");
va_set_hid_port(runtime, "COM3");
va_open_dxgi(runtime, 0, 0, 0);
va_set_output_enabled(runtime, 1);
```

State that the current RTX 5060 cannot run a TensorRT 8.6 engine and that hardware acceptance belongs on the 1080 Ti host.

- [ ] **Step 4: Verify the old Python GPU runtime paths are gone**

Run:

```powershell
rg -n "torch_lib|tensorrt_libs|site-packages/tensorrt" xmake.lua README.md runtime.example.cfg
```

Expected: no matches in `xmake.lua`; documentation may mention only the prohibition, not a runtime path.

- [ ] **Step 5: Commit deployment isolation and docs**

```powershell
git add xmake.lua runtime.example.cfg README.md
git commit -m "docs: define GTX 1080 Ti runtime deployment"
```

### Task 6: Full local verification and production handoff

**Files:**

- Verify only; modify files only if a verification failure exposes a defect.

- [ ] **Step 1: Run the full xmake build and tests**

```powershell
xmake build -r vision_runtime vision_analyzer vision_analyzer_tests vision_runtime_c_api_tests
xmake run vision_analyzer_tests
xmake run vision_runtime_c_api_tests
```

Expected: clean build; algorithm and C API executables report success with no failures.

- [ ] **Step 2: Run a real-model CPU smoke test**

Use the explicit CPU backend because the development RTX 5060 cannot validate the production TensorRT 8.6 stack:

```powershell
xmake run vision_analyzer --backend opencv-onnx --model D:\project\cs2-vision-trainer\runs\detect\train\weights\best.onnx --schema D:\project\cs2-vision-trainer\runs\detect\train\weights\best.onnx.schema.json --video D:\project\cs2-vision-trainer\videos\02.mp4 --dry-run --warmup-frames 0 --max-frames 3
```

Expected: three frames process successfully and the reported backend is `opencv-onnx`.

- [ ] **Step 3: Run the full CMake build and CTest suite**

```powershell
cmake -S . -B build-cmake -DONNXRUNTIME_ROOT=D:\Tool\onnxruntime-win-x64-gpu-1.24.4 -DBUILD_TESTING=ON
cmake --build build-cmake --config Release
ctest --test-dir build-cmake -C Release --output-on-failure
```

Expected: DLL, CLI, and both test executables build; CTest reports `100% tests passed, 0 tests failed out of 2`.

- [ ] **Step 4: Audit requirements and repository state**

```powershell
git diff --check
git status --short --branch
rg -n "trt_fp16_enable.*1|IbInputSimulator|tensorrt_libs" include src xmake.lua runtime.example.cfg README.md
```

Expected: no whitespace errors; no unintended changes; no enabled FP16, IbInputSimulator, or implicit TensorRT 11 runtime path.

- [ ] **Step 5: Record the 1080 Ti hardware validation commands for handoff**

Run these on the production host after installing the fixed stack and setting `ONNXRUNTIME_ROOT` to its ORT 1.17.x SDK:

```powershell
$env:ONNXRUNTIME_ROOT='D:\runtime\sm61\onnxruntime-win-x64-gpu-1.17.3'
$env:PATH='D:\runtime\sm61\TensorRT-8.6.1.6\lib;D:\runtime\sm61\cudnn-8.9\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin;' + $env:PATH
xmake f -c --onnxruntime_root=$env:ONNXRUNTIME_ROOT -m release
xmake build vision_runtime vision_analyzer
xmake run vision_analyzer --backend ort-tensorrt --tensorrt-cache-path D:\runtime\sm61\cache --model D:\project\cs2-vision-trainer\runs\detect\train\weights\best.onnx --schema D:\project\cs2-vision-trainer\runs\detect\train\weights\best.onnx.schema.json --input dxgi --dxgi-roi 640 220 640 640 --dry-run --warmup-frames 3 --max-frames 300
```

Expected on the 1080 Ti host: TensorRT session creation succeeds, the cache directory gains an engine file, a second run reuses it, and 300 frames report preprocess/inference/postprocess timing. Do not claim hardware acceptance until these commands have actually run on that host.

Also verify the captured dimensions before the timed run:

```powershell
xmake run vision_analyzer --input dxgi --dxgi-roi 640 220 640 640 --verify-input --dry-run
```

Expected: `width=640 height=640`. RP2350 arming must then be checked through the DLL host: process one frame while disabled and observe no movement, enable with `va_set_output_enabled(runtime, 1)` and observe output, then disable and confirm output stops immediately.

- [ ] **Step 6: Commit any verification-only correction**

If verification required a source correction, first add a failing regression test, reproduce RED, apply the minimal fix, rerun the affected and full suites, then commit only those files with a defect-specific message. If no correction was required, make no empty commit.
