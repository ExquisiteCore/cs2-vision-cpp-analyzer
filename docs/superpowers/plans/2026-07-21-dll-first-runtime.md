# DLL-First Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `vision_runtime.dll` and its C API the primary product, remove the archived UI/EUI dependency, and keep the CLI only as a diagnostic executable.

**Architecture:** Build all reusable runtime implementation in `vision_analyzer_core`, then link that static library into the DLL, CLI, and algorithm tests. Keep `vision_runtime_c_api.cpp` exclusive to the DLL and `main.cpp` exclusive to the CLI so DLL consumers never pull in CLI or UI code.

**Tech Stack:** C++17, OpenCV 4 DNN/VideoIO, optional ONNX Runtime, optional RP2350 HID SDK, xmake, CMake/CTest, MSVC on Windows.

---

### Task 1: Record the baseline and structural RED state

**Files:**
- Inspect: `CMakeLists.txt`
- Inspect: `xmake.lua`
- Inspect: `src/ui_app.cpp`
- Inspect: `README.md`

- [ ] **Step 1: Verify the existing behavior baseline**

Run:

```powershell
xmake -r
xmake run vision_analyzer_tests
xmake run vision_runtime_c_api_tests
```

Expected: build succeeds, `algorithm tests passed`, and `C API tests passed`.

- [ ] **Step 2: Run the structural acceptance check and observe RED**

Run:

```powershell
rg -n "EUI_NEO|EUI-NEO|ui_app|vision_analyzer_ui" CMakeLists.txt xmake.lua src README.md
```

Expected: matches in `CMakeLists.txt`, `xmake.lua`, `src/ui_app.cpp`, and `README.md`, proving the DLL-first cleanup is not yet implemented.

### Task 2: Separate runtime validation from the CLI translation unit

**Files:**
- Create: `src/runtime_options.cpp`
- Modify: `src/main.cpp:205-269`
- Test: `tests/test_algorithms.cpp`

- [ ] **Step 1: Move `validate_options` into a core-owned source file**

Create `src/runtime_options.cpp` with the existing behavior unchanged:

```cpp
#include "vision_analyzer/runtime.hpp"

#include <cmath>
#include <stdexcept>

namespace vision_analyzer {

void validate_options(const Options& options) {
    if (options.dxgi_adapter < 0 || options.dxgi_output < 0) {
        throw std::runtime_error("--dxgi-adapter and --dxgi-output must be greater than or equal to 0");
    }
    if (options.dxgi_timeout_ms <= 0) {
        throw std::runtime_error("--dxgi-timeout must be greater than 0");
    }
    if (options.dxgi_roi.x < 0 || options.dxgi_roi.y < 0 ||
        options.dxgi_roi.width < 0 || options.dxgi_roi.height < 0) {
        throw std::runtime_error("--dxgi-roi values must be greater than or equal to 0");
    }
    if ((options.dxgi_roi.width == 0) != (options.dxgi_roi.height == 0)) {
        throw std::runtime_error("--dxgi-roi width and height must either both be set or both be 0");
    }
    if (!std::isfinite(options.hid_move_gain)) {
        throw std::runtime_error("--hid-gain must be finite");
    }
    if (options.hid_max_step < 0) {
        throw std::runtime_error("--hid-max-step must be greater than or equal to 0");
    }
    if (!std::isfinite(options.hid_deadzone_px) || options.hid_deadzone_px < 0.0F) {
        throw std::runtime_error("--hid-deadzone must be finite and greater than or equal to 0");
    }
    if (!std::isfinite(options.tuning.body_head_anchor_ratio) ||
        options.tuning.body_head_anchor_ratio <= 0.0F ||
        options.tuning.body_head_anchor_ratio >= 0.5F) {
        throw std::runtime_error("--body-head-anchor-ratio must be finite and between 0 and 0.5");
    }
    if (!std::isfinite(options.tuning.kalman_process_noise) || options.tuning.kalman_process_noise <= 0.0F ||
        !std::isfinite(options.tuning.kalman_measurement_noise) || options.tuning.kalman_measurement_noise <= 0.0F ||
        !std::isfinite(options.tuning.kalman_error_covariance) || options.tuning.kalman_error_covariance <= 0.0F) {
        throw std::runtime_error("Kalman tuning values must be finite and greater than 0");
    }
    if (options.list_dxgi_outputs || options.probe_dxgi_outputs || options.verify_input) {
        return;
    }
    if (options.test_hid_move) {
        if (options.hid_port.empty()) {
            throw std::runtime_error("--test-hid-move requires --hid-port COMx");
        }
        return;
    }
    if (options.calibrate_hid) {
        if (options.input_source != InputSource::Dxgi) {
            throw std::runtime_error("--calibrate-hid requires DXGI input");
        }
        if (options.hid_port.empty()) {
            throw std::runtime_error("--calibrate-hid requires --hid-port COMx");
        }
        if (options.calibration_step_counts <= 0 || options.calibration_repeats <= 0 ||
            options.calibration_noise_samples < 0 || options.calibration_settle_ms < 0) {
            throw std::runtime_error("calibration step/repeats must be greater than 0; noise samples and settle must be non-negative");
        }
        return;
    }
    if (options.hid_port.empty() && !options.dry_run) {
        throw std::runtime_error("use --hid-port COMx for live SDK output or --dry-run for tuning");
    }
    if (!options.dry_run && options.player_side == PlayerSide::Unknown) {
        throw std::runtime_error("live SDK output requires --player-side ct or --player-side t");
    }
    if (options.status_every_frames <= 0) {
        throw std::runtime_error("--status-every must be greater than 0");
    }
}

}  // namespace vision_analyzer
```

Delete the identical function body from `src/main.cpp`. Do not change validation rules in this refactor.

- [ ] **Step 2: Rebuild with the current xmake wildcard source graph**

Run:

```powershell
xmake -r
xmake run vision_analyzer_tests
xmake run vision_runtime_c_api_tests
```

Expected: all targets link and both test executables pass, proving the extraction preserved behavior.

- [ ] **Step 3: Commit the source-boundary refactor**

```powershell
git add src/main.cpp src/runtime_options.cpp
git commit -m "refactor: separate runtime option validation"
```

### Task 3: Make xmake use one reusable runtime core

**Files:**
- Modify: `xmake.lua`
- Test: `tests/test_algorithms.cpp`
- Test: `tests/test_c_api.cpp`

- [ ] **Step 1: Replace wildcard compilation with an explicit target graph**

Replace `xmake.lua` with the following complete target graph:

```lua
add_rules("mode.debug", "mode.release")

set_project("cs2_vision_cpp_analyzer")
set_languages("c++17")

add_requires("opencv 4.x", {configs = {dnn = true, ffmpeg = false}})

option("onnxruntime_root")
    set_showmenu(true)
    set_default(os.getenv("ONNXRUNTIME_ROOT") or "")
    set_description("ONNX Runtime SDK root")

option("hid_sdk_root")
    set_showmenu(true)
    set_default(os.getenv("RP2350_HID_BRIDGE_SDK") or "")
    set_description("RP2350 HID bridge C++ SDK root")

local ort_root = get_config("onnxruntime_root") or ""
local ort_include = path.join(ort_root, "include")
local ort_lib = path.join(ort_root, "lib")
local has_ort = ort_root ~= "" and os.isdir(ort_include) and os.isdir(ort_lib)
local torch_lib = path.join(os.projectdir(), "../../.venv/Lib/site-packages/torch/lib")
local tensorrt_libs = path.join(os.projectdir(), "../../.venv/Lib/site-packages/tensorrt_libs")
local hid_sdk_root = get_config("hid_sdk_root") or ""
if hid_sdk_root == "" then
    hid_sdk_root = path.join(os.projectdir(), "../rp2350_hid_bridge_cpp")
end
local hid_sdk_include = path.join(hid_sdk_root, "include")
local has_hid_sdk = hid_sdk_root ~= "" and os.isdir(hid_sdk_include)

local runtime_core_files = {
    "src/aim_controller.cpp",
    "src/calibration.cpp",
    "src/calibration_fit.cpp",
    "src/detector.cpp",
    "src/frame_source.cpp",
    "src/hid_output.cpp",
    "src/model_schema.cpp",
    "src/postprocess.cpp",
    "src/runtime_config.cpp",
    "src/runtime_options.cpp",
    "src/runtime_session.cpp",
    "src/tracking.cpp",
    "src/types.cpp",
}

local function add_runtime_runenvs()
    if has_ort then
        add_runenvs("PATH", ort_lib)
    end
    if os.isdir(torch_lib) then
        add_runenvs("PATH", torch_lib)
    end
    if os.isdir(tensorrt_libs) then
        add_runenvs("PATH", tensorrt_libs)
    end
end

local function copy_ort_runtime(target)
    if has_ort then
        os.cp(path.join(ort_lib, "*.dll"), target:targetdir())
    end
end

target("vision_analyzer_core")
    set_kind("static")
    add_includedirs("include", {public = true})
    add_packages("opencv", {public = true})
    for _, source_file in ipairs(runtime_core_files) do
        add_files(source_file)
    end
    if has_ort then
        add_includedirs(ort_include, {public = true})
        add_defines("VISION_ANALYZER_WITH_ORT", {public = true})
        add_linkdirs(ort_lib, {public = true})
        add_links("onnxruntime", {public = true})
    end
    if has_hid_sdk then
        add_includedirs(hid_sdk_include, {public = true})
        add_defines("VISION_ANALYZER_WITH_RP2350_HID", {public = true})
    end
    if is_plat("windows") then
        add_cxflags("/utf-8")
        add_syslinks("d3d11", "dxgi", {public = true})
    end

target("vision_analyzer")
    set_kind("binary")
    add_files("src/main.cpp")
    add_deps("vision_analyzer_core")
    add_runtime_runenvs()
    if is_plat("windows") then
        add_cxflags("/utf-8")
    end
    after_build(copy_ort_runtime)

target("vision_runtime")
    set_kind("shared")
    add_files("src/vision_runtime_c_api.cpp")
    add_deps("vision_analyzer_core")
    add_defines("VISION_RUNTIME_BUILD_DLL")
    add_includedirs("include", {public = true})
    add_runtime_runenvs()
    if is_plat("windows") then
        add_cxflags("/utf-8")
    end
    after_build(copy_ort_runtime)

target("vision_analyzer_tests")
    set_kind("binary")
    add_files("tests/test_algorithms.cpp")
    add_deps("vision_analyzer_core")
    add_runtime_runenvs()
    if is_plat("windows") then
        add_cxflags("/utf-8")
    end

target("vision_runtime_c_api_tests")
    set_kind("binary")
    add_files("tests/test_c_api.cpp")
    add_deps("vision_runtime")
    add_runtime_runenvs()
    if is_plat("windows") then
        add_cxflags("/utf-8")
    end
```

- [ ] **Step 2: Verify the xmake target graph and outputs**

Run:

```powershell
xmake -r
xmake show -l targets
xmake run vision_analyzer_tests
xmake run vision_runtime_c_api_tests
Get-Item build\windows\x64\release\vision_runtime.dll,build\windows\x64\release\vision_runtime.lib,build\windows\x64\release\vision_analyzer.exe
```

Expected: five targets including the internal core are listed; both test suites pass; DLL, import library, and CLI exist.

- [ ] **Step 3: Commit the xmake graph**

```powershell
git add xmake.lua
git commit -m "build: share runtime core across xmake targets"
```

### Task 4: Replace the archived UI CMake graph

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `.gitignore`

- [ ] **Step 1: Rewrite CMake around the same core/DLL/CLI/test graph**

Replace `CMakeLists.txt` with this complete configuration:

```cmake
cmake_minimum_required(VERSION 3.14)
project(cs2_vision_runtime LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(CTest)

if(NOT OpenCV_DIR AND DEFINED ENV{LOCALAPPDATA})
    file(GLOB XMAKE_OPENCV_CONFIGS
        "$ENV{LOCALAPPDATA}/.xmake/packages/o/opencv/*/*/OpenCVConfig.cmake"
    )
    if(XMAKE_OPENCV_CONFIGS)
        list(GET XMAKE_OPENCV_CONFIGS 0 XMAKE_OPENCV_CONFIG)
        get_filename_component(XMAKE_OPENCV_DIR "${XMAKE_OPENCV_CONFIG}" DIRECTORY)
        set(OpenCV_DIR "${XMAKE_OPENCV_DIR}" CACHE PATH "OpenCV package config directory" FORCE)
        set(OpenCV_STATIC ON CACHE BOOL "Use the static OpenCV package resolved by xmake" FORCE)
    endif()
endif()
find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime SDK root")
set(HID_SDK_ROOT "$ENV{RP2350_HID_BRIDGE_SDK}" CACHE PATH "RP2350 HID bridge C++ SDK root")
if(HID_SDK_ROOT STREQUAL "")
    set(HID_SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../rp2350_hid_bridge_cpp" CACHE PATH "RP2350 HID bridge C++ SDK root" FORCE)
endif()

set(VISION_ANALYZER_CORE_SOURCES
    src/aim_controller.cpp
    src/calibration.cpp
    src/calibration_fit.cpp
    src/detector.cpp
    src/frame_source.cpp
    src/hid_output.cpp
    src/model_schema.cpp
    src/postprocess.cpp
    src/runtime_config.cpp
    src/runtime_options.cpp
    src/runtime_session.cpp
    src/tracking.cpp
    src/types.cpp
)

add_library(vision_analyzer_core STATIC ${VISION_ANALYZER_CORE_SOURCES})
set_target_properties(vision_analyzer_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(vision_analyzer_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(vision_analyzer_core PUBLIC ${OpenCV_LIBS})

if(EXISTS "${ONNXRUNTIME_ROOT}/include" AND EXISTS "${ONNXRUNTIME_ROOT}/lib")
    target_compile_definitions(vision_analyzer_core PUBLIC VISION_ANALYZER_WITH_ORT)
    target_include_directories(vision_analyzer_core PUBLIC "${ONNXRUNTIME_ROOT}/include")
    target_link_directories(vision_analyzer_core PUBLIC "${ONNXRUNTIME_ROOT}/lib")
    target_link_libraries(vision_analyzer_core PUBLIC onnxruntime)
endif()

if(EXISTS "${HID_SDK_ROOT}/include")
    target_compile_definitions(vision_analyzer_core PUBLIC VISION_ANALYZER_WITH_RP2350_HID)
    target_include_directories(vision_analyzer_core PUBLIC "${HID_SDK_ROOT}/include")
endif()

if(WIN32)
    target_link_libraries(vision_analyzer_core PUBLIC d3d11 dxgi)
endif()

function(vision_analyzer_enable_utf8 target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /utf-8)
    endif()
endfunction()

function(vision_analyzer_copy_ort_runtime target_name)
    if(EXISTS "${ONNXRUNTIME_ROOT}/lib")
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${ONNXRUNTIME_ROOT}/lib"
                    "$<TARGET_FILE_DIR:${target_name}>"
        )
    endif()
endfunction()

vision_analyzer_enable_utf8(vision_analyzer_core)

add_library(vision_runtime SHARED src/vision_runtime_c_api.cpp)
target_compile_definitions(vision_runtime PRIVATE VISION_RUNTIME_BUILD_DLL)
target_include_directories(vision_runtime PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(vision_runtime PRIVATE vision_analyzer_core)
vision_analyzer_enable_utf8(vision_runtime)
vision_analyzer_copy_ort_runtime(vision_runtime)

add_executable(vision_analyzer src/main.cpp)
target_link_libraries(vision_analyzer PRIVATE vision_analyzer_core)
vision_analyzer_enable_utf8(vision_analyzer)
vision_analyzer_copy_ort_runtime(vision_analyzer)

if(BUILD_TESTING)
    add_executable(vision_analyzer_tests tests/test_algorithms.cpp)
    target_link_libraries(vision_analyzer_tests PRIVATE vision_analyzer_core)
    vision_analyzer_enable_utf8(vision_analyzer_tests)
    add_test(NAME vision_analyzer_tests COMMAND vision_analyzer_tests)

    add_executable(vision_runtime_c_api_tests tests/test_c_api.cpp)
    target_link_libraries(vision_runtime_c_api_tests PRIVATE vision_runtime)
    vision_analyzer_enable_utf8(vision_runtime_c_api_tests)
    add_test(NAME vision_runtime_c_api_tests COMMAND vision_runtime_c_api_tests)
endif()
```

Append this exact line to `.gitignore` for the new clean verification directory:

```gitignore
build-cmake/
```

- [ ] **Step 2: Configure, build, and run CTest without EUI**

Run:

```powershell
cmake -S . -B build-cmake -A x64
cmake --build build-cmake --config Release --clean-first
ctest --test-dir build-cmake -C Release --output-on-failure
```

Expected: configuration contains no EUI lookup, DLL/CLI/tests build, and both CTest entries pass.

- [ ] **Step 3: Commit the CMake replacement**

```powershell
git add CMakeLists.txt .gitignore
git commit -m "build: replace archived ui cmake graph"
```

### Task 5: Remove UI source and make documentation DLL-first

**Files:**
- Delete: `src/ui_app.cpp`
- Modify: `README.md:1-75`

- [ ] **Step 1: Delete the archived UI implementation**

Delete `src/ui_app.cpp`. Do not delete the ignored local `build-ui/` directory.

- [ ] **Step 2: Rewrite the README introduction and build workflow**

The introduction must identify these artifacts explicitly:

```text
vision_runtime.dll  primary product artifact and stable C API implementation
vision_runtime.lib  MSVC import library
vision_analyzer.exe optional diagnostic CLI
```

Document both supported builds:

```powershell
xmake f -m release
xmake
xmake run vision_analyzer_tests
xmake run vision_runtime_c_api_tests

cmake -S . -B build-cmake -A x64
cmake --build build-cmake --config Release
ctest --test-dir build-cmake -C Release --output-on-failure
```

Remove all language describing an archived UI or requiring EUI-NEO. Keep the existing model contract, dry-run, DXGI, HID, calibration, and C API usage documentation.

- [ ] **Step 3: Verify structural GREEN**

Run:

```powershell
rg -n "EUI_NEO|EUI-NEO|ui_app|vision_analyzer_ui" CMakeLists.txt xmake.lua src README.md
```

Expected: no matches and exit code 1.

- [ ] **Step 4: Commit UI removal and documentation**

```powershell
git add README.md src/ui_app.cpp
git commit -m "refactor: remove archived runtime ui"
```

### Task 6: Run final cross-build and runtime verification

**Files:**
- Verify: `CMakeLists.txt`
- Verify: `xmake.lua`
- Verify: `README.md`
- Verify: `include/vision_analyzer/vision_runtime_c_api.h`

- [ ] **Step 1: Run a fresh xmake verification**

```powershell
xmake -r
xmake run vision_analyzer_tests
xmake run vision_runtime_c_api_tests
```

Expected: clean build and 33 total test functions passing.

- [ ] **Step 2: Run a real-model CLI dry-run**

```powershell
xmake run vision_analyzer --backend opencv-onnx --model D:\project\cs2-vision-trainer\runs\detect\train\weights\best.onnx --video D:\project\cs2-vision-trainer\videos\02.mp4 --player-side unknown --dry-run --warmup-frames 0 --start-frame 5101 --max-frames 3 --status-every 1
```

Expected: schema validates, three frames process, and the process exits without HID output.

- [ ] **Step 3: Run a fresh CMake/CTest verification**

```powershell
cmake --build build-cmake --config Release --clean-first
ctest --test-dir build-cmake -C Release --output-on-failure
```

Expected: build succeeds and both CTest tests pass.

- [ ] **Step 4: Verify scope and repository state**

```powershell
git diff HEAD~4 --stat
git status --short
rg -n "EUI_NEO|EUI-NEO|ui_app|vision_analyzer_ui" CMakeLists.txt xmake.lua src README.md
```

Expected: only the planned files changed, worktree is clean after the implementation commits, and the structural search has no matches.
