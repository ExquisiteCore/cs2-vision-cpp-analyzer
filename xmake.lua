add_rules("mode.debug", "mode.release")

set_project("cs2_vision_cpp_analyzer")
set_languages("c++17")

add_requires("opencv 4.x", {configs = {dnn = true, ffmpeg = false}})

local ort_root = os.getenv("ONNXRUNTIME_ROOT") or "D:/Tool/onnxruntime-win-x64-gpu-1.22.1"
local ort_include = path.join(ort_root, "include")
local ort_lib = path.join(ort_root, "lib")
local has_ort = os.isdir(ort_include) and os.isdir(ort_lib)
local torch_lib = path.join(os.projectdir(), "../../.venv/Lib/site-packages/torch/lib")
local tensorrt_libs = path.join(os.projectdir(), "../../.venv/Lib/site-packages/tensorrt_libs")
local hid_sdk_root = os.getenv("RP2350_HID_BRIDGE_SDK") or "D:/project/pi/test/sdk/cpp"
local hid_sdk_include = path.join(hid_sdk_root, "include")

target("vision_analyzer")
    set_kind("binary")
    add_includedirs("include")
    if has_ort then
        add_includedirs(ort_include)
        add_defines("VISION_ANALYZER_WITH_ORT")
        add_linkdirs(ort_lib)
        add_links("onnxruntime")
        add_runenvs("PATH", ort_lib)
        after_build(function (target)
            os.cp(path.join(ort_lib, "*.dll"), target:targetdir())
        end)
    end
    if os.isdir(hid_sdk_include) then
        add_includedirs(hid_sdk_include)
        add_defines("VISION_ANALYZER_WITH_RP2350_HID")
    end
    add_files("src/*.cpp")
    add_packages("opencv")
    add_runenvs("PATH", torch_lib)
    add_runenvs("PATH", tensorrt_libs)
    if is_plat("windows") then
        add_cxflags("/utf-8")
    end

target("vision_analyzer_tests")
    set_kind("binary")
    add_includedirs("include")
    add_files("tests/test_algorithms.cpp")
    add_files("src/types.cpp", "src/postprocess.cpp", "src/tracking.cpp", "src/hid_output.cpp", "src/aim_controller.cpp")
    add_packages("opencv")
    if is_plat("windows") then
        add_cxflags("/utf-8")
    end
