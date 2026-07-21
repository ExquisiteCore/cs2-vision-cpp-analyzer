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
    "src/dxgi_roi.cpp",
    "src/frame_source.cpp",
    "src/hid_output.cpp",
    "src/model_input.cpp",
    "src/model_schema.cpp",
    "src/postprocess.cpp",
    "src/runtime_config.cpp",
    "src/runtime_options.cpp",
    "src/runtime_session.cpp",
    "src/tensorrt_provider_config.cpp",
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
