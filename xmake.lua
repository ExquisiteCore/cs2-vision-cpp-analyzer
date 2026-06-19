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
    if has_hid_sdk then
        add_includedirs(hid_sdk_include)
        add_defines("VISION_ANALYZER_WITH_RP2350_HID")
    end
    add_files("src/*.cpp")
    remove_files("src/ui_app.cpp")
    add_packages("opencv")
    add_runenvs("PATH", torch_lib)
    add_runenvs("PATH", tensorrt_libs)
    if is_plat("windows") then
        add_cxflags("/utf-8")
        add_syslinks("d3d11", "dxgi")
    end

target("vision_analyzer_tests")
    set_kind("binary")
    add_includedirs("include")
    add_files("tests/test_algorithms.cpp")
    add_files(
        "src/types.cpp",
        "src/postprocess.cpp",
        "src/tracking.cpp",
        "src/hid_output.cpp",
        "src/aim_controller.cpp",
        "src/runtime_config.cpp",
        "src/model_schema.cpp",
        "src/calibration_fit.cpp"
    )
    add_packages("opencv")
    if is_plat("windows") then
        add_cxflags("/utf-8")
    end
