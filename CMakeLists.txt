cmake_minimum_required(VERSION 3.14)
project(cs2_vision_cpp_ui LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EUI_NEO_ROOT "D:/project/EUI-NEO" CACHE PATH "Path to EUI-NEO")
if(NOT EXISTS "${EUI_NEO_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR "EUI_NEO_ROOT does not point to EUI-NEO: ${EUI_NEO_ROOT}")
endif()

set(EUI_BUILD_APPS OFF CACHE BOOL "" FORCE)
set(EUI_BUILD_TEST_FIXTURES OFF CACHE BOOL "" FORCE)
set(EUI_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory("${EUI_NEO_ROOT}" "${CMAKE_BINARY_DIR}/_deps/eui_neo")

if(NOT OpenCV_DIR AND DEFINED ENV{LOCALAPPDATA})
    file(GLOB XMAKE_OPENCV_CONFIGS
        "$ENV{LOCALAPPDATA}/.xmake/packages/o/opencv/*/*/OpenCVConfig.cmake"
    )
    if(XMAKE_OPENCV_CONFIGS)
        list(GET XMAKE_OPENCV_CONFIGS 0 XMAKE_OPENCV_CONFIG)
        get_filename_component(XMAKE_OPENCV_DIR "${XMAKE_OPENCV_CONFIG}" DIRECTORY)
        set(OpenCV_DIR "${XMAKE_OPENCV_DIR}" CACHE PATH "OpenCV package config directory" FORCE)
    endif()
endif()
find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime SDK root")
set(HID_SDK_ROOT "$ENV{RP2350_HID_BRIDGE_SDK}" CACHE PATH "RP2350 HID bridge C++ SDK root")
if(HID_SDK_ROOT STREQUAL "")
    set(HID_SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../rp2350_hid_bridge_cpp" CACHE PATH "RP2350 HID bridge C++ SDK root" FORCE)
endif()

add_library(vision_analyzer_runtime STATIC
    src/aim_controller.cpp
    src/calibration.cpp
    src/calibration_fit.cpp
    src/detector.cpp
    src/frame_source.cpp
    src/hid_output.cpp
    src/main.cpp
    src/model_schema.cpp
    src/postprocess.cpp
    src/runtime_config.cpp
    src/tracking.cpp
    src/types.cpp
)
target_include_directories(vision_analyzer_runtime PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${OpenCV_INCLUDE_DIRS}
)
target_compile_definitions(vision_analyzer_runtime PRIVATE VISION_ANALYZER_NO_CLI_MAIN)
target_link_libraries(vision_analyzer_runtime PUBLIC ${OpenCV_LIBS})

if(EXISTS "${ONNXRUNTIME_ROOT}/include" AND EXISTS "${ONNXRUNTIME_ROOT}/lib")
    target_compile_definitions(vision_analyzer_runtime PUBLIC VISION_ANALYZER_WITH_ORT)
    target_include_directories(vision_analyzer_runtime PUBLIC "${ONNXRUNTIME_ROOT}/include")
    target_link_directories(vision_analyzer_runtime PUBLIC "${ONNXRUNTIME_ROOT}/lib")
    target_link_libraries(vision_analyzer_runtime PUBLIC onnxruntime)
endif()

if(EXISTS "${HID_SDK_ROOT}/include")
    target_compile_definitions(vision_analyzer_runtime PUBLIC VISION_ANALYZER_WITH_RP2350_HID)
    target_include_directories(vision_analyzer_runtime PUBLIC "${HID_SDK_ROOT}/include")
endif()

if(WIN32)
    target_link_libraries(vision_analyzer_runtime PUBLIC d3d11 dxgi)
endif()

add_executable(vision_analyzer_ui
    "${EUI_NEO_ROOT}/core/app/glfw_app_main.cpp"
    src/ui_app.cpp
)
target_link_libraries(vision_analyzer_ui PRIVATE vision_analyzer_runtime)

target_compile_definitions(vision_analyzer_ui PRIVATE
    VISION_ANALYZER_CPP_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)

if(MSVC)
    target_compile_options(vision_analyzer_ui PRIVATE /utf-8)
endif()

eui_neo_configure_app(vision_analyzer_ui)

if(EXISTS "${ONNXRUNTIME_ROOT}/lib")
    add_custom_command(TARGET vision_analyzer_ui POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${ONNXRUNTIME_ROOT}/lib"
                "$<TARGET_FILE_DIR:vision_analyzer_ui>"
    )
endif()
