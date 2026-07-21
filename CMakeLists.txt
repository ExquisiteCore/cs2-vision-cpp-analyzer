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
    src/dxgi_roi.cpp
    src/frame_source.cpp
    src/hid_output.cpp
    src/model_input.cpp
    src/model_schema.cpp
    src/postprocess.cpp
    src/runtime_config.cpp
    src/runtime_options.cpp
    src/runtime_session.cpp
    src/runtime_status.cpp
    src/tensorrt_provider_config.cpp
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
