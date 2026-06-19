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

add_executable(vision_analyzer_ui
    "${EUI_NEO_ROOT}/core/app/glfw_app_main.cpp"
    src/ui_app.cpp
)

target_compile_definitions(vision_analyzer_ui PRIVATE
    VISION_ANALYZER_CPP_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)

if(MSVC)
    target_compile_options(vision_analyzer_ui PRIVATE /utf-8)
endif()

eui_neo_configure_app(vision_analyzer_ui)
