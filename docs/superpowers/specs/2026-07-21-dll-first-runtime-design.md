# DLL-First Runtime Design

## Goal

Make `vision_runtime.dll` and its C API the primary product artifact, remove the archived UI and EUI-NEO dependency, and retain `vision_analyzer.exe` only as a diagnostic and end-to-end validation tool.

## Chosen approach

Create one reusable static core target that contains capture, inference, tracking, calibration, configuration, and HID orchestration. Link that core into the DLL, CLI, and algorithm tests. Compile the C API only into the DLL and the CLI entry point only into the executable.

This is preferred over two alternatives:

- Minimal deletion: remove `ui_app.cpp` and EUI-NEO but keep duplicated source lists. This is smaller initially, but the DLL, CLI, and tests can drift.
- DLL-only delivery: remove both UI and CLI. This produces the smallest product surface, but loses the most useful model/video troubleshooting path.

## Architecture

The resulting target graph is:

```text
vision_analyzer_core (static)
├── vision_runtime (shared DLL + C API)
├── vision_analyzer (diagnostic CLI)
└── vision_analyzer_tests

vision_runtime_c_api_tests ──links──> vision_runtime
```

`vision_analyzer_core` owns all non-entry-point implementation. The DLL exposes only the functions declared in `include/vision_analyzer/vision_runtime_c_api.h`. The CLI remains optional and must not be required by DLL consumers.

## File changes

- Delete `src/ui_app.cpp`.
- Rewrite `CMakeLists.txt` around the core, DLL, CLI, and two test executables; remove every EUI-NEO reference.
- Restructure `xmake.lua` to define the same target graph and stop compiling `src/*.cpp` indiscriminately into both the CLI and DLL.
- Extract `validate_options` from `src/main.cpp` into `src/runtime_options.cpp`, allowing the DLL to validate options without linking CLI-only help, preview, and process entry-point code.
- Keep `src/main.cpp` as the diagnostic executable entry point.
- Update `README.md` so DLL integration is the primary workflow and no archived UI wording or EUI build path remains.
- Keep the ignored local `build-ui/` directory untouched because it is generated state outside the product sources.

## Data and API behavior

Runtime processing behavior and the existing C ABI remain unchanged. A caller still creates an opaque `VaRuntime`, configures it through `va_set_*`, opens video or DXGI input, calls `va_process_next`, and finally calls `va_close`/`va_destroy`.

No C API functions or fields are added, removed, or reordered in this change.

## Error handling

Build configuration must fail with clear messages when OpenCV is unavailable. Optional ONNX Runtime and RP2350 HID support remain feature-detected exactly as before. Removing the UI must not alter runtime error propagation through `va_last_error`.

## Testing and acceptance

The change is accepted when all of the following are true:

1. No tracked source or build definition references `ui_app`, EUI-NEO, or `vision_analyzer_ui`.
2. A clean xmake rebuild produces `vision_runtime.dll`, `vision_runtime.lib`, and the diagnostic CLI.
3. The 29 algorithm tests and 4 C API tests pass.
4. A real-model video dry-run still processes frames through the CLI.
5. CMake configures and builds the DLL, CLI, and tests without an EUI checkout.
6. `git status` contains only the intended source, build-definition, documentation, and deletion changes.

Because most changes are build configuration and source relocation rather than new runtime behavior, verification will use existing behavior tests plus clean builds instead of adding artificial runtime tests that merely restate existing behavior.
