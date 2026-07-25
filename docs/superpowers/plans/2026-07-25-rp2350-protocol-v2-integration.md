# RP2350 Protocol v2 Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the DLL around the RP2350 reliable protocol v2 SDK, guarantee no-throw shutdown, reject legacy HID builds during packaging, and deliver an ORT 1.17.3 GTX 1080 Ti package plus a small overlay patch.

**Architecture:** Keep the existing `HidClient` boundary and C ABI. Add a pure protocol-health parser, make HID close explicit and no-throw, and enforce protocol v2 at compile and package time. Rebuild the unchanged detection pipeline against the fixed SM61 dependency profile.

**Tech Stack:** C++17, header-only `rp2350_hid_bridge_cpp`, xmake, CMake/CTest, Windows PowerShell 5.1, ONNX Runtime GPU 1.17.3, TensorRT 8.6.1.6, CUDA 11.8, cuDNN 8.9.7.

---

## File map

- `include/vision_analyzer/hid_output.hpp`: internal health result, health parser, HID close contract.
- `src/hid_output.cpp`: protocol v2 compile guard, `ping/info/caps`, concrete no-throw close.
- `include/vision_analyzer/runtime_session.hpp`: no-throw runtime shutdown declaration.
- `src/runtime_session.cpp`: exception-safe shutdown order.
- `tests/test_algorithms.cpp`: protocol payload and shutdown regression tests.
- `CMakeLists.txt`: CMake 3.20 and SDK thread dependency.
- `packaging/sm61/PackageTools.psm1`: unit-testable binary protocol-v2 gate.
- `packaging/sm61/build-portable-package.ps1`: gate release DLL and record the v2 component.
- `packaging/sm61/package/scripts/common.ps1`: validate v2 manifest metadata on production machines.
- `packaging/sm61/package/scripts/verify-runtime.ps1`: invoke the manifest v2 check.
- `packaging/sm61/tests/run-tests.ps1`: package, build-definition, and manifest regressions.
- `README.md`, `packaging/sm61/package/README_中文.md`: operator-facing protocol v2 requirements.

### Task 1: Parse and require RP2350 protocol v2 health

**Files:**
- Modify: `tests/test_algorithms.cpp`
- Modify: `include/vision_analyzer/hid_output.hpp`
- Modify: `src/hid_output.cpp`

- [ ] **Step 1: Write failing payload tests**

Add calls for these tests to `main()` and define them near the existing HID sender tests:

```cpp
void test_rp2350_v2_health_accepts_required_capabilities() {
    const std::vector<std::uint8_t> info{2, 0, 240, 3};
    const std::vector<std::uint8_t> caps{2, 0, 240, 0, 0x7F, 1, 1, 0, 8, 20};
    const HidDeviceHealth health = parse_rp2350_v2_health(info, caps);
    require(health.protocol_version == 2, "health must report protocol v2");
    require(health.capabilities == 0x7F, "health must preserve capability bits");
}

void test_rp2350_v2_health_rejects_legacy_or_incomplete_devices() {
    bool rejected_legacy = false;
    try {
        (void)parse_rp2350_v2_health({1, 0, 240, 3}, {1, 0, 240, 0, 0x0F});
    } catch (const std::runtime_error& error) {
        rejected_legacy = std::string(error.what()).find("protocol v2") != std::string::npos;
    }
    require(rejected_legacy, "legacy firmware must be rejected with a v2 error");

    bool rejected_caps = false;
    try {
        (void)parse_rp2350_v2_health({2, 0, 240, 3}, {2, 0, 240, 0, 0x02});
    } catch (const std::runtime_error&) {
        rejected_caps = true;
    }
    require(rejected_caps, "firmware without retry, lease, and cancellation must be rejected");
}
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
xmake -r -P . vision_analyzer_tests
```

Expected: compilation fails because `HidDeviceHealth` and `parse_rp2350_v2_health` do not exist.

- [ ] **Step 3: Add the minimal health API**

Add to `hid_output.hpp`:

```cpp
struct HidDeviceHealth {
    std::uint8_t protocol_version = 0;
    std::uint16_t capabilities = 0;
};

[[nodiscard]] HidDeviceHealth parse_rp2350_v2_health(
    const std::vector<std::uint8_t>& info,
    const std::vector<std::uint8_t>& caps
);
```

Include `<vector>`. Implement in `hid_output.cpp` with protocol version `2` and required capability mask
`0x0072` (`mouse | retry-safe | lease | cancellation`). All validation failures must include the exact
marker `RP2350 protocol v2 capabilities are required`.

- [ ] **Step 4: Use the health check during connection**

Inside the Windows SDK block, add:

```cpp
static_assert(
    rp2350_hid_bridge::PROTOCOL_VERSION == 2,
    "RP2350 HID SDK protocol v2 is required"
);
```

Construct `HidBridge` through `HidBridgeOptions`, call `open()`, `ping()`, `info()`, and `caps()`, then
pass both payloads to `parse_rp2350_v2_health()` before the client is returned.

- [ ] **Step 5: Run GREEN tests**

Run:

```powershell
xmake -r -P . vision_analyzer_tests
```

Expected: `algorithm tests passed`.

- [ ] **Step 6: Commit**

```powershell
git add include/vision_analyzer/hid_output.hpp src/hid_output.cpp tests/test_algorithms.cpp
git commit -m "feat: require RP2350 protocol v2 health"
```

### Task 2: Guarantee no-throw HID and runtime shutdown

**Files:**
- Modify: `tests/test_algorithms.cpp`
- Modify: `include/vision_analyzer/hid_output.hpp`
- Modify: `src/hid_output.cpp`
- Modify: `include/vision_analyzer/runtime_session.hpp`
- Modify: `src/runtime_session.cpp`

- [ ] **Step 1: Write the failing shutdown tests**

Extend `RecordingHidClient`:

```cpp
void stop_all() override {
    ++stop_calls;
    if (throw_on_stop) {
        throw std::runtime_error("simulated stop failure");
    }
}

void close() noexcept override {
    ++close_calls;
}

bool throw_on_stop = false;
int close_calls = 0;
```

Add:

```cpp
static_assert(noexcept(std::declval<RuntimeSession&>().close()));

void test_hid_close_continues_after_stop_failure() {
    RecordingHidClient client;
    client.throw_on_stop = true;
    close_hid_client_noexcept(&client);
    require(client.stop_calls == 1, "shutdown must attempt STOP_ALL once");
    require(client.close_calls == 1, "shutdown must close after STOP_ALL failure");
}
```

- [ ] **Step 2: Run the test and verify RED**

Run `xmake -r -P . vision_analyzer_tests`.

Expected: compilation fails because `HidClient::close`, `close_hid_client_noexcept`, and the noexcept
runtime declaration are absent.

- [ ] **Step 3: Implement the no-throw HID close boundary**

Add to `HidClient`:

```cpp
virtual void close() noexcept = 0;
```

Declare and implement:

```cpp
void close_hid_client_noexcept(HidClient* client) noexcept {
    if (client == nullptr) {
        return;
    }
    try {
        client->stop_all();
    } catch (...) {
    }
    client->close();
}
```

Implement `Rp2350HidClient::close() noexcept` by calling `bridge_.close()`. Make its destructor call only
`close()`, because the outer helper and SDK close path already provide best-effort STOP_ALL.

- [ ] **Step 4: Make `RuntimeSession::close()` no-throw**

Change declarations to:

```cpp
~RuntimeSession() noexcept;
void close() noexcept;
```

In `close()`, disable fire, set `open_ = false`, reset `hid_sender_`, call
`close_hid_client_noexcept(hid_client_.get())`, reset the client, catch exceptions from frame-source
release, and then reset the remaining resources. No exception may escape.

- [ ] **Step 5: Run GREEN and C API tests**

```powershell
xmake -r -P .
xmake run -P . vision_analyzer_tests
xmake run -P . vision_runtime_c_api_tests
```

Expected: both test binaries pass.

- [ ] **Step 6: Commit**

```powershell
git add include/vision_analyzer/hid_output.hpp src/hid_output.cpp include/vision_analyzer/runtime_session.hpp src/runtime_session.cpp tests/test_algorithms.cpp
git commit -m "fix: make RP2350 shutdown exception safe"
```

### Task 3: Pin build definitions to SDK v2

**Files:**
- Modify: `packaging/sm61/tests/run-tests.ps1`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write a failing build-definition test**

Add a package-tool test that reads repository `CMakeLists.txt` and `src/hid_output.cpp`, then asserts:

```powershell
Assert-True ($cmake -match 'cmake_minimum_required\(VERSION 3\.20\)') 'CMake must match SDK minimum version'
Assert-True ($cmake.Contains('find_package(Threads REQUIRED)')) 'CMake must resolve SDK thread support'
Assert-True ($cmake.Contains('Threads::Threads')) 'HID-enabled core must link the thread target'
Assert-True ($hidOutput.Contains('PROTOCOL_VERSION == 2')) 'the compiled runtime must reject an old SDK'
```

- [ ] **Step 2: Run the package tests and verify RED**

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File .\packaging\sm61\tests\run-tests.ps1
```

Expected: the CMake version/thread assertions fail.

- [ ] **Step 3: Update CMake**

Set `cmake_minimum_required(VERSION 3.20)`. Inside the HID SDK condition call
`find_package(Threads REQUIRED)` and link `Threads::Threads` publicly to `vision_analyzer_core`.

- [ ] **Step 4: Run GREEN package tests and CMake build**

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File .\packaging\sm61\tests\run-tests.ps1
cmake -S . -B build-cmake-v2 -A x64 -DONNXRUNTIME_ROOT=D:\Tool\onnxruntime-win-x64-gpu-1.17.3 -DHID_SDK_ROOT=D:\project\cs2-vision-trainer\tools\rp2350_hid_bridge_cpp
cmake --build build-cmake-v2 --config Release --parallel
ctest --test-dir build-cmake-v2 -C Release --output-on-failure
```

Expected: package tests pass and CTest reports 2/2.

- [ ] **Step 5: Commit**

```powershell
git add CMakeLists.txt packaging/sm61/tests/run-tests.ps1
git commit -m "build: pin RP2350 SDK v2 requirements"
```

### Task 4: Reject legacy DLLs and require v2 package metadata

**Files:**
- Modify: `packaging/sm61/tests/run-tests.ps1`
- Modify: `packaging/sm61/PackageTools.psm1`
- Modify: `packaging/sm61/build-portable-package.ps1`
- Modify: `packaging/sm61/package/scripts/common.ps1`
- Modify: `packaging/sm61/package/scripts/verify-runtime.ps1`

- [ ] **Step 1: Write failing binary-gate tests**

Create temporary files containing legacy and v2 marker text, then require:

```powershell
Assert-Throws {
    Assert-Rp2350ProtocolV2Binary -LiteralPath $legacyDll
} 'protocol v2' 'legacy HID release DLL must be rejected'

Assert-Rp2350ProtocolV2Binary -LiteralPath $v2Dll
```

The accepted fixture contains `RP2350 protocol v2 capabilities are required`; the rejected fixture
contains only unrelated bytes.

- [ ] **Step 2: Write failing manifest tests**

Dot-source package `common.ps1` and call `Assert-Rp2350ProtocolV2Manifest` with three fixtures: no HID
component, `protocol-v1`, and exactly one `protocol-v2`. The first two must throw and the last must pass.

- [ ] **Step 3: Run tests and verify RED**

Run the Windows PowerShell package test command.

Expected: missing helper functions fail the new tests.

- [ ] **Step 4: Implement the DLL gate**

Add `Assert-Rp2350ProtocolV2Binary` to `PackageTools.psm1`. It reads the binary as ASCII, rejects the
existing “SDK unavailable” marker, and requires the exact v2 health marker. Replace the builder's inline
string check with this helper.

- [ ] **Step 5: Record and validate manifest metadata**

Append this component before `Write-PackageManifest`:

```powershell
$manifestComponents += [pscustomobject][ordered]@{
    id = 'rp2350-hid-sdk'
    version = 'protocol-v2'
    sourceMode = 'header-only-build'
}
```

Add `Assert-Rp2350ProtocolV2Manifest` to package `common.ps1` and call it immediately after manifest
profile validation in `verify-runtime.ps1`.

- [ ] **Step 6: Run GREEN tests**

Run package tests under Windows PowerShell 5.1.

Expected: all package tool tests pass with zero failures.

- [ ] **Step 7: Commit**

```powershell
git add packaging/sm61/PackageTools.psm1 packaging/sm61/build-portable-package.ps1 packaging/sm61/package/scripts/common.ps1 packaging/sm61/package/scripts/verify-runtime.ps1 packaging/sm61/tests/run-tests.ps1
git commit -m "build: reject legacy RP2350 runtime packages"
```

### Task 5: Update operator documentation

**Files:**
- Modify: `README.md`
- Modify: `packaging/sm61/package/README_中文.md`

- [ ] **Step 1: Add documentation assertions first**

Extend the package template test to require `协议 v2`, `ping/info/caps`, `500 ms` and `两秒` in the
package README.

- [ ] **Step 2: Verify RED**

Run package tests and confirm the README assertion fails.

- [ ] **Step 3: Document the new contract**

Explain that the board must run matching v2 firmware, startup performs read-only health checks, the SDK
sends a 500 ms heartbeat, the firmware has a two-second safety lease, and a legacy DLL/manifest is
rejected before production testing.

- [ ] **Step 4: Verify GREEN and commit**

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File .\packaging\sm61\tests\run-tests.ps1
git add README.md packaging/sm61/package/README_中文.md packaging/sm61/tests/run-tests.ps1
git commit -m "docs: document RP2350 protocol v2 runtime"
```

### Task 6: Run source-level verification with the production dependency profile

**Files:**
- No source changes expected.

- [ ] **Step 1: Verify the SDK**

```powershell
cmake --build D:\project\cs2-vision-trainer\tools\rp2350_hid_bridge_cpp\build --config Release
ctest --test-dir D:\project\cs2-vision-trainer\tools\rp2350_hid_bridge_cpp\build -C Release --output-on-failure
```

Expected: protocol test 1/1 passes.

- [ ] **Step 2: Clean-configure xmake for ORT 1.17.3**

```powershell
xmake f -c -m release --onnxruntime_root=D:\Tool\onnxruntime-win-x64-gpu-1.17.3 --hid_sdk_root=D:\project\cs2-vision-trainer\tools\rp2350_hid_bridge_cpp -P .
xmake -r -P .
xmake run -P . vision_analyzer_tests
xmake run -P . vision_runtime_c_api_tests
```

Expected: build succeeds and both tests pass.

- [ ] **Step 3: Verify binary markers and ORT linkage inputs**

Confirm the release DLL contains the v2 marker and xmake target information points to ORT 1.17.3, not
1.24.4.

- [ ] **Step 4: Run CMake/CTest and package tests**

Run CMake clean build, CTest 2/2, and Windows PowerShell package tests. Record exact counts.

### Task 7: Build and verify the v2 production package and overlay

**Files:**
- Generated: `D:/project/cs2-vision-trainer/dist/cs2-vision-runtime-sm61-v2/`
- Generated: `D:/project/cs2-vision-trainer/dist/cs2-vision-runtime-sm61-v2.zip`
- Generated: `C:/Users/xiaol/Downloads/cs2-vision-runtime-sm61-v2-hotfix.zip`

- [ ] **Step 1: Build the full package without replacing the old delivery**

Run `packaging/sm61/build-portable-package.ps1` with explicit release root, ORT 1.17.3 root, TensorRT
8.6.1.6 archive, Python project root, `-OutputRoot ...sm61-v2`, and `-OutputZip ...sm61-v2.zip`.

- [ ] **Step 2: Verify package metadata**

Require profile `sm61-ort1173-trt861-fp32`, component `rp2350-hid-sdk=protocol-v2`, all immutable
hashes valid, and no CUDA 12/cuDNN 9/TensorRT 10+ files.

- [ ] **Step 3: Run safe package tests**

Run static verification with unsupported local GPU allowed, two TensorRT video passes where supported,
and a clean extraction smoke test. The one-click path remains dry-run and never opens a COM port.

- [ ] **Step 4: Generate the small overlay**

Compare old and v2 manifests, copy every changed/new immutable file plus the new manifest into a staging
tree named `cs2-vision-runtime-sm61`, create a Zip archive, and verify that overlaying it on a clean old
extraction produces a valid v2 manifest.

- [ ] **Step 5: Final verification**

Run `git diff --check`, confirm the worktree is clean, list commit IDs, ZIP byte sizes and SHA256 hashes,
and state that actual COM3 health/calibration still requires the production board.
