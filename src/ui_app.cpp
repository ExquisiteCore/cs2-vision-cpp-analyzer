#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "eui_neo.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace app {
namespace {

constexpr const char* kProjectRoot = VISION_ANALYZER_CPP_ROOT;

constexpr eui::Color kBg{0.075f, 0.082f, 0.090f, 1.0f};
constexpr eui::Color kPanel{0.118f, 0.130f, 0.143f, 1.0f};
constexpr eui::Color kPanel2{0.145f, 0.158f, 0.172f, 1.0f};
constexpr eui::Color kField{0.085f, 0.095f, 0.107f, 1.0f};
constexpr eui::Color kBorder{0.255f, 0.300f, 0.335f, 0.92f};
constexpr eui::Color kText{0.930f, 0.950f, 0.960f, 1.0f};
constexpr eui::Color kMuted{0.570f, 0.620f, 0.650f, 1.0f};
constexpr eui::Color kAccent{0.075f, 0.560f, 0.480f, 1.0f};
constexpr eui::Color kBlue{0.220f, 0.430f, 0.840f, 1.0f};
constexpr eui::Color kAmber{0.890f, 0.580f, 0.180f, 1.0f};
constexpr eui::Color kRose{0.820f, 0.260f, 0.360f, 1.0f};
constexpr eui::Color kClear{0.0f, 0.0f, 0.0f, 0.0f};

struct UiState {
    std::string analyzerExe;
    std::string configPath;
    std::string modelPath;
    std::string schemaPath;
    std::string videoPath;
    std::string actionLog = "actions.txt";
    std::string calibrationOutput = "hid-calibration.txt";
    std::string calibrationConfig = "hid-tuned.cfg";
    std::string hidPort = "COM4";
    std::string hidGain = "1.0";
    std::string hidMaxStep = "120";
    std::string hidDeadzone = "1.5";
    std::string confidence = "0.25";
    std::string nms = "0.45";
    std::string dxgiAdapter = "0";
    std::string dxgiOutput = "0";
    std::string calibrationStep = "40";
    std::string calibrationNoiseSamples = "2";
    std::string testMoveDx = "300";
    std::string testMoveDy = "0";
    int backend = 0;
    int input = 0;
    int side = 0;
    bool preview = true;
    bool hidClick = false;
};

UiState state;
std::mutex logMutex;
std::vector<std::string> logLines;
std::atomic<bool> processRunning{false};

#if defined(_WIN32)
std::mutex processMutex;
HANDLE processHandle = nullptr;
#endif

std::filesystem::path rootPath() {
    return std::filesystem::path(kProjectRoot);
}

std::string rootFile(const std::string& relative) {
    return (rootPath() / relative).lexically_normal().string();
}

void initializeState() {
    static bool initialized = false;
    if (initialized) {
        return;
    }
    initialized = true;
    state.analyzerExe = rootFile("build/windows/x64/release/vision_analyzer.exe");
    state.configPath = rootFile("runtime.example.cfg");
    state.modelPath = (rootPath() / "../../runs/detect/train/weights/best.onnx").lexically_normal().string();
    state.schemaPath.clear();
    state.videoPath = (rootPath() / "../../videos/02.mp4").lexically_normal().string();
}

components::theme::ThemeColorTokens themeTokens() {
    return {kBg, kAccent, kPanel2, {0.180f, 0.205f, 0.220f, 1.0f}, {0.100f, 0.120f, 0.135f, 1.0f}, kText, kBorder, true};
}

eui::Transition transition() {
    return eui::Transition::make(0.16f, eui::Ease::OutCubic);
}

eui::Color alpha(eui::Color color, float value) {
    color.a = std::clamp(value, 0.0f, 1.0f);
    return color;
}

void appendLog(std::string line) {
    std::lock_guard<std::mutex> lock(logMutex);
    logLines.push_back(std::move(line));
    if (logLines.size() > 180u) {
        logLines.erase(logLines.begin(), logLines.begin() + static_cast<std::ptrdiff_t>(logLines.size() - 180u));
    }
}

std::vector<std::string> snapshotLog() {
    std::lock_guard<std::mutex> lock(logMutex);
    return logLines;
}

std::string quoteArg(const std::string& value) {
    std::string out = "\"";
    for (char ch : value) {
        if (ch == '"') {
            out += "\\\"";
        } else {
            out.push_back(ch);
        }
    }
    out.push_back('"');
    return out;
}

void addArg(std::vector<std::string>& args, const std::string& key, const std::string& value) {
    if (!value.empty()) {
        args.push_back(key);
        args.push_back(value);
    }
}

const char* backendName() {
    static constexpr const char* kNames[] = {"opencv-onnx", "opencv-cuda", "ort-cuda", "ort-tensorrt"};
    return kNames[std::clamp(state.backend, 0, 3)];
}

const char* inputName() {
    return state.input == 0 ? "dxgi" : "video";
}

const char* sideName() {
    static constexpr const char* kNames[] = {"ct", "t", "unknown"};
    return kNames[std::clamp(state.side, 0, 2)];
}

std::vector<std::string> commonArgs() {
    std::vector<std::string> args;
    addArg(args, "--config", state.configPath);
    addArg(args, "--backend", backendName());
    addArg(args, "--model", state.modelPath);
    addArg(args, "--schema", state.schemaPath);
    if (state.input == 0) {
        addArg(args, "--input", "dxgi");
        addArg(args, "--dxgi-adapter", state.dxgiAdapter);
        addArg(args, "--dxgi-output", state.dxgiOutput);
    } else {
        addArg(args, "--video", state.videoPath);
    }
    addArg(args, "--player-side", sideName());
    addArg(args, "--hid-port", state.hidPort);
    addArg(args, "--hid-gain", state.hidGain);
    addArg(args, "--hid-max-step", state.hidMaxStep);
    addArg(args, "--hid-deadzone", state.hidDeadzone);
    addArg(args, "--conf", state.confidence);
    addArg(args, "--nms", state.nms);
    addArg(args, "--action-log", state.actionLog);
    if (state.preview) {
        args.push_back("--preview");
    }
    if (state.hidClick) {
        args.push_back("--hid-click");
    }
    return args;
}

std::string commandLine(const std::vector<std::string>& args) {
    std::string command = quoteArg(state.analyzerExe);
    for (const auto& arg : args) {
        command.push_back(' ');
        command += quoteArg(arg);
    }
    return command;
}

std::string buildVerifyCommand() {
    std::vector<std::string> args;
    addArg(args, "--input", inputName());
    if (state.input == 0) {
        addArg(args, "--dxgi-adapter", state.dxgiAdapter);
        addArg(args, "--dxgi-output", state.dxgiOutput);
    } else {
        addArg(args, "--video", state.videoPath);
    }
    args.push_back("--verify-input");
    return commandLine(args);
}

std::string buildHidProbeCommand() {
    std::vector<std::string> args;
    addArg(args, "--hid-port", state.hidPort);
    args.push_back("--test-hid-move");
    args.push_back(state.testMoveDx);
    args.push_back(state.testMoveDy);
    return commandLine(args);
}

std::string buildCalibrationCommand() {
    std::vector<std::string> args;
    args.push_back("--calibrate-hid");
    addArg(args, "--hid-port", state.hidPort);
    addArg(args, "--dxgi-adapter", state.dxgiAdapter);
    addArg(args, "--dxgi-output", state.dxgiOutput);
    addArg(args, "--calibration-step", state.calibrationStep);
    addArg(args, "--calibration-noise-samples", state.calibrationNoiseSamples);
    addArg(args, "--calibration-output", state.calibrationOutput);
    addArg(args, "--calibration-config-output", state.calibrationConfig);
    return commandLine(args);
}

std::string buildDryRunCommand() {
    auto args = commonArgs();
    args.push_back("--dry-run");
    return commandLine(args);
}

std::string buildLiveCommand() {
    return commandLine(commonArgs());
}

#if defined(_WIN32)
void clearProcessHandle() {
    std::lock_guard<std::mutex> lock(processMutex);
    processHandle = nullptr;
}

void runProcess(std::string command) {
    if (processRunning.exchange(true)) {
        appendLog("runner: process already running");
        return;
    }

    appendLog("> " + command);
    std::thread([command = std::move(command)] {
        SECURITY_ATTRIBUTES security{};
        security.nLength = sizeof(security);
        security.bInheritHandle = TRUE;

        HANDLE readPipe = nullptr;
        HANDLE writePipe = nullptr;
        if (!CreatePipe(&readPipe, &writePipe, &security, 0)) {
            appendLog("runner: CreatePipe failed");
            processRunning = false;
            return;
        }
        SetHandleInformation(readPipe, HANDLE_FLAG_INHERIT, 0);

        STARTUPINFOA startup{};
        startup.cb = sizeof(startup);
        startup.dwFlags = STARTF_USESTDHANDLES;
        startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
        startup.hStdOutput = writePipe;
        startup.hStdError = writePipe;

        PROCESS_INFORMATION process{};
        std::vector<char> mutableCommand(command.begin(), command.end());
        mutableCommand.push_back('\0');

        const BOOL ok = CreateProcessA(
            nullptr,
            mutableCommand.data(),
            nullptr,
            nullptr,
            TRUE,
            CREATE_NO_WINDOW,
            nullptr,
            kProjectRoot,
            &startup,
            &process
        );
        CloseHandle(writePipe);

        if (!ok) {
            appendLog("runner: CreateProcess failed, check analyzer path");
            CloseHandle(readPipe);
            processRunning = false;
            return;
        }

        {
            std::lock_guard<std::mutex> lock(processMutex);
            processHandle = process.hProcess;
        }

        std::string pending;
        char buffer[512]{};
        DWORD read = 0;
        while (ReadFile(readPipe, buffer, sizeof(buffer) - 1, &read, nullptr) && read > 0) {
            buffer[read] = '\0';
            pending.append(buffer, read);
            std::size_t newline = std::string::npos;
            while ((newline = pending.find('\n')) != std::string::npos) {
                std::string line = pending.substr(0, newline);
                if (!line.empty() && line.back() == '\r') {
                    line.pop_back();
                }
                appendLog(line);
                pending.erase(0, newline + 1);
            }
        }
        if (!pending.empty()) {
            appendLog(pending);
        }

        WaitForSingleObject(process.hProcess, INFINITE);
        DWORD exitCode = 0;
        GetExitCodeProcess(process.hProcess, &exitCode);
        appendLog("runner: exited with code " + std::to_string(exitCode));

        clearProcessHandle();
        CloseHandle(process.hThread);
        CloseHandle(process.hProcess);
        CloseHandle(readPipe);
        processRunning = false;
    }).detach();
}

void stopProcess() {
    std::lock_guard<std::mutex> lock(processMutex);
    if (processHandle != nullptr) {
        TerminateProcess(processHandle, 130);
        appendLog("runner: terminate requested");
    }
}
#else
void runProcess(std::string command) {
    appendLog("runner: process launch is implemented for Windows only");
    appendLog("> " + command);
}

void stopProcess() {
    appendLog("runner: stop is implemented for Windows only");
}
#endif

void label(eui::Ui& ui,
           const std::string& id,
           float x,
           float y,
           float width,
           float height,
           const std::string& text,
           float fontSize,
           eui::Color color,
           eui::HorizontalAlign align = eui::HorizontalAlign::Left) {
    ui.text(id)
        .x(x)
        .y(y)
        .size(width, height)
        .text(text)
        .fontSize(fontSize)
        .lineHeight(fontSize + 5.0f)
        .color(color)
        .horizontalAlign(align)
        .verticalAlign(eui::VerticalAlign::Center)
        .build();
}

void panel(eui::Ui& ui, const std::string& id, float x, float y, float width, float height) {
    ui.rect(id)
        .x(x)
        .y(y)
        .size(width, height)
        .color(kPanel)
        .radius(8.0f)
        .border(1.0f, alpha(kBorder, 0.78f))
        .build();
}

void field(eui::Ui& ui,
           const std::string& id,
           float x,
           float y,
           float width,
           const std::string& name,
           std::string& value,
           const std::string& placeholder = {}) {
    std::string* target = &value;
    label(ui, id + ".label", x, y, width, 18.0f, name, 11.0f, kMuted);
    ui.stack(id + ".wrap")
        .x(x)
        .y(y + 20.0f)
        .size(width, 32.0f)
        .content([&] {
            components::input(ui, id)
                .theme(themeTokens())
                .size(width, 32.0f)
                .fontSize(12.0f)
                .placeholder(placeholder)
                .value(value)
                .onChange([target](const std::string& next) {
                    *target = next;
                })
                .build();
        })
        .build();
}

void button(eui::Ui& ui,
            const std::string& id,
            float x,
            float y,
            float width,
            const std::string& text,
            unsigned int icon,
            eui::Color color,
            std::function<void()> onClick,
            bool disabled = false) {
    const eui::Color hover = eui::mixColor(color, {1.0f, 1.0f, 1.0f, 1.0f}, 0.12f);
    const eui::Color pressed = eui::mixColor(color, {0.0f, 0.0f, 0.0f, 1.0f}, 0.12f);
    ui.stack(id + ".wrap")
        .x(x)
        .y(y)
        .size(width, 34.0f)
        .content([&] {
            components::button(ui, id)
                .size(width, 34.0f)
                .icon(icon)
                .iconSize(13.0f)
                .fontSize(12.0f)
                .text(text)
                .colors(color, hover, pressed)
                .textColor(kText)
                .iconColor(kText)
                .radius(8.0f)
                .border(1.0f, alpha(color, 0.65f))
                .shadow(0.0f, 0.0f, 0.0f, kClear)
                .transition(transition())
                .disabled(disabled)
                .onClick(std::move(onClick))
                .build();
        })
        .build();
}

void toggle(eui::Ui& ui, const std::string& id, float x, float y, float width, const std::string& text, bool& value) {
    bool* target = &value;
    ui.stack(id + ".wrap")
        .x(x)
        .y(y)
        .size(width, 30.0f)
        .content([&] {
            components::toggleSwitch(ui, id)
                .theme(themeTokens())
                .size(width, 30.0f)
                .trackSize(38.0f, 20.0f)
                .fontSize(12.0f)
                .text(text)
                .checked(value)
                .onChange([target](bool next) {
                    *target = next;
                })
                .build();
        })
        .build();
}

void segmented(eui::Ui& ui,
               const std::string& id,
               float x,
               float y,
               float width,
               const std::vector<std::string>& items,
               int& selected) {
    int* target = &selected;
    ui.stack(id + ".wrap")
        .x(x)
        .y(y)
        .size(width, 32.0f)
        .content([&] {
            components::segmented(ui, id)
                .theme(themeTokens())
                .size(width, 32.0f)
                .items(items)
                .selected(selected)
                .fontSize(12.0f)
                .transition(transition())
                .onChange([target](int next) {
                    *target = next;
                })
                .build();
        })
        .build();
}

void composeHeader(eui::Ui& ui, float x, float y, float width) {
    label(ui, "header.title", x, y, 360.0f, 38.0f, "CS2 Vision Runtime", 24.0f, kText);
    label(ui, "header.sub", x, y + 32.0f, 520.0f, 22.0f, "EUI launcher for DXGI input, model runtime, HID bridge, and calibration.", 12.0f, kMuted);

    const bool running = processRunning.load();
    ui.rect("header.status.bg")
        .x(x + width - 156.0f)
        .y(y + 8.0f)
        .size(156.0f, 34.0f)
        .color(running ? alpha(kAmber, 0.14f) : alpha(kAccent, 0.13f))
        .radius(8.0f)
        .border(1.0f, running ? alpha(kAmber, 0.42f) : alpha(kAccent, 0.42f))
        .build();
    label(ui, "header.status", x + width - 156.0f, y + 8.0f, 156.0f, 34.0f, running ? "Process running" : "Idle", 13.0f, running ? kAmber : kAccent, eui::HorizontalAlign::Center);
}

void composeConfig(eui::Ui& ui, float x, float y, float width, float height) {
    panel(ui, "config.panel", x, y, width, height);
    label(ui, "config.title", x + 14.0f, y + 10.0f, width - 28.0f, 24.0f, "Runtime Configuration", 15.0f, kText);

    const float pad = 14.0f;
    const float gap = 12.0f;
    const float fieldW = (width - pad * 2.0f - gap) * 0.5f;
    const float left = x + pad;
    const float right = left + fieldW + gap;
    float rowY = y + 46.0f;

    field(ui, "cfg.exe", left, rowY, width - pad * 2.0f, "Analyzer executable", state.analyzerExe);
    rowY += 60.0f;
    field(ui, "cfg.config", left, rowY, width - pad * 2.0f, "Runtime config", state.configPath);
    rowY += 60.0f;
    field(ui, "cfg.model", left, rowY, width - pad * 2.0f, "Model", state.modelPath);
    rowY += 60.0f;
    field(ui, "cfg.schema", left, rowY, width - pad * 2.0f, "Schema override", state.schemaPath, "optional");
    rowY += 60.0f;

    label(ui, "cfg.backend.label", left, rowY, fieldW, 18.0f, "Backend", 11.0f, kMuted);
    segmented(ui, "cfg.backend", left, rowY + 20.0f, fieldW, {"onnx", "cuda", "ort", "trt"}, state.backend);
    label(ui, "cfg.input.label", right, rowY, fieldW, 18.0f, "Input", 11.0f, kMuted);
    segmented(ui, "cfg.input", right, rowY + 20.0f, fieldW, {"dxgi", "video"}, state.input);
    rowY += 62.0f;

    label(ui, "cfg.side.label", left, rowY, fieldW, 18.0f, "Player side", 11.0f, kMuted);
    segmented(ui, "cfg.side", left, rowY + 20.0f, fieldW, {"ct", "t", "unknown"}, state.side);
    field(ui, "cfg.video", right, rowY, fieldW, "Video path", state.videoPath);
    rowY += 62.0f;

    field(ui, "cfg.dxgi.adapter", left, rowY, fieldW, "DXGI adapter", state.dxgiAdapter);
    field(ui, "cfg.dxgi.output", right, rowY, fieldW, "DXGI output", state.dxgiOutput);
    rowY += 62.0f;

    field(ui, "cfg.hid.port", left, rowY, fieldW, "HID port", state.hidPort);
    field(ui, "cfg.action.log", right, rowY, fieldW, "Action log", state.actionLog);
}

void composeTuning(eui::Ui& ui, float x, float y, float width, float height) {
    panel(ui, "tuning.panel", x, y, width, height);
    label(ui, "tuning.title", x + 14.0f, y + 10.0f, width - 28.0f, 24.0f, "Movement and Calibration", 15.0f, kText);

    const float pad = 14.0f;
    const float gap = 10.0f;
    const float fieldW = (width - pad * 2.0f - gap * 2.0f) / 3.0f;
    const float x0 = x + pad;
    const float x1 = x0 + fieldW + gap;
    const float x2 = x1 + fieldW + gap;
    float rowY = y + 46.0f;

    field(ui, "tune.gain", x0, rowY, fieldW, "HID gain", state.hidGain);
    field(ui, "tune.step", x1, rowY, fieldW, "Max step", state.hidMaxStep);
    field(ui, "tune.deadzone", x2, rowY, fieldW, "Deadzone px", state.hidDeadzone);
    rowY += 62.0f;

    field(ui, "tune.conf", x0, rowY, fieldW, "Confidence", state.confidence);
    field(ui, "tune.nms", x1, rowY, fieldW, "NMS", state.nms);
    field(ui, "tune.calstep", x2, rowY, fieldW, "Cal step", state.calibrationStep);
    rowY += 62.0f;

    field(ui, "tune.noise", x0, rowY, fieldW, "Noise samples", state.calibrationNoiseSamples);
    field(ui, "tune.probedx", x1, rowY, fieldW, "Probe dx", state.testMoveDx);
    field(ui, "tune.probedy", x2, rowY, fieldW, "Probe dy", state.testMoveDy);
    rowY += 64.0f;

    field(ui, "tune.calout", x0, rowY, width - pad * 2.0f, "Calibration sample output", state.calibrationOutput);
    rowY += 60.0f;
    field(ui, "tune.calconfig", x0, rowY, width - pad * 2.0f, "Tuned config output", state.calibrationConfig);
    rowY += 62.0f;

    toggle(ui, "tune.preview", x0, rowY, 140.0f, "Preview", state.preview);
    toggle(ui, "tune.click", x0 + 150.0f, rowY, 150.0f, "Left click", state.hidClick);
}

void composeActions(eui::Ui& ui, float x, float y, float width, float height) {
    panel(ui, "actions.panel", x, y, width, height);
    label(ui, "actions.title", x + 14.0f, y + 10.0f, width - 28.0f, 24.0f, "Run Controls", 15.0f, kText);
    const bool running = processRunning.load();
    const float pad = 14.0f;
    const float gap = 10.0f;
    const float buttonW = (width - pad * 2.0f - gap * 2.0f) / 3.0f;
    const float x0 = x + pad;
    const float x1 = x0 + buttonW + gap;
    const float x2 = x1 + buttonW + gap;
    const float row0 = y + 46.0f;
    const float row1 = row0 + 44.0f;

    button(ui, "act.verify", x0, row0, buttonW, "Verify Input", 0xF06E, kBlue, [] { runProcess(buildVerifyCommand()); }, running);
    button(ui, "act.probe", x1, row0, buttonW, "HID Probe", 0xF11C, kAmber, [] { runProcess(buildHidProbeCommand()); }, running);
    button(ui, "act.calibrate", x2, row0, buttonW, "Calibrate", 0xF1EC, kAccent, [] { runProcess(buildCalibrationCommand()); }, running);

    button(ui, "act.dry", x0, row1, buttonW, "Dry Run", 0xF04B, kBlue, [] { runProcess(buildDryRunCommand()); }, running);
    button(ui, "act.live", x1, row1, buttonW, "Start Live", 0xF05B, kAccent, [] { runProcess(buildLiveCommand()); }, running);
    button(ui, "act.stop", x2, row1, buttonW, "Stop", 0xF04D, kRose, [] { stopProcess(); }, !running);
}

void composeLog(eui::Ui& ui, float x, float y, float width, float height) {
    panel(ui, "log.panel", x, y, width, height);
    label(ui, "log.title", x + 14.0f, y + 10.0f, width - 28.0f, 24.0f, "Process Output", 15.0f, kText);

    const auto lines = snapshotLog();
    const int maxLines = std::max(1, static_cast<int>((height - 52.0f) / 18.0f));
    const int start = std::max(0, static_cast<int>(lines.size()) - maxLines);
    float lineY = y + 44.0f;
    for (int index = start; index < static_cast<int>(lines.size()); ++index) {
        std::string line = lines[static_cast<std::size_t>(index)];
        if (line.size() > 156u) {
            line.resize(153u);
            line += "...";
        }
        label(ui, "log.line." + std::to_string(index), x + 14.0f, lineY, width - 28.0f, 18.0f, line, 11.0f, index == start && lines.size() > static_cast<std::size_t>(maxLines) ? kMuted : kText);
        lineY += 18.0f;
    }
}

}  // namespace

const DslAppConfig& dslAppConfig() {
    static const DslAppConfig config = DslAppConfig{}
        .title("CS2 Vision Runtime")
        .pageId("cs2_vision_runtime")
        .windowSize(1180, 760)
        .background(kBg)
        .fps(60.0)
        .showDebugStatsInTitle(false);
    return config;
}

void compose(eui::Ui& ui, const eui::Screen& screen) {
    initializeState();

    ui.rect("root.bg")
        .size(screen.width, screen.height)
        .color(kBg)
        .build();

    const float margin = 22.0f;
    const float width = std::max(980.0f, screen.width - margin * 2.0f);
    const float height = std::max(680.0f, screen.height - margin * 2.0f);
    const float x = (screen.width - width) * 0.5f;
    const float y = margin;
    const float gap = 14.0f;
    const float leftW = std::clamp(width * 0.56f, 540.0f, 700.0f);
    const float rightW = width - leftW - gap;

    composeHeader(ui, x, y, width);
    composeConfig(ui, x, y + 70.0f, leftW, height - 70.0f);
    composeTuning(ui, x + leftW + gap, y + 70.0f, rightW, 344.0f);
    composeActions(ui, x + leftW + gap, y + 428.0f, rightW, 142.0f);
    composeLog(ui, x + leftW + gap, y + 584.0f, rightW, height - 584.0f);
}

}  // namespace app
