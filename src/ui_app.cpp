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
        appendLog("运行器：已有进程在执行");
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
            appendLog("运行器：创建输出管道失败");
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
            appendLog("运行器：启动进程失败，请检查分析器程序路径");
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
        appendLog("运行器：进程退出，代码 " + std::to_string(exitCode));

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
        appendLog("运行器：已请求停止当前进程");
    }
}
#else
void runProcess(std::string command) {
    appendLog("运行器：进程启动目前只支持 Windows");
    appendLog("> " + command);
}

void stopProcess() {
    appendLog("运行器：停止进程目前只支持 Windows");
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
    label(ui, id + ".label", x, y, width, 16.0f, name, 10.5f, kMuted);
    ui.stack(id + ".wrap")
        .x(x)
        .y(y + 18.0f)
        .size(width, 30.0f)
        .content([&] {
            components::input(ui, id)
                .theme(themeTokens())
                .size(width, 30.0f)
                .fontSize(11.5f)
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
    const float statusW = 156.0f;
    const float titleW = std::max(220.0f, width - statusW - 18.0f);
    label(ui, "header.title", x, y, titleW, 38.0f, "CS2 视觉运行控制台", 24.0f, kText);
    label(ui, "header.sub", x, y + 32.0f, titleW, 22.0f, "统一管理模型、DXGI 输入、HID 标定、干跑预览和实机运行。", 12.0f, kMuted);

    const bool running = processRunning.load();
    ui.rect("header.status.bg")
        .x(x + width - statusW)
        .y(y + 8.0f)
        .size(statusW, 34.0f)
        .color(running ? alpha(kAmber, 0.14f) : alpha(kAccent, 0.13f))
        .radius(8.0f)
        .border(1.0f, running ? alpha(kAmber, 0.42f) : alpha(kAccent, 0.42f))
        .build();
    label(ui, "header.status", x + width - statusW, y + 8.0f, statusW, 34.0f, running ? "运行中" : "空闲", 13.0f, running ? kAmber : kAccent, eui::HorizontalAlign::Center);
}

void composeConfig(eui::Ui& ui, float x, float y, float width, float height) {
    panel(ui, "config.panel", x, y, width, height);
    label(ui, "config.title", x + 14.0f, y + 10.0f, width - 28.0f, 24.0f, "模型与输入", 15.0f, kText);

    const float pad = 14.0f;
    const float gap = 12.0f;
    const float fieldW = (width - pad * 2.0f - gap) * 0.5f;
    const float left = x + pad;
    const float right = left + fieldW + gap;
    float rowY = y + 46.0f;

    field(ui, "cfg.exe", left, rowY, width - pad * 2.0f, "分析器程序", state.analyzerExe);
    rowY += 54.0f;
    field(ui, "cfg.config", left, rowY, width - pad * 2.0f, "运行配置", state.configPath);
    rowY += 54.0f;
    field(ui, "cfg.model", left, rowY, width - pad * 2.0f, "ONNX 模型", state.modelPath);
    rowY += 54.0f;
    field(ui, "cfg.schema", left, rowY, width - pad * 2.0f, "Schema 文件", state.schemaPath, "可留空，默认读取模型旁边的 schema");
    rowY += 54.0f;

    label(ui, "cfg.backend.label", left, rowY, fieldW, 18.0f, "推理后端", 11.0f, kMuted);
    segmented(ui, "cfg.backend", left, rowY + 20.0f, fieldW, {"ONNX", "CUDA", "ORT", "TRT"}, state.backend);
    label(ui, "cfg.input.label", right, rowY, fieldW, 18.0f, "输入源", 11.0f, kMuted);
    segmented(ui, "cfg.input", right, rowY + 20.0f, fieldW, {"DXGI", "视频"}, state.input);
    rowY += 56.0f;

    label(ui, "cfg.side.label", left, rowY, fieldW, 18.0f, "我方阵营", 11.0f, kMuted);
    segmented(ui, "cfg.side", left, rowY + 20.0f, fieldW, {"我是 CT", "我是 T", "未知"}, state.side);
    field(ui, "cfg.video", right, rowY, fieldW, "视频文件", state.videoPath);
    rowY += 56.0f;

    field(ui, "cfg.dxgi.adapter", left, rowY, fieldW, "DXGI 适配器", state.dxgiAdapter);
    field(ui, "cfg.dxgi.output", right, rowY, fieldW, "显示输出", state.dxgiOutput);

    label(ui, "cfg.note", left, y + height - 30.0f, width - pad * 2.0f, 18.0f, "实机运行会强制校验 Schema；视频或 DXGI 干跑可先不接 HID。", 11.0f, kMuted);
}

void composeTuning(eui::Ui& ui, float x, float y, float width, float height) {
    panel(ui, "tuning.panel", x, y, width, height);
    label(ui, "tuning.title", x + 14.0f, y + 10.0f, width - 28.0f, 24.0f, "HID 与调参", 15.0f, kText);

    const float pad = 14.0f;
    const float gap = 10.0f;
    const float fieldW = (width - pad * 2.0f - gap * 2.0f) / 3.0f;
    const float x0 = x + pad;
    const float x1 = x0 + fieldW + gap;
    const float x2 = x1 + fieldW + gap;
    float rowY = y + 46.0f;

    field(ui, "tune.port", x0, rowY, fieldW, "HID 端口", state.hidPort);
    field(ui, "tune.action.log", x1, rowY, fieldW * 2.0f + gap, "动作日志", state.actionLog);
    rowY += 54.0f;

    field(ui, "tune.gain", x0, rowY, fieldW, "HID 增益", state.hidGain);
    field(ui, "tune.step", x1, rowY, fieldW, "单帧限幅", state.hidMaxStep);
    field(ui, "tune.deadzone", x2, rowY, fieldW, "死区像素", state.hidDeadzone);
    rowY += 54.0f;

    field(ui, "tune.conf", x0, rowY, fieldW, "置信度", state.confidence);
    field(ui, "tune.nms", x1, rowY, fieldW, "NMS 阈值", state.nms);
    field(ui, "tune.calstep", x2, rowY, fieldW, "标定步长", state.calibrationStep);
    rowY += 54.0f;

    field(ui, "tune.noise", x0, rowY, fieldW, "噪声样本", state.calibrationNoiseSamples);
    field(ui, "tune.probedx", x1, rowY, fieldW, "探针 dx", state.testMoveDx);
    field(ui, "tune.probedy", x2, rowY, fieldW, "探针 dy", state.testMoveDy);
    rowY += 54.0f;

    field(ui, "tune.calout", x0, rowY, width - pad * 2.0f, "标定样本输出", state.calibrationOutput);
    rowY += 54.0f;
    field(ui, "tune.calconfig", x0, rowY, width - pad * 2.0f, "调参配置输出", state.calibrationConfig);
    rowY += 56.0f;

    toggle(ui, "tune.preview", x0, rowY, 136.0f, "预览窗口", state.preview);
    toggle(ui, "tune.click", x0 + 148.0f, rowY, 150.0f, "允许开火", state.hidClick);
}

void composeActions(eui::Ui& ui, float x, float y, float width, float height) {
    panel(ui, "actions.panel", x, y, width, height);
    label(ui, "actions.title", x + 14.0f, y + 10.0f, width - 28.0f, 24.0f, "运行控制", 15.0f, kText);
    const bool running = processRunning.load();
    const float pad = 14.0f;
    const float gap = 10.0f;
    const float buttonW = (width - pad * 2.0f - gap * 2.0f) / 3.0f;
    const float x0 = x + pad;
    const float x1 = x0 + buttonW + gap;
    const float x2 = x1 + buttonW + gap;
    const float row0 = y + 46.0f;
    const float row1 = row0 + 44.0f;

    button(ui, "act.verify", x0, row0, buttonW, "验证输入", 0xF06E, kBlue, [] { runProcess(buildVerifyCommand()); }, running);
    button(ui, "act.probe", x1, row0, buttonW, "HID 探针", 0xF11C, kAmber, [] { runProcess(buildHidProbeCommand()); }, running);
    button(ui, "act.calibrate", x2, row0, buttonW, "执行标定", 0xF1EC, kAccent, [] { runProcess(buildCalibrationCommand()); }, running);

    button(ui, "act.dry", x0, row1, buttonW, "干跑预览", 0xF04B, kBlue, [] { runProcess(buildDryRunCommand()); }, running);
    button(ui, "act.live", x1, row1, buttonW, "启动实机", 0xF05B, kAccent, [] { runProcess(buildLiveCommand()); }, running);
    button(ui, "act.stop", x2, row1, buttonW, "停止", 0xF04D, kRose, [] { stopProcess(); }, !running);
}

void composeLog(eui::Ui& ui, float x, float y, float width, float height) {
    panel(ui, "log.panel", x, y, width, height);
    label(ui, "log.title", x + 14.0f, y + 10.0f, width - 28.0f, 24.0f, "输出日志", 15.0f, kText);

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
        .title("CS2 视觉运行控制台")
        .pageId("cs2_vision_runtime")
        .windowSize(1180, 760)
        .background(kBg)
        .textFont("assets/JingNanJunJunTi-JinNanJunJunTi-Bold-2.ttf")
        .iconFont("assets/Font Awesome 7 Free-Solid-900.otf")
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

    const float margin = std::clamp(screen.width * 0.020f, 10.0f, 22.0f);
    const float width = std::max(320.0f, screen.width - margin * 2.0f);
    const float height = std::max(360.0f, screen.height - margin * 2.0f);
    const float x = margin;
    const float y = margin;
    const float gap = std::clamp(width * 0.014f, 8.0f, 14.0f);
    const float contentY = y + 72.0f;
    const float contentH = std::max(240.0f, height - 72.0f);
    float rightW = std::clamp(width * 0.38f, 320.0f, 440.0f);
    float leftW = width - rightW - gap;
    if (leftW < 360.0f) {
        leftW = std::max(300.0f, width - gap - 320.0f);
        rightW = std::max(280.0f, width - leftW - gap);
    }
    const float configLimit = std::max(250.0f, contentH - 150.0f - gap);
    const float configH = std::min(std::clamp(contentH * 0.67f, 360.0f, 450.0f), configLimit);
    const float logY = contentY + configH + gap;
    const float logH = std::max(120.0f, contentY + contentH - logY);
    const float tuningLimit = std::max(250.0f, contentH - 150.0f - gap);
    const float tuningH = std::min(std::clamp(contentH * 0.68f, 360.0f, 460.0f), tuningLimit);
    const float actionsY = contentY + tuningH + gap;
    const float actionsH = std::max(120.0f, contentY + contentH - actionsY);

    composeHeader(ui, x, y, width);
    composeConfig(ui, x, contentY, leftW, configH);
    composeLog(ui, x, logY, leftW, logH);
    composeTuning(ui, x + leftW + gap, contentY, rightW, tuningH);
    composeActions(ui, x + leftW + gap, actionsY, rightW, actionsH);
}

}  // namespace app
