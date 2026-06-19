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

constexpr eui::Color kBg{0.030f, 0.038f, 0.047f, 1.0f};
constexpr eui::Color kPanel{0.070f, 0.084f, 0.098f, 1.0f};
constexpr eui::Color kPanel2{0.095f, 0.113f, 0.132f, 1.0f};
constexpr eui::Color kField{0.045f, 0.055f, 0.067f, 1.0f};
constexpr eui::Color kFieldHover{0.118f, 0.140f, 0.160f, 1.0f};
constexpr eui::Color kBorder{0.205f, 0.255f, 0.295f, 0.92f};
constexpr eui::Color kText{0.930f, 0.955f, 0.970f, 1.0f};
constexpr eui::Color kMuted{0.590f, 0.650f, 0.690f, 1.0f};
constexpr eui::Color kDim{0.390f, 0.455f, 0.500f, 1.0f};
constexpr eui::Color kAccent{0.075f, 0.630f, 0.540f, 1.0f};
constexpr eui::Color kBlue{0.250f, 0.455f, 0.870f, 1.0f};
constexpr eui::Color kAmber{0.920f, 0.580f, 0.200f, 1.0f};
constexpr eui::Color kRose{0.850f, 0.255f, 0.360f, 1.0f};
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
    return {kBg, kAccent, kPanel2, kFieldHover, kField, kText, kBorder, true};
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
        .fontFamily("Microsoft YaHei")
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
        .radius(10.0f)
        .border(1.0f, alpha(kBorder, 0.70f))
        .shadow(18.0f, 0.0f, 10.0f, alpha({0.0f, 0.0f, 0.0f, 1.0f}, 0.10f))
        .build();
    ui.rect(id + ".top")
        .x(x + 1.0f)
        .y(y + 1.0f)
        .size(width - 2.0f, 1.0f)
        .color(alpha(kText, 0.055f))
        .build();
}

void sectionTitle(eui::Ui& ui, const std::string& id, float x, float y, float width, const std::string& title, const std::string& meta = {}) {
    label(ui, id + ".title", x, y, width, 24.0f, title, 15.0f, kText);
    if (!meta.empty()) {
        label(ui, id + ".meta", x, y + 22.0f, width, 18.0f, meta, 10.5f, kDim);
    }
}

void badge(eui::Ui& ui, const std::string& id, float x, float y, float width, const std::string& text, eui::Color color) {
    ui.rect(id + ".bg")
        .x(x)
        .y(y)
        .size(width, 32.0f)
        .color(alpha(color, 0.12f))
        .radius(8.0f)
        .border(1.0f, alpha(color, 0.34f))
        .build();
    label(ui, id + ".text", x, y, width, 32.0f, text, 12.0f, color, eui::HorizontalAlign::Center);
}

void stepCard(eui::Ui& ui,
              const std::string& id,
              float x,
              float y,
              float width,
              const std::string& number,
              const std::string& title,
              const std::string& detail,
              eui::Color color) {
    ui.rect(id + ".bg")
        .x(x)
        .y(y)
        .size(width, 60.0f)
        .color(kPanel2)
        .radius(9.0f)
        .border(1.0f, alpha(color, 0.26f))
        .build();
    ui.rect(id + ".num.bg")
        .x(x + 12.0f)
        .y(y + 13.0f)
        .size(28.0f, 28.0f)
        .color(alpha(color, 0.16f))
        .radius(8.0f)
        .border(1.0f, alpha(color, 0.36f))
        .build();
    label(ui, id + ".num", x + 12.0f, y + 13.0f, 28.0f, 28.0f, number, 13.0f, color, eui::HorizontalAlign::Center);
    label(ui, id + ".title", x + 50.0f, y + 10.0f, width - 60.0f, 22.0f, title, 13.0f, kText);
    label(ui, id + ".detail", x + 50.0f, y + 32.0f, width - 60.0f, 18.0f, detail, 10.5f, kMuted);
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
    label(ui, id + ".label", x, y, width, 15.0f, name, 10.2f, kMuted);
    ui.stack(id + ".wrap")
        .x(x)
        .y(y + 16.0f)
        .size(width, 28.0f)
        .content([&] {
            components::input(ui, id)
                .theme(themeTokens())
                .size(width, 28.0f)
                .fontSize(11.0f)
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
            float height,
            const std::string& text,
            unsigned int icon,
            eui::Color color,
            std::function<void()> onClick,
            bool disabled = false,
            bool primary = false) {
    const eui::Color normal = primary ? color : alpha(color, 0.16f);
    const eui::Color hover = primary
        ? eui::mixColor(color, {1.0f, 1.0f, 1.0f, 1.0f}, 0.12f)
        : alpha(eui::mixColor(color, {1.0f, 1.0f, 1.0f, 1.0f}, 0.10f), 0.24f);
    const eui::Color pressed = primary
        ? eui::mixColor(color, {0.0f, 0.0f, 0.0f, 1.0f}, 0.12f)
        : alpha(color, 0.28f);
    const eui::Color textColor = primary ? kText : eui::mixColor(color, kText, 0.50f);
    ui.stack(id + ".wrap")
        .x(x)
        .y(y)
        .size(width, height)
        .content([&] {
            components::button(ui, id)
                .size(width, height)
                .icon(icon)
                .iconSize(primary ? 14.0f : 13.0f)
                .fontSize(primary ? 13.0f : 12.0f)
                .text(text)
                .colors(normal, hover, pressed)
                .textColor(textColor)
                .iconColor(textColor)
                .radius(8.0f)
                .border(1.0f, primary ? alpha(color, 0.70f) : alpha(color, 0.38f))
                .shadow(0.0f, 0.0f, 0.0f, kClear)
                .transition(transition())
                .disabled(disabled)
                .onClick(std::move(onClick))
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
            bool disabled = false,
            bool primary = false) {
    button(ui, id, x, y, width, 34.0f, text, icon, color, std::move(onClick), disabled, primary);
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
    const float badgesW = 332.0f;
    const float titleW = std::max(260.0f, width - badgesW - 18.0f);
    label(ui, "header.title", x, y - 1.0f, titleW, 38.0f, "CS2 视觉运行台", 25.0f, kText);
    label(ui, "header.sub", x, y + 32.0f, titleW, 22.0f, "训练在 Python，运行在这里：模型输入、HID 标定、预览和实机控制。", 12.0f, kMuted);

    const bool running = processRunning.load();
    const float bx = x + width - badgesW;
    badge(ui, "header.status", bx, y + 10.0f, 92.0f, running ? "运行中" : "空闲", running ? kAmber : kAccent);
    badge(ui, "header.input", bx + 102.0f, y + 10.0f, 100.0f, state.input == 0 ? "DXGI 输入" : "视频输入", kBlue);
    badge(ui, "header.port", bx + 212.0f, y + 10.0f, 120.0f, "HID " + state.hidPort, state.hidClick ? kRose : kMuted);
}

void composeWorkflow(eui::Ui& ui, float x, float y, float width) {
    panel(ui, "workflow.panel", x, y, width, 88.0f);

    const float pad = 14.0f;
    const float titleW = 116.0f;
    sectionTitle(ui, "workflow", x + pad, y + 13.0f, titleW, "运行流程", "按这个顺序点");

    const float gap = 10.0f;
    const float sx = x + pad + titleW + 12.0f;
    const float stepW = (width - pad * 2.0f - titleW - 12.0f - gap * 3.0f) / 4.0f;
    const float sy = y + 14.0f;
    stepCard(ui, "workflow.input", sx, sy, stepW, "1", "验证输入", "先确认能抓画面", kBlue);
    stepCard(ui, "workflow.hid", sx + (stepW + gap), sy, stepW, "2", "HID 探针", "看板子能不能动", kAmber);
    stepCard(ui, "workflow.dry", sx + (stepW + gap) * 2.0f, sy, stepW, "3", "干跑预览", "只看识别和路线", kAccent);
    stepCard(ui, "workflow.live", sx + (stepW + gap) * 3.0f, sy, stepW, "4", "启动实机", "确认后再开左键", kRose);
}

void composeConfig(eui::Ui& ui, float x, float y, float width, float height) {
    panel(ui, "config.panel", x, y, width, height);
    sectionTitle(ui, "config", x + 14.0f, y + 10.0f, width - 28.0f, "路径与识别", "模型、输入源、阵营和截图来源");

    const float pad = 14.0f;
    const float gap = 12.0f;
    const float fieldW = (width - pad * 2.0f - gap) * 0.5f;
    const float left = x + pad;
    const float right = left + fieldW + gap;
    float rowY = y + 56.0f;

    field(ui, "cfg.exe", left, rowY, width - pad * 2.0f, "运行端 EXE", state.analyzerExe);
    rowY += 48.0f;
    field(ui, "cfg.config", left, rowY, width - pad * 2.0f, "配置文件 CFG", state.configPath);
    rowY += 48.0f;
    field(ui, "cfg.model", left, rowY, width - pad * 2.0f, "模型 ONNX", state.modelPath);
    rowY += 48.0f;
    field(ui, "cfg.schema", left, rowY, width - pad * 2.0f, "Schema JSON", state.schemaPath, "best.onnx.schema.json，实机必须有");
    rowY += 48.0f;

    label(ui, "cfg.backend.label", left, rowY, fieldW, 18.0f, "推理后端", 11.0f, kMuted);
    segmented(ui, "cfg.backend", left, rowY + 20.0f, fieldW, {"ONNX", "CUDA", "ORT", "TRT"}, state.backend);
    label(ui, "cfg.input.label", right, rowY, fieldW, 18.0f, "输入源", 11.0f, kMuted);
    segmented(ui, "cfg.input", right, rowY + 20.0f, fieldW, {"DXGI", "视频"}, state.input);
    rowY += 50.0f;

    label(ui, "cfg.side.label", left, rowY, fieldW, 18.0f, "我方阵营", 11.0f, kMuted);
    segmented(ui, "cfg.side", left, rowY + 20.0f, fieldW, {"CT", "T", "未知"}, state.side);
    field(ui, "cfg.video", right, rowY, fieldW, "视频样本", state.videoPath);
    rowY += 50.0f;

    field(ui, "cfg.dxgi.adapter", left, rowY, fieldW, "DXGI 适配器", state.dxgiAdapter);
    field(ui, "cfg.dxgi.output", right, rowY, fieldW, "DXGI 显示器", state.dxgiOutput);

    if (height >= 428.0f) {
        ui.rect("cfg.note.bg")
            .x(left)
            .y(y + height - 36.0f)
            .size(width - pad * 2.0f, 24.0f)
            .color(alpha(kBlue, 0.08f))
            .radius(7.0f)
            .border(1.0f, alpha(kBlue, 0.18f))
            .build();
        label(ui, "cfg.note", left + 10.0f, y + height - 36.0f, width - pad * 2.0f - 20.0f, 24.0f, "实机运行会强制校验 Schema；先用视频或 DXGI 干跑排错。", 11.0f, kMuted);
    }
}

void composeTuning(eui::Ui& ui, float x, float y, float width, float height) {
    panel(ui, "tuning.panel", x, y, width, height);
    sectionTitle(ui, "tuning", x + 14.0f, y + 10.0f, width - 28.0f, "HID 与标定", "外置键鼠、动作 TXT、滤波参数");

    const float pad = 14.0f;
    const float gap = 10.0f;
    const float fieldW = (width - pad * 2.0f - gap * 2.0f) / 3.0f;
    const float x0 = x + pad;
    const float x1 = x0 + fieldW + gap;
    const float x2 = x1 + fieldW + gap;
    float rowY = y + 56.0f;

    field(ui, "tune.port", x0, rowY, fieldW, "HID 串口", state.hidPort);
    field(ui, "tune.action.log", x1, rowY, fieldW * 2.0f + gap, "动作 TXT", state.actionLog);
    rowY += 48.0f;

    field(ui, "tune.gain", x0, rowY, fieldW, "移动增益", state.hidGain);
    field(ui, "tune.step", x1, rowY, fieldW, "最大步长", state.hidMaxStep);
    field(ui, "tune.deadzone", x2, rowY, fieldW, "死区 px", state.hidDeadzone);
    rowY += 48.0f;

    field(ui, "tune.conf", x0, rowY, fieldW, "检测置信度", state.confidence);
    field(ui, "tune.nms", x1, rowY, fieldW, "NMS 阈值", state.nms);
    field(ui, "tune.calstep", x2, rowY, fieldW, "标定步长", state.calibrationStep);
    rowY += 48.0f;

    field(ui, "tune.noise", x0, rowY, fieldW, "噪声采样", state.calibrationNoiseSamples);
    field(ui, "tune.probedx", x1, rowY, fieldW, "探针 X", state.testMoveDx);
    field(ui, "tune.probedy", x2, rowY, fieldW, "探针 Y", state.testMoveDy);
    rowY += 48.0f;

    field(ui, "tune.calout", x0, rowY, width - pad * 2.0f, "标定样本 TXT", state.calibrationOutput);
    rowY += 48.0f;
    field(ui, "tune.calconfig", x0, rowY, width - pad * 2.0f, "调参配置 CFG", state.calibrationConfig);
    rowY += 50.0f;

    toggle(ui, "tune.preview", x0, rowY, 128.0f, "预览窗口", state.preview);
    toggle(ui, "tune.click", x0 + 140.0f, rowY, 142.0f, "允许左键", state.hidClick);
}

void composeActions(eui::Ui& ui, float x, float y, float width, float height) {
    panel(ui, "actions.panel", x, y, width, height);
    sectionTitle(ui, "actions", x + 14.0f, y + 10.0f, width - 28.0f, "执行控制", processRunning.load() ? "当前有任务在跑" : "空闲，可以启动任务");
    const bool running = processRunning.load();
    const float pad = 14.0f;
    const float gap = 10.0f;
    const float buttonW = (width - pad * 2.0f - gap * 2.0f) / 3.0f;
    const float x0 = x + pad;
    const float x1 = x0 + buttonW + gap;
    const float x2 = x1 + buttonW + gap;
    const float row0 = y + 58.0f;
    const float row1 = row0 + 44.0f;

    button(ui, "act.verify", x0, row0, buttonW, "验证输入", 0xF06E, kBlue, [] { runProcess(buildVerifyCommand()); }, running);
    button(ui, "act.probe", x1, row0, buttonW, "HID 探针", 0xF11C, kAmber, [] { runProcess(buildHidProbeCommand()); }, running);
    button(ui, "act.calibrate", x2, row0, buttonW, "执行标定", 0xF1EC, kAccent, [] { runProcess(buildCalibrationCommand()); }, running);

    button(ui, "act.dry", x0, row1, buttonW, "干跑预览", 0xF04B, kBlue, [] { runProcess(buildDryRunCommand()); }, running);
    button(ui, "act.live", x1, row1, buttonW, "启动实机", 0xF05B, kAccent, [] { runProcess(buildLiveCommand()); }, running, true);
    button(ui, "act.stop", x2, row1, buttonW, "停止", 0xF04D, kRose, [] { stopProcess(); }, !running);
}

void composeLog(eui::Ui& ui, float x, float y, float width, float height) {
    panel(ui, "log.panel", x, y, width, height);
    sectionTitle(ui, "log", x + 14.0f, y + 10.0f, width - 28.0f, "输出日志", "命令、识别、标定和错误都会写在这里");

    const auto lines = snapshotLog();
    if (lines.empty()) {
        ui.rect("log.empty.bg")
            .x(x + 14.0f)
            .y(y + 58.0f)
            .size(width - 28.0f, 54.0f)
            .color(alpha(kFieldHover, 0.36f))
            .radius(8.0f)
            .border(1.0f, alpha(kBorder, 0.52f))
            .build();
        label(ui, "log.empty.title", x + 14.0f, y + 61.0f, width - 28.0f, 24.0f, "还没有输出", 13.0f, kMuted, eui::HorizontalAlign::Center);
        label(ui, "log.empty.sub", x + 14.0f, y + 84.0f, width - 28.0f, 20.0f, "点击验证、干跑或启动后，这里会显示结果。", 11.0f, kDim, eui::HorizontalAlign::Center);
        return;
    }

    const int maxLines = std::max(1, static_cast<int>((height - 62.0f) / 20.0f));
    const int start = std::max(0, static_cast<int>(lines.size()) - maxLines);
    const int maxChars = std::max(28, static_cast<int>((width - 42.0f) / 7.0f));
    float lineY = y + 54.0f;
    for (int index = start; index < static_cast<int>(lines.size()); ++index) {
        std::string line = lines[static_cast<std::size_t>(index)];
        if (line.size() > static_cast<std::size_t>(maxChars)) {
            line.resize(static_cast<std::size_t>(std::max(0, maxChars - 3)));
            line += "...";
        }
        const bool faded = index == start && lines.size() > static_cast<std::size_t>(maxLines);
        ui.rect("log.line.bg." + std::to_string(index))
            .x(x + 12.0f)
            .y(lineY)
            .size(width - 24.0f, 18.0f)
            .color(index % 2 == 0 ? alpha(kFieldHover, 0.22f) : alpha(kField, 0.20f))
            .radius(5.0f)
            .build();
        label(ui, "log.line." + std::to_string(index), x + 20.0f, lineY, width - 40.0f, 18.0f, line, 10.8f, faded ? kMuted : kText);
        lineY += 20.0f;
    }
}

}  // namespace

const DslAppConfig& dslAppConfig() {
    static const DslAppConfig config = DslAppConfig{}
        .title("CS2 视觉运行控制台")
        .pageId("cs2_vision_runtime")
        .windowSize(1180, 760)
        .background(kBg)
        .textFont("C:/Windows/Fonts/msyh.ttc")
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
    const float workflowY = y + 66.0f;
    const float workflowH = 88.0f;
    const float contentY = workflowY + workflowH + gap;
    const float contentH = std::max(300.0f, y + height - contentY);
    float rightW = std::clamp(width * 0.38f, 320.0f, 440.0f);
    float leftW = width - rightW - gap;
    if (leftW < 360.0f) {
        leftW = std::max(300.0f, width - gap - 320.0f);
        rightW = std::max(280.0f, width - leftW - gap);
    }
    const float configLimit = std::max(320.0f, contentH - 124.0f - gap);
    const float configH = std::min(std::clamp(contentH * 0.72f, 382.0f, 420.0f), configLimit);
    const float logY = contentY + configH + gap;
    const float logH = std::max(120.0f, contentY + contentH - logY);
    const float tuningLimit = std::max(330.0f, contentH - 136.0f - gap);
    const float tuningH = std::min(std::clamp(contentH * 0.72f, 374.0f, 400.0f), tuningLimit);
    const float actionsY = contentY + tuningH + gap;
    const float actionsH = std::max(120.0f, contentY + contentH - actionsY);

    composeHeader(ui, x, y, width);
    composeWorkflow(ui, x, workflowY, width);
    composeConfig(ui, x, contentY, leftW, configH);
    composeLog(ui, x, logY, leftW, logH);
    composeTuning(ui, x + leftW + gap, contentY, rightW, tuningH);
    composeActions(ui, x + leftW + gap, actionsY, rightW, actionsH);
}

}  // namespace app
