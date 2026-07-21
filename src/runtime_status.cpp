#include "vision_analyzer/runtime_status.hpp"

#include <iomanip>
#include <sstream>

namespace vision_analyzer {

std::string format_runtime_status(const RuntimeStepResult& step) {
    const auto& report = step.report;
    const auto& timing = report.timing;
    const auto& command = step.command;

    std::ostringstream status;
    status << "frame=" << report.frame_index
           << " fps=" << std::fixed << std::setprecision(2) << report.fps
           << " preprocess_ms=" << timing.preprocess_ms
           << " inference_ms=" << timing.inference_ms
           << " postprocess_ms=" << timing.postprocess_ms
           << " total_ms=" << timing.total_ms()
           << " det=" << report.detection_count
           << " target=" << (command.has_target ? 1 : 0)
           << " dx=" << command.dx
           << " dy=" << command.dy
           << " click=" << (command.click_left ? 1 : 0)
           << " lock=" << lock_state_name(command.lock_state);
    return status.str();
}

}  // namespace vision_analyzer
