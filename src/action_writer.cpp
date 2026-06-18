#include "vision_analyzer/action_writer.hpp"

#include <filesystem>
#include <iomanip>
#include <stdexcept>

namespace fs = std::filesystem;

namespace vision_analyzer {

ActionWriter::ActionWriter(const std::string& output_path) {
    const auto parent = fs::path(output_path).parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent);
    }
    output_.open(output_path, std::ios::binary);
    if (!output_) {
        throw std::runtime_error("failed to open actions output: " + output_path);
    }
}

void ActionWriter::write_report(const FrameReport& report) {
    output_ << "frame=" << report.frame_index
            << " time_ms=" << std::fixed << std::setprecision(3) << report.timestamp_ms;

    if (!report.target.has_value()) {
        output_ << " action=hold"
                << " reason=no_target"
                << " move_to=none"
                << " move_delta=(0.000,0.000)"
                << " left_button=hold"
                << " lock_state=idle\n";
        return;
    }

    const auto& target = *report.target;
    output_ << " action=move"
            << " target_id=" << target.id
            << " target_class=" << target.detection.label
            << " confidence=" << std::fixed << std::setprecision(6) << target.detection.confidence
            << " move_to=(" << std::fixed << std::setprecision(3)
            << target.analysis_point.x << ',' << target.analysis_point.y << ')'
            << " aim_to=(" << target.predicted_center.x << ',' << target.predicted_center.y << ')'
            << " move_delta=(" << target.offset.x << ',' << target.offset.y << ')'
            << " distance=" << target.distance
            << " velocity=(" << target.velocity.x << ',' << target.velocity.y << ')'
            << " acceleration=(" << target.acceleration.x << ',' << target.acceleration.y << ')'
            << " stability=" << target.stability
            << " switched=" << (target.switched ? 1 : 0)
            << " lock_state=" << lock_state_name(target.lock_state)
            << " left_button=" << (target.fire_candidate ? "press_candidate" : "hold")
            << '\n';
}

}  // namespace vision_analyzer
