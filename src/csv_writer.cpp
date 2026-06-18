#include "vision_analyzer/csv_writer.hpp"

#include <filesystem>
#include <iomanip>
#include <stdexcept>

namespace fs = std::filesystem;

namespace vision_analyzer {

std::string escape_csv(const std::string& value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) {
        return value;
    }
    std::string escaped = "\"";
    for (char ch : value) {
        if (ch == '"') {
            escaped += "\"\"";
        } else {
            escaped += ch;
        }
    }
    escaped += '"';
    return escaped;
}

CsvWriter::CsvWriter(const std::string& output_path) {
    const auto parent = fs::path(output_path).parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent);
    }
    output_.open(output_path, std::ios::binary);
    if (!output_) {
        throw std::runtime_error("failed to open output: " + output_path);
    }
}

void CsvWriter::write_header() {
    output_
        << "frame_index,timestamp_ms,fps,latency_ms,detection_count,"
        << "preprocess_ms,inference_ms,postprocess_ms,"
        << "target_id,target_class,target_confidence,"
        << "target_x1,target_y1,target_x2,target_y2,target_cx,target_cy,"
        << "offset_x,offset_y,distance,switched,analysis_x,analysis_y,"
        << "predicted_x,predicted_y,velocity_x,velocity_y,accel_x,accel_y,"
        << "stability,tracking_error,lock_state,fire_candidate\n";
}

void CsvWriter::write_report(const FrameReport& report) {
    output_ << report.frame_index << ','
            << std::fixed << std::setprecision(3) << report.timestamp_ms << ','
            << std::fixed << std::setprecision(3) << report.fps << ','
            << std::fixed << std::setprecision(3) << report.timing.total_ms() << ','
            << report.detection_count << ','
            << std::fixed << std::setprecision(3) << report.timing.preprocess_ms << ','
            << std::fixed << std::setprecision(3) << report.timing.inference_ms << ','
            << std::fixed << std::setprecision(3) << report.timing.postprocess_ms << ',';

    if (!report.target.has_value()) {
        output_ << "-1,,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,idle,0\n";
        return;
    }

    const auto& target = *report.target;
    const auto& box = target.detection.box;
    output_ << target.id << ','
            << escape_csv(target.detection.label) << ','
            << std::fixed << std::setprecision(6) << target.detection.confidence << ','
            << box.x << ',' << box.y << ',' << (box.x + box.width) << ',' << (box.y + box.height) << ','
            << std::fixed << std::setprecision(3) << target.center.x << ','
            << std::fixed << std::setprecision(3) << target.center.y << ','
            << std::fixed << std::setprecision(3) << target.offset.x << ','
            << std::fixed << std::setprecision(3) << target.offset.y << ','
            << std::fixed << std::setprecision(3) << target.distance << ','
            << (target.switched ? 1 : 0) << ','
            << std::fixed << std::setprecision(3) << target.analysis_point.x << ','
            << std::fixed << std::setprecision(3) << target.analysis_point.y << ','
            << std::fixed << std::setprecision(3) << target.predicted_center.x << ','
            << std::fixed << std::setprecision(3) << target.predicted_center.y << ','
            << std::fixed << std::setprecision(3) << target.velocity.x << ','
            << std::fixed << std::setprecision(3) << target.velocity.y << ','
            << std::fixed << std::setprecision(3) << target.acceleration.x << ','
            << std::fixed << std::setprecision(3) << target.acceleration.y << ','
            << std::fixed << std::setprecision(3) << target.stability << ','
            << std::fixed << std::setprecision(3) << target.tracking_error << ','
            << lock_state_name(target.lock_state) << ','
            << (target.fire_candidate ? 1 : 0) << '\n';
}

}  // namespace vision_analyzer
