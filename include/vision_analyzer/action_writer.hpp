#pragma once

#include <fstream>
#include <string>

#include "vision_analyzer/types.hpp"

namespace vision_analyzer {

class ActionWriter {
public:
    explicit ActionWriter(const std::string& output_path);

    void write_report(const FrameReport& report);

private:
    std::ofstream output_;
};

}  // namespace vision_analyzer
