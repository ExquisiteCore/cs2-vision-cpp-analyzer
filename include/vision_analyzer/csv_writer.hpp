#pragma once

#include <fstream>
#include <string>

#include "vision_analyzer/types.hpp"

namespace vision_analyzer {

class CsvWriter {
public:
    explicit CsvWriter(const std::string& output_path);

    void write_header();
    void write_report(const FrameReport& report);

private:
    std::ofstream output_;
};

[[nodiscard]] std::string escape_csv(const std::string& value);

}  // namespace vision_analyzer
