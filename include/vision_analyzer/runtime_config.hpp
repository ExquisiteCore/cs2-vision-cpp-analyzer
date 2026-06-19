#pragma once

#include <string>

#include "vision_analyzer/types.hpp"

namespace vision_analyzer {

void apply_runtime_config_file(Options& options, const std::string& path);

}  // namespace vision_analyzer
