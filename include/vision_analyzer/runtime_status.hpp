#pragma once

#include <string>

#include "vision_analyzer/runtime_session.hpp"

namespace vision_analyzer {

[[nodiscard]] std::string format_runtime_status(const RuntimeStepResult& step);

}  // namespace vision_analyzer
