#pragma once

#include <optional>
#include <string>
#include <vector>

#include "vision_analyzer/types.hpp"

namespace vision_analyzer {

struct ModelSchema {
    std::vector<std::string> classes;
};

[[nodiscard]] std::string default_model_schema_path(const std::string& model_path);
[[nodiscard]] ModelSchema load_model_schema(const std::string& path);
void validate_model_schema(const ModelSchema& schema);
void validate_configured_model_schema(const Options& options);

}  // namespace vision_analyzer
