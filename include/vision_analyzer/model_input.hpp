#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace vision_analyzer {

enum class ModelElementType {
    Float32,
    Unsupported,
};

struct ModelInputSpec {
    int width = 0;
    int height = 0;
    std::array<std::int64_t, 4> shape{};
};

[[nodiscard]] ModelInputSpec parse_model_input_spec(
    const std::vector<std::int64_t>& shape,
    ModelElementType element_type
);

}  // namespace vision_analyzer
