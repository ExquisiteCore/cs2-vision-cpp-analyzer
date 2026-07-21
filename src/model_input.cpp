#include "vision_analyzer/model_input.hpp"

#include <limits>
#include <sstream>
#include <stdexcept>

namespace vision_analyzer {
namespace {

[[nodiscard]] std::string format_shape(const std::vector<std::int64_t>& shape) {
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) {
            output << ", ";
        }
        output << shape[index];
    }
    output << ']';
    return output.str();
}

}  // namespace

ModelInputSpec parse_model_input_spec(const std::vector<std::int64_t>& shape, ModelElementType element_type) {
    if (element_type != ModelElementType::Float32) {
        throw std::runtime_error("ORT model input must be FP32, got shape " + format_shape(shape));
    }
    if (shape.size() != 4 || shape[0] != 1 || shape[1] != 3) {
        throw std::runtime_error("ORT model input must be static NCHW [1, 3, H, W], got " + format_shape(shape));
    }
    if (shape[2] <= 0 || shape[3] <= 0 ||
        shape[2] > std::numeric_limits<int>::max() || shape[3] > std::numeric_limits<int>::max()) {
        throw std::runtime_error(
            "ORT model input height and width must be positive static integers, got " + format_shape(shape)
        );
    }
    return ModelInputSpec{
        static_cast<int>(shape[3]),
        static_cast<int>(shape[2]),
        {shape[0], shape[1], shape[2], shape[3]},
    };
}

}  // namespace vision_analyzer
