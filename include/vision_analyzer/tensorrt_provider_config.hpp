#pragma once

#include <string>
#include <vector>

namespace vision_analyzer {

struct TensorRtProviderOption {
    std::string key;
    std::string value;
};

[[nodiscard]] std::vector<TensorRtProviderOption> make_sm61_tensorrt_provider_options(
    const std::string& cache_path
);

}  // namespace vision_analyzer
