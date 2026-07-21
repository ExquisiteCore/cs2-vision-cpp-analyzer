#include "vision_analyzer/tensorrt_provider_config.hpp"

#include <stdexcept>

namespace vision_analyzer {

std::vector<TensorRtProviderOption> make_sm61_tensorrt_provider_options(const std::string& cache_path) {
    if (cache_path.empty()) {
        throw std::runtime_error("TensorRT cache path must not be empty");
    }
    return {
        {"device_id", "0"},
        {"trt_fp16_enable", "0"},
        {"trt_engine_cache_enable", "1"},
        {"trt_engine_cache_path", cache_path},
        {"trt_max_workspace_size", "2147483648"},
        {"trt_min_subgraph_size", "1"},
    };
}

}  // namespace vision_analyzer
