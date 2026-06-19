#pragma once

#include <atomic>

#include "vision_analyzer/types.hpp"

namespace vision_analyzer {

[[nodiscard]] Options parse_args(int argc, char** argv);
void validate_options(const Options& options);
void verify_input(const Options& options);
void test_hid_move(const Options& options);
void run(const Options& options, const std::atomic_bool* stop_requested = nullptr);

}  // namespace vision_analyzer
