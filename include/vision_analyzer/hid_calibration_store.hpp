#pragma once

#include <filesystem>

#include "vision_analyzer/hid_calibration_profile.hpp"

namespace vision_analyzer {

[[nodiscard]] HidCalibrationProfile load_hid_calibration_profile(
    const std::filesystem::path& path
);
void save_hid_calibration_profile_atomic(
    const std::filesystem::path& path,
    const HidCalibrationProfile& profile
);

}  // namespace vision_analyzer
