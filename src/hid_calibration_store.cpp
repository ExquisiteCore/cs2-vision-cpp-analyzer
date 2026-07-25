#include "vision_analyzer/hid_calibration_store.hpp"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>

#include <opencv2/core.hpp>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace vision_analyzer {
namespace {

std::atomic<std::uint64_t> temporary_counter{0};

template <typename Value>
void read_required_scalar(
    const cv::FileStorage& storage,
    const char* key,
    Value& value
) {
    const cv::FileNode node = storage[key];
    if (node.empty() || node.isSeq() || node.isMap()) {
        throw std::runtime_error(std::string("invalid HID calibration field: ") + key);
    }
    node >> value;
}

void read_required_curve(
    const cv::FileStorage& storage,
    const char* key,
    std::array<float, kHidCalibrationLevels>& values
) {
    const cv::FileNode node = storage[key];
    if (node.empty() || !node.isSeq() || node.size() != kHidCalibrationLevels) {
        throw std::runtime_error(std::string("invalid HID calibration field: ") + key);
    }
    for (std::size_t index = 0; index < values.size(); ++index) {
        node[static_cast<int>(index)] >> values[index];
    }
}

void write_curve(
    cv::FileStorage& storage,
    const char* key,
    const std::array<float, kHidCalibrationLevels>& values
) {
    storage << key << "[";
    for (float value : values) {
        storage << value;
    }
    storage << "]";
}

[[nodiscard]] std::uint64_t process_identifier() {
#if defined(_WIN32)
    return static_cast<std::uint64_t>(GetCurrentProcessId());
#else
    return static_cast<std::uint64_t>(getpid());
#endif
}

[[nodiscard]] std::filesystem::path temporary_sibling(
    const std::filesystem::path& destination
) {
    std::ostringstream name;
    name << destination.filename().string()
         << ".tmp."
         << process_identifier()
         << '.'
         << temporary_counter.fetch_add(1, std::memory_order_relaxed);
    return destination.parent_path() / name.str();
}

void replace_file_write_through(
    const std::filesystem::path& temporary,
    const std::filesystem::path& destination
) {
#if defined(_WIN32)
    if (MoveFileExW(
            temporary.c_str(),
            destination.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH
        ) == FALSE) {
        const DWORD error = GetLastError();
        throw std::runtime_error(
            "failed to atomically replace HID calibration profile: win32=" +
            std::to_string(error)
        );
    }
#else
    std::filesystem::rename(temporary, destination);
#endif
}

}  // namespace

HidCalibrationProfile load_hid_calibration_profile(
    const std::filesystem::path& path
) {
    if (path.empty()) {
        throw std::runtime_error("HID calibration path must not be empty");
    }

    try {
        cv::FileStorage storage(
            path.string(),
            cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON
        );
        if (!storage.isOpened()) {
            throw std::runtime_error("failed to open HID calibration profile: " + path.string());
        }

        int schema_version = 0;
        read_required_scalar(storage, "schema_version", schema_version);
        if (schema_version != kHidCalibrationSchemaVersion) {
            throw std::runtime_error("invalid HID calibration field: schema_version");
        }

        HidCalibrationProfile profile;
        profile.valid = true;
        read_required_scalar(storage, "frame_width", profile.frame_width);
        read_required_scalar(storage, "frame_height", profile.frame_height);
        read_required_curve(storage, "x_shift_px", profile.x.shift_px);
        read_required_curve(storage, "x_counts_per_pixel", profile.x.counts_per_pixel);
        read_required_curve(storage, "y_shift_px", profile.y.shift_px);
        read_required_curve(storage, "y_counts_per_pixel", profile.y.counts_per_pixel);
        read_required_scalar(storage, "deadzone_px", profile.deadzone_px);
        read_required_scalar(storage, "max_step", profile.max_step);
        read_required_scalar(storage, "noise_px", profile.noise_px);
        read_required_scalar(storage, "quality", profile.quality);
        read_required_scalar(storage, "accepted_samples", profile.accepted_samples);

        if (!valid_hid_calibration_profile(profile)) {
            throw std::runtime_error("invalid HID calibration profile values");
        }
        return profile;
    } catch (const cv::Exception& error) {
        throw std::runtime_error(
            "failed to parse HID calibration profile: " + std::string(error.what())
        );
    }
}

void save_hid_calibration_profile_atomic(
    const std::filesystem::path& path,
    const HidCalibrationProfile& profile
) {
    if (path.empty()) {
        throw std::runtime_error("HID calibration path must not be empty");
    }
    if (!valid_hid_calibration_profile(profile)) {
        throw std::runtime_error("refusing to save invalid HID calibration profile");
    }

    const std::filesystem::path temporary = temporary_sibling(path);
    try {
        cv::FileStorage storage(
            temporary.string(),
            cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON
        );
        if (!storage.isOpened()) {
            throw std::runtime_error(
                "failed to create temporary HID calibration profile: " + temporary.string()
            );
        }
        storage << "schema_version" << kHidCalibrationSchemaVersion;
        storage << "frame_width" << profile.frame_width;
        storage << "frame_height" << profile.frame_height;
        write_curve(storage, "x_shift_px", profile.x.shift_px);
        write_curve(storage, "x_counts_per_pixel", profile.x.counts_per_pixel);
        write_curve(storage, "y_shift_px", profile.y.shift_px);
        write_curve(storage, "y_counts_per_pixel", profile.y.counts_per_pixel);
        storage << "deadzone_px" << profile.deadzone_px;
        storage << "max_step" << profile.max_step;
        storage << "noise_px" << profile.noise_px;
        storage << "quality" << profile.quality;
        storage << "accepted_samples" << profile.accepted_samples;
        storage.release();

        (void)load_hid_calibration_profile(temporary);
        replace_file_write_through(temporary, path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

}  // namespace vision_analyzer
