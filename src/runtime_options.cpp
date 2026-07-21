#include "vision_analyzer/runtime.hpp"

#include <cmath>
#include <stdexcept>

namespace vision_analyzer {

void validate_options(const Options& options) {
    if (options.dxgi_adapter < 0 || options.dxgi_output < 0) {
        throw std::runtime_error("--dxgi-adapter and --dxgi-output must be greater than or equal to 0");
    }
    if (options.dxgi_timeout_ms <= 0) {
        throw std::runtime_error("--dxgi-timeout must be greater than 0");
    }
    if (options.dxgi_roi.x < 0 || options.dxgi_roi.y < 0 ||
        options.dxgi_roi.width < 0 || options.dxgi_roi.height < 0) {
        throw std::runtime_error("--dxgi-roi values must be greater than or equal to 0");
    }
    if ((options.dxgi_roi.width == 0) != (options.dxgi_roi.height == 0)) {
        throw std::runtime_error("--dxgi-roi width and height must either both be set or both be 0");
    }
    if (!std::isfinite(options.hid_move_gain)) {
        throw std::runtime_error("--hid-gain must be finite");
    }
    if (options.hid_max_step < 0) {
        throw std::runtime_error("--hid-max-step must be greater than or equal to 0");
    }
    if (!std::isfinite(options.hid_deadzone_px) || options.hid_deadzone_px < 0.0F) {
        throw std::runtime_error("--hid-deadzone must be finite and greater than or equal to 0");
    }
    if (!std::isfinite(options.fire_policy.head_confidence) ||
        options.fire_policy.head_confidence < 0.0F ||
        options.fire_policy.head_confidence > 1.0F) {
        throw std::runtime_error("head fire confidence must be finite and in [0, 1]");
    }
    if (!std::isfinite(options.fire_policy.body_confidence) ||
        options.fire_policy.body_confidence < 0.0F ||
        options.fire_policy.body_confidence > 1.0F) {
        throw std::runtime_error("body fire confidence must be finite and in [0, 1]");
    }
    if (options.fire_policy.cooldown_frames < 0) {
        throw std::runtime_error("fire cooldown frames must be greater than or equal to 0");
    }
    if (!std::isfinite(options.tuning.body_head_anchor_ratio) ||
        options.tuning.body_head_anchor_ratio <= 0.0F ||
        options.tuning.body_head_anchor_ratio >= 0.5F) {
        throw std::runtime_error("--body-head-anchor-ratio must be finite and between 0 and 0.5");
    }
    if (!std::isfinite(options.tuning.kalman_process_noise) || options.tuning.kalman_process_noise <= 0.0F ||
        !std::isfinite(options.tuning.kalman_measurement_noise) || options.tuning.kalman_measurement_noise <= 0.0F ||
        !std::isfinite(options.tuning.kalman_error_covariance) || options.tuning.kalman_error_covariance <= 0.0F) {
        throw std::runtime_error("Kalman tuning values must be finite and greater than 0");
    }
    if (options.list_dxgi_outputs || options.probe_dxgi_outputs || options.verify_input) {
        return;
    }
    if (options.test_hid_move) {
        if (options.hid_port.empty()) {
            throw std::runtime_error("--test-hid-move requires --hid-port COMx");
        }
        return;
    }
    if (options.calibrate_hid) {
        if (options.input_source != InputSource::Dxgi) {
            throw std::runtime_error("--calibrate-hid requires DXGI input");
        }
        if (options.hid_port.empty()) {
            throw std::runtime_error("--calibrate-hid requires --hid-port COMx");
        }
        if (options.calibration_step_counts <= 0 || options.calibration_repeats < 2 ||
            options.calibration_noise_samples < 0 || options.calibration_settle_ms < 0) {
            throw std::runtime_error("calibration repeats must be at least 2; step must be positive; noise samples and settle must be non-negative");
        }
        return;
    }
    if (options.hid_port.empty() && !options.dry_run) {
        throw std::runtime_error("use --hid-port COMx for live SDK output or --dry-run for tuning");
    }
    if (!options.dry_run && options.player_side == PlayerSide::Unknown) {
        throw std::runtime_error("live SDK output requires --player-side ct or --player-side t");
    }
    if (options.status_every_frames <= 0) {
        throw std::runtime_error("--status-every must be greater than 0");
    }
}

}  // namespace vision_analyzer
