#include "vision_analyzer/tracking.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace vision_analyzer {
namespace {

constexpr double kPi = 3.14159265358979323846;

[[nodiscard]] float point_distance(const cv::Point2f& left, const cv::Point2f& right) {
    return static_cast<float>(cv::norm(left - right));
}

[[nodiscard]] float frame_diagonal(const cv::Size& frame_size) {
    return std::sqrt(static_cast<float>(frame_size.width * frame_size.width + frame_size.height * frame_size.height));
}

[[nodiscard]] float clamp01(float value) {
    return std::clamp(value, 0.0F, 1.0F);
}

}  // namespace

cv::Point2f box_center(const cv::Rect& box) {
    return {box.x + box.width / 2.0F, box.y + box.height / 2.0F};
}

cv::Point2f target_anchor_point(const Detection& detection, float body_head_anchor_ratio) {
    const cv::Rect& box = detection.box;
    const float clamped_ratio = std::clamp(body_head_anchor_ratio, 0.05F, 0.45F);
    if (is_body(detection.class_id)) {
        return {box.x + box.width / 2.0F, box.y + box.height * clamped_ratio};
    }
    return box_center(box);
}

float intersection_over_union(const cv::Rect& left, const cv::Rect& right) {
    const int x1 = std::max(left.x, right.x);
    const int y1 = std::max(left.y, right.y);
    const int x2 = std::min(left.x + left.width, right.x + right.width);
    const int y2 = std::min(left.y + left.height, right.y + right.height);
    const int intersection_width = std::max(0, x2 - x1);
    const int intersection_height = std::max(0, y2 - y1);
    const float intersection = static_cast<float>(intersection_width * intersection_height);
    const float union_area = static_cast<float>(left.area() + right.area()) - intersection;
    if (union_area <= 0.0F) {
        return 0.0F;
    }
    return intersection / union_area;
}

float TrackManager::match_score(const Track& track, const Detection& detection, const cv::Size& frame_size) const {
    const auto detection_center = box_center(detection.box);
    const float iou = intersection_over_union(track.detection.box, detection.box);
    const float distance = point_distance(track.predicted_center, detection_center);
    const float distance_gate = std::max(48.0F, frame_diagonal(frame_size) * 0.085F);
    const float proximity = clamp01(1.0F - distance / distance_gate);
    const float class_bonus = track.detection.class_id == detection.class_id ? 0.08F : -0.04F;
    return iou * 0.62F + proximity * 0.34F + class_bonus;
}

TrackedDetection TrackManager::export_track(const Track& track) const {
    const float hit_score = clamp01(static_cast<float>(track.hits) / 10.0F);
    const float miss_penalty = clamp01(static_cast<float>(track.missed) / static_cast<float>(max_missed_ + 1));
    return TrackedDetection{
        track.id,
        track.detection,
        track.center,
        track.predicted_center,
        track.velocity,
        track.age,
        track.hits,
        track.missed,
        track.confidence_ema,
        clamp01(hit_score * (1.0F - miss_penalty)),
    };
}

std::vector<TrackedDetection> TrackManager::update(const std::vector<Detection>& detections, const cv::Size& frame_size) {
    for (auto& track : tracks_) {
        track.predicted_center = track.center + track.velocity;
        ++track.age;
        ++track.missed;
    }

    struct Candidate {
        int track_index = -1;
        int detection_index = -1;
        float score = 0.0F;
    };

    std::vector<Candidate> candidates;
    for (int track_index = 0; track_index < static_cast<int>(tracks_.size()); ++track_index) {
        for (int detection_index = 0; detection_index < static_cast<int>(detections.size()); ++detection_index) {
            const float score = match_score(
                tracks_[static_cast<std::size_t>(track_index)],
                detections[static_cast<std::size_t>(detection_index)],
                frame_size
            );
            if (score >= 0.22F) {
                candidates.push_back({track_index, detection_index, score});
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& left, const Candidate& right) {
        return left.score > right.score;
    });

    std::vector<bool> used_tracks(tracks_.size(), false);
    std::vector<bool> used_detections(detections.size(), false);
    for (const auto& candidate : candidates) {
        if (used_tracks[static_cast<std::size_t>(candidate.track_index)] ||
            used_detections[static_cast<std::size_t>(candidate.detection_index)]) {
            continue;
        }

        auto& track = tracks_[static_cast<std::size_t>(candidate.track_index)];
        const auto& detection = detections[static_cast<std::size_t>(candidate.detection_index)];
        const cv::Point2f new_center = box_center(detection.box);
        const cv::Point2f observed_velocity = new_center - track.center;
        track.velocity = track.hits <= 1
            ? observed_velocity
            : track.velocity * 0.65F + observed_velocity * 0.35F;
        track.center = new_center;
        track.predicted_center = track.center + track.velocity;
        track.detection = detection;
        track.missed = 0;
        ++track.hits;
        track.confidence_ema = track.confidence_ema <= 0.0F
            ? detection.confidence
            : track.confidence_ema * 0.82F + detection.confidence * 0.18F;
        used_tracks[static_cast<std::size_t>(candidate.track_index)] = true;
        used_detections[static_cast<std::size_t>(candidate.detection_index)] = true;
    }

    for (int detection_index = 0; detection_index < static_cast<int>(detections.size()); ++detection_index) {
        if (used_detections[static_cast<std::size_t>(detection_index)]) {
            continue;
        }
        const auto& detection = detections[static_cast<std::size_t>(detection_index)];
        const cv::Point2f center = box_center(detection.box);
        tracks_.push_back(Track{
            next_id_++,
            detection,
            center,
            center,
            {0.0F, 0.0F},
            1,
            1,
            0,
            detection.confidence,
        });
    }

    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(), [&](const Track& track) {
            return track.missed > max_missed_;
        }),
        tracks_.end()
    );

    std::vector<TrackedDetection> visible;
    for (const auto& track : tracks_) {
        if (track.missed == 0) {
            visible.push_back(export_track(track));
        }
    }
    return visible;
}

TargetSelector::TargetSelector(RuntimeTuningConfig tuning)
    : tuning_(tuning) {}

float TargetSelector::score(
    const TrackedDetection& track,
    const cv::Size& frame_size,
    std::optional<int> active_track_id
) const {
    const cv::Point2f frame_center(frame_size.width / 2.0F, frame_size.height / 2.0F);
    const cv::Point2f anchor_delta =
        target_anchor_point(track.detection, tuning_.body_head_anchor_ratio) - box_center(track.detection.box);
    const cv::Point2f predicted_anchor = track.predicted_center + anchor_delta;
    const float distance = point_distance(predicted_anchor, frame_center);
    const float normalized_distance = distance / std::max(1.0F, frame_diagonal(frame_size));
    const float class_bias = is_head(track.detection.class_id) ? 0.70F : 1.00F;
    const float confidence_factor = 1.0F / std::max(0.05F, track.confidence_ema);
    const float stability_factor = 1.0F - std::min(0.28F, track.stability * 0.28F);
    const float switch_factor = active_track_id.has_value() && *active_track_id != track.track_id ? 1.22F : 0.88F;
    const float area = static_cast<float>(track.detection.box.area());
    const float frame_area = static_cast<float>(std::max(1, frame_size.width * frame_size.height));
    const float size_factor = area / frame_area < 0.00045F ? 1.20F : 1.0F;
    return normalized_distance * class_bias * confidence_factor * stability_factor * switch_factor * size_factor;
}

std::optional<TrackedDetection> TargetSelector::select(
    const std::vector<TrackedDetection>& tracks,
    const cv::Size& frame_size,
    std::optional<int> active_track_id
) const {
    if (tracks.empty()) {
        return std::nullopt;
    }
    return *std::min_element(tracks.begin(), tracks.end(), [&](const TrackedDetection& left, const TrackedDetection& right) {
        return score(left, frame_size, active_track_id) < score(right, frame_size, active_track_id);
    });
}

OneEuroFilter::OneEuroFilter(double min_cutoff, double beta, double derivative_cutoff)
    : min_cutoff_(min_cutoff),
      beta_(beta),
      derivative_cutoff_(derivative_cutoff) {}

double OneEuroFilter::smoothing_alpha(double cutoff, double dt) const {
    const double tau = 1.0 / (2.0 * kPi * cutoff);
    return 1.0 / (1.0 + tau / std::max(1e-6, dt));
}

double OneEuroFilter::update(double value, double timestamp_seconds) {
    if (!initialized_) {
        initialized_ = true;
        previous_timestamp_ = timestamp_seconds;
        previous_value_ = value;
        previous_derivative_ = 0.0;
        return value;
    }

    const double dt = std::max(1e-3, timestamp_seconds - previous_timestamp_);
    const double derivative = (value - previous_value_) / dt;
    const double derivative_alpha = smoothing_alpha(derivative_cutoff_, dt);
    const double derivative_hat = derivative_alpha * derivative + (1.0 - derivative_alpha) * previous_derivative_;
    const double cutoff = min_cutoff_ + beta_ * std::abs(derivative_hat);
    const double value_alpha = smoothing_alpha(cutoff, dt);
    const double value_hat = value_alpha * value + (1.0 - value_alpha) * previous_value_;

    previous_timestamp_ = timestamp_seconds;
    previous_value_ = value_hat;
    previous_derivative_ = derivative_hat;
    return value_hat;
}

void OneEuroFilter::reset() {
    initialized_ = false;
    previous_timestamp_ = 0.0;
    previous_value_ = 0.0;
    previous_derivative_ = 0.0;
}

MotionFilter2D::MotionFilter2D(RuntimeTuningConfig tuning)
    : filter_(4, 2, 0, CV_32F),
      tuning_(tuning) {
    configure_filter(1.0F / 60.0F);
    cv::setIdentity(filter_.errorCovPost, cv::Scalar::all(tuning_.kalman_error_covariance));
}

cv::Point2f MotionFilter2D::update(const cv::Point2f& measurement, double timestamp_ms) {
    float dt = 1.0F / 60.0F;
    if (initialized_) {
        dt = static_cast<float>(std::clamp((timestamp_ms - previous_timestamp_ms_) / 1000.0, 1e-3, 0.25));
    }
    configure_filter(dt);

    if (!initialized_) {
        filter_.statePost.at<float>(0) = measurement.x;
        filter_.statePost.at<float>(1) = measurement.y;
        filter_.statePost.at<float>(2) = 0.0F;
        filter_.statePost.at<float>(3) = 0.0F;
        cv::setIdentity(filter_.errorCovPost, cv::Scalar::all(tuning_.kalman_error_covariance));
        previous_point_ = measurement;
        previous_timestamp_ms_ = timestamp_ms;
        velocity_ = {0.0F, 0.0F};
        acceleration_ = {0.0F, 0.0F};
        initialized_ = true;
        return measurement;
    }

    filter_.predict();
    cv::Mat measurement_vector(2, 1, CV_32F);
    measurement_vector.at<float>(0) = measurement.x;
    measurement_vector.at<float>(1) = measurement.y;
    const cv::Mat corrected = filter_.correct(measurement_vector);

    const cv::Point2f filtered(corrected.at<float>(0), corrected.at<float>(1));
    const cv::Point2f new_velocity(corrected.at<float>(2), corrected.at<float>(3));
    acceleration_ = (new_velocity - velocity_) / dt;
    velocity_ = new_velocity;

    previous_timestamp_ms_ = timestamp_ms;
    previous_point_ = filtered;
    return filtered;
}

cv::Point2f MotionFilter2D::predict(double lookahead_ms) const {
    if (!initialized_) {
        return previous_point_;
    }
    const float dt = static_cast<float>(std::max(0.0, lookahead_ms) / 1000.0);
    return {
        filter_.statePost.at<float>(0) + filter_.statePost.at<float>(2) * dt,
        filter_.statePost.at<float>(1) + filter_.statePost.at<float>(3) * dt,
    };
}

void MotionFilter2D::reset() {
    configure_filter(1.0F / 60.0F);
    cv::setIdentity(filter_.errorCovPost, cv::Scalar::all(tuning_.kalman_error_covariance));
    initialized_ = false;
    previous_timestamp_ms_ = 0.0;
    previous_point_ = {0.0F, 0.0F};
    velocity_ = {0.0F, 0.0F};
    acceleration_ = {0.0F, 0.0F};
}

cv::Point2f MotionFilter2D::velocity() const {
    return velocity_;
}

cv::Point2f MotionFilter2D::acceleration() const {
    return acceleration_;
}

bool MotionFilter2D::initialized() const {
    return initialized_;
}

void MotionFilter2D::configure_filter(float dt_seconds) {
    filter_.transitionMatrix = (cv::Mat_<float>(4, 4) <<
        1.0F, 0.0F, dt_seconds, 0.0F,
        0.0F, 1.0F, 0.0F, dt_seconds,
        0.0F, 0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 1.0F);
    filter_.measurementMatrix = cv::Mat::zeros(2, 4, CV_32F);
    filter_.measurementMatrix.at<float>(0, 0) = 1.0F;
    filter_.measurementMatrix.at<float>(1, 1) = 1.0F;
    cv::setIdentity(filter_.processNoiseCov, cv::Scalar::all(tuning_.kalman_process_noise));
    cv::setIdentity(filter_.measurementNoiseCov, cv::Scalar::all(tuning_.kalman_measurement_noise));
}

AnalysisState::AnalysisState(RuntimeTuningConfig tuning)
    : tuning_(tuning),
      filter_(tuning) {}

TargetFrame AnalysisState::update(
    const TrackedDetection& selected,
    const cv::Size& frame_size,
    double timestamp_ms,
    double latency_ms
) {
    const bool switched = !active_track_id_.has_value() || *active_track_id_ != selected.track_id;
    if (switched) {
        filter_.reset();
        locked_frames_ = 0;
    }

    active_track_id_ = selected.track_id;
    missing_frames_ = 0;
    if (previous_timestamp_ms_.has_value() && timestamp_ms > *previous_timestamp_ms_) {
        const double interval = std::clamp(timestamp_ms - *previous_timestamp_ms_, 1.0, 250.0);
        frame_interval_ema_ms_ = frame_interval_ema_ms_ * 0.88 + interval * 0.12;
    }
    previous_timestamp_ms_ = timestamp_ms;

    const cv::Point2f frame_center(frame_size.width / 2.0F, frame_size.height / 2.0F);
    const float latency_frames = static_cast<float>(
        std::clamp(latency_ms / std::max(1.0, frame_interval_ema_ms_), 0.0, 2.0)
    );
    const cv::Point2f anchor = target_anchor_point(selected.detection, tuning_.body_head_anchor_ratio);
    const cv::Point2f predicted = anchor + selected.velocity * latency_frames;
    (void)filter_.update(predicted, timestamp_ms);
    const cv::Point2f analysis_point = filter_.predict(latency_ms);
    const cv::Point2f offset = analysis_point - frame_center;
    const float distance = point_distance(analysis_point, frame_center);
    const float lock_radius = is_head(selected.detection.class_id) ? 18.0F : 30.0F;

    if (distance <= lock_radius) {
        ++locked_frames_;
    } else {
        locked_frames_ = 0;
    }

    LockState lock_state = LockState::Acquiring;
    if (distance <= lock_radius && selected.stability >= 0.35F) {
        lock_state = LockState::Locked;
    } else if (selected.stability >= 0.45F) {
        lock_state = LockState::Tracking;
    }

    const bool fire_candidate =
        is_head(selected.detection.class_id) &&
        lock_state == LockState::Locked &&
        locked_frames_ >= 3 &&
        selected.confidence_ema >= 0.48F;

    return TargetFrame{
        selected.track_id,
        selected.detection,
        anchor,
        predicted,
        offset,
        distance,
        switched,
        analysis_point,
        filter_.velocity(),
        filter_.acceleration(),
        selected.stability,
        point_distance(analysis_point, predicted),
        lock_state,
        fire_candidate,
    };
}

void AnalysisState::mark_no_target() {
    ++missing_frames_;
    if (missing_frames_ > 2) {
        active_track_id_.reset();
        filter_.reset();
        locked_frames_ = 0;
    }
}

std::optional<int> AnalysisState::active_track_id() const {
    return active_track_id_;
}

}  // namespace vision_analyzer
