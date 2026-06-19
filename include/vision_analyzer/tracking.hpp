#pragma once

#include <optional>
#include <vector>

#include <opencv2/video/tracking.hpp>

#include "vision_analyzer/types.hpp"

namespace vision_analyzer {

[[nodiscard]] cv::Point2f box_center(const cv::Rect& box);
[[nodiscard]] cv::Point2f target_anchor_point(const Detection& detection, float body_head_anchor_ratio = 0.18F);
[[nodiscard]] std::vector<Detection> fuse_head_body_detections(
    const std::vector<Detection>& detections,
    RuntimeTuningConfig tuning = {}
);
[[nodiscard]] float intersection_over_union(const cv::Rect& left, const cv::Rect& right);

class TrackManager {
public:
    [[nodiscard]] std::vector<TrackedDetection> update(
        const std::vector<Detection>& detections,
        const cv::Size& frame_size,
        RuntimeTuningConfig tuning = {}
    );

private:
    struct Track {
        int id = -1;
        Detection detection;
        cv::Point2f center;
        cv::Point2f predicted_center;
        cv::Point2f velocity;
        int age = 0;
        int hits = 0;
        int missed = 0;
        float confidence_ema = 0.0F;
    };

    [[nodiscard]] float match_score(
        const Track& track,
        const Detection& detection,
        const cv::Size& frame_size,
        RuntimeTuningConfig tuning
    ) const;
    [[nodiscard]] TrackedDetection export_track(const Track& track) const;

    std::vector<Track> tracks_;
    int next_id_ = 1;
    int max_missed_ = 8;
};

class TargetSelector {
public:
    explicit TargetSelector(RuntimeTuningConfig tuning = {});

    [[nodiscard]] std::optional<TrackedDetection> select(
        const std::vector<TrackedDetection>& tracks,
        const cv::Size& frame_size,
        std::optional<int> active_track_id
    ) const;

private:
    [[nodiscard]] float score(
        const TrackedDetection& track,
        const cv::Size& frame_size,
        std::optional<int> active_track_id
    ) const;

    RuntimeTuningConfig tuning_;
};

class OneEuroFilter {
public:
    OneEuroFilter(double min_cutoff, double beta, double derivative_cutoff);

    [[nodiscard]] double update(double value, double timestamp_seconds);
    void reset();

private:
    [[nodiscard]] double smoothing_alpha(double cutoff, double dt) const;

    double min_cutoff_ = 1.0;
    double beta_ = 0.0;
    double derivative_cutoff_ = 1.0;
    bool initialized_ = false;
    double previous_timestamp_ = 0.0;
    double previous_value_ = 0.0;
    double previous_derivative_ = 0.0;
};

class MotionFilter2D {
public:
    explicit MotionFilter2D(RuntimeTuningConfig tuning = {});

    [[nodiscard]] cv::Point2f update(const cv::Point2f& measurement, double timestamp_ms);
    [[nodiscard]] cv::Point2f predict(double lookahead_ms) const;
    void reset();

    [[nodiscard]] cv::Point2f velocity() const;
    [[nodiscard]] cv::Point2f acceleration() const;
    [[nodiscard]] bool initialized() const;

private:
    void configure_filter(float dt_seconds);

    cv::KalmanFilter filter_;
    RuntimeTuningConfig tuning_;
    bool initialized_ = false;
    double previous_timestamp_ms_ = 0.0;
    cv::Point2f previous_point_;
    cv::Point2f velocity_;
    cv::Point2f acceleration_;
};

class AnalysisState {
public:
    explicit AnalysisState(RuntimeTuningConfig tuning = {});

    [[nodiscard]] TargetFrame update(
        const TrackedDetection& selected,
        const cv::Size& frame_size,
        double timestamp_ms,
        double latency_ms
    );

    void mark_no_target();
    [[nodiscard]] std::optional<int> active_track_id() const;

private:
    std::optional<int> active_track_id_;
    RuntimeTuningConfig tuning_;
    MotionFilter2D filter_;
    int locked_frames_ = 0;
    int missing_frames_ = 0;
    std::optional<double> previous_timestamp_ms_;
    double frame_interval_ema_ms_ = 33.333;
};

}  // namespace vision_analyzer
