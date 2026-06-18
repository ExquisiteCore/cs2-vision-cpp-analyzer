#include "vision_analyzer/detector.hpp"

#include <array>
#include <chrono>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#if defined(VISION_ANALYZER_WITH_ORT)
#include <onnxruntime_cxx_api.h>
#endif

#include "vision_analyzer/postprocess.hpp"

namespace vision_analyzer {
namespace {

[[nodiscard]] double elapsed_ms(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

class OpenCvDetector final : public Detector {
public:
    OpenCvDetector(const std::string& model_path, bool use_cuda)
        : use_cuda_(use_cuda) {
        net_ = cv::dnn::readNetFromONNX(model_path);
        if (net_.empty()) {
            throw std::runtime_error("failed to load ONNX model: " + model_path);
        }

        if (use_cuda_) {
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
        } else {
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
    }

    DetectionResult detect(const cv::Mat& frame, float confidence, float nms_threshold) override {
        const auto preprocess_start = std::chrono::steady_clock::now();
        const LetterboxResult prepared = letterbox(frame, kInputSize);
        cv::Mat blob = cv::dnn::blobFromImage(
            prepared.image,
            1.0 / 255.0,
            cv::Size(kInputSize, kInputSize),
            cv::Scalar(),
            true,
            false
        );
        const auto preprocess_end = std::chrono::steady_clock::now();

        const auto inference_start = std::chrono::steady_clock::now();
        net_.setInput(blob);
        cv::Mat output;
        try {
            output = net_.forward();
        } catch (const cv::Exception& error) {
            if (use_cuda_) {
                throw std::runtime_error(
                    "OpenCV CUDA DNN backend failed. This usually means the OpenCV package was built without CUDA DNN. "
                    "Use --backend opencv-onnx, or build OpenCV with CUDA, cuDNN, and DNN CUDA support. OpenCV error: " +
                    std::string(error.what())
                );
            }
            throw;
        }
        const auto inference_end = std::chrono::steady_clock::now();

        const auto postprocess_start = std::chrono::steady_clock::now();
        const int sizes[] = {1, output.size[1], output.size[2]};
        cv::Mat shaped(3, sizes, CV_32F, output.ptr<float>());
        auto detections = decode_yolo_output(shaped, prepared, frame.size(), confidence, nms_threshold);
        const auto postprocess_end = std::chrono::steady_clock::now();

        return DetectionResult{
            std::move(detections),
            InferenceTiming{
                elapsed_ms(preprocess_start, preprocess_end),
                elapsed_ms(inference_start, inference_end),
                elapsed_ms(postprocess_start, postprocess_end),
            },
        };
    }

    std::string name() const override {
        return use_cuda_ ? "opencv-cuda" : "opencv-onnx";
    }

private:
    cv::dnn::Net net_;
    bool use_cuda_ = false;
};

#if defined(VISION_ANALYZER_WITH_ORT)

[[nodiscard]] std::wstring widen_path(const std::string& path) {
    return std::filesystem::path(path).wstring();
}

[[nodiscard]] std::vector<const char*> name_ptrs(const std::vector<std::string>& names) {
    std::vector<const char*> ptrs;
    ptrs.reserve(names.size());
    for (const auto& name : names) {
        ptrs.push_back(name.c_str());
    }
    return ptrs;
}

[[nodiscard]] std::vector<std::string> get_input_names(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> names;
    const std::size_t count = session.GetInputCount();
    names.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        auto name = session.GetInputNameAllocated(index, allocator);
        names.emplace_back(name.get());
    }
    return names;
}

[[nodiscard]] std::vector<std::string> get_output_names(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> names;
    const std::size_t count = session.GetOutputCount();
    names.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        auto name = session.GetOutputNameAllocated(index, allocator);
        names.emplace_back(name.get());
    }
    return names;
}

void append_cuda_provider(Ort::SessionOptions& session_options) {
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0;
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
    cuda_options.gpu_mem_limit = 2ULL * 1024ULL * 1024ULL * 1024ULL;
    cuda_options.do_copy_in_default_stream = 1;
    session_options.AppendExecutionProvider_CUDA(cuda_options);
}

void append_tensorrt_provider(Ort::SessionOptions& session_options) {
    OrtTensorRTProviderOptionsV2* trt_options = nullptr;
    const auto& api = Ort::GetApi();
    Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&trt_options));

    const std::filesystem::path cache_path = std::filesystem::absolute("../../runs/ort_trt_cache");
    std::filesystem::create_directories(cache_path);
    const std::string cache = cache_path.string();
    const std::array<const char*, 6> keys = {
        "device_id",
        "trt_fp16_enable",
        "trt_engine_cache_enable",
        "trt_engine_cache_path",
        "trt_max_workspace_size",
        "trt_min_subgraph_size",
    };
    const std::string workspace = std::to_string(1ULL * 1024ULL * 1024ULL * 1024ULL);
    const std::array<const char*, 6> values = {
        "0",
        "1",
        "1",
        cache.c_str(),
        workspace.c_str(),
        "1",
    };

    try {
        Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(trt_options, keys.data(), values.data(), keys.size()));
        session_options.AppendExecutionProvider_TensorRT_V2(*trt_options);
    } catch (...) {
        api.ReleaseTensorRTProviderOptions(trt_options);
        throw;
    }
    api.ReleaseTensorRTProviderOptions(trt_options);
}

class OrtDetector final : public Detector {
public:
    OrtDetector(const std::string& model_path, Backend backend)
        : backend_(backend),
          env_(ORT_LOGGING_LEVEL_WARNING, "vision_analyzer"),
          session_(nullptr),
          input_data_(3 * kInputSize * kInputSize) {
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetInterOpNumThreads(1);

        if (backend_ == Backend::OrtTensorRt) {
            append_tensorrt_provider(session_options_);
            append_cuda_provider(session_options_);
        } else {
            append_cuda_provider(session_options_);
        }

        const auto wide_model_path = widen_path(model_path);
        session_ = Ort::Session(env_, wide_model_path.c_str(), session_options_);
        input_names_ = get_input_names(session_);
        output_names_ = get_output_names(session_);
        if (input_names_.empty() || output_names_.empty()) {
            throw std::runtime_error("ORT model has no input or output tensors");
        }
    }

    DetectionResult detect(const cv::Mat& frame, float confidence, float nms_threshold) override {
        const auto preprocess_start = std::chrono::steady_clock::now();
        const LetterboxResult prepared = letterbox(frame, kInputSize);
        fill_input(prepared.image);
        const auto preprocess_end = std::chrono::steady_clock::now();

        const auto inference_start = std::chrono::steady_clock::now();
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<int64_t, 4> input_shape = {1, 3, kInputSize, kInputSize};
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_data_.data(),
            input_data_.size(),
            input_shape.data(),
            input_shape.size()
        );
        auto input_ptrs = name_ptrs(input_names_);
        auto output_ptrs = name_ptrs(output_names_);
        auto outputs = session_.Run(
            Ort::RunOptions{nullptr},
            input_ptrs.data(),
            &input_tensor,
            1,
            output_ptrs.data(),
            output_ptrs.size()
        );
        const auto inference_end = std::chrono::steady_clock::now();

        const auto postprocess_start = std::chrono::steady_clock::now();
        if (outputs.empty() || !outputs[0].IsTensor()) {
            throw std::runtime_error("ORT model did not return a tensor output");
        }
        auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 3) {
            throw std::runtime_error("unexpected ORT YOLO output rank: " + std::to_string(shape.size()));
        }
        std::vector<int> sizes;
        sizes.reserve(shape.size());
        for (const auto dimension : shape) {
            if (dimension <= 0 || dimension > static_cast<int64_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("unexpected ORT YOLO output dimension");
            }
            sizes.push_back(static_cast<int>(dimension));
        }
        float* output_data = outputs[0].GetTensorMutableData<float>();
        cv::Mat shaped(static_cast<int>(sizes.size()), sizes.data(), CV_32F, output_data);
        auto detections = decode_yolo_output(shaped, prepared, frame.size(), confidence, nms_threshold);
        const auto postprocess_end = std::chrono::steady_clock::now();

        return DetectionResult{
            std::move(detections),
            InferenceTiming{
                elapsed_ms(preprocess_start, preprocess_end),
                elapsed_ms(inference_start, inference_end),
                elapsed_ms(postprocess_start, postprocess_end),
            },
        };
    }

    std::string name() const override {
        return backend_ == Backend::OrtTensorRt ? "ort-tensorrt" : "ort-cuda";
    }

private:
    void fill_input(const cv::Mat& image) {
        const int plane = kInputSize * kInputSize;
        for (int y = 0; y < kInputSize; ++y) {
            const auto* row = image.ptr<cv::Vec3b>(y);
            for (int x = 0; x < kInputSize; ++x) {
                const int offset = y * kInputSize + x;
                const cv::Vec3b bgr = row[x];
                input_data_[offset] = static_cast<float>(bgr[2]) / 255.0F;
                input_data_[plane + offset] = static_cast<float>(bgr[1]) / 255.0F;
                input_data_[2 * plane + offset] = static_cast<float>(bgr[0]) / 255.0F;
            }
        }
    }

    Backend backend_ = Backend::OrtCuda;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<float> input_data_;
};

#else

class OrtUnavailableDetector final : public Detector {
public:
    explicit OrtUnavailableDetector(Backend backend)
        : backend_(backend) {}

    DetectionResult detect(const cv::Mat&, float, float) override {
        throw std::runtime_error("ONNX Runtime backend is not compiled in this build.");
    }

    std::string name() const override {
        return backend_ == Backend::OrtTensorRt ? "ort-tensorrt-unavailable" : "ort-cuda-unavailable";
    }

private:
    Backend backend_;
};

#endif

class TensorRtUnavailableDetector final : public Detector {
public:
    explicit TensorRtUnavailableDetector(std::string model_path)
        : model_path_(std::move(model_path)) {}

    DetectionResult detect(const cv::Mat&, float, float) override {
        throw std::runtime_error(
            "TensorRT C++ backend is not compiled in this build. "
            "The engine exists, but this machine needs TensorRT C++ headers/libraries such as NvInfer.h and nvinfer import libs. "
            "Use --backend opencv-onnx for now, or install the TensorRT C++ SDK and rebuild."
        );
    }

    std::string name() const override {
        return "tensorrt-unavailable:" + model_path_;
    }

private:
    std::string model_path_;
};

}  // namespace

Backend parse_backend(const std::string& value) {
    if (value == "opencv-onnx") {
        return Backend::OpenCvOnnx;
    }
    if (value == "opencv-cuda") {
        return Backend::OpenCvCuda;
    }
    if (value == "ort-cuda") {
        return Backend::OrtCuda;
    }
    if (value == "ort-tensorrt") {
        return Backend::OrtTensorRt;
    }
    if (value == "tensorrt") {
        return Backend::TensorRt;
    }
    throw std::runtime_error("unknown backend: " + value);
}

std::unique_ptr<Detector> create_detector(Backend backend, const std::string& model_path) {
    switch (backend) {
    case Backend::OpenCvOnnx:
        return std::make_unique<OpenCvDetector>(model_path, false);
    case Backend::OpenCvCuda:
        return std::make_unique<OpenCvDetector>(model_path, true);
    case Backend::OrtCuda:
#if defined(VISION_ANALYZER_WITH_ORT)
        return std::make_unique<OrtDetector>(model_path, backend);
#else
        return std::make_unique<OrtUnavailableDetector>(backend);
#endif
    case Backend::OrtTensorRt:
#if defined(VISION_ANALYZER_WITH_ORT)
        return std::make_unique<OrtDetector>(model_path, backend);
#else
        return std::make_unique<OrtUnavailableDetector>(backend);
#endif
    case Backend::TensorRt:
        return std::make_unique<TensorRtUnavailableDetector>(model_path);
    }
    throw std::runtime_error("unknown backend");
}

}  // namespace vision_analyzer
