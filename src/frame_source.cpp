#include "vision_analyzer/frame_source.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d11.h>
#include <dxgi1_6.h>
#include <windows.h>
#include <wrl/client.h>
#endif

namespace vision_analyzer {
namespace {

[[nodiscard]] double now_ms() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(now).count();
}

class VideoFrameSource final : public FrameSource {
public:
    explicit VideoFrameSource(const Options& options)
        : capture_(options.video_path) {
        if (!capture_.isOpened()) {
            throw std::runtime_error("failed to open video: " + options.video_path);
        }
        if (options.start_time_seconds > 0.0) {
            capture_.set(cv::CAP_PROP_POS_MSEC, options.start_time_seconds * 1000.0);
        } else if (options.start_frame > 0) {
            capture_.set(cv::CAP_PROP_POS_FRAMES, options.start_frame);
        }
    }

    bool read(CapturedFrame& frame) override {
        cv::Mat image;
        if (!capture_.read(image)) {
            return false;
        }
        frame = CapturedFrame{
            std::move(image),
            static_cast<int>(capture_.get(cv::CAP_PROP_POS_FRAMES)) - 1,
            capture_.get(cv::CAP_PROP_POS_MSEC),
        };
        return true;
    }

    void release() override {
        capture_.release();
    }

    std::string name() const override {
        return "video";
    }

    bool is_live() const override {
        return false;
    }

private:
    cv::VideoCapture capture_;
};

#if defined(_WIN32)

template <typename T>
void throw_if_failed(T result, const char* message) {
    if (FAILED(result)) {
        std::ostringstream stream;
        stream << message << " hresult=0x"
               << std::hex << std::uppercase << static_cast<unsigned long>(static_cast<HRESULT>(result));
        throw std::runtime_error(stream.str());
    }
}

[[nodiscard]] std::string narrow_utf8(const wchar_t* value) {
    const int required = WideCharToMultiByte(CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
    if (required <= 1) {
        return {};
    }
    std::string result(static_cast<std::size_t>(required - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value, -1, result.data(), required, nullptr, nullptr);
    return result;
}

[[nodiscard]] std::string hresult_hex(HRESULT result) {
    std::ostringstream stream;
    stream << "0x" << std::hex << std::uppercase << static_cast<unsigned long>(result);
    return stream.str();
}

[[nodiscard]] const char* output_rotation_name(DXGI_MODE_ROTATION rotation) {
    switch (rotation) {
    case DXGI_MODE_ROTATION_UNSPECIFIED:
        return "unspecified";
    case DXGI_MODE_ROTATION_IDENTITY:
        return "identity";
    case DXGI_MODE_ROTATION_ROTATE90:
        return "rotate90";
    case DXGI_MODE_ROTATION_ROTATE180:
        return "rotate180";
    case DXGI_MODE_ROTATION_ROTATE270:
        return "rotate270";
    }
    return "unknown";
}

class DxgiFrameSource final : public FrameSource {
public:
    explicit DxgiFrameSource(const Options& options)
        : adapter_index_(options.dxgi_adapter),
          output_index_(options.dxgi_output),
          timeout_ms_(options.dxgi_timeout_ms),
          roi_(options.dxgi_roi),
          debug_(options.dxgi_debug),
          start_time_ms_(now_ms()) {
        initialize();
    }

    bool read(CapturedFrame& frame) override {
        for (;;) {
            DXGI_OUTDUPL_FRAME_INFO frame_info{};
            Microsoft::WRL::ComPtr<IDXGIResource> desktop_resource;
            const HRESULT acquire_result = duplication_->AcquireNextFrame(
                static_cast<UINT>(timeout_ms_),
                &frame_info,
                &desktop_resource
            );
            if (acquire_result == DXGI_ERROR_WAIT_TIMEOUT) {
                continue;
            }
            if (acquire_result == DXGI_ERROR_ACCESS_LOST) {
                initialize();
                continue;
            }
            throw_if_failed(acquire_result, "DXGI AcquireNextFrame failed");
            if (debug_ && !debug_frame_info_printed_) {
                std::cout << "dxgi_frame_info"
                          << " accumulated=" << frame_info.AccumulatedFrames
                          << " rects_coalesced=" << (frame_info.RectsCoalesced ? 1 : 0)
                          << " protected_content_masked=" << (frame_info.ProtectedContentMaskedOut ? 1 : 0)
                          << " last_present=" << frame_info.LastPresentTime.QuadPart
                          << " last_mouse=" << frame_info.LastMouseUpdateTime.QuadPart
                          << '\n';
                debug_frame_info_printed_ = true;
            }
            if (frame_info.AccumulatedFrames == 0 && frame_index_ == 0) {
                duplication_->ReleaseFrame();
                continue;
            }

            try {
                Microsoft::WRL::ComPtr<ID3D11Texture2D> desktop_texture;
                throw_if_failed(desktop_resource.As(&desktop_texture), "DXGI frame is not a D3D11 texture");
                copy_texture_to_frame(desktop_texture.Get(), frame);
                duplication_->ReleaseFrame();
                return true;
            } catch (...) {
                duplication_->ReleaseFrame();
                throw;
            }
        }
    }

    void release() override {
        staging_texture_.Reset();
        duplication_.Reset();
        context_.Reset();
        device_.Reset();
    }

    std::string name() const override {
        if (roi_.width > 0 && roi_.height > 0) {
            return "dxgi-roi";
        }
        return "dxgi";
    }

    bool is_live() const override {
        return true;
    }

private:
    void initialize() {
        staging_texture_.Reset();
        duplication_.Reset();
        context_.Reset();
        device_.Reset();

        Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
        throw_if_failed(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1 failed");

        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
        if (factory->EnumAdapters1(static_cast<UINT>(adapter_index_), &adapter) == DXGI_ERROR_NOT_FOUND) {
            throw std::runtime_error("DXGI adapter index is unavailable");
        }

        Microsoft::WRL::ComPtr<IDXGIOutput> output;
        if (adapter->EnumOutputs(static_cast<UINT>(output_index_), &output) == DXGI_ERROR_NOT_FOUND) {
            throw std::runtime_error("DXGI output index is unavailable");
        }

        Microsoft::WRL::ComPtr<IDXGIOutput1> output1;
        throw_if_failed(output.As(&output1), "DXGI output does not support desktop duplication");

        const D3D_FEATURE_LEVEL requested_levels[] = {
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_10_0,
        };
        D3D_FEATURE_LEVEL selected_level{};
        throw_if_failed(
            D3D11CreateDevice(
                adapter.Get(),
                D3D_DRIVER_TYPE_UNKNOWN,
                nullptr,
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                requested_levels,
                static_cast<UINT>(std::size(requested_levels)),
                D3D11_SDK_VERSION,
                &device_,
                &selected_level,
                &context_
            ),
            "D3D11CreateDevice failed"
        );
        (void)selected_level;

        throw_if_failed(output1->DuplicateOutput(device_.Get(), &duplication_), "DXGI DuplicateOutput failed");
    }

    void ensure_staging_texture(const D3D11_TEXTURE2D_DESC& source_desc) {
        if (staging_texture_) {
            D3D11_TEXTURE2D_DESC current_desc{};
            staging_texture_->GetDesc(&current_desc);
            if (current_desc.Width == source_desc.Width && current_desc.Height == source_desc.Height &&
                current_desc.Format == source_desc.Format) {
                return;
            }
            staging_texture_.Reset();
        }

        D3D11_TEXTURE2D_DESC staging_desc = source_desc;
        staging_desc.BindFlags = 0;
        staging_desc.MiscFlags = 0;
        staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        staging_desc.Usage = D3D11_USAGE_STAGING;
        throw_if_failed(device_->CreateTexture2D(&staging_desc, nullptr, &staging_texture_), "CreateTexture2D staging failed");
    }

    void copy_texture_to_frame(ID3D11Texture2D* texture, CapturedFrame& frame) {
        D3D11_TEXTURE2D_DESC desc{};
        texture->GetDesc(&desc);
        ensure_staging_texture(desc);
        context_->CopyResource(staging_texture_.Get(), texture);

        D3D11_MAPPED_SUBRESOURCE mapped{};
        throw_if_failed(context_->Map(staging_texture_.Get(), 0, D3D11_MAP_READ, 0, &mapped), "Map DXGI staging texture failed");
        if (debug_ && !debug_printed_) {
            const auto* bytes = static_cast<const unsigned char*>(mapped.pData);
            const std::size_t sample_size = std::min<std::size_t>(
                static_cast<std::size_t>(mapped.RowPitch) * static_cast<std::size_t>(desc.Height),
                64U * 1024U
            );
            std::size_t nonzero = 0;
            unsigned char max_value = 0;
            for (std::size_t index = 0; index < sample_size; ++index) {
                if (bytes[index] != 0) {
                    ++nonzero;
                    max_value = std::max(max_value, bytes[index]);
                }
            }
            std::cout << "dxgi_debug"
                      << " width=" << desc.Width
                      << " height=" << desc.Height
                      << " format=" << desc.Format
                      << " row_pitch=" << mapped.RowPitch
                      << " sample_bytes=" << sample_size
                      << " sample_nonzero=" << nonzero
                      << " sample_max=" << static_cast<int>(max_value)
                      << '\n';
            debug_printed_ = true;
        }
        cv::Mat bgra(static_cast<int>(desc.Height), static_cast<int>(desc.Width), CV_8UC4, mapped.pData, mapped.RowPitch);
        cv::Mat bgr;
        cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
        context_->Unmap(staging_texture_.Get(), 0);

        if (roi_.width > 0 && roi_.height > 0) {
            const cv::Rect frame_rect(0, 0, bgr.cols, bgr.rows);
            const cv::Rect clipped = roi_ & frame_rect;
            if (clipped.width <= 0 || clipped.height <= 0) {
                throw std::runtime_error("DXGI ROI is outside the captured frame");
            }
            bgr = bgr(clipped).clone();
        }

        frame = CapturedFrame{
            std::move(bgr),
            frame_index_++,
            now_ms() - start_time_ms_,
        };
    }

    int adapter_index_ = 0;
    int output_index_ = 0;
    int timeout_ms_ = 16;
    cv::Rect roi_;
    bool debug_ = false;
    bool debug_printed_ = false;
    bool debug_frame_info_printed_ = false;
    int frame_index_ = 0;
    double start_time_ms_ = 0.0;
    Microsoft::WRL::ComPtr<ID3D11Device> device_;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> context_;
    Microsoft::WRL::ComPtr<IDXGIOutputDuplication> duplication_;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> staging_texture_;
};

#else

class DxgiFrameSource final : public FrameSource {
public:
    explicit DxgiFrameSource(const Options&) {
        throw std::runtime_error("DXGI input is only available on Windows");
    }

    bool read(CapturedFrame&) override {
        return false;
    }

    void release() override {}

    std::string name() const override {
        return "dxgi-unavailable";
    }

    bool is_live() const override {
        return true;
    }
};

#endif

}  // namespace

void print_dxgi_outputs(std::ostream& output) {
#if defined(_WIN32)
    Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
    throw_if_failed(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1 failed");

    for (UINT adapter_index = 0;; ++adapter_index) {
        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
        if (factory->EnumAdapters1(adapter_index, &adapter) == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        DXGI_ADAPTER_DESC1 adapter_desc{};
        adapter->GetDesc1(&adapter_desc);
        output << "adapter=" << adapter_index
               << " name=\"" << narrow_utf8(adapter_desc.Description) << "\""
               << " vendor=0x" << std::hex << std::uppercase << adapter_desc.VendorId
               << " device=0x" << adapter_desc.DeviceId
               << std::dec
               << " flags=" << adapter_desc.Flags
               << " dedicated_video_mb=" << adapter_desc.DedicatedVideoMemory / (1024 * 1024)
               << " shared_system_mb=" << adapter_desc.SharedSystemMemory / (1024 * 1024)
               << '\n';

        bool had_output = false;
        for (UINT output_index = 0;; ++output_index) {
            Microsoft::WRL::ComPtr<IDXGIOutput> dxgi_output;
            if (adapter->EnumOutputs(output_index, &dxgi_output) == DXGI_ERROR_NOT_FOUND) {
                break;
            }
            had_output = true;
            DXGI_OUTPUT_DESC output_desc{};
            dxgi_output->GetDesc(&output_desc);
            const RECT& rect = output_desc.DesktopCoordinates;
            output << "  output=" << output_index
                   << " output_name=\"" << narrow_utf8(output_desc.DeviceName) << "\""
                   << " desktop_left=" << rect.left
                   << " desktop_top=" << rect.top
                   << " desktop_right=" << rect.right
                   << " desktop_bottom=" << rect.bottom
                   << " attached=" << (output_desc.AttachedToDesktop ? 1 : 0)
                   << " rotation=" << output_rotation_name(output_desc.Rotation)
                   << '\n';
        }
        if (!had_output) {
            output << "  output=none\n";
        }
    }
#else
    output << "DXGI output enumeration is only available on Windows\n";
#endif
}

void apply_dxgi_gpu_preference(DxgiGpuPreference preference) {
#if defined(_WIN32)
    if (preference == DxgiGpuPreference::Default) {
        return;
    }
    DXGI_GPU_PREFERENCE dxgi_preference = DXGI_GPU_PREFERENCE_UNSPECIFIED;
    switch (preference) {
    case DxgiGpuPreference::Default:
        dxgi_preference = DXGI_GPU_PREFERENCE_UNSPECIFIED;
        break;
    case DxgiGpuPreference::MinimumPower:
        dxgi_preference = DXGI_GPU_PREFERENCE_MINIMUM_POWER;
        break;
    case DxgiGpuPreference::HighPerformance:
        dxgi_preference = DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE;
        break;
    }
    using SetProcessDefaultGpuPreferenceFn = HRESULT(WINAPI*)(DXGI_GPU_PREFERENCE);
    HMODULE dxgi_module = GetModuleHandleA("dxgi.dll");
    if (dxgi_module == nullptr) {
        dxgi_module = LoadLibraryA("dxgi.dll");
    }
    if (dxgi_module == nullptr) {
        throw std::runtime_error("failed to load dxgi.dll for GPU preference");
    }
    const auto function = reinterpret_cast<SetProcessDefaultGpuPreferenceFn>(
        GetProcAddress(dxgi_module, "SetProcessDefaultGpuPreference")
    );
    if (function == nullptr) {
        std::cout << "dxgi_gpu_preference=unavailable requested=" << dxgi_gpu_preference_name(preference) << '\n';
        return;
    }
    const HRESULT result = function(dxgi_preference);
    if (FAILED(result)) {
        std::cout << "dxgi_gpu_preference=failed requested=" << dxgi_gpu_preference_name(preference)
                  << " hresult=" << hresult_hex(result) << '\n';
        return;
    }
    std::cout << "dxgi_gpu_preference=applied requested=" << dxgi_gpu_preference_name(preference) << '\n';
#else
    (void)preference;
#endif
}

void probe_dxgi_outputs(std::ostream& output) {
#if defined(_WIN32)
    Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
    throw_if_failed(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1 failed");

    for (UINT adapter_index = 0;; ++adapter_index) {
        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
        if (factory->EnumAdapters1(adapter_index, &adapter) == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        DXGI_ADAPTER_DESC1 adapter_desc{};
        adapter->GetDesc1(&adapter_desc);
        output << "probe_adapter=" << adapter_index
               << " name=\"" << narrow_utf8(adapter_desc.Description) << "\""
               << " flags=" << adapter_desc.Flags
               << '\n';

        for (UINT output_index = 0;; ++output_index) {
            Microsoft::WRL::ComPtr<IDXGIOutput> base_output;
            if (adapter->EnumOutputs(output_index, &base_output) == DXGI_ERROR_NOT_FOUND) {
                break;
            }
            DXGI_OUTPUT_DESC output_desc{};
            base_output->GetDesc(&output_desc);

            Microsoft::WRL::ComPtr<IDXGIOutput1> output1;
            const HRESULT as_output1 = base_output.As(&output1);
            output << "  probe_output=" << output_index
                   << " name=\"" << narrow_utf8(output_desc.DeviceName) << "\""
                   << " attached=" << (output_desc.AttachedToDesktop ? 1 : 0)
                   << " as_output1=" << hresult_hex(as_output1);
            if (FAILED(as_output1)) {
                output << '\n';
                continue;
            }

            const D3D_FEATURE_LEVEL requested_levels[] = {
                D3D_FEATURE_LEVEL_11_1,
                D3D_FEATURE_LEVEL_11_0,
                D3D_FEATURE_LEVEL_10_1,
                D3D_FEATURE_LEVEL_10_0,
            };
            D3D_FEATURE_LEVEL selected_level{};
            Microsoft::WRL::ComPtr<ID3D11Device> device;
            Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;
            const HRESULT create_device = D3D11CreateDevice(
                adapter.Get(),
                D3D_DRIVER_TYPE_UNKNOWN,
                nullptr,
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                requested_levels,
                static_cast<UINT>(std::size(requested_levels)),
                D3D11_SDK_VERSION,
                &device,
                &selected_level,
                &context
            );
            output << " create_device=" << hresult_hex(create_device);
            if (FAILED(create_device)) {
                output << '\n';
                continue;
            }

            Microsoft::WRL::ComPtr<IDXGIOutputDuplication> duplication;
            const HRESULT duplicate = output1->DuplicateOutput(device.Get(), &duplication);
            output << " duplicate_output=" << hresult_hex(duplicate);

            Microsoft::WRL::ComPtr<IDXGIOutput5> output5;
            const HRESULT as_output5 = base_output.As(&output5);
            output << " as_output5=" << hresult_hex(as_output5);
            if (SUCCEEDED(as_output5)) {
                const DXGI_FORMAT formats[] = {
                    DXGI_FORMAT_B8G8R8A8_UNORM,
                    DXGI_FORMAT_R8G8B8A8_UNORM,
                };
                Microsoft::WRL::ComPtr<IDXGIOutputDuplication> duplication1;
                const HRESULT duplicate1 = output5->DuplicateOutput1(
                    device.Get(),
                    0,
                    static_cast<UINT>(std::size(formats)),
                    formats,
                    &duplication1
                );
                output << " duplicate_output1=" << hresult_hex(duplicate1);
            }
            output << '\n';
        }
    }
#else
    output << "DXGI output probing is only available on Windows\n";
#endif
}

std::unique_ptr<FrameSource> create_frame_source(const Options& options) {
    switch (options.input_source) {
    case InputSource::Video:
        return std::make_unique<VideoFrameSource>(options);
    case InputSource::Dxgi:
        return std::make_unique<DxgiFrameSource>(options);
    }
    throw std::runtime_error("unknown input source");
}

}  // namespace vision_analyzer
