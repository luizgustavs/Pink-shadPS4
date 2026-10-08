// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <mutex>

#include "common/logging/log.h"
#include "video_core/renderer_vulkan/host_passes/dlssnr_pass.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

#ifdef _WIN32
#include <array>
#include <cstring>
#include <filesystem>
#include <string_view>
#include <fmt/format.h>

#include "common/path_util.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/host_passes/dlssnr_bridge/dlssnr_bridge.h"

#include <windows.h>

#include <d3d12.h>
#include <dxgi1_4.h>
#include <vulkan/vulkan_win32.h>
#include <wrl/client.h>
#endif

namespace Vulkan::HostPasses {

namespace {

std::atomic<bool> g_enabled{false};
std::atomic<bool> g_running{false};
std::mutex g_status_lock;
// Why the pass cannot run (it stays off for the session), or why it is paused
std::string g_problem;
bool g_problem_permanent{false};

void SetProblem(std::string problem, bool permanent) {
    std::scoped_lock lock{g_status_lock};
    if (g_problem_permanent) {
        return;
    }
    g_problem = std::move(problem);
    g_problem_permanent = permanent;
}

void ClearTransientProblem() {
    std::scoped_lock lock{g_status_lock};
    if (!g_problem_permanent) {
        g_problem.clear();
    }
}

} // namespace

void DlssNrPass::SetEnabled(bool enabled) {
    g_enabled = enabled;
}

void DlssNrPass::Toggle() {
    const bool enabled = !g_enabled.load();
    g_enabled = enabled;
    LOG_INFO(Render_Vulkan, "DLSS Neural Rendering {}", enabled ? "on" : "off");
}

bool DlssNrPass::IsEnabled() {
    return g_enabled;
}

std::string DlssNrPass::StatusText() {
    if (!g_enabled) {
        return "DLSS 5: off";
    }
    std::scoped_lock lock{g_status_lock};
    if (!g_problem.empty()) {
        return (g_problem_permanent ? "DLSS 5 unavailable: " : "DLSS 5 paused: ") + g_problem;
    }
    return g_running ? "DLSS 5: on" : "DLSS 5: starting";
}

#ifdef _WIN32

using Microsoft::WRL::ComPtr;

namespace {

// Allocators in flight on the D3D12 queue; one per frame
constexpr u32 kNumAllocators = 3;
constexpr vk::Format kSharedFormat = vk::Format::eR16G16B16A16Sfloat;
constexpr DXGI_FORMAT kSharedDxgiFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;

constexpr vk::ImageSubresourceRange kColorRange = {
    .aspectMask = vk::ImageAspectFlagBits::eColor,
    .baseMipLevel = 0,
    .levelCount = 1,
    .baseArrayLayer = 0,
    .layerCount = 1,
};

D3D12_RESOURCE_BARRIER Transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                                  D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    return barrier;
}

std::filesystem::path ExecutableDir() {
    std::wstring path(MAX_PATH, L'\0');
    while (true) {
        const DWORD size =
            GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
        if (size < path.size()) {
            path.resize(size);
            break;
        }
        path.resize(path.size() * 2);
    }
    return std::filesystem::path{path}.parent_path();
}

} // namespace

struct DlssNrPass::Impl {
    struct SharedImage {
        ComPtr<ID3D12Resource> resource;
        vk::Image image;
        vk::DeviceMemory memory;
        // Not yet used by Vulkan: its first barrier comes from the undefined layout
        bool fresh{true};
    };

    const Instance* instance{};
    Scheduler* scheduler{};
    vk::Device device;

    bool init_done{};
    bool broken{};

    HMODULE bridge{};
    PFN_DlssNrBridgeInit bridge_init{};
    PFN_DlssNrBridgeCreate bridge_create{};
    PFN_DlssNrBridgeEvaluate bridge_evaluate{};
    PFN_DlssNrBridgeRelease bridge_release{};
    PFN_DlssNrBridgeShutdown bridge_shutdown{};
    PFN_DlssNrBridgeReadLog bridge_read_log{};
    bool bridge_ready{};

    PFN_vkGetMemoryWin32HandlePropertiesKHR get_memory_handle_properties{};
    PFN_vkImportSemaphoreWin32HandleKHR import_semaphore_handle{};

    ComPtr<ID3D12Device> d3d_device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence> fence;
    HANDLE fence_event{};
    std::array<ComPtr<ID3D12CommandAllocator>, kNumAllocators> allocators;
    std::array<u64, kNumAllocators> allocator_values{};
    u32 allocator_index{};
    ComPtr<ID3D12GraphicsCommandList> command_list;
    // The fence is shared: Vulkan signals the odd steps (frame copied in), D3D12 the others
    u64 fence_value{};
    vk::Semaphore semaphore;

    u32 width{};
    u32 height{};
    SharedImage input;
    SharedImage output;
    ComPtr<ID3D12Resource> depth;
    ComPtr<ID3D12Resource> motion;
    void* feature{};
    Settings feature_settings{};
    // Scheduler tick of the last frame that used the shared images
    u64 last_use_tick{};

    vk::Format checked_format{};
    bool format_blittable{};

    ~Impl() {
        Destroy();
    }

    void Fail(std::string reason) {
        LOG_ERROR(Render_Vulkan, "DLSS Neural Rendering disabled: {}", reason);
        DrainBridgeLog();
        SetProblem(std::move(reason), true);
        broken = true;
        g_running = false;
    }

    void DrainBridgeLog() {
        if (!bridge_read_log) {
            return;
        }
        std::array<char, 4096> buffer{};
        while (bridge_read_log(buffer.data(), static_cast<unsigned>(buffer.size())) > 0) {
            std::string_view text{buffer.data()};
            while (!text.empty()) {
                const size_t end = text.find('\n');
                const std::string_view line = text.substr(0, end);
                if (!line.empty()) {
                    LOG_INFO(Render_Vulkan, "{}", line);
                }
                if (end == std::string_view::npos) {
                    break;
                }
                text.remove_prefix(end + 1);
            }
        }
    }

    bool WaitD3D(u64 value) {
        if (fence->GetCompletedValue() >= value) {
            return true;
        }
        if (FAILED(fence->SetEventOnCompletion(value, fence_event))) {
            return false;
        }
        WaitForSingleObject(fence_event, INFINITE);
        return true;
    }

    /// Resets the next allocator once the D3D12 work that last used it has finished
    bool BeginCommands() {
        const u32 index = allocator_index;
        allocator_index = (allocator_index + 1) % kNumAllocators;
        if (!WaitD3D(allocator_values[index])) {
            return false;
        }
        return SUCCEEDED(allocators[index]->Reset()) &&
               SUCCEEDED(command_list->Reset(allocators[index].Get(), nullptr));
    }

    /// Submits the recorded commands after the fence reaches wait_value, then signals the next
    /// value
    bool SubmitCommands(u64 wait_value) {
        if (FAILED(command_list->Close())) {
            return false;
        }
        if (wait_value != 0 && FAILED(queue->Wait(fence.Get(), wait_value))) {
            return false;
        }
        ID3D12CommandList* lists[] = {command_list.Get()};
        queue->ExecuteCommandLists(1, lists);
        const u64 value = ++fence_value;
        allocator_values[(allocator_index + kNumAllocators - 1) % kNumAllocators] = value;
        return SUCCEEDED(queue->Signal(fence.Get(), value));
    }

    bool Initialize() {
        init_done = true;
        device = instance->GetDevice();

        if (!instance->IsD3D12InteropSupported()) {
            Fail("the Vulkan driver cannot share memory with D3D12");
            return false;
        }
        if (instance->GetVendorID() != 0x10DE) {
            Fail("needs an NVIDIA RTX GPU");
            return false;
        }

        const std::filesystem::path exe_dir = ExecutableDir();
        const std::filesystem::path model_path = exe_dir / DLSSNR_MODEL_DLL_NAME;
        const std::filesystem::path bridge_path = exe_dir / DLSSNR_BRIDGE_DLL_NAME;
        if (!std::filesystem::exists(model_path)) {
            Fail("nvngx_dlssnr.dll not found next to shadps4.exe");
            return false;
        }
        bridge = LoadLibraryExW(bridge_path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!bridge) {
            Fail(fmt::format("could not load {} (error {})", bridge_path.string(),
                             GetLastError()));
            return false;
        }
        bridge_init =
            reinterpret_cast<PFN_DlssNrBridgeInit>(GetProcAddress(bridge, "DlssNrBridgeInit"));
        bridge_create = reinterpret_cast<PFN_DlssNrBridgeCreate>(
            GetProcAddress(bridge, "DlssNrBridgeCreate"));
        bridge_evaluate = reinterpret_cast<PFN_DlssNrBridgeEvaluate>(
            GetProcAddress(bridge, "DlssNrBridgeEvaluate"));
        bridge_release = reinterpret_cast<PFN_DlssNrBridgeRelease>(
            GetProcAddress(bridge, "DlssNrBridgeRelease"));
        bridge_shutdown = reinterpret_cast<PFN_DlssNrBridgeShutdown>(
            GetProcAddress(bridge, "DlssNrBridgeShutdown"));
        bridge_read_log = reinterpret_cast<PFN_DlssNrBridgeReadLog>(
            GetProcAddress(bridge, "DlssNrBridgeReadLog"));
        if (!bridge_init || !bridge_create || !bridge_evaluate || !bridge_release ||
            !bridge_shutdown || !bridge_read_log) {
            Fail(fmt::format("{} is incomplete", bridge_path.string()));
            return false;
        }

        const auto get_device_proc = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;
        get_memory_handle_properties = reinterpret_cast<PFN_vkGetMemoryWin32HandlePropertiesKHR>(
            get_device_proc(device, "vkGetMemoryWin32HandlePropertiesKHR"));
        import_semaphore_handle = reinterpret_cast<PFN_vkImportSemaphoreWin32HandleKHR>(
            get_device_proc(device, "vkImportSemaphoreWin32HandleKHR"));
        if (!get_memory_handle_properties || !import_semaphore_handle) {
            Fail("the Vulkan driver lacks the win32 handle entry points");
            return false;
        }

        return CreateD3D12Device() && CreateSharedFence() && InitializeModel();
    }

    bool CreateD3D12Device() {
        const auto props_chain =
            instance->GetPhysicalDevice()
                .getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceIDProperties>();
        const auto& id_props = props_chain.get<vk::PhysicalDeviceIDProperties>();
        if (!id_props.deviceLUIDValid) {
            Fail("the Vulkan device reports no LUID");
            return false;
        }
        LUID luid{};
        std::memcpy(&luid, id_props.deviceLUID.data(), sizeof(luid));

        HMODULE dxgi = LoadLibraryW(L"dxgi.dll");
        HMODULE d3d12 = LoadLibraryW(L"d3d12.dll");
        const auto create_factory = reinterpret_cast<HRESULT(WINAPI*)(REFIID, void**)>(
            dxgi ? GetProcAddress(dxgi, "CreateDXGIFactory1") : nullptr);
        const auto create_device = reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(
            d3d12 ? GetProcAddress(d3d12, "D3D12CreateDevice") : nullptr);
        if (!create_factory || !create_device) {
            Fail("D3D12 is not available");
            return false;
        }

        ComPtr<IDXGIFactory4> factory;
        ComPtr<IDXGIAdapter1> adapter;
        if (FAILED(create_factory(IID_PPV_ARGS(&factory))) ||
            FAILED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter)))) {
            Fail("no DXGI adapter matches the Vulkan device");
            return false;
        }
        if (FAILED(create_device(adapter.Get(), D3D_FEATURE_LEVEL_12_0,
                                 IID_PPV_ARGS(&d3d_device)))) {
            Fail("could not create a D3D12 device");
            return false;
        }

        const D3D12_COMMAND_QUEUE_DESC queue_desc = {
            .Type = D3D12_COMMAND_LIST_TYPE_DIRECT,
        };
        if (FAILED(d3d_device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)))) {
            Fail("could not create a D3D12 queue");
            return false;
        }
        for (auto& allocator : allocators) {
            if (FAILED(d3d_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                          IID_PPV_ARGS(&allocator)))) {
                Fail("could not create a D3D12 command allocator");
                return false;
            }
        }
        if (FAILED(d3d_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                 allocators[0].Get(), nullptr,
                                                 IID_PPV_ARGS(&command_list))) ||
            FAILED(command_list->Close())) {
            Fail("could not create a D3D12 command list");
            return false;
        }
        return true;
    }

    bool CreateSharedFence() {
        if (FAILED(d3d_device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence)))) {
            Fail("could not create a shared D3D12 fence");
            return false;
        }
        fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        HANDLE handle{};
        if (!fence_event || FAILED(d3d_device->CreateSharedHandle(fence.Get(), nullptr,
                                                                  GENERIC_ALL, nullptr, &handle))) {
            Fail("could not share the D3D12 fence");
            return false;
        }

        const vk::SemaphoreTypeCreateInfo type_info = {
            .semaphoreType = vk::SemaphoreType::eTimeline,
            .initialValue = 0,
        };
        auto [result, created] = device.createSemaphore({.pNext = &type_info});
        if (result != vk::Result::eSuccess) {
            CloseHandle(handle);
            Fail("could not create the shared timeline semaphore");
            return false;
        }
        semaphore = created;
        SetObjectName(device, semaphore, "DLSS NR shared fence");

        const VkImportSemaphoreWin32HandleInfoKHR import_info = {
            .sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR,
            .semaphore = semaphore,
            .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT,
            .handle = handle,
        };
        const VkResult import_result = import_semaphore_handle(device, &import_info);
        // Importing does not take ownership of an NT handle
        CloseHandle(handle);
        if (import_result != VK_SUCCESS) {
            Fail(fmt::format("could not import the D3D12 fence ({})",
                             vk::to_string(vk::Result{import_result})));
            return false;
        }
        return true;
    }

    bool InitializeModel() {
        const std::filesystem::path model_path = ExecutableDir() / DLSSNR_MODEL_DLL_NAME;
        const std::filesystem::path data_path =
            Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "ngx";
        std::error_code ec;
        std::filesystem::create_directories(data_path, ec);

        const DlssNrBridgeInitInfo info = {
            .device = d3d_device.Get(),
            .model_path = model_path.c_str(),
            .data_path = data_path.c_str(),
            .arch_spoof = 1,
        };
        const int result = bridge_init(&info);
        DrainBridgeLog();
        if (result != DLSSNR_BRIDGE_SUCCESS) {
            Fail(fmt::format("the model did not initialize ({:#010x}), see the log",
                             static_cast<u32>(result)));
            return false;
        }
        bridge_ready = true;
        LOG_INFO(Render_Vulkan, "DLSS Neural Rendering ready ({})", model_path.string());
        return true;
    }

    bool CreateSharedImage(SharedImage& shared, const wchar_t* name) {
        const D3D12_HEAP_PROPERTIES heap = {.Type = D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_RESOURCE_DESC desc = {
            .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            .Width = width,
            .Height = height,
            .DepthOrArraySize = 1,
            .MipLevels = 1,
            .Format = kSharedDxgiFormat,
            .SampleDesc = {.Count = 1},
            .Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
            // Simultaneous access keeps the texture uncompressed, so both APIs read the same bytes
            .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS |
                     D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS,
        };
        if (FAILED(d3d_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc,
                                                       D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                       IID_PPV_ARGS(&shared.resource)))) {
            return false;
        }
        shared.resource->SetName(name);
        HANDLE handle{};
        if (FAILED(d3d_device->CreateSharedHandle(shared.resource.Get(), nullptr, GENERIC_ALL,
                                                  nullptr, &handle))) {
            return false;
        }

        const vk::ExternalMemoryImageCreateInfo external_info = {
            .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eD3D12Resource,
        };
        const vk::ImageCreateInfo image_info = {
            .pNext = &external_info,
            .imageType = vk::ImageType::e2D,
            .format = kSharedFormat,
            .extent = {width, height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined,
        };
        auto [image_result, image] = device.createImage(image_info);
        if (image_result != vk::Result::eSuccess) {
            CloseHandle(handle);
            return false;
        }
        shared.image = image;
        shared.fresh = true;

        const vk::MemoryRequirements requirements = device.getImageMemoryRequirements(image);
        VkMemoryWin32HandlePropertiesKHR handle_props = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR,
        };
        if (get_memory_handle_properties(device,
                                         VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
                                         handle, &handle_props) != VK_SUCCESS) {
            CloseHandle(handle);
            return false;
        }
        const u32 type_bits = requirements.memoryTypeBits & handle_props.memoryTypeBits;
        const auto& memory_props = instance->GetMemoryProperties();
        u32 type_index = VK_MAX_MEMORY_TYPES;
        for (u32 i = 0; i < memory_props.memoryTypeCount; ++i) {
            if (!(type_bits & (1u << i))) {
                continue;
            }
            if (type_index == VK_MAX_MEMORY_TYPES ||
                (memory_props.memoryTypes[i].propertyFlags &
                 vk::MemoryPropertyFlagBits::eDeviceLocal)) {
                type_index = i;
            }
            if (memory_props.memoryTypes[i].propertyFlags &
                vk::MemoryPropertyFlagBits::eDeviceLocal) {
                break;
            }
        }
        if (type_index == VK_MAX_MEMORY_TYPES) {
            CloseHandle(handle);
            return false;
        }

        const vk::MemoryDedicatedAllocateInfo dedicated_info = {.image = image};
        const VkImportMemoryWin32HandleInfoKHR import_info = {
            .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR,
            .pNext = &dedicated_info,
            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
            .handle = handle,
        };
        auto [memory_result, memory] = device.allocateMemory({
            .pNext = &import_info,
            .allocationSize = requirements.size,
            .memoryTypeIndex = type_index,
        });
        // Importing does not take ownership of an NT handle
        CloseHandle(handle);
        if (memory_result != vk::Result::eSuccess) {
            return false;
        }
        shared.memory = memory;
        return device.bindImageMemory(image, memory, 0) == vk::Result::eSuccess;
    }

    void DestroySharedImage(SharedImage& shared) {
        if (shared.image) {
            device.destroyImage(shared.image);
        }
        if (shared.memory) {
            device.freeMemory(shared.memory);
        }
        shared = {};
    }

    /// Depth and motion stay zero: no motion vectors, no depth
    bool CreateZeroInputs() {
        const D3D12_HEAP_PROPERTIES heap = {.Type = D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC desc = {
            .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            .Width = width,
            .Height = height,
            .DepthOrArraySize = 1,
            .MipLevels = 1,
            .Format = DXGI_FORMAT_R32_FLOAT,
            .SampleDesc = {.Count = 1},
            .Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
        };
        if (FAILED(d3d_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                       D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                       IID_PPV_ARGS(&depth)))) {
            return false;
        }
        desc.Format = DXGI_FORMAT_R16G16_FLOAT;
        if (FAILED(d3d_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                       D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                       IID_PPV_ARGS(&motion)))) {
            return false;
        }
        depth->SetName(L"DLSS NR depth");
        motion->SetName(L"DLSS NR motion");

        std::array<ID3D12Resource*, 2> targets = {depth.Get(), motion.Get()};
        std::array<D3D12_PLACED_SUBRESOURCE_FOOTPRINT, 2> footprints{};
        u64 total = 0;
        for (size_t i = 0; i < targets.size(); ++i) {
            const D3D12_RESOURCE_DESC target_desc = targets[i]->GetDesc();
            u64 size = 0;
            d3d_device->GetCopyableFootprints(&target_desc, 0, 1, total, &footprints[i], nullptr,
                                              nullptr, &size);
            total = (footprints[i].Offset + size + D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT - 1) &
                    ~u64(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT - 1);
        }

        const D3D12_HEAP_PROPERTIES upload_heap = {.Type = D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_RESOURCE_DESC buffer_desc = {
            .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
            .Width = total,
            .Height = 1,
            .DepthOrArraySize = 1,
            .MipLevels = 1,
            .Format = DXGI_FORMAT_UNKNOWN,
            .SampleDesc = {.Count = 1},
            .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        };
        ComPtr<ID3D12Resource> upload;
        if (FAILED(d3d_device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE,
                                                       &buffer_desc,
                                                       D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                       IID_PPV_ARGS(&upload)))) {
            return false;
        }
        void* mapped = nullptr;
        if (FAILED(upload->Map(0, nullptr, &mapped))) {
            return false;
        }
        std::memset(mapped, 0, total);
        upload->Unmap(0, nullptr);

        if (!BeginCommands()) {
            return false;
        }
        std::array<D3D12_RESOURCE_BARRIER, 2> barriers{};
        for (size_t i = 0; i < targets.size(); ++i) {
            D3D12_TEXTURE_COPY_LOCATION dst{};
            dst.pResource = targets[i];
            dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            dst.SubresourceIndex = 0;
            D3D12_TEXTURE_COPY_LOCATION src{};
            src.pResource = upload.Get();
            src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            src.PlacedFootprint = footprints[i];
            command_list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            barriers[i] = Transition(targets[i], D3D12_RESOURCE_STATE_COPY_DEST,
                                     D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        command_list->ResourceBarrier(static_cast<UINT>(barriers.size()), barriers.data());
        return SubmitCommands(0) && WaitD3D(fence_value);
    }

    bool CreateFeature(const Settings& settings) {
        if (feature) {
            bridge_release(feature);
            feature = nullptr;
        }
        const DlssNrModelParams params = {
            .preset = 0,
            .intensity = settings.intensity,
            .style = settings.style,
            .local_structure = 1.0f,
            .local_tone = 1.0f,
            .skin_structure = -1.0f,
            .auto_mask = 0,
            .ui_correction = settings.ui_correction ? 1u : 0u,
        };
        if (!BeginCommands()) {
            return false;
        }
        feature = bridge_create(command_list.Get(), width, height, &params);
        // The command list has to be closed and run even when creation failed
        const bool submitted = SubmitCommands(0) && WaitD3D(fence_value);
        DrainBridgeLog();
        if (!feature || !submitted) {
            return false;
        }
        feature_settings = settings;
        LOG_INFO(Render_Vulkan,
                 "DLSS Neural Rendering feature created at {}x{} (style {}, intensity {:.2f}, UI "
                 "correction {})",
                 width, height, settings.style, settings.intensity, settings.ui_correction);
        return true;
    }

    /// Waits until neither API uses the shared images
    void WaitIdle() {
        if (last_use_tick != 0) {
            scheduler->Wait(last_use_tick);
        }
        if (fence) {
            WaitD3D(fence_value);
        }
    }

    bool PrepareResources(u32 new_width, u32 new_height, const Settings& settings) {
        if (feature && new_width == width && new_height == height && settings == feature_settings) {
            return true;
        }
        WaitIdle();
        if (new_width != width || new_height != height || !input.image) {
            if (feature) {
                bridge_release(feature);
                feature = nullptr;
            }
            DestroySharedImage(input);
            DestroySharedImage(output);
            depth.Reset();
            motion.Reset();
            width = new_width;
            height = new_height;
            if (!CreateSharedImage(input, L"DLSS NR color") ||
                !CreateSharedImage(output, L"DLSS NR output")) {
                Fail(fmt::format("could not share a {}x{} image between Vulkan and D3D12", width,
                                 height));
                return false;
            }
            SetObjectName(device, input.image, "DLSS NR color");
            SetObjectName(device, output.image, "DLSS NR output");
            if (!CreateZeroInputs()) {
                Fail("could not create the depth and motion inputs");
                return false;
            }
        }
        if (!CreateFeature(settings)) {
            Fail(fmt::format("the model refused a {}x{} feature, see the log", width, height));
            return false;
        }
        return true;
    }

    bool IsBlittable(vk::Format format) {
        if (format != checked_format) {
            checked_format = format;
            const auto physical = instance->GetPhysicalDevice();
            const auto blit = vk::FormatFeatureFlagBits::eBlitSrc |
                              vk::FormatFeatureFlagBits::eBlitDst;
            format_blittable =
                (physical.getFormatProperties(format).optimalTilingFeatures & blit) == blit &&
                (physical.getFormatProperties(kSharedFormat).optimalTilingFeatures & blit) ==
                    blit;
        }
        return format_blittable;
    }

    /// Barrier that hands a shared image to D3D12 or takes it back. Shared images stay in the
    /// general layout, so only the first hand-off leaves the undefined one
    vk::ImageMemoryBarrier2 Ownership(const SharedImage& shared, bool acquire,
                                      vk::PipelineStageFlags2 stage, vk::AccessFlags2 access,
                                      vk::ImageLayout old_layout = vk::ImageLayout::eGeneral) const {
        const u32 family = instance->GetGraphicsQueueFamilyIndex();
        vk::ImageMemoryBarrier2 barrier = {
            .oldLayout = old_layout,
            .newLayout = vk::ImageLayout::eGeneral,
            .srcQueueFamilyIndex = acquire ? VK_QUEUE_FAMILY_EXTERNAL : family,
            .dstQueueFamilyIndex = acquire ? family : VK_QUEUE_FAMILY_EXTERNAL,
            .image = shared.image,
            .subresourceRange = kColorRange,
        };
        if (acquire) {
            barrier.dstStageMask = stage;
            barrier.dstAccessMask = access;
        } else {
            barrier.srcStageMask = stage;
            barrier.srcAccessMask = access;
        }
        return barrier;
    }

    vk::ImageMemoryBarrier2 FrameBarrier(vk::Image image, vk::ImageLayout old_layout,
                                         vk::ImageLayout new_layout,
                                         vk::PipelineStageFlags2 src_stage,
                                         vk::AccessFlags2 src_access,
                                         vk::PipelineStageFlags2 dst_stage,
                                         vk::AccessFlags2 dst_access) const {
        return {
            .srcStageMask = src_stage,
            .srcAccessMask = src_access,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_access,
            .oldLayout = old_layout,
            .newLayout = new_layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = kColorRange,
        };
    }

    vk::ImageBlit FullBlit() const {
        const vk::ImageSubresourceLayers layers = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };
        const std::array<vk::Offset3D, 2> offsets = {
            vk::Offset3D{0, 0, 0},
            vk::Offset3D{static_cast<s32>(width), static_cast<s32>(height), 1},
        };
        return {
            .srcSubresource = layers,
            .srcOffsets = offsets,
            .dstSubresource = layers,
            .dstOffsets = offsets,
        };
    }

    bool Render(vk::Image image, vk::Format format, u32 frame_width, u32 frame_height,
                const Settings& settings, SubmitInfo& info) {
        if (!init_done && !Initialize()) {
            return false;
        }
        if (broken) {
            return false;
        }
        if (!IsBlittable(format)) {
            SetProblem(fmt::format("{} frames cannot be copied", vk::to_string(format)), false);
            return false;
        }
        if (!PrepareResources(frame_width, frame_height, settings)) {
            return false;
        }

        // Copy the frame into the shared colour image and hand it to D3D12
        vk::CommandBuffer cmdbuf = scheduler->CommandBuffer();
        {
            std::array<vk::ImageMemoryBarrier2, 3> barriers;
            u32 count = 0;
            barriers[count++] = FrameBarrier(
                image, vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal,
                vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryWrite,
                vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
            if (input.fresh) {
                barriers[count++] = FrameBarrier(
                    input.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
                    vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eNone,
                    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
            } else {
                barriers[count++] =
                    Ownership(input, true, vk::PipelineStageFlagBits2::eTransfer,
                              vk::AccessFlagBits2::eTransferWrite);
            }
            if (output.fresh) {
                // D3D12 writes it before Vulkan first reads it, so it has to leave the undefined
                // layout now
                barriers[count++] =
                    Ownership(output, false, vk::PipelineStageFlagBits2::eAllCommands,
                              vk::AccessFlagBits2::eNone, vk::ImageLayout::eUndefined);
            }
            cmdbuf.pipelineBarrier2({
                .imageMemoryBarrierCount = count,
                .pImageMemoryBarriers = barriers.data(),
            });
        }
        cmdbuf.blitImage(image, vk::ImageLayout::eTransferSrcOptimal, input.image,
                         vk::ImageLayout::eGeneral, FullBlit(), vk::Filter::eNearest);
        {
            const std::array barriers = {
                Ownership(input, false, vk::PipelineStageFlagBits2::eTransfer,
                          vk::AccessFlagBits2::eTransferWrite),
                FrameBarrier(image, vk::ImageLayout::eTransferSrcOptimal,
                             vk::ImageLayout::eTransferDstOptimal,
                             vk::PipelineStageFlagBits2::eTransfer,
                             vk::AccessFlagBits2::eTransferRead,
                             vk::PipelineStageFlagBits2::eTransfer,
                             vk::AccessFlagBits2::eTransferWrite),
            };
            cmdbuf.pipelineBarrier2({
                .imageMemoryBarrierCount = static_cast<u32>(barriers.size()),
                .pImageMemoryBarriers = barriers.data(),
            });
        }
        input.fresh = false;
        output.fresh = false;

        const u64 copied_value = ++fence_value;
        SubmitInfo copy_info{};
        copy_info.AddSignal(semaphore, copied_value);
        scheduler->Flush(copy_info);

        // Run the model on D3D12 once the copy is done
        bool evaluated = BeginCommands();
        if (evaluated) {
            const std::array before = {
                Transition(input.resource.Get(), D3D12_RESOURCE_STATE_COMMON,
                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                Transition(output.resource.Get(), D3D12_RESOURCE_STATE_COMMON,
                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            };
            command_list->ResourceBarrier(static_cast<UINT>(before.size()), before.data());
            // Without motion vectors the history would smear moving images, so every frame starts
            // over
            const int result = bridge_evaluate(command_list.Get(), feature, input.resource.Get(),
                                               depth.Get(), motion.Get(), output.resource.Get(),
                                               width, height, 1);
            evaluated = result == DLSSNR_BRIDGE_SUCCESS;
            const std::array after = {
                Transition(input.resource.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                           D3D12_RESOURCE_STATE_COMMON),
                Transition(output.resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                           D3D12_RESOURCE_STATE_COMMON),
            };
            command_list->ResourceBarrier(static_cast<UINT>(after.size()), after.data());
            // Submit even after a failed evaluation so the D3D12 side stays in step
            evaluated = SubmitCommands(copied_value) && evaluated;
        }
        const u64 done_value = fence_value;

        cmdbuf = scheduler->CommandBuffer();
        if (!evaluated) {
            // Leave the frame as it was, in the layout the presenter expects
            const vk::ImageMemoryBarrier2 restore = FrameBarrier(
                image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eGeneral,
                vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                vk::PipelineStageFlagBits2::eAllCommands,
                vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
            cmdbuf.pipelineBarrier2({
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &restore,
            });
            last_use_tick = scheduler->CurrentTick();
            Fail("the model failed to run, see the log");
            return false;
        }

        // Take the result back into the frame
        {
            const vk::ImageMemoryBarrier2 acquire = Ownership(
                output, true, vk::PipelineStageFlagBits2::eTransfer,
                vk::AccessFlagBits2::eTransferRead);
            cmdbuf.pipelineBarrier2({
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &acquire,
            });
        }
        cmdbuf.blitImage(output.image, vk::ImageLayout::eGeneral, image,
                         vk::ImageLayout::eTransferDstOptimal, FullBlit(), vk::Filter::eNearest);
        {
            const std::array barriers = {
                Ownership(output, false, vk::PipelineStageFlagBits2::eTransfer,
                          vk::AccessFlagBits2::eTransferRead),
                FrameBarrier(image, vk::ImageLayout::eTransferDstOptimal,
                             vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eTransfer,
                             vk::AccessFlagBits2::eTransferWrite,
                             vk::PipelineStageFlagBits2::eAllCommands,
                             vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite),
            };
            cmdbuf.pipelineBarrier2({
                .imageMemoryBarrierCount = static_cast<u32>(barriers.size()),
                .pImageMemoryBarriers = barriers.data(),
            });
        }
        info.AddWait(semaphore, done_value);
        last_use_tick = scheduler->CurrentTick();
        return true;
    }

    void Destroy() {
        if (!init_done) {
            return;
        }
        if (fence) {
            WaitD3D(fence_value);
        }
        if (feature && bridge_release) {
            bridge_release(feature);
            feature = nullptr;
        }
        if (bridge_ready && bridge_shutdown) {
            bridge_shutdown();
            bridge_ready = false;
        }
        DrainBridgeLog();
        if (device) {
            DestroySharedImage(input);
            DestroySharedImage(output);
            if (semaphore) {
                device.destroySemaphore(semaphore);
                semaphore = nullptr;
            }
        }
        depth.Reset();
        motion.Reset();
        command_list.Reset();
        for (auto& allocator : allocators) {
            allocator.Reset();
        }
        fence.Reset();
        queue.Reset();
        d3d_device.Reset();
        if (fence_event) {
            CloseHandle(fence_event);
            fence_event = nullptr;
        }
        init_done = false;
    }
};

DlssNrPass::DlssNrPass() : impl{std::make_unique<Impl>()} {}

DlssNrPass::~DlssNrPass() = default;

void DlssNrPass::Create(const Instance& instance, Scheduler& scheduler) {
    impl->instance = &instance;
    impl->scheduler = &scheduler;
}

bool DlssNrPass::Render(vk::Image image, vk::Format format, u32 width, u32 height, bool hdr,
                        const Settings& settings, SubmitInfo& info) {
    if (!g_enabled) {
        g_running = false;
        return false;
    }
    if (hdr) {
        // The model was trained on SDR frames
        SetProblem("HDR output is not supported", false);
        g_running = false;
        return false;
    }
    ClearTransientProblem();
    const bool ran = impl->Render(image, format, width, height, settings, info);
    g_running = ran;
    return ran;
}

void DlssNrPass::Destroy() {
    impl->Destroy();
}

#else

struct DlssNrPass::Impl {};

DlssNrPass::DlssNrPass() = default;

DlssNrPass::~DlssNrPass() = default;

void DlssNrPass::Create(const Instance&, Scheduler&) {
    SetProblem("needs Windows and an NVIDIA RTX GPU", true);
}

bool DlssNrPass::Render(vk::Image, vk::Format, u32, u32, bool, const Settings&, SubmitInfo&) {
    return false;
}

void DlssNrPass::Destroy() {}

#endif

} // namespace Vulkan::HostPasses
