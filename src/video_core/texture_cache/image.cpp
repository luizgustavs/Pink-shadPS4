// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <mutex>
#include <ranges>
#include "common/assert.h"
#include "common/logging/events.h"
#include "common/perf_stats.h"
#include "core/emulator_settings.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/blit_helper.h"
#include "video_core/texture_cache/image.h"

#include <vk_mem_alloc.h>

namespace VideoCore {

using namespace Vulkan;

Common::IncrementalIdProvider<u64> Image::global_image_uid{};

static vk::ImageUsageFlags ImageUsageFlags(const Vulkan::Instance& instance,
                                           const ImageInfo& info) {
    vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eTransferSrc |
                                vk::ImageUsageFlagBits::eTransferDst |
                                vk::ImageUsageFlagBits::eSampled;
    if (!info.props.is_block) {
        if (info.props.is_depth) {
            usage |= vk::ImageUsageFlagBits::eDepthStencilAttachment;
        } else {
            usage |= vk::ImageUsageFlagBits::eColorAttachment;
            if (instance.IsAttachmentFeedbackLoopLayoutSupported()) {
                usage |= vk::ImageUsageFlagBits::eAttachmentFeedbackLoopEXT;
            }
            // Always create images with storage flag to avoid needing re-creation in case of e.g
            // compute clears This sacrifices a bit of performance but is less work. ExtendedUsage
            // flag is also used. The exception here is for multisample images when storage is not
            // supported, where even with ExtendedUsage we may get only one supported sample back.
            if (info.num_samples == 1 || instance.IsMultisampleStorageImageSupported())
                usage |= vk::ImageUsageFlagBits::eStorage;
        }
    } else {
        // Similarly to above, we specify storage usage. This is typically not supported by
        // compressed formats, but may be used for uncompressed views. In order to satisfy this,
        // we will also specify the extended usage bit.
        usage |= vk::ImageUsageFlagBits::eStorage;
    }

    return usage;
}

static vk::ImageType ConvertImageType(AmdGpu::ImageType type) noexcept {
    switch (type) {
    case AmdGpu::ImageType::Color1D:
    case AmdGpu::ImageType::Color1DArray:
        return vk::ImageType::e1D;
    case AmdGpu::ImageType::Color2D:
    case AmdGpu::ImageType::Color2DMsaa:
    case AmdGpu::ImageType::Color2DArray:
        return vk::ImageType::e2D;
    case AmdGpu::ImageType::Color3D:
        return vk::ImageType::e3D;
    default:
        UNREACHABLE();
    }
}

static vk::FormatFeatureFlags2 FormatFeatureFlags(const vk::ImageUsageFlags usage_flags) {
    vk::FormatFeatureFlags2 feature_flags{};
    if (usage_flags & vk::ImageUsageFlagBits::eTransferSrc) {
        feature_flags |= vk::FormatFeatureFlagBits2::eTransferSrc;
    }
    if (usage_flags & vk::ImageUsageFlagBits::eTransferDst) {
        feature_flags |= vk::FormatFeatureFlagBits2::eTransferDst;
    }
    if (usage_flags & vk::ImageUsageFlagBits::eSampled) {
        feature_flags |= vk::FormatFeatureFlagBits2::eSampledImage;
    }
    if (usage_flags & vk::ImageUsageFlagBits::eColorAttachment) {
        feature_flags |= vk::FormatFeatureFlagBits2::eColorAttachment;
    }
    if (usage_flags & vk::ImageUsageFlagBits::eDepthStencilAttachment) {
        feature_flags |= vk::FormatFeatureFlagBits2::eDepthStencilAttachment;
    }
    // Note: StorageImage is intentionally ignored for now since it is always set, and can mess up
    // compatibility checks.
    return feature_flags;
}

UniqueImage::~UniqueImage() {
    if (image) {
        vmaDestroyImage(allocator, image, allocation);
    }
}

void UniqueImage::Destroy() {
    if (image) {
        vmaDestroyImage(allocator, image, allocation);
        image = vk::Image{};
        allocation = {};
    }
}

namespace {
// Keep one VMA pool per memory type for the small images handled by image_memory_pool
// Custom pool allocations stay inside those blocks instead of using dedicated memory
// Creating or freeing these images can then avoid repeated vkAllocateMemory and vkFreeMemory calls
// The pool supplies the backing memory while each image still follows its normal lifetime
constexpr VkDeviceSize PooledImageMaxBytes = 16ULL << 20;
constexpr VkDeviceSize ImagePoolBlockBytes = 64ULL << 20;
std::mutex image_pools_mutex;
std::array<VmaPool, VK_MAX_MEMORY_TYPES> image_pools{};

VmaPool ImagePool(VmaAllocator allocator, u32 memory_type_bits) {
    // VMA AUTO usages need image information to choose memory, so use the device-local choice they
    // would make for these pooled images
    const VmaAllocationCreateInfo type_ci = {
        .usage = VMA_MEMORY_USAGE_UNKNOWN,
        .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    };
    u32 type_index{};
    if (vmaFindMemoryTypeIndex(allocator, memory_type_bits, &type_ci, &type_index) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    std::scoped_lock lk{image_pools_mutex};
    auto& pool = image_pools[type_index];
    if (!pool) {
        const VmaPoolCreateInfo pool_ci = {
            .memoryTypeIndex = type_index,
            .blockSize = ImagePoolBlockBytes,
        };
        if (vmaCreatePool(allocator, &pool_ci, &pool) != VK_SUCCESS) {
            pool = VK_NULL_HANDLE;
        }
    }
    return pool;
}
} // Anonymous namespace

bool UniqueImage::TryCreatePooled(const vk::ImageCreateInfo& image_ci) {
    static const bool enabled = [] {
        const bool on = EmulatorSettings.IsImageMemoryPool();
        if (on) {
            LOG_WARNING(Render_Vulkan, "Workaround image_memory_pool enabled");
        }
        return on;
    }();
    static const bool follows_toggle = Common::PerfStats::AbToggleFollows("image_memory_pool");
    if (!enabled || !Common::PerfStats::AbToggleOn(follows_toggle)) {
        return false;
    }
    const auto give_up = [](const char* reason) {
        static std::atomic<u32> reported{};
        if (reported.fetch_add(1) < 8) {
            LOG_WARNING(Render_Vulkan, "image_memory_pool: not pooled ({})", reason);
        }
        return false;
    };
    // A moved-from UniqueImage no longer owns a device, so leave it without a device to use during
    // cleanup
    if (!device) {
        return give_up("no device");
    }
    auto [create_result, new_image] = device.createImage(image_ci);
    if (create_result != vk::Result::eSuccess) {
        return give_up("createImage");
    }
    const auto requirements = device.getImageMemoryRequirements(new_image);
    const VmaPool pool = requirements.size <= PooledImageMaxBytes
                             ? ImagePool(allocator, requirements.memoryTypeBits)
                             : VK_NULL_HANDLE;
    if (!pool) {
        device.destroyImage(new_image);
        return requirements.size <= PooledImageMaxBytes ? give_up("no pool") : false;
    }
    const VmaAllocationCreateInfo alloc_ci = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
        .pool = pool,
    };
    VmaAllocationInfo alloc_info{};
    if (vmaAllocateMemoryForImage(allocator, static_cast<VkImage>(new_image), &alloc_ci,
                                  &allocation, &alloc_info) != VK_SUCCESS) {
        device.destroyImage(new_image);
        allocation = {};
        return give_up("pool allocation");
    }
    if (vmaBindImageMemory(allocator, allocation, static_cast<VkImage>(new_image)) != VK_SUCCESS) {
        vmaFreeMemory(allocator, allocation);
        device.destroyImage(new_image);
        allocation = {};
        return false;
    }
    Common::PerfStats::Add(Common::PerfStats::Id::PooledImages);
    image = new_image;
    size_bytes = alloc_info.size;
    return true;
}

void UniqueImage::Create(const vk::ImageCreateInfo& image_ci) {
    this->image_ci = image_ci;
    ASSERT(!image);
    if (TryCreatePooled(image_ci)) {
        return;
    }
    const VmaAllocationCreateInfo alloc_ci = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = 0,
        .preferredFlags = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    const VkImageCreateInfo image_ci_unsafe = static_cast<VkImageCreateInfo>(image_ci);
    VkImage unsafe_image{};
    VmaAllocationInfo alloc_info{};
    VkResult result = vmaCreateImage(allocator, &image_ci_unsafe, &alloc_ci, &unsafe_image,
                                     &allocation, &alloc_info);
    // Print the image description so an implausible (garbage T#) request is diagnosable
    ASSERT_MSG(result == VK_SUCCESS,
               "Failed allocating image with error {}: {}x{}x{} layers={} levels={} samples={} "
               "format={}",
               vk::to_string(vk::Result{result}), image_ci.extent.width, image_ci.extent.height,
               image_ci.extent.depth, image_ci.arrayLayers, image_ci.mipLevels,
               vk::to_string(image_ci.samples), vk::to_string(image_ci.format));
    image = vk::Image{unsafe_image};
    size_bytes = alloc_info.size;
}

Image::Image(const Vulkan::Instance& instance, Vulkan::Runtime& runtime_,
             Common::SlotVector<ImageView>& slot_image_views_, const ImageInfo& info_)
    : runtime{&runtime_}, slot_image_views{&slot_image_views_}, info{info_} {
    if (info.pixel_format == vk::Format::eUndefined) {
        return;
    }

    image_uid = global_image_uid.Next();
    mip_hashes.resize(info.resources.levels);
    vk::ImageCreateFlags flags{vk::ImageCreateFlagBits::eMutableFormat |
                               vk::ImageCreateFlagBits::eExtendedUsage};
    if (info.props.is_volume) {
        flags |= vk::ImageCreateFlagBits::e2DArrayCompatible;
        if (instance.Is2dViewOf3dSupported()) {
            flags |= vk::ImageCreateFlagBits::e2DViewCompatibleEXT;
        }
    }
    if (info.props.is_block && instance.IsBlockTexelViewSupported()) {
        flags |= vk::ImageCreateFlagBits::eBlockTexelViewCompatible;
    }

    usage_flags = ImageUsageFlags(instance, info);
    format_features = FormatFeatureFlags(usage_flags);
    if (info.props.is_depth) {
        aspect_mask = vk::ImageAspectFlagBits::eDepth;
        if (info.props.has_stencil) {
            aspect_mask |= vk::ImageAspectFlagBits::eStencil;
        }
    }

    constexpr auto tiling = vk::ImageTiling::eOptimal;
    const auto supported_format = instance.GetSupportedFormat(info.pixel_format, format_features);
    const vk::PhysicalDeviceImageFormatInfo2 format_info{
        .format = supported_format,
        .type = ConvertImageType(info.type),
        .tiling = tiling,
        .usage = usage_flags,
        .flags = flags,
    };
    const auto image_format_properties =
        instance.GetPhysicalDevice().getImageFormatProperties2(format_info);
    if (image_format_properties.result == vk::Result::eErrorFormatNotSupported) {
        LOG_RENDER_PROBLEM(Render_Vulkan, Error,
                           "image format {} type {} is not supported (flags {}, usage {})",
                           vk::to_string(supported_format), vk::to_string(format_info.type),
                           vk::to_string(format_info.flags), vk::to_string(format_info.usage));
    }
    supported_samples = image_format_properties.result == vk::Result::eSuccess
                            ? image_format_properties.value.imageFormatProperties.sampleCounts
                            : vk::SampleCountFlagBits::e1;

    const vk::ImageCreateInfo image_ci = {
        .flags = flags,
        .imageType = ConvertImageType(info.type),
        .format = supported_format,
        .extent{
            .width = info.size.width,
            .height = info.size.height,
            .depth = info.size.depth,
        },
        .mipLevels = static_cast<u32>(info.resources.levels),
        .arrayLayers = static_cast<u32>(info.resources.layers),
        .samples = LiverpoolToVK::NumSamples(info.num_samples, supported_samples),
        .tiling = tiling,
        .usage = usage_flags,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    backing = &backing_images.emplace_back();
    backing->num_samples = info.num_samples;
    backing->image = UniqueImage{instance.GetDevice(), instance.GetAllocator()};
    backing->image.Create(image_ci);

    Vulkan::SetObjectName(instance.GetDevice(), GetImage(),
                          "Image {}x{}x{} {} {} {:#x}:{:#x} L:{} M:{} S:{}", info.size.width,
                          info.size.height, info.size.depth, AmdGpu::NameOf(info.tile_mode),
                          vk::to_string(info.pixel_format), info.guest_address, info.guest_size,
                          info.resources.layers, info.resources.levels, info.num_samples);
}

Image::~Image() = default;

ImageView& Image::FindView(const ImageViewInfo& view_info, bool ensure_guest_samples) {
    if (ensure_guest_samples && backing->num_samples > 1 != info.num_samples > 1) {
        runtime->SetBackingSamples(this, info.num_samples);
    }
    const auto& view_infos = backing->image_view_infos;
    const auto it = std::ranges::find(view_infos, view_info);
    if (it != view_infos.end()) {
        const auto view_id = backing->image_view_ids[std::distance(view_infos.begin(), it)];
        return (*slot_image_views)[view_id];
    }
    const auto view_id = slot_image_views->insert(runtime->GetInstance(), view_info, *this);
    backing->image_view_infos.emplace_back(view_info);
    backing->image_view_ids.emplace_back(view_id);
    return (*slot_image_views)[view_id];
}

void Image::GetBarriers(Barriers& barriers, vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
                        vk::PipelineStageFlags2 dst_stage,
                        std::optional<SubresourceRange> subres_range) {
    GetImageBarriers(GetImage(), aspect_mask, info.resources, backing->state,
                     backing->subresource_states, barriers, dst_layout, dst_mask, dst_stage,
                     subres_range);
}

} // namespace VideoCore
