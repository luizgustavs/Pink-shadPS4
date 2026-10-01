// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_semaphore.h"

#include "common/assert.h"
#include "common/perf_stats.h"
#include "core/emulator_settings.h"

#include <vk_mem_alloc.h>

namespace Vulkan {

constexpr u64 WAIT_TIMEOUT = std::numeric_limits<u64>::max();

/// Separate marker slots by a cache line so polling one dword does not share a line with another
/// marker write
constexpr u32 MARKER_STRIDE = 64;
constexpr u32 MARKER_SLOTS = 2;

bool SpinUntil(u32 spin_us, const std::function<bool()>& done) {
    using namespace Common::PerfStats;
    static const bool follows_toggle = AbToggleFollows("wait_spin_us");
    if (spin_us == 0 || !AbToggleOn(follows_toggle)) {
        return false;
    }
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    const auto deadline = start + std::chrono::microseconds(spin_us);
    u64 polls = 0;
    bool hit = false;
    auto now = start;
    while (now < deadline) {
        ++polls;
        if (done()) {
            hit = true;
            break;
        }
#if defined(_M_X64) || defined(__x86_64__)
        _mm_pause();
#endif
        now = Clock::now();
    }
    if (Enabled()) {
        Add(Id::WaitSpins);
        Add(Id::WaitSpinPolls, polls);
        Add(Id::WaitSpinNs,
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
        if (hit) {
            Add(Id::WaitSpinHits);
        }
    }
    return hit;
}

Semaphore::Semaphore(const Instance& instance_)
    : instance{instance_}, spin_us{EmulatorSettings.GetWaitSpinUs()},
      marker_mode{EmulatorSettings.GetWaitMarker()} {
    const vk::StructureChain semaphore_chain = {
        vk::SemaphoreCreateInfo{},
        vk::SemaphoreTypeCreateInfo{
            .semaphoreType = vk::SemaphoreType::eTimeline,
            .initialValue = 0,
        },
    };
    auto [semaphore_result, sem] =
        instance.GetDevice().createSemaphoreUnique(semaphore_chain.get());
    ASSERT_MSG(semaphore_result == vk::Result::eSuccess, "Failed to create master semaphore: {}",
               vk::to_string(semaphore_result));
    semaphore = std::move(sem);

    if (marker_mode == 0) {
        return;
    }
    if (spin_us == 0) {
        LOG_WARNING(Render_Vulkan, "wait_marker without wait_spin_us: there is no polling to "
                                   "replace, so the key has no effect");
        return;
    }
    // Allow marker writes on the graphics queue and on the transfer queue used for ahead copies so
    // both submission paths can report completion
    const std::array<u32, 2> families = {instance.GetGraphicsQueueFamilyIndex(),
                                         instance.GetTransferQueueFamilyIndex()};
    const bool shared =
        static_cast<bool>(instance.GetTransferQueue()) && families[0] != families[1];
    const VkBufferCreateInfo buffer_ci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = MARKER_STRIDE * MARKER_SLOTS,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = shared ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = shared ? static_cast<u32>(families.size()) : 0U,
        .pQueueFamilyIndices = shared ? families.data() : nullptr,
    };
    // Use cached host memory so polling usually reads the CPU cache until the GPU write makes the
    // new marker value visible
    const VmaAllocationCreateInfo alloc_ci = {
        .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        .preferredFlags = VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
    };
    VkBuffer buffer{};
    VmaAllocationInfo alloc_info{};
    const VkResult result = vmaCreateBuffer(instance.GetAllocator(), &buffer_ci, &alloc_ci,
                                            &buffer, &marker_allocation, &alloc_info);
    if (result != VK_SUCCESS || !alloc_info.pMappedData) {
        LOG_WARNING(Render_Vulkan, "wait_marker: no host marker buffer ({}), polling the driver",
                    vk::to_string(vk::Result{result}));
        if (result == VK_SUCCESS) {
            vmaDestroyBuffer(instance.GetAllocator(), buffer, marker_allocation);
        }
        marker_allocation = {};
        return;
    }
    marker_buffer = vk::Buffer{buffer};
    std::memset(alloc_info.pMappedData, 0, MARKER_STRIDE * MARKER_SLOTS);
    marker_data = static_cast<const volatile u32*>(alloc_info.pMappedData);
    VkMemoryPropertyFlags props{};
    vmaGetAllocationMemoryProperties(instance.GetAllocator(), marker_allocation, &props);
    LOG_WARNING(Render_Vulkan, "Workaround wait_marker {} enabled: host marker buffer in {} memory",
                marker_mode, vk::to_string(vk::MemoryPropertyFlags{props}));
}

Semaphore::~Semaphore() {
    if (marker_buffer) {
        vmaDestroyBuffer(instance.GetAllocator(), static_cast<VkBuffer>(marker_buffer),
                         marker_allocation);
    }
}

bool Semaphore::UseMarker() const {
    static const bool follows_toggle = Common::PerfStats::AbToggleFollows("wait_marker");
    return marker_buffer && Common::PerfStats::AbToggleOn(follows_toggle);
}

void Semaphore::RecordMarker(vk::CommandBuffer cmdbuf, MarkerSlot slot, u32 value) const {
    // Write the marker after all earlier commands finish and make their writes visible to the host
    // Finish readback copies rely on this because they have no separate host barrier
    // Only later clear-stage operations wait here, and marker writes remain ordered
    // That includes fills, buffer updates and image clears without making every later command wait
    // at this barrier
    const vk::MemoryBarrier2 pre_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eClear | vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eHostRead,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &pre_barrier,
    });
    cmdbuf.fillBuffer(marker_buffer, static_cast<u32>(slot) * MARKER_STRIDE, sizeof(u32), value);
    const vk::MemoryBarrier2 post_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eClear,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &post_barrier,
    });
}

bool Semaphore::MarkerReached(MarkerSlot slot, u64 value) const noexcept {
    const u32 current = marker_data[static_cast<u32>(slot) * MARKER_STRIDE / sizeof(u32)];
    std::atomic_thread_fence(std::memory_order_acquire);
    return static_cast<s32>(current - static_cast<u32>(value)) >= 0;
}

void Semaphore::Refresh() {
    u64 this_tick{};
    u64 counter{};
    do {
        this_tick = gpu_tick.load(std::memory_order_acquire);
        auto [counter_result, cntr] = instance.GetDevice().getSemaphoreCounterValue(*semaphore);
        if (counter_result == vk::Result::eErrorDeviceLost) {
            instance.ReportDeviceFault("Semaphore::Refresh");
        }
        ASSERT_MSG(counter_result == vk::Result::eSuccess,
                   "Failed to get master semaphore value: {}", vk::to_string(counter_result));
        counter = cntr;
        if (counter < this_tick) {
            return;
        }
    } while (!gpu_tick.compare_exchange_weak(this_tick, counter, std::memory_order_release,
                                             std::memory_order_relaxed));
}

void Semaphore::WaitHostRead(u64 tick) {
    if (!TrustMarker() || IsFree(tick) || tick >= CurrentTick()) {
        Wait(tick);
        return;
    }
    if (MarkerReached(MarkerSlot::Timeline, tick) ||
        SpinUntil(spin_us, [&] { return MarkerReached(MarkerSlot::Timeline, tick); })) {
        if (Common::PerfStats::Enabled()) {
            Common::PerfStats::Add(Common::PerfStats::Id::WaitMarkerEarly);
        }
        return;
    }
    Wait(tick);
}

void Semaphore::Wait(u64 tick) {
    // No need to wait if the GPU is ahead of the tick
    if (IsFree(tick)) {
        return;
    }
    // Update the GPU tick and try again
    Refresh();
    if (IsFree(tick)) {
        return;
    }
    // Do not poll a tick that has not been submitted yet because pending operations may still
    // depend on the recording tick
    // With wait_marker, ask the driver only after the GPU writes the marker
    // The semaphore signal follows that write and remains the source of the known GPU tick
    // Seeing a marker alone must not advance that tick or let resources from the submit be reused
    const bool marker = UseMarker();
    const bool perf = Common::PerfStats::Enabled();
    bool late = false;
    const bool spun = tick < CurrentTick() && SpinUntil(spin_us, [&] {
                          if (marker && !MarkerReached(MarkerSlot::Timeline, tick)) {
                              return false;
                          }
                          if (perf) {
                              Common::PerfStats::Add(Common::PerfStats::Id::WaitKernelPolls);
                          }
                          Refresh();
                          if (IsFree(tick)) {
                              return true;
                          }
                          late = marker;
                          return false;
                      });
    if (late && perf) {
        Common::PerfStats::Add(Common::PerfStats::Id::WaitMarkerLate);
    }
    if (spun) {
        return;
    }

    // If none of the above is hit, fallback to a regular wait
    const vk::SemaphoreWaitInfo wait_info = {
        .semaphoreCount = 1,
        .pSemaphores = &semaphore.get(),
        .pValues = &tick,
    };

    vk::Result result;
    while ((result = instance.GetDevice().waitSemaphores(&wait_info, WAIT_TIMEOUT)) !=
           vk::Result::eSuccess) {
        // A lost device never signals; spinning here would hang without any report
        if (result == vk::Result::eErrorDeviceLost) {
            instance.ReportDeviceFault("Semaphore::Wait");
            UNREACHABLE_MSG("Device lost while waiting for tick {}", tick);
        }
    }
    Refresh();
}

} // namespace Vulkan
