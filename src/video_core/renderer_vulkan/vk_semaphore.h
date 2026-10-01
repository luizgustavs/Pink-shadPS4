// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <thread>
#include <queue>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

VK_DEFINE_HANDLE(VmaAllocation)

namespace Vulkan {

class Instance;
class Scheduler;

/// Polls briefly before a blocking GPU wait
bool SpinUntil(u32 spin_us, const std::function<bool()>& done);

class Semaphore {
public:
    explicit Semaphore(const Instance& instance_);
    ~Semaphore();

    [[nodiscard]] u64 CurrentTick() const noexcept {
        return current_tick.load(std::memory_order_acquire);
    }

    [[nodiscard]] u64 KnownGpuTick() const noexcept {
        return gpu_tick.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool IsFree(u64 tick) const noexcept {
        return KnownGpuTick() >= tick;
    }

    [[nodiscard]] u64 NextTick() noexcept {
        return current_tick.fetch_add(1, std::memory_order_release);
    }

    [[nodiscard]] vk::Semaphore Handle() const noexcept {
        return semaphore.get();
    }

    /// Refresh the known GPU tick
    void Refresh();

    /// Waits for a tick to be hit on the GPU
    void Wait(u64 tick);

    /// Wait until the host can read GPU writes through tick, returning on the marker in wait_marker
    /// mode 2
    /// Keep the known GPU tick unchanged until the driver confirms completion so this early read
    /// cannot cause resources used by that submit to be reused while their fence or timeline signal
    /// is still pending
    void WaitHostRead(u64 tick);

    /// Use one host-visible dword per marker slot, with each slot on its own cache line for
    /// independent polling
    enum class MarkerSlot : u32 {
        Timeline = 0, ///< Low 32 bits of the timeline tick written at the end of each work submission
        Ahead = 1,    ///< Sequence number written at the end of each ahead copy submission when readback_ahead is used
    };

    /// Record marker writes only when the setting is enabled and waits are configured to spin
    /// before blocking
    [[nodiscard]] bool HasMarker() const noexcept {
        return static_cast<bool>(marker_buffer);
    }

    /// Check whether this poll should read the GPU marker before asking the driver, including the
    /// current SHADPS4_AB_TOGGLE interval
    [[nodiscard]] bool UseMarker() const;

    /// Allow host reads to return on the GPU marker only in wait_marker mode 2 when marker polling
    /// is active
    [[nodiscard]] bool TrustMarker() const {
        return marker_mode >= 2 && UseMarker();
    }

    /// Write value into slot after every earlier command in submission order and make that write
    /// visible to the host
    /// Call this outside a render pass on a queue family that supports transfers
    void RecordMarker(vk::CommandBuffer cmdbuf, MarkerSlot slot, u32 value) const;

    /// Check whether the slot reached value or a later marker, using a comparison that handles
    /// wrapping 32-bit counters
    [[nodiscard]] bool MarkerReached(MarkerSlot slot, u64 value) const noexcept;

protected:
    const Instance& instance;
    vk::UniqueSemaphore semaphore;    ///< Timeline semaphore.
    std::atomic<u64> gpu_tick{0};     ///< Current known GPU tick.
    std::atomic<u64> current_tick{1}; ///< Current logical tick.
    const u32 spin_us;                ///< wait_spin_us
    const u32 marker_mode;            ///< Buffer allocation used to hold the GPU-written markers that wait_marker polls from the host
    vk::Buffer marker_buffer{};       ///< Host-visible marker words updated by the GPU and read during wait_marker polling
    VmaAllocation marker_allocation{};
    const volatile u32* marker_data{};
};

} // namespace Vulkan
