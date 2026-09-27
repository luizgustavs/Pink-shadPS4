// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <limits>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_semaphore.h"

#include "common/assert.h"
#include "common/perf_stats.h"
#include "core/emulator_settings.h"

namespace Vulkan {

constexpr u64 WAIT_TIMEOUT = std::numeric_limits<u64>::max();

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
    : instance{instance_}, spin_us{EmulatorSettings.GetWaitSpinUs()} {
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
}

Semaphore::~Semaphore() = default;

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
    // A tick not submitted yet cannot signal while polling (pending ops wait on the recording tick)
    if (tick < CurrentTick() && SpinUntil(spin_us, [&] {
            Refresh();
            return IsFree(tick);
        })) {
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
