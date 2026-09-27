// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <string>

#include "common/types.h"

// SHADPS4_PERF_STATS gathers frame costs with relaxed atomic counters and reports them every 10 seconds
// Disabled calls take one predictable branch. Timers nest, so draw time includes walker, fault and readback
// work. Report names match tools/harness/perf_report.py
namespace Common::PerfStats {

enum class Id : u32 {
    // GPU tracker faults and time spent handling them, including readback and protection repair
    FaultRead,
    FaultWrite,
    FaultNs,
    FaultGpuThread,
    FaultGpuThreadNs,
    // Write faults on GPU-modified memory: the write waits for a readback on the GPU thread
    FaultWriteReadback,
    // SRT walker read faults completed from guest memory without a readback
    // (srt_walker_clean_reads). They are not counted in FaultRead/FaultGpuThread
    SrtCleanReads,
    SrtCleanReadNs,
    // Walker loads served the same way by the generated code, without a fault
    SrtCleanLoads,
    // Cached shader code ranges compared through the backing (shader_code_clean_reads)
    ShaderCodeCleanReads,
    // Tracker readbacks on the GPU thread, each draining Vulkan before copying data to guest memory
    Readbacks,
    ReadbackNs,
    ReadbackBytes,
    ImageDownloads,
    // Vulkan drains, timeline waits, submits and GPU time from rasterizer command buffer timestamps
    VkFinish,
    VkFinishNs,
    VkWait,
    VkWaitNs,
    VkSubmits,
    GpuNs,
    // Time guest threads spend in WaitGpuIdle until the command processor catches up
    GnmSubmitDone,
    GnmIdleWait,
    GnmIdleWaitNs,
    // Command processor busy time, VO label sleeps and PM4 wait loops that yielded
    CpBusyNs,
    CpVoWaitNs,
    CpWaitYields,
    // Rasterizer work recorded by the command processor
    Draws,
    DrawNs,
    Dispatches,
    DispatchNs,
    SrtWalks,
    SrtWalkNs,
    ShaderCompiles,
    ShaderCompileNs,
    PipelineCreates,
    PipelineCreateNs,
    TextureUploads,
    TextureUploadNs,
    TextureUploadBytes,
    BufferUploadBytes,
    // Buffer cache sync batch flushes (uploads of CPU-modified ranges before GPU use)
    SyncFlushes,
    // Address-space churn: tracker protection calls and Windows view splits
    Protects,
    Splits,
    // Guest write fault and protection costs; split protection time by thread to show upload work
    ProtectNs,
    ProtectGpuThread,
    ProtectGpuThreadNs,
    FaultWriteGuestNs,
    Invalidates,
    InvalidateBufferNs,
    InvalidateTextureNs,
    FaultRepairCalls,
    FaultRepairNs,
    // Distinct 4 KiB pages hit by guest write faults in the reporting interval
    FaultWritePages,
    // bpe_heap_guard: guards built for readbacks and the build time
    HeapGuardBuilds,
    HeapGuardBuildNs,
    // Idea B probe (B0): newest GPU writer of the bytes a readback copies, in the command buffer being
    // recorded (or a binding not yet closed by its command), or in one already submitted (busy = still
    // running when the readback starts)
    ReadbackWriterCurrent,
    ReadbackWriterOld,
    ReadbackWriterOldBusy,
    // Idea B (readback_ahead): readbacks copied by a command buffer submitted ahead of the current one, and
    // the wait for it; idle = every writer had already completed
    ReadbackAhead,
    ReadbackAheadNs,
    ReadbackAheadIdle,
    ReadbackAheadIdleNs,
    // periodic_flush_commands: submits after N guest commands, and those that cut a render pass
    FlushPeriodic,
    FlushPeriodicInPass,
    // EOP/EOS/RELEASE_MEM writes, their submit-to-write lag and the longest graphics submit queue
    LabelWrites,
    LabelLagNs,
    LabelLagMaxNs,
    GfxQueueMax,
    Count,
};

bool Enabled();

namespace Detail {
std::atomic<u64>& Slot(Id id);
}

inline void Add(Id id, u64 value = 1) {
    if (Enabled()) {
        Detail::Slot(id).fetch_add(value, std::memory_order_relaxed);
    }
}

/// Keeps the largest value seen since the previous report
inline void Max(Id id, u64 value) {
    if (Enabled()) {
        auto& slot = Detail::Slot(id);
        u64 current = slot.load(std::memory_order_relaxed);
        while (value > current &&
               !slot.compare_exchange_weak(current, value, std::memory_order_relaxed)) {
        }
    }
}

/// Adds one to `count` and the elapsed nanoseconds to `ns` when it goes out of scope
class ScopedTimer {
public:
    ScopedTimer(Id count, Id ns) : count_id{count}, ns_id{ns}, active{Enabled()} {
        if (active) {
            start = std::chrono::steady_clock::now();
        }
    }
    ~ScopedTimer() {
        if (active) {
            const auto elapsed = std::chrono::steady_clock::now() - start;
            Detail::Slot(count_id).fetch_add(1, std::memory_order_relaxed);
            Detail::Slot(ns_id).fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count(),
                std::memory_order_relaxed);
        }
    }
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    Id count_id;
    Id ns_id;
    bool active;
    std::chrono::steady_clock::time_point start;
};

/// Adds the elapsed nanoseconds to `ns` when it goes out of scope (no count)
class ScopedNs {
public:
    explicit ScopedNs(Id ns) : ns_id{ns}, active{Enabled()} {
        if (active) {
            start = std::chrono::steady_clock::now();
        }
    }
    ~ScopedNs() {
        if (active) {
            Detail::Slot(ns_id).fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                              std::chrono::steady_clock::now() - start)
                                              .count(),
                                          std::memory_order_relaxed);
        }
    }
    ScopedNs(const ScopedNs&) = delete;
    ScopedNs& operator=(const ScopedNs&) = delete;

private:
    Id ns_id;
    bool active;
    std::chrono::steady_clock::time_point start;
};

/// Marks the calling thread as the GPU command processor thread
void MarkGpuThread();
bool IsGpuThread();

/// Formats the counters accumulated since the previous call and resets them
std::string TakeReport();

/// Counts a handled fault against its 1 MiB region, split by faulting thread and access. Guest write faults
/// also count the distinct pages they hit (FaultWritePages)
void RecordFaultRegion(u64 address, bool is_write);

/// Top regions since the previous call ("gpu=0x242300000:1234,... guest_r=... guest_w=..."), then resets them
std::string TakeFaultRegionReport();

} // namespace Common::PerfStats
