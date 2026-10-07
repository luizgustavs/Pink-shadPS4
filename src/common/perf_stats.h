// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <string>
#include <string_view>

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
    // readback_hot_regions: recent readback windows downloaded along with a
    // readback that drained the GPU, and the command processor time spent
    // recording and writing them
    ReadbackHotRegions,
    ReadbackHotBytes,
    ReadbackHotNs,
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
    // Tiled texture uploads detiled on the GPU, and those whose source is a host staging copy of guest memory
    // (not GPU-modified) with its bytes
    TextureDetiles,
    TextureDetilesHost,
    TextureDetileHostBytes,
    // SHADPS4_AB_TOGGLE: 1 in each report interval that ran with the toggled keys on
    AbToggleOn,
    // Pipeline binds Scheduler::BindPipeline dropped because the command buffer already had the pipeline
    PipelineBindsSkipped,
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
    // Newest writer state for bytes copied by a readback
    ReadbackWriterCurrent,
    ReadbackWriterOld,
    ReadbackWriterOldBusy,
    // Ahead readbacks and their wait time
    ReadbackAhead,
    ReadbackAheadNs,
    ReadbackAheadIdle,
    ReadbackAheadIdleNs,
    // periodic_flush_commands: submits after N guest commands, and those that cut a render pass
    FlushPeriodic,
    FlushPeriodicInPass,
    // Guest fault readbacks, waits and writer state
    GuestReadbacks,
    GuestReadbackNs,
    GuestReadbackWaitNs,
    GuestReadbackWrite,
    GuestReadbackEmpty,
    GuestReadbackFinish,
    GuestReadbackCurrent,
    GuestReadbackPageDirty,
    // EOP/EOS/RELEASE_MEM writes, their submit-to-write lag and the longest graphics submit queue
    LabelWrites,
    LabelLagNs,
    LabelLagMaxNs,
    GfxQueueMax,
    // Stack and GPU-written bytes in sync uploads
    BufferUploadStackBytes,
    BufferUploadWrittenBytes,
    // Pipeline binds and repeated binds in the same command buffer
    PipelineBinds,
    PipelineBindsSame,
    // DMA sweeps and their guest stack bytes
    DmaSweeps,
    DmaSweepStackBytes,
    // Count the DMA range sweeps avoided by dma_sync_once_per_batch after the current sync batch
    // has already been filled
    DmaSweepsSkipped,
    // Count V# and T# bindings reused from the previous call of the same pipeline, along with the
    // checks made when verification is enabled
    BindReusedBuffers,
    BindReusedImages,
    BindVerifyChecks,
    BindVerifyMismatches,
    // Count T# cache hits, missing entries and entries whose pages gained or lost an image since
    // the last lookup
    // Also count incremental_bind slots kept valid by checking only the pages used by their image
    TsharpCacheHits,
    TsharpCacheNew,
    TsharpCacheStale,
    TsharpCacheSlotPages,
    // Short GPU wait polling stats
    WaitSpins,
    WaitSpinHits,
    WaitSpinPolls,
    WaitSpinNs,
    // Count driver queries during marker polling and waits where the GPU marker arrived before the
    // driver reported the tick or fence
    // For mode 2, also count readback waits that returned as soon as their marker arrived
    WaitKernelPolls,
    WaitMarkerLate,
    WaitMarkerEarly,
    // Count images whose memory was sub-allocated from image_memory_pool instead of using an
    // individual allocation
    PooledImages,
    // Count Vulkan calls sent to the recording thread and the time the command processor spends
    // waiting for it
    // Also track how long the recording thread stays busy processing those calls
    RecorderCommands,
    RecorderDrains,
    RecorderDrainNs,
    RecorderBusyNs,
    // Count command processor waits that need the recording thread to submit a tick before Finish
    // or Wait can continue
    RecorderSubmitWaits,
    RecorderSubmitWaitNs,
    // Count times the command processor wakes a sleeping recording thread after making another
    // command batch available
    RecorderWakes,
    Count,
};

bool Enabled();

namespace Detail {
std::atomic<u64>& Slot(Id id);
extern std::atomic<bool> ab_toggle_on;
}

/// Returns the active side of the in-run A/B toggle
inline bool AbToggleActive() {
    return Detail::ab_toggle_on.load(std::memory_order_relaxed);
}

/// Whether the key follows SHADPS4_AB_TOGGLE: always with the toggle unless SHADPS4_AB_TOGGLE_KEYS lists other
/// keys (comma separated), so keys held on both sides of an in-run A/B stay on. Read once, at key setup
bool AbToggleFollows(std::string_view key);

/// Check whether SHADPS4_AB_TOGGLE_KEYS includes this key while SHADPS4_AB_TOGGLE is enabled
/// Settings needing startup work, such as cp_record_thread, prepare it for the run even if their
/// initial interval is off
/// The thread must already exist when a later interval enables recording
bool AbToggleLists(std::string_view key);

/// Whether a key that follows the toggle (AbToggleFollows) runs in this report interval
inline bool AbToggleOn(bool follows) {
    return !follows || AbToggleActive();
}

/// Called after each perf report with the seconds since the first one; flips SHADPS4_AB_TOGGLE
void OnReportEmitted(double seconds_since_start);

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

/// Adds a sync-batch upload (or a texture upload) to its 1 MiB region in KiB, reported as "upload=" (or
/// "tex_upload=") in the fault region line
void RecordUploadRegion(u64 address, u64 bytes, bool texture = false);

/// Top regions since the previous call ("gpu=0x242300000:1234,... guest_r=... guest_w=..."), then resets them
std::string TakeFaultRegionReport();

} // namespace Common::PerfStats
