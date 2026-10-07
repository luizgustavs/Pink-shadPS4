// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>

#include "common/perf_stats.h"

namespace Common::PerfStats {

namespace {

enum class Unit { Count, Ms, Kb };

struct Field {
    Id id;
    const char* name;
    Unit unit;
};

// Report order and names; the harness parses "name=value" pairs, so names must stay stable
constexpr std::array<Field, static_cast<size_t>(Id::Count)> Fields{{
    {Id::FaultRead, "fault_r", Unit::Count},
    {Id::FaultWrite, "fault_w", Unit::Count},
    {Id::FaultNs, "fault_ms", Unit::Ms},
    {Id::FaultGpuThread, "fault_gpu", Unit::Count},
    {Id::FaultGpuThreadNs, "fault_gpu_ms", Unit::Ms},
    {Id::FaultWriteReadback, "fault_w_readback", Unit::Count},
    {Id::SrtCleanReads, "srt_clean", Unit::Count},
    {Id::SrtCleanReadNs, "srt_clean_ms", Unit::Ms},
    {Id::SrtCleanLoads, "srt_clean_nofault", Unit::Count},
    {Id::ShaderCodeCleanReads, "code_clean", Unit::Count},
    {Id::Readbacks, "readbacks", Unit::Count},
    {Id::ReadbackNs, "readback_ms", Unit::Ms},
    {Id::ReadbackBytes, "readback_kb", Unit::Kb},
    {Id::ReadbackHotRegions, "rb_hot", Unit::Count},
    {Id::ReadbackHotBytes, "rb_hot_kb", Unit::Kb},
    {Id::ReadbackHotNs, "rb_hot_ms", Unit::Ms},
    {Id::ImageDownloads, "image_downloads", Unit::Count},
    {Id::VkFinish, "vk_finish", Unit::Count},
    {Id::VkFinishNs, "vk_finish_ms", Unit::Ms},
    {Id::VkWait, "vk_wait", Unit::Count},
    {Id::VkWaitNs, "vk_wait_ms", Unit::Ms},
    {Id::VkSubmits, "vk_submits", Unit::Count},
    {Id::GpuNs, "gpu_ms", Unit::Ms},
    {Id::GnmSubmitDone, "gnm_submit_done", Unit::Count},
    {Id::GnmIdleWait, "gnm_idle_wait", Unit::Count},
    {Id::GnmIdleWaitNs, "gnm_idle_wait_ms", Unit::Ms},
    {Id::CpBusyNs, "cp_busy_ms", Unit::Ms},
    {Id::CpVoWaitNs, "cp_vo_wait_ms", Unit::Ms},
    {Id::CpWaitYields, "cp_wait_yields", Unit::Count},
    {Id::Draws, "draws", Unit::Count},
    {Id::DrawNs, "draw_ms", Unit::Ms},
    {Id::Dispatches, "dispatches", Unit::Count},
    {Id::DispatchNs, "dispatch_ms", Unit::Ms},
    {Id::SrtWalks, "srt_walks", Unit::Count},
    {Id::SrtWalkNs, "srt_walk_ms", Unit::Ms},
    {Id::ShaderCompiles, "shader_compiles", Unit::Count},
    {Id::ShaderCompileNs, "shader_compile_ms", Unit::Ms},
    {Id::PipelineCreates, "pipeline_creates", Unit::Count},
    {Id::PipelineCreateNs, "pipeline_create_ms", Unit::Ms},
    {Id::TextureUploads, "tex_uploads", Unit::Count},
    {Id::TextureUploadNs, "tex_upload_ms", Unit::Ms},
    {Id::TextureUploadBytes, "tex_upload_kb", Unit::Kb},
    {Id::TextureDetiles, "tex_detiles", Unit::Count},
    {Id::TextureDetilesHost, "tex_detiles_host", Unit::Count},
    {Id::TextureDetileHostBytes, "tex_detile_host_kb", Unit::Kb},
    {Id::AbToggleOn, "ab_on", Unit::Count},
    {Id::PipelineBindsSkipped, "pipeline_binds_skipped", Unit::Count},
    {Id::BufferUploadBytes, "buf_upload_kb", Unit::Kb},
    {Id::SyncFlushes, "sync_flushes", Unit::Count},
    {Id::Protects, "protects", Unit::Count},
    {Id::Splits, "splits", Unit::Count},
    {Id::ProtectNs, "protect_ms", Unit::Ms},
    {Id::ProtectGpuThread, "protect_gpu", Unit::Count},
    {Id::ProtectGpuThreadNs, "protect_gpu_ms", Unit::Ms},
    {Id::FaultWriteGuestNs, "fault_w_ms", Unit::Ms},
    {Id::Invalidates, "invalidates", Unit::Count},
    {Id::InvalidateBufferNs, "inval_buf_ms", Unit::Ms},
    {Id::InvalidateTextureNs, "inval_tex_ms", Unit::Ms},
    {Id::FaultRepairCalls, "fault_repairs", Unit::Count},
    {Id::FaultRepairNs, "fault_repair_ms", Unit::Ms},
    {Id::FaultWritePages, "fault_w_pages", Unit::Count},
    {Id::HeapGuardBuilds, "guard_builds", Unit::Count},
    {Id::HeapGuardBuildNs, "guard_ms", Unit::Ms},
    {Id::ReadbackWriterCurrent, "rb_wr_cur", Unit::Count},
    {Id::ReadbackWriterOld, "rb_wr_old", Unit::Count},
    {Id::ReadbackWriterOldBusy, "rb_wr_old_busy", Unit::Count},
    {Id::ReadbackAhead, "rb_ahead", Unit::Count},
    {Id::ReadbackAheadNs, "rb_ahead_ms", Unit::Ms},
    {Id::ReadbackAheadIdle, "rb_ahead_idle", Unit::Count},
    {Id::ReadbackAheadIdleNs, "rb_ahead_idle_ms", Unit::Ms},
    {Id::FlushPeriodic, "flush_periodic", Unit::Count},
    {Id::FlushPeriodicInPass, "flush_periodic_pass", Unit::Count},
    {Id::GuestReadbacks, "guest_rb", Unit::Count},
    {Id::GuestReadbackNs, "guest_rb_ms", Unit::Ms},
    {Id::GuestReadbackWaitNs, "guest_rb_wait_ms", Unit::Ms},
    {Id::GuestReadbackWrite, "guest_rb_w", Unit::Count},
    {Id::GuestReadbackEmpty, "guest_rb_empty", Unit::Count},
    {Id::GuestReadbackFinish, "guest_rb_finish", Unit::Count},
    {Id::GuestReadbackCurrent, "guest_rb_cur", Unit::Count},
    {Id::GuestReadbackPageDirty, "guest_rb_page_dirty", Unit::Count},
    {Id::LabelWrites, "labels", Unit::Count},
    {Id::LabelLagNs, "label_lag_ms", Unit::Ms},
    {Id::LabelLagMaxNs, "label_lag_max_ms", Unit::Ms},
    {Id::GfxQueueMax, "gfx_queue_max", Unit::Count},
    {Id::BufferUploadStackBytes, "buf_upload_stack_kb", Unit::Kb},
    {Id::BufferUploadWrittenBytes, "buf_upload_written_kb", Unit::Kb},
    {Id::PipelineBinds, "pipeline_binds", Unit::Count},
    {Id::PipelineBindsSame, "pipeline_binds_same", Unit::Count},
    {Id::DmaSweeps, "dma_sweeps", Unit::Count},
    {Id::DmaSweepStackBytes, "dma_sweep_stack_kb", Unit::Kb},
    {Id::DmaSweepsSkipped, "dma_sweeps_skipped", Unit::Count},
    {Id::BindReusedBuffers, "bind_reuse_v", Unit::Count},
    {Id::BindReusedImages, "bind_reuse_t", Unit::Count},
    {Id::BindVerifyChecks, "bind_verify", Unit::Count},
    {Id::BindVerifyMismatches, "bind_verify_bad", Unit::Count},
    {Id::TsharpCacheHits, "tsharp_hit", Unit::Count},
    {Id::TsharpCacheNew, "tsharp_new", Unit::Count},
    {Id::TsharpCacheStale, "tsharp_stale", Unit::Count},
    {Id::TsharpCacheSlotPages, "tsharp_slot_pages", Unit::Count},
    {Id::WaitSpins, "wait_spins", Unit::Count},
    {Id::WaitSpinHits, "wait_spin_hits", Unit::Count},
    {Id::WaitSpinPolls, "wait_spin_polls", Unit::Count},
    {Id::WaitSpinNs, "wait_spin_ms", Unit::Ms},
    {Id::WaitKernelPolls, "wait_kernel_polls", Unit::Count},
    {Id::WaitMarkerLate, "wait_marker_late", Unit::Count},
    {Id::WaitMarkerEarly, "wait_marker_early", Unit::Count},
    {Id::PooledImages, "pooled_images", Unit::Count},
    {Id::RecorderCommands, "rec_cmds", Unit::Count},
    {Id::RecorderDrains, "rec_drains", Unit::Count},
    {Id::RecorderDrainNs, "rec_drain_ms", Unit::Ms},
    {Id::RecorderBusyNs, "rec_busy_ms", Unit::Ms},
    {Id::RecorderSubmitWaits, "rec_submit_waits", Unit::Count},
    {Id::RecorderSubmitWaitNs, "rec_submit_wait_ms", Unit::Ms},
    {Id::RecorderWakes, "rec_wakes", Unit::Count},
    {Id::DepthTargetSampled, "depth_tgt_sampled", Unit::Count},
    {Id::GcnUnorderedPoints, "gcn_points", Unit::Count},
    {Id::GcnUnorderedSkips, "gcn_skips", Unit::Count},
}};

constexpr bool FieldsMatchIds() {
    for (size_t i = 0; i < Fields.size(); ++i) {
        if (Fields[i].id != static_cast<Id>(i)) {
            return false;
        }
    }
    return true;
}
static_assert(FieldsMatchIds(), "Fields must list every Id in enum order");

std::array<std::atomic<u64>, static_cast<size_t>(Id::Count)> slots{};
thread_local bool is_gpu_thread = false;

// Fault regions: only touched on handled faults, which already cost microseconds or more
constexpr u32 RegionBits = 20;
constexpr size_t TopRegions = 6;
enum RegionKind : size_t { GpuThread, GuestRead, GuestWrite, Upload, TexUpload, NumRegionKinds };
std::mutex region_mutex;
std::array<std::unordered_map<u64, u64>, NumRegionKinds> region_counts;
constexpr u32 PageBits = 12;
std::unordered_set<u64> guest_write_pages;

} // Anonymous namespace

bool Enabled() {
    static const bool enabled = std::getenv("SHADPS4_PERF_STATS") != nullptr;
    return enabled;
}

std::atomic<u64>& Detail::Slot(Id id) {
    return slots[static_cast<size_t>(id)];
}

namespace {
// Seconds of reports before SHADPS4_AB_TOGGLE starts flipping, or a negative value without it
double AbToggleStart() {
    const char* env = std::getenv("SHADPS4_AB_TOGGLE");
    if (!env || !std::getenv("SHADPS4_PERF_STATS")) {
        return -1.0;
    }
    return std::max(std::strtod(env, nullptr), 0.0);
}
const double ab_toggle_start = AbToggleStart();
} // Anonymous namespace

std::atomic<bool> Detail::ab_toggle_on{ab_toggle_start < 0.0};

bool AbToggleLists(std::string_view key) {
    if (ab_toggle_start < 0.0) {
        return false;
    }
    const char* env = std::getenv("SHADPS4_AB_TOGGLE_KEYS");
    std::string_view list{env ? env : ""};
    while (!list.empty()) {
        const size_t comma = list.find(',');
        std::string_view item = list.substr(0, comma);
        while (!item.empty() && item.front() == ' ') {
            item.remove_prefix(1);
        }
        while (!item.empty() && item.back() == ' ') {
            item.remove_suffix(1);
        }
        if (item == key) {
            return true;
        }
        list = comma == std::string_view::npos ? std::string_view{} : list.substr(comma + 1);
    }
    return false;
}

bool AbToggleFollows(std::string_view key) {
    if (ab_toggle_start < 0.0) {
        return false;
    }
    const char* env = std::getenv("SHADPS4_AB_TOGGLE_KEYS");
    return !env || !*env || AbToggleLists(key);
}

void OnReportEmitted(double seconds_since_start) {
    if (ab_toggle_start < 0.0 || seconds_since_start < ab_toggle_start) {
        return;
    }
    const bool on = !Detail::ab_toggle_on.load(std::memory_order_relaxed);
    Detail::ab_toggle_on.store(on, std::memory_order_relaxed);
    if (on) {
        Add(Id::AbToggleOn);
    }
}

void MarkGpuThread() {
    is_gpu_thread = true;
}

bool IsGpuThread() {
    return is_gpu_thread;
}

void RecordFaultRegion(u64 address, bool is_write) {
    if (!Enabled()) {
        return;
    }
    const RegionKind kind = is_gpu_thread ? GpuThread : is_write ? GuestWrite : GuestRead;
    std::scoped_lock lk{region_mutex};
    ++region_counts[kind][address >> RegionBits];
    if (kind == GuestWrite && guest_write_pages.insert(address >> PageBits).second) {
        Detail::Slot(Id::FaultWritePages).fetch_add(1, std::memory_order_relaxed);
    }
}

void RecordUploadRegion(u64 address, u64 bytes, bool texture) {
    if (!Enabled()) {
        return;
    }
    std::scoped_lock lk{region_mutex};
    region_counts[texture ? TexUpload : Upload][address >> RegionBits] += bytes >> 10;
}

std::string TakeFaultRegionReport() {
    static constexpr std::array<const char*, NumRegionKinds> names{"gpu", "guest_r", "guest_w",
                                                                   "upload", "tex_upload"};
    std::array<std::unordered_map<u64, u64>, NumRegionKinds> counts;
    {
        std::scoped_lock lk{region_mutex};
        counts.swap(region_counts);
    }
    std::string report;
    for (size_t kind = 0; kind < NumRegionKinds; ++kind) {
        std::vector<std::pair<u64, u64>> top(counts[kind].begin(), counts[kind].end());
        const size_t shown = std::min(top.size(), TopRegions);
        std::partial_sort(top.begin(), top.begin() + shown, top.end(),
                          [](const auto& a, const auto& b) { return a.second > b.second; });
        if (!report.empty()) {
            report += ' ';
        }
        report += fmt::format("{}=", names[kind]);
        for (size_t i = 0; i < shown; ++i) {
            report += fmt::format("{}{:#x}:{}", i ? "," : "", top[i].first << RegionBits,
                                  top[i].second);
        }
        if (shown == 0) {
            report += '-';
        }
    }
    return report;
}

std::string TakeReport() {
    {
        std::scoped_lock lk{region_mutex};
        guest_write_pages.clear();
    }
    std::string report;
    for (const auto& field : Fields) {
        const u64 value =
            slots[static_cast<size_t>(field.id)].exchange(0, std::memory_order_relaxed);
        if (!report.empty()) {
            report += ' ';
        }
        switch (field.unit) {
        case Unit::Count:
            report += fmt::format("{}={}", field.name, value);
            break;
        case Unit::Ms:
            report += fmt::format("{}={:.1f}", field.name, static_cast<double>(value) / 1e6);
            break;
        case Unit::Kb:
            report += fmt::format("{}={}", field.name, value >> 10);
            break;
        }
    }
    return report;
}

} // namespace Common::PerfStats
