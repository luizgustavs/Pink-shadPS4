// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <magic_enum/magic_enum.hpp>

#include "common/alignment.h"
#include "common/guest_write_journal.h"
#include "common/logging/events.h"
#include "common/perf_stats.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/bpe_heap_guard.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

#include <vk_mem_alloc.h>

namespace VideoCore {

static constexpr size_t GDS_BUFFER_SIZE = 64_KB;
static constexpr size_t STREAM_BUFFER_SIZE = 128_MB;

static constexpr auto ARENA_USAGE =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;

std::optional<u32> FindMemoryType(const vk::PhysicalDeviceMemoryProperties& properties,
                                  vk::MemoryPropertyFlags wanted, u32 memory_type_bits) {
    for (u32 i = 0; i < properties.memoryTypeCount; ++i) {
        if (((memory_type_bits >> i) & 1) == 0) {
            continue;
        }
        const auto flags = properties.memoryTypes[i].propertyFlags;
        if ((flags & wanted) == wanted) {
            return i;
        }
    }
    return std::nullopt;
}

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         Vulkan::Runtime& runtime_, AmdGpu::Liverpool* liverpool_,
                         TextureCache& texture_cache_, PageManager& tracker)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
      staging_pool{runtime_.GetStagingPool()}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_},
      memory_tracker{std::make_unique<MemoryTracker>(tracker)},
      stream_buffer{instance, scheduler, MemoryType::Stream, STREAM_BUFFER_SIZE},
      gds_buffer{instance, 0, GDS_BUFFER_SIZE, MemoryType::Stream, "GDS Buffer"},
      loop_cap_buffer{instance, 0, 256, MemoryType::HostCached, "Loop Cap Buffer"},
      readback_ahead{EmulatorSettings.IsReadbackAhead()},
      track_writers{readback_ahead || Common::PerfStats::Enabled()}, memory_semaphore{instance},
      dma_sync_once{EmulatorSettings.IsDmaSyncOncePerBatch()},
      dma_sync_once_toggle{Common::PerfStats::AbToggleFollows("dma_sync_once_per_batch")},
      dma_sync_once_verify{dma_sync_once &&
                           std::getenv("SHADPS4_DMA_SYNC_ONCE_VERIFY") != nullptr},
      sweep_skip_stacks{EmulatorSettings.IsDmaSweepSkipStacks()},
      sweep_skip_toggle{Common::PerfStats::AbToggleFollows("dma_sweep_skip_stacks")} {
    const vk::BufferCreateInfo probe_ci = {
        .flags =
            vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency,
        .size = ARENA_PAGE_SIZE,
        .usage = ARENA_USAGE,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    const vk::DeviceBufferMemoryRequirements req_info = {
        .pCreateInfo = &probe_ci,
    };
    const auto device = instance.GetDevice();
    const auto reqs = device.getBufferMemoryRequirements(req_info).memoryRequirements;
    block_size = Common::AlignUp(std::max<u64>(reqs.alignment, MIN_BLOCK_SIZE), reqs.alignment);
    ASSERT_MSG(std::popcount(block_size) == 1, "Sparse block size {} is not a power of 2",
               block_size);
    block_shift = std::bit_width(block_size) - 1;
    blocks_per_arena_page = ARENA_PAGE_SIZE / block_size;
    blocks_per_arena_page_shift = ARENA_PAGE_BITS - block_shift;
    arena_memory_type_index =
        FindMemoryType(instance.GetMemoryProperties(), vk::MemoryPropertyFlagBits::eDeviceLocal,
                       reqs.memoryTypeBits)
            .value();

    const u64 bda_pagetable_size =
        (blocks_per_arena_page * NUM_ARENA_PAGES) * sizeof(vk::DeviceAddress);
    fault_manager = std::make_unique<FaultManager>(instance, scheduler, *this, block_shift,
                                                   blocks_per_arena_page * NUM_ARENA_PAGES);
    // Shaders atomically set bits in the fault bitmap. Device-local allocation contents are undefined, so the
    // initial state must be made explicitly empty
    const auto* fault_buffer = fault_manager->GetFaultBuffer();
    runtime.FillBuffer(fault_buffer, 0u, fault_buffer->SizeBytes(), 0u);
    bda_pagetable_buffer = std::make_unique<Buffer>(
        instance, 0, bda_pagetable_size, MemoryType::DeviceLocal, "BDA Page Table Buffer");
    runtime.FillBuffer(bda_pagetable_buffer.get(), 0u, bda_pagetable_size, 0u);

    std::memset(loop_cap_buffer.mapped_data.data(), 0, loop_cap_buffer.mapped_data.size());
    loop_cap_buffer.Flush(0, loop_cap_buffer.SizeBytes());
    if (const u32 cap = EmulatorSettings.GetComputeLoopCap()) {
        LOG_WARNING(Render_Vulkan, "Workaround compute_loop_cap enabled: {} back edges", cap);
    }
    if (readback_ahead) {
        LOG_WARNING(Render_Vulkan, "Workaround readback_ahead enabled");
    }
    if (sweep_skip_stacks && !EmulatorSettings.IsCpuAuthoritativeStacks()) {
        LOG_WARNING(Render_Vulkan, "dma_sweep_skip_stacks without cpu_authoritative_stacks: no "
                                   "stack is registered, so the DMA sweep is unchanged");
    }
    if (dma_sync_once) {
        LOG_WARNING(Render_Vulkan, "Workaround dma_sync_once_per_batch enabled{}",
                    dma_sync_once_verify ? " (SHADPS4_DMA_SYNC_ONCE_VERIFY)" : "");
    }
}

BufferCache::~BufferCache() = default;

void BufferCache::TickFrame() {
    if (std::exchange(fault_process_pending, false)) {
        fault_manager->ProcessFaultBuffer();
    }
    ReportLoopCapHits();
    DebugState.num_batches_per_frame = std::exchange(num_flushes_per_frame, 0u);
}

static void ReportHeapGuardHit(VAddr addr, u64 bytes) {
    // Only the GPU command processor thread downloads memory
    static u64 hits = 0;
    static u64 total = 0;
    static auto last_report = std::chrono::steady_clock::now();
    ++hits;
    total += bytes;
    const auto now = std::chrono::steady_clock::now();
    if (hits == 1) {
        LOG_WARNING(Render, "Workaround bpe_heap_guard: kept CPU heap metadata under a readback "
                            "at {:#x} ({:#x} differing bytes)",
                    addr, bytes);
        last_report = now;
    } else if (now - last_report >= std::chrono::minutes{1}) {
        LOG_WARNING(Render,
                    "Workaround bpe_heap_guard: {} readbacks kept heap metadata so far ({:#x} "
                    "bytes), last at {:#x}",
                    hits, total, addr);
        last_report = now;
    }
}

void BufferCache::ReportLoopCapHits() {
    const u32 cap = EmulatorSettings.GetComputeLoopCap();
    if (cap == 0) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - loop_cap_last_check < std::chrono::seconds{1}) {
        return;
    }
    loop_cap_last_check = now;
    // The shaders only increment the counter, so the CPU never writes it back and cannot lose hits that land
    // between the read and a reset
    loop_cap_buffer.Invalidate(0, loop_cap_buffer.SizeBytes());
    std::array<u32, 2> words{};
    std::memcpy(words.data(), loop_cap_buffer.mapped_data.data(), sizeof(words));
    const u32 new_hits = words[0] - loop_cap_hits_reported;
    if (new_hits != 0 && loop_cap_hits_reported == 0) {
        LOG_WARNING(Render_Vulkan,
                    "Workaround compute_loop_cap hit: cs {:#x} stopped after {} back edges",
                    words[1], cap);
        loop_cap_last_report = now;
    }
    loop_cap_hits_reported = words[0];
    loop_cap_hits_minute += new_hits;
    if (loop_cap_hits_minute != 0 && now - loop_cap_last_report >= std::chrono::minutes{1}) {
        LOG_WARNING(Render_Vulkan,
                    "Workaround compute_loop_cap: {} capped invocations in the last minute "
                    "(total {}, last cs {:#x})",
                    loop_cap_hits_minute, loop_cap_hits_reported, words[1]);
        loop_cap_hits_minute = 0;
        loop_cap_last_report = now;
    }
}

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks) {
    memory_tracker->InvalidateRegion(device_addr, size, [this, device_addr, size, assume_locks] {
        Common::PerfStats::Add(Common::PerfStats::Id::FaultWriteReadback);
        ReadMemory(device_addr, size, true, assume_locks);
    });
}

void BufferCache::ReleaseCpuAuthoritativeRange(VAddr device_addr, u64 size) {
    // A released stack range goes back to normal tracking as CPU-modified: whatever the GPU wrote there
    // before it became a stack is stale
    Common::GuestWriteJournal::Record(Common::GuestWriteJournal::Source::CpuAuthoritativeRelease,
                                      device_addr, size);
    liverpool->SendCommand([this, device_addr, size] {
        memory_tracker->UnmarkRegionAsGpuModified(device_addr, size, true);
        gpu_modified_ranges.Subtract(device_addr, size);
        InvalidateCleanPages(device_addr, size);
    });
}

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write, bool assume_locks) {
    const auto flush_request = [this, device_addr, size, is_write, assume_locks] {
        const Common::PerfStats::ScopedTimer perf_timer{Common::PerfStats::Id::Readbacks,
                                                        Common::PerfStats::Id::ReadbackNs};
        // J0 probe: requests from guest threads come through SendCommand, the command processor's own
        // faults run inline
        guest_readback = !assume_locks && Common::PerfStats::Enabled();
        std::optional<Common::PerfStats::ScopedTimer> guest_timer;
        if (guest_readback) {
            using Common::PerfStats::Id;
            guest_timer.emplace(Id::GuestReadbacks, Id::GuestReadbackNs);
            if (is_write) {
                Common::PerfStats::Add(Id::GuestReadbackWrite);
            }
            if (gpu_modified_ranges.Intersects(Common::AlignDown(device_addr, 4_KB), 4_KB)) {
                Common::PerfStats::Add(Id::GuestReadbackPageDirty);
            }
        }
        const u32 first_block = device_addr >> block_shift;
        const u32 last_block = (device_addr + size - 1) >> block_shift;
        const auto* arena = GetArena(first_block, last_block);

        // GPU-modified ranges come as many small scattered islands,
        // so the download is widened to a window around the request
        constexpr u64 WindowSize = 512_KB;
        const VAddr arena_end = arena->cpu_addr + arena->size_bytes;
        const VAddr window_start =
            std::max<VAddr>(Common::AlignDown(device_addr, WindowSize), arena->cpu_addr);
        const VAddr window_end = std::min<VAddr>(
            std::max<VAddr>(window_start + WindowSize, device_addr + size), arena_end);
        DownloadMemory(arena, window_start, window_end - window_start);
        guest_readback = false;
        if (is_write) {
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        }
    };
    if (assume_locks) {
        flush_request();
    } else {
        const Common::PerfStats::ScopedNs wait_timer{Common::PerfStats::Id::GuestReadbackWaitNs};
        liverpool->SendCommand<true>(std::move(flush_request));
    }
}

void BufferCache::DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size) {
    boost::container::small_vector<vk::BufferCopy, 1> copies;
    u64 total_size_bytes = 0;
    const VAddr arena_base = arena->cpu_addr;
    memory_tracker->ForEachDownloadRange<false>(device_addr, size, [&](u64 address, u64 size) {
        const auto add_download = [&](VAddr start, VAddr end) {
            const u64 new_offset = start - arena_base;
            const u64 new_size = end - start;
            copies.push_back(vk::BufferCopy{
                .srcOffset = new_offset,
                .dstOffset = total_size_bytes,
                .size = new_size,
            });
            // Align up to avoid cache conflicts
            constexpr u64 align = 64ULL;
            constexpr u64 mask = ~(align - 1ULL);
            total_size_bytes += (new_size + align - 1) & mask;
        };
        // Stack pages are CPU-authoritative (cpu_authoritative_stacks): never download over them
        gpu_modified_ranges.ForEachInRange(address, size, [&](VAddr start, VAddr end) {
            for (const auto& [lo, len] : memory->SubtractStackRanges(start, end - start)) {
                add_download(lo, lo + len);
            }
        });
        gpu_modified_ranges.Subtract(address, size);
        InvalidateCleanPages(address, size);
    });
    if (total_size_bytes == 0) {
        if (guest_readback) {
            Common::PerfStats::Add(Common::PerfStats::Id::GuestReadbackEmpty);
        }
        memory_tracker->UnmarkRegionAsGpuModified(device_addr, size, false);
        return;
    }
    Common::PerfStats::Add(Common::PerfStats::Id::ReadbackBytes, total_size_bytes);
    const auto download = staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached);
    for (auto& copy : copies) {
        copy.dstOffset += download.offset;
    }
    // Submit the copy ahead when the current command buffer cannot write these bytes
    u64 wait_tick = 0;
    const WriterState writers =
        track_writers ? ClassifyWriters(arena_base, copies, &wait_tick) : WriterState::Current;
    wait_tick = std::max(wait_tick, migration_bind_tick);
    if (guest_readback && writers == WriterState::Current) {
        Common::PerfStats::Add(Common::PerfStats::Id::GuestReadbackCurrent);
    }
    if (readback_ahead && writers != WriterState::Current && pending_binds.empty()) {
        const auto cmdbuf = scheduler.BeginAhead();
        const vk::MemoryBarrier2 pre_barrier = {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &pre_barrier,
        });
        cmdbuf.copyBuffer(arena->Handle(), download.buffer->Handle(), copies);
        const vk::MemoryBarrier2 host_barrier = {
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eHost,
            .dstAccessMask = vk::AccessFlagBits2::eHostRead,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &host_barrier,
        });
        // With cp_record_thread, wait until the writer has actually submitted its work before
        // SubmitAhead queues the copy that depends on it
        scheduler.SubmitAhead(wait_tick);
        scheduler.WaitAhead(writers == WriterState::Done);
        static const bool verify = std::getenv("SHADPS4_READBACK_AHEAD_VERIFY") != nullptr;
        if (verify) {
            download.buffer->Invalidate(download.offset, download.size);
            VerifyAheadCopy(arena, copies, download.mapped, download.offset);
        }
    } else {
        if (guest_readback) {
            Common::PerfStats::Add(Common::PerfStats::Id::GuestReadbackFinish);
        }
        runtime.CopyBuffer(arena, download.buffer, copies);
        scheduler.FinishHostRead();
    }

    download.buffer->Invalidate(download.offset, download.size);
    // Per-game bpe_heap_guard: GPU output of dispatches running on stale descriptors must not overwrite the
    // allocator metadata of the game's heap (SotC AVs at eboot+0x48xxx)
    std::optional<BpeHeapGuard> heap_guard;
    if (const u64 heap = EmulatorSettings.GetBpeHeapGuardAddress(); heap != 0) {
        const Common::PerfStats::ScopedTimer perf_timer{Common::PerfStats::Id::HeapGuardBuilds,
                                                        Common::PerfStats::Id::HeapGuardBuildNs};
        heap_guard.emplace(memory, heap);
        if (!heap_guard->IsValid()) {
            heap_guard.reset();
        }
    }
    for (const auto& copy : copies) {
        const VAddr copy_addr = arena_base + copy.srcOffset;
        const u8* src = download.mapped + (copy.dstOffset - download.offset);
        // A page can become a stack while the GPU drains: subtract the stacks again
        for (const auto& [lo, len] : memory->SubtractStackRanges(copy_addr, copy.size)) {
            Common::GuestWriteJournal::Record(Common::GuestWriteJournal::Source::BufferReadback,
                                              lo, len, src + (lo - copy_addr), device_addr);
            if (heap_guard) {
                if (const u64 kept = heap_guard->Write(lo, src + (lo - copy_addr), len);
                    kept != 0) {
                    Common::GuestWriteJournal::Record(
                        Common::GuestWriteJournal::Source::HeapGuardKept, lo, len, nullptr, kept);
                    ReportHeapGuardHit(lo, kept);
                }
                continue;
            }
            memory->TryWriteBacking(std::bit_cast<u8*>(lo), src + (lo - copy_addr), len);
        }
    }
    memory_tracker->UnmarkRegionAsGpuModified(device_addr, size, false);
}

std::pair<const Buffer*, u64> BufferCache::ObtainBuffer(VAddr device_addr, u32 size,
                                                        bool is_written, bool is_texel_buffer) {
    // For read-only buffers use device local stream buffer to reduce renderpass breaks.
    if (!is_written && size <= STREAM_THRESHOLD && !IsRegionGpuModified(device_addr, size)) {
        const auto [data, offset] = stream_buffer.Map(size, instance.UniformMinAlignment());
        memory->CopySparseMemory(device_addr, data, size);
        stream_buffer.Commit();
        return {&stream_buffer, offset};
    }
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    EnsureResident(arena, first_block, last_block);
    sync_batch.Add(device_addr, device_addr + size, is_written);
    if (is_texel_buffer && !is_written) {
        SynchronizeMemoryFromImage(arena, device_addr, size);
    }
    if (is_written) {
        for (const auto& [lo, len] : memory->SubtractStackRanges(device_addr, size)) {
            gpu_modified_ranges.Add(lo, len);
            InvalidateCleanPages(lo, len);
        }
        RecordGpuWrite(device_addr, size, true);
    }
    return {arena, arena->Offset(device_addr)};
}

BufferCache::CleanRead BufferCache::ReadCleanPage(VAddr addr, void* out, u32 size) {
    const VAddr page_addr = addr & ~(CleanPageSize - 1);
    const auto& page = clean_pages[(addr >> CleanPageBits) % NumCleanPages];
    if (page.page_addr != page_addr ||
        page.map_generation != clean_map_generation.load(std::memory_order_acquire)) {
        return CleanRead::Miss;
    }
    const u32 offset = static_cast<u32>(addr - page_addr);
    if (offset < page.dirty_hi && offset + size > page.dirty_lo &&
        gpu_modified_ranges.Intersects(addr, size)) {
        return CleanRead::GpuWritten;
    }
    std::memcpy(out, page.backing + offset, size);
    return CleanRead::Served;
}

bool BufferCache::FillCleanPage(VAddr page_addr, u64 generation) {
    const u8* backing = memory->GetBackingPointer(page_addr);
    if (!backing) {
        return false;
    }
    u32 dirty_lo = CleanPageSize;
    u32 dirty_hi = 0;
    gpu_modified_ranges.ForEachInRange(page_addr, CleanPageSize, [&](VAddr start, VAddr end) {
        dirty_lo = std::min<u32>(dirty_lo, static_cast<u32>(start - page_addr));
        dirty_hi = std::max<u32>(dirty_hi, static_cast<u32>(end - page_addr));
    });
    clean_pages_used = true;
    clean_pages[(page_addr >> CleanPageBits) % NumCleanPages] = {
        .page_addr = page_addr,
        .map_generation = generation,
        .backing = backing,
        .dirty_lo = dirty_lo,
        .dirty_hi = dirty_hi,
    };
    return true;
}

void BufferCache::InvalidateCleanPages(VAddr addr, u64 size) {
    if (!clean_pages_used) {
        // Without clean reads the page cache stays empty: written binds and readbacks pay one branch
        return;
    }
    const u64 first = addr >> CleanPageBits;
    const u64 last = (addr + size - 1) >> CleanPageBits;
    if (last - first >= NumCleanPages) {
        for (auto& page : clean_pages) {
            const u64 index = page.page_addr >> CleanPageBits;
            if (index >= first && index <= last) {
                page.page_addr = ~0ULL;
            }
        }
        return;
    }
    for (u64 index = first; index <= last; ++index) {
        auto& page = clean_pages[index % NumCleanPages];
        if (page.page_addr == index << CleanPageBits) {
            page.page_addr = ~0ULL;
        }
    }
}

void BufferCache::RecordGpuWrite(VAddr addr, u64 size, bool open) {
    if (!track_writers || size == 0) {
        return;
    }
    if (open) {
        open_writes.emplace_back(addr, size);
        return;
    }
    // A readback only needs to know whether the writer is in the current tick; older ticks are kept for
    // the busy/done split of the probe
    constexpr size_t TrackedTicks = 8;
    const u64 tick = scheduler.CurrentTick();
    if (writer_ticks.empty() || writer_ticks.back().tick != tick) {
        if (writer_ticks.size() == TrackedTicks) {
            auto recycled = std::move(writer_ticks.front());
            writer_ticks.pop_front();
            recycled.ranges.Clear();
            recycled.tick = tick;
            writer_ticks.push_back(std::move(recycled));
        } else {
            writer_ticks.emplace_back().tick = tick;
        }
    }
    writer_ticks.back().ranges.Add(addr, size);
}

void BufferCache::CloseGpuWrites() {
    // The tick is taken after the command is recorded, so a stored tick is never older than the real one
    for (const auto& [addr, size] : open_writes) {
        RecordGpuWrite(addr, size);
    }
    open_writes.clear();
}

BufferCache::WriterState BufferCache::ClassifyWriters(VAddr arena_base,
                                                      std::span<const vk::BufferCopy> copies,
                                                      u64* wait_tick) {
    using Common::PerfStats::Id;
    const auto overlaps = [&](VAddr lo, u64 size) {
        return std::ranges::any_of(copies, [&](const vk::BufferCopy& copy) {
            const VAddr copy_lo = arena_base + copy.srcOffset;
            return lo < copy_lo + copy.size && copy_lo < lo + size;
        });
    };
    const auto intersects = [&](const RangeSet& ranges) {
        return std::ranges::any_of(copies, [&](const vk::BufferCopy& copy) {
            return ranges.Intersects(arena_base + copy.srcOffset, copy.size);
        });
    };
    if (std::ranges::any_of(open_writes,
                            [&](const auto& write) { return overlaps(write.first, write.second); })) {
        Common::PerfStats::Add(Id::ReadbackWriterCurrent);
        return WriterState::Current;
    }
    const u64 current = scheduler.CurrentTick();
    for (auto it = writer_ticks.rbegin(); it != writer_ticks.rend(); ++it) {
        if (!intersects(it->ranges)) {
            continue;
        }
        if (it->tick == current) {
            Common::PerfStats::Add(Id::ReadbackWriterCurrent);
            return WriterState::Current;
        }
        Common::PerfStats::Add(Id::ReadbackWriterOld);
        *wait_tick = it->tick;
        if (!scheduler.IsFree(it->tick)) {
            Common::PerfStats::Add(Id::ReadbackWriterOldBusy);
            return WriterState::Busy;
        }
        return WriterState::Done;
    }
    // Written before the tracked ticks. Only the last few ticks with writes are tracked, so the writer may be
    // any tick before the oldest tracked one, and it may still be running
    Common::PerfStats::Add(Id::ReadbackWriterOld);
    *wait_tick = writer_ticks.empty() ? current - 1 : writer_ticks.front().tick - 1;
    return WriterState::Done;
}

void BufferCache::VerifyAheadCopy(const Buffer* arena, std::span<const vk::BufferCopy> copies,
                                  const u8* ahead_data, u64 ahead_offset) {
    // Gate only: every ahead copy also pays the Finish it was meant to avoid. Nothing was recorded since
    // the ahead copy, so a byte that differs after the Finish had a writer the tracking missed
    static u64 checks = 0;
    static u64 mismatches = 0;
    static auto last_report = std::chrono::steady_clock::now();
    u64 total = 0;
    for (const auto& copy : copies) {
        total = std::max<u64>(total, copy.dstOffset - ahead_offset + copy.size);
    }
    const auto check = staging_pool.Request(total, VideoCore::MemoryType::HostCached);
    boost::container::small_vector<vk::BufferCopy, 1> check_copies(copies.begin(), copies.end());
    for (auto& copy : check_copies) {
        copy.dstOffset = copy.dstOffset - ahead_offset + check.offset;
    }
    runtime.CopyBuffer(arena, check.buffer, check_copies);
    scheduler.Finish();
    check.buffer->Invalidate(check.offset, check.size);
    ++checks;
    for (const auto& copy : copies) {
        const u64 offset = copy.dstOffset - ahead_offset;
        if (std::memcmp(ahead_data + offset, check.mapped + offset, copy.size) == 0) {
            continue;
        }
        ++mismatches;
        if (mismatches <= 16) {
            LOG_ERROR(Render_Vulkan, "Readback ahead verify: mismatch at {:#x} ({:#x} bytes), tick {}",
                      arena->cpu_addr + copy.srcOffset, copy.size, scheduler.CurrentTick());
        }
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - last_report >= std::chrono::minutes{1}) {
        LOG_WARNING(Render_Vulkan, "Readback ahead verify: {} ahead copies checked, {} mismatches",
                    checks, mismatches);
        last_report = now;
    }
}

std::pair<const Buffer*, u64> BufferCache::ObtainBufferForImage(VAddr device_addr, u32 size) {
    if (IsRegionGpuModified(device_addr, size)) {
        return ObtainBuffer(device_addr, size, false);
    }
    const auto staging = staging_pool.Request(size, VideoCore::MemoryType::HostUncached,
                                              instance.StorageMinAlignment());
    memory->CopySparseMemory(device_addr, staging.mapped, staging.size);
    staging.Flush();
    return {staging.buffer, staging.offset};
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionCpuModified(addr, size);
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    if (memory_tracker->IsRegionGpuModified(addr, size)) {
        return true;
    }
    const VAddr start = Common::AlignDown(addr, BYTES_PER_PAGE);
    const VAddr end = Common::AlignUp(addr + size, BYTES_PER_PAGE);
    auto it = std::ranges::upper_bound(sync_batch, start, {}, &SyncRange::end);
    for (; it != sync_batch.end() && it->start < end; ++it) {
        if (it->written) {
            return true;
        }
    }
    return false;
}

void BufferCache::SynchronizeDmaBuffers() {
    fault_process_pending = true;
    // Stack pages are never write-watched (cpu_authoritative_stacks), so they stay CPU-modified and a sweep
    // over them uploads every resident stack byte again. Like the FaultManager, which never turns stack pages
    // into GPU buffers, the sweep leaves them out; bindings whose V# covers a stack still upload it
    const bool skip_stacks = sweep_skip_stacks && Common::PerfStats::AbToggleOn(sweep_skip_toggle);
    const u64 stack_generation = skip_stacks ? memory->StackRangesGeneration() : 0;
    // Every compute dispatch that uses DMA lands here, and re-adding all resident ranges to a batch that
    // already covers them changes nothing: SyncRange::Dominant keeps written ranges written
    if (dma_sync_once && Common::PerfStats::AbToggleOn(dma_sync_once_toggle) && dma_sync_covered &&
        stack_generation == dma_sync_stack_generation) {
        if (dma_sync_once_verify) [[unlikely]] {
            VerifyDmaSyncCovered(skip_stacks);
        }
        if (Common::PerfStats::Enabled()) {
            Common::PerfStats::Add(Common::PerfStats::Id::DmaSweepsSkipped);
        }
        return;
    }
    dma_sync_covered = true;
    dma_sync_stack_generation = stack_generation;
    const bool perf = Common::PerfStats::Enabled();
    if (perf) {
        Common::PerfStats::Add(Common::PerfStats::Id::DmaSweeps);
    }
    if (!skip_stacks) {
        for (const auto& range : resident_ranges) {
            const VAddr device_addr = range.start << block_shift;
            sync_batch.Add(device_addr, range.end << block_shift, false);
        }
    } else {
        if (!dma_sweep_pieces_valid || dma_sweep_pieces_generation != stack_generation) {
            dma_sweep_pieces.clear();
            for (const auto& range : resident_ranges) {
                const VAddr device_addr = range.start << block_shift;
                const u64 size = (range.end - range.start) << block_shift;
                for (const auto& piece : memory->SubtractStackRanges(device_addr, size)) {
                    dma_sweep_pieces.push_back(piece);
                }
            }
            dma_sweep_pieces_generation = stack_generation;
            dma_sweep_pieces_valid = true;
        }
        for (const auto& [lo, len] : dma_sweep_pieces) {
            sync_batch.Add(lo, lo + len, false);
        }
    }
    if (perf) {
        // Resident stack bytes, left out or not
        for (const auto& range : resident_ranges) {
            const VAddr device_addr = range.start << block_shift;
            const u64 size = (range.end - range.start) << block_shift;
            for (const auto& [lo, len] : memory->GetStackRangesIn(device_addr, size)) {
                Common::PerfStats::Add(Common::PerfStats::Id::DmaSweepStackBytes, len);
            }
        }
    }
}

void BufferCache::VerifyDmaSyncCovered(bool skip_stacks) {
    // Only the GPU command processor thread synchronizes DMA buffers here, so the batch tracking
    // stays on that thread
    static u64 checks = 0;
    static u64 mismatches = 0;
    static auto last_report = std::chrono::steady_clock::now();
    ++checks;
    const auto check = [&](VAddr lo, VAddr hi) {
        if (!sync_batch.Contains(lo, hi) && ++mismatches <= 16) {
            LOG_ERROR(Render_Vulkan,
                      "dma_sync_once_per_batch: skipped sync, but the batch does not cover {:#x}-{:#x}",
                      lo, hi);
        }
    };
    if (skip_stacks) {
        for (const auto& [lo, len] : dma_sweep_pieces) {
            check(lo, lo + len);
        }
    } else {
        for (const auto& range : resident_ranges) {
            check(range.start << block_shift, range.end << block_shift);
        }
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - last_report >= std::chrono::seconds{60}) {
        last_report = now;
        LOG_WARNING(Render_Vulkan, "dma_sync_once_per_batch verify: {} skipped syncs, {} mismatches",
                    checks, mismatches);
    }
}

const Buffer* BufferCache::GetArena(u64 first_block, u64 last_block) {
    const u64 first_page = first_block >> blocks_per_arena_page_shift;
    const u64 last_page = last_block >> blocks_per_arena_page_shift;
    ASSERT_MSG(last_page - first_page <= 1,
               "Buffer request cannot span more than two VA arena pages");

    const auto* first_arena = address_space[first_page];
    const auto* last_arena = address_space[last_page];
    if (first_arena == last_arena) {
        if (!first_arena) {
            const u64 base_block = Common::AlignDownPow2<u64>(first_block, blocks_per_arena_page);
            const u64 num_pages = last_page - first_page + 1;
            const auto* new_arena =
                &arenas.emplace_back(instance, base_block << block_shift,
                                     num_pages << ARENA_PAGE_BITS, MemoryType::Sparse);
            arena_generation.fetch_add(1, std::memory_order_release);
            address_space[first_page] = new_arena;
            address_space[last_page] = new_arena;
        }
        return address_space[first_page];
    }

    LOG_WARNING(Render, "Migrating arena");

    const u64 first_addr = first_arena ? first_arena->cpu_addr : (first_page << ARENA_PAGE_BITS);
    const u64 first_size = first_arena ? first_arena->size_bytes : ARENA_PAGE_SIZE;
    const u64 last_size = last_arena ? last_arena->size_bytes : ARENA_PAGE_SIZE;

    const u64 base_block = first_addr >> block_shift;
    const u64 total_size = first_size + last_size;
    const u64 end_block = (first_addr + total_size) >> block_shift;
    auto* new_arena = &arenas.emplace_back(instance, first_addr, total_size, MemoryType::Sparse);
    arena_generation.fetch_add(1, std::memory_order_release);
    auto* bind = BindsForArena(new_arena);
    migration_binds_pending = true;
    resident_ranges.ForEachInRange(base_block, end_block, [&](const Backing& backing) {
        const u64 start = std::max(base_block, backing.start);
        const u64 end = std::min(end_block, backing.end);
        bind->binds.push_back(vk::SparseMemoryBind{
            .resourceOffset = (start - base_block) << block_shift,
            .size = (end - start) << block_shift,
            .memory = backing.memory,
            .memoryOffset = backing.offset + ((start - backing.start) << block_shift),
        });
    });

    u64 base_page = first_addr >> ARENA_PAGE_BITS;
    for (u32 page = 0; page < (first_size >> ARENA_PAGE_BITS); ++page) {
        address_space[base_page + page] = new_arena;
    }
    base_page = last_page;
    for (u32 page = 0; page < (last_size >> ARENA_PAGE_BITS); ++page) {
        address_space[base_page + page] = new_arena;
    }
    return new_arena;
}

bool BufferCache::CheckReusedBuffer(const Buffer* arena, u64 offset, VAddr device_addr, u32 size) {
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    if (address_space[first_block >> blocks_per_arena_page_shift] != arena ||
        address_space[last_block >> blocks_per_arena_page_shift] != arena ||
        arena->Offset(device_addr) != offset) {
        return false;
    }
    bool resident = true;
    resident_ranges.ForEachGap(first_block, last_block + 1, [&](u64, u64) { resident = false; });
    return resident && sync_batch.Contains(device_addr, device_addr + size);
}

void BufferCache::EnsureResident(const Buffer* arena, u64 first_block, u64 last_block) {
    u32 resident_blocks{};
    IntervalList bind_ranges;
    resident_ranges.ForEachGap(first_block, last_block + 1, [&](u64 start, u64 end) {
        resident_blocks += end - start;
        bind_ranges.Add({start, end});
    });

    if (bind_ranges.Empty()) {
        return;
    }

    const vk::MemoryAllocateInfo alloc_info = {
        .allocationSize = resident_blocks << block_shift,
        .memoryTypeIndex = arena_memory_type_index,
    };
    const auto device_memory = Vulkan::Check(instance.GetDevice().allocateMemory(alloc_info));

    boost::container::small_vector<vk::BufferCopy, 8> copies;
    const auto staging =
        staging_pool.Request(resident_blocks * sizeof(vk::DeviceAddress), MemoryType::HostUncached);

    u64 memory_offset{};
    ArenaBinds* binds = BindsForArena(arena);
    auto* bda_addrs = reinterpret_cast<vk::DeviceAddress*>(staging.mapped);
    u64 offset = staging.offset;
    for (const auto& range : bind_ranges) {
        Backing backing;
        backing.start = range.start;
        backing.end = range.end;
        backing.memory = device_memory;
        backing.offset = memory_offset;
        resident_ranges.Add(backing);
        dma_sync_covered = false;
        dma_sweep_pieces_valid = false;

        LOG_INFO(Render, "Making range start={}, end={} resident", backing.start, backing.end);

        const auto& bind = binds->binds.emplace_back(vk::SparseMemoryBind{
            .resourceOffset = (range.start << block_shift) - arena->cpu_addr,
            .size = (range.end - range.start) << block_shift,
            .memory = device_memory,
            .memoryOffset = memory_offset,
        });
        memory_offset += bind.size;

        for (u32 block = 0; block < bind.size; block += block_size) {
            *(bda_addrs++) = arena->BufferDeviceAddress() + bind.resourceOffset + block;
        }
        const u64 copy_size = (backing.end - backing.start) * sizeof(vk::DeviceAddress);
        copies.emplace_back(offset, backing.start * sizeof(vk::DeviceAddress), copy_size);
        offset += copy_size;
    }

    staging.Flush();
    runtime.CopyBuffer(staging.buffer, bda_pagetable_buffer.get(), copies);
}

bool BufferCache::SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size) {
    if (auto type = texture_cache.IsMeta(device_addr)) {
        if (*type == TextureCache::MetaType::HTile) {
            static constexpr u32 ZmaskUncompressed = 0xf;
            runtime.FillBuffer(arena, arena->Offset(device_addr), size, ZmaskUncompressed);
            RecordGpuWrite(device_addr, size);
            return true;
        } else {
            LOG_RENDER_PROBLEM(Render_Vulkan, Warning, "Unhandled metadata type {}",
                               magic_enum::enum_name(*type));
        }
    }
    const ImageId image_id = texture_cache.FindImageFromRange(device_addr, size);
    if (!image_id) {
        return false;
    }
    Image& image = texture_cache.GetImage(image_id);
    ASSERT_MSG(device_addr == image.info.guest_address,
               "Texel buffer aliases image subresources {:x} : {:x}", device_addr,
               image.info.guest_address);
    const u64 arena_offset = arena->Offset(device_addr);
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    for (u32 mip = 0; mip < image.info.resources.levels; mip++) {
        const auto& mip_info = image.info.mips_layout[mip];
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth = std::max(image.info.size.depth >> mip, 1u);
        if (arena_offset + mip_info.offset + mip_info.size > arena->size_bytes) {
            break;
        }
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = mip_info.offset,
            .bufferRowLength = mip_info.pitch,
            .bufferImageHeight = mip_info.height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, depth},
        });
    }
    if (buffer_copies.empty()) {
        return false;
    }
    auto& tile_manager = texture_cache.GetTileManager();
    tile_manager.TileImage(image, buffer_copies, arena, arena_offset);
    RecordGpuWrite(device_addr, std::max<u64>(size, image.info.guest_size));
    return true;
}

void BufferCache::SubmitPendingArenaBinds(Vulkan::SubmitInfo& info) {
    if (pending_binds.empty()) {
        return;
    }

    const u64 signal_tick = memory_semaphore.NextTick();
    const auto signal_sema = memory_semaphore.Handle();

    info.AddWait(signal_sema, signal_tick);
    if (migration_binds_pending) {
        // Called from the submit callback: the tick being submitted is the one before the current
        migration_bind_tick = scheduler.CurrentTick() - 1;
        migration_binds_pending = false;
    }

    // Queue the sparse bind immediately before the submit that waits for it
    // When cp_record_thread moves that work to another thread, the closure must own the bind lists
    // until it runs
    scheduler.RunOnQueue([this, binds = std::move(pending_binds), signal_tick, signal_sema] {
        std::vector<vk::SparseBufferMemoryBindInfo> buffer_binds;
        buffer_binds.reserve(binds.size());

        for (const auto& arena_binds : binds) {
            buffer_binds.emplace_back(vk::SparseBufferMemoryBindInfo{
                .buffer = arena_binds.arena->Handle(),
                .bindCount = static_cast<u32>(arena_binds.binds.size()),
                .pBinds = arena_binds.binds.data(),
            });
        }

        const vk::TimelineSemaphoreSubmitInfo timeline_si = {
            .signalSemaphoreValueCount = 1u,
            .pSignalSemaphoreValues = &signal_tick,
        };

        const vk::BindSparseInfo sparse_info = {
            .pNext = &timeline_si,
            .bufferBindCount = static_cast<u32>(buffer_binds.size()),
            .pBufferBinds = buffer_binds.data(),
            .signalSemaphoreCount = 1u,
            .pSignalSemaphores = &signal_sema,
        };

        auto submit_result = instance.GetGraphicsQueue().bindSparse(sparse_info);
        if (submit_result == vk::Result::eErrorDeviceLost) {
            instance.ReportDeviceFault("SubmitPendingArenaBinds");
        }
        ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");
    });

    pending_binds.clear();
}

void BufferCache::FlushSyncBatch(bool from_scheduler) {
    boost::container::small_vector<vk::BufferCopy, 32> copies;
    size_t total_size_bytes = 0;
    for (const auto& range : sync_batch) {
        memory_tracker->ForEachUploadRange(
            range.start, range.end - range.start, range.written, [&](u64 addr, u64 range_size) {
                copies.emplace_back(total_size_bytes, addr, range_size);
                total_size_bytes += range_size;
                if (Common::PerfStats::Enabled()) {
                    Common::PerfStats::RecordUploadRegion(addr, range_size);
                    if (range.written) {
                        Common::PerfStats::Add(Common::PerfStats::Id::BufferUploadWrittenBytes,
                                               range_size);
                    }
                    for (const auto& [lo, len] : memory->GetStackRangesIn(addr, range_size)) {
                        Common::PerfStats::Add(Common::PerfStats::Id::BufferUploadStackBytes, len);
                    }
                }
            });
    }
    sync_batch.Clear();
    sync_batch_generation.fetch_add(1, std::memory_order_release);
    dma_sync_covered = false;
    if (copies.empty()) {
        return;
    }
    Common::PerfStats::Add(Common::PerfStats::Id::SyncFlushes);
    Common::PerfStats::Add(Common::PerfStats::Id::BufferUploadBytes, total_size_bytes);
    const auto staging = staging_pool.Request(total_size_bytes, MemoryType::HostUncached);
    for (auto& copy : copies) {
        RecordGpuWrite(copy.dstOffset, copy.size);
        memory->CopySparseMemory(copy.dstOffset, staging.mapped + copy.srcOffset, copy.size);
        Common::GuestWriteJournal::Record(Common::GuestWriteJournal::Source::Upload,
                                          copy.dstOffset, copy.size,
                                          staging.mapped + copy.srcOffset);
        copy.srcOffset += staging.offset;
    }
    staging.Flush();

    const auto cmdbuf = scheduler.UploadCommandBuffer();

    u32 batch_start = 0;
    u32 batch_end = 0;

    while (true) {
        batch_start = batch_end;
        const auto& copy = copies[batch_start];
        const auto* arena = address_space[copy.dstOffset >> ARENA_PAGE_BITS];

        const auto expand_batch = [&] {
            const auto& copy = copies[batch_end];
            const auto end_page = (copy.dstOffset + copy.size - 1) >> ARENA_PAGE_BITS;
            return address_space[end_page] == arena;
        };
        const auto flush_batch = [&] {
            const auto regions = std::span{copies}.subspan(batch_start, batch_end - batch_start);
            for (auto& copy : regions) {
                copy.dstOffset -= arena->cpu_addr;
            }
            cmdbuf.copyBuffer(staging.buffer->Handle(), arena->Handle(), regions);
        };

        while (batch_end < copies.size() && expand_batch()) {
            ++batch_end;
        }

        // No more copies to examine.
        if (batch_end == copies.size()) {
            flush_batch();
            break;
        }

        // Next copy does not overlap with the current buffer.
        auto end_copy = copies[batch_end];
        const auto* end_arena = address_space[end_copy.dstOffset >> ARENA_PAGE_BITS];
        if (end_arena != arena) {
            flush_batch();
            continue;
        }

        // Next copy partially overlaps with buffer.
        const auto copy_size = end_arena->cpu_addr + end_arena->size_bytes - end_copy.dstOffset;
        copies[batch_end].size = copy_size;
        end_copy.srcOffset += copy_size;
        end_copy.dstOffset += copy_size;
        end_copy.size -= copy_size;
        ++batch_end;
        flush_batch();
        --batch_end;
        copies[batch_end] = end_copy;
    }

    const vk::MemoryBarrier2 memory_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .memoryBarrierCount = 1u,
        .pMemoryBarriers = &memory_barrier,
    });
    num_flushes_per_frame++;

    if (!from_scheduler) {
        scheduler.BeginSession();
    }
}

} // namespace VideoCore
