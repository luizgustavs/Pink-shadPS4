// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <deque>
#include <span>
#include <boost/container/small_vector.hpp>

#include "common/interval_set.h"
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/fault_manager.h"
#include "video_core/buffer_cache/range_set.h"
#include "video_core/renderer_vulkan/vk_semaphore.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {
class GraphicsPipeline;
struct SubmitInfo;
class Runtime;
class StagingBufferPool;
} // namespace Vulkan

namespace VideoCore {

class TextureCache;
class MemoryTracker;
class BpeHeapGuard;
class PageManager;

class BufferCache {
    static constexpr u64 ADDRESS_SPACE_BITS = 40;
    static constexpr u64 ARENA_PAGE_BITS = 32;
    static constexpr u64 ARENA_PAGE_SIZE = u64{1} << ARENA_PAGE_BITS;
    static constexpr u64 NUM_ARENA_PAGES = u64{1} << (ADDRESS_SPACE_BITS - ARENA_PAGE_BITS);
    static constexpr u64 MIN_BLOCK_SIZE = 16_KB;
    static constexpr u64 STREAM_THRESHOLD = 16_KB;

public:
    explicit BufferCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         Vulkan::Runtime& runtime, AmdGpu::Liverpool* liverpool,
                         TextureCache& texture_cache, PageManager& tracker);
    ~BufferCache();

    /// Returns a pointer to GDS device local buffer.
    [[nodiscard]] const Buffer* GetGdsBuffer() const noexcept {
        return &gds_buffer;
    }

    /// Retrieves the device local DBA page table buffer.
    [[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept {
        return bda_pagetable_buffer.get();
    }

    /// Retrieves the fault buffer.
    [[nodiscard]] Buffer* GetFaultBuffer() noexcept {
        return fault_manager->GetFaultBuffer();
    }

    /// Retrieves the compute_loop_cap hit counter buffer
    [[nodiscard]] const Buffer* GetLoopCapBuffer() const noexcept {
        return &loop_cap_buffer;
    }

    /// Retrieves the stream buffer.
    [[nodiscard]] StreamBuffer& GetStreamBuffer() noexcept {
        return stream_buffer;
    }

    /// Return true when a region has a pending synchronization request.
    [[nodiscard]] bool IsRegionInSyncBatch(VAddr addr, size_t size) const noexcept {
        return sync_batch.Overlaps(addr, addr + size);
    }

    /// Returns minimum granularity of a sparse memory bind.
    u32 GetSparsePageShift() const noexcept {
        return block_shift;
    }

    void TickFrame();

    /// Invalidates any buffer in the logical page range.
    void InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks = false);

    /// Flushes any GPU modified buffer in the logical page range back to CPU memory.
    void ReadMemory(VAddr device_addr, u64 size, bool is_write = false, bool assume_locks = false);

    /// Returns a guest stack range that stopped being a stack to normal tracking, CPU-modified
    void ReleaseCpuAuthoritativeRange(VAddr device_addr, u64 size);

    /// Finds a buffer for the specified region.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBuffer(VAddr device_addr, u32 size,
                                                             bool is_written,
                                                             bool is_texel_buffer = false);

    /// Attempts to obtain a buffer without modifying the cache contents.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBufferForImage(VAddr device_addr, u32 size);

    /// Return true when a region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(VAddr addr, size_t size);

    /// Return true when a region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(VAddr addr, size_t size);

    /// Byte-precise: true when a written binding covered any byte of the range since its last readback.
    /// Readbacks copy only these bytes, so the backing already holds the value of every other byte.
    /// Command processor thread only
    [[nodiscard]] bool IsRangeGpuWritten(VAddr addr, size_t size) const {
        return gpu_modified_ranges.Intersects(addr, size);
    }

    enum class CleanRead { Served, GpuWritten, Miss };
    /// Clean reads (Rasterizer::ReadCleanMemory) of [addr, addr + size), within one 4 KiB page, from a cache
    /// of pages that holds their backing pointer and the hull of their GPU-written bytes. On Miss the caller
    /// checks that the page is GPU mapped and calls FillCleanPage. Command processor thread only
    CleanRead ReadCleanPage(VAddr addr, void* out, u32 size);
    /// `generation` is MappingGeneration() read before checking that the page is GPU mapped
    bool FillCleanPage(VAddr page_addr, u64 generation);
    [[nodiscard]] u64 MappingGeneration() const {
        return clean_map_generation.load(std::memory_order_acquire);
    }
    /// A GPU mapping was removed or added: cached backing pointers may be stale. Any thread, after the mapped
    /// ranges changed
    void OnMappingChanged() {
        clean_map_generation.fetch_add(1, std::memory_order_release);
    }

    /// Idea B: the command that uses the written bindings obtained since the last call has been recorded,
    /// so their writes belong to the current command buffer from now on
    void CloseGpuWrites();

    /// Count sync batch flushes so reused bindings can tell whether their ranges are still in the
    /// same growing batch
    [[nodiscard]] u64 SyncBatchGeneration() const noexcept {
        return sync_batch_generation.load(std::memory_order_acquire);
    }

    /// Count arena creations so reused ObtainBuffer results can tell whether the arena layout has
    /// changed since their last binding
    [[nodiscard]] u64 ArenaGeneration() const noexcept {
        return arena_generation.load(std::memory_order_acquire);
    }

    /// Do the work ObtainBuffer still needs for a reused read-only arena binding with unchanged
    /// arenas and sync batch
    /// This includes copying an aliased image into its texel buffer because the GPU may have
    /// written to the image since the last bind
    /// Reusing the buffer location does not remove the need to refresh that image data
    void ReuseReadBuffer(const Buffer* arena, VAddr device_addr, u32 size, bool is_texel_buffer) {
        if (is_texel_buffer) {
            SynchronizeMemoryFromImage(arena, device_addr, size);
        }
    }

    /// Check the assumptions behind a reused arena binding by confirming its offset and range,
    /// residency and presence in the sync batch
    /// The result is reusable only while all of those conditions still match what ObtainBuffer
    /// expects
    [[nodiscard]] bool CheckReusedBuffer(const Buffer* arena, u64 offset, VAddr device_addr, u32 size);

    /// Synchronizes all buffers needed for DMA.
    void SynchronizeDmaBuffers();
    /// Check that the current sync batch already contains every range a DMA sync would add before
    /// we skip that sync
    void VerifyDmaSyncCovered(bool skip_stacks);

    /// Commits pending sparse buffer memory binds. Must be called before every scheduler submit.
    void SubmitPendingArenaBinds(Vulkan::SubmitInfo& info);

    /// Flushes pending synchronization requests
    void FlushSyncBatch(bool from_scheduler = false);

private:
    struct ArenaBinds {
        const Buffer* arena;
        boost::container::small_vector<vk::SparseMemoryBind, 32> binds;
    };

    ArenaBinds* BindsForArena(const Buffer* arena) {
        auto it = std::ranges::find(pending_binds, arena, &ArenaBinds::arena);
        if (it != pending_binds.end()) {
            return std::addressof(*it);
        }
        return &pending_binds.emplace_back(arena);
    }

    const Buffer* GetArena(u64 first_block, u64 last_block);

    void EnsureResident(const Buffer* arena, u64 first_block, u64 last_block);

    void DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size);

    /// readback_hot_regions: a readback window requested now;
    /// returns whether the key is on
    bool TouchHotReadback(const Buffer* arena, VAddr addr, u64 size);

    /// readback_hot_regions: records copies of the GPU-written bytes of the other
    /// recent readback windows into hot_buffer, before the readback drains the GPU
    void RecordHotDownloads(VAddr skip_addr);

    /// readback_hot_regions: after the drain, writes the hot downloads to guest memory
    void ApplyHotDownloads(BpeHeapGuard* heap_guard);

    bool SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size);

    /// Logs compute_loop_cap hits: the first one, then a count per minute
    void ReportLoopCapHits();

    /// gpu_modified_ranges changed in [addr, addr + size): drop the clean pages there
    void InvalidateCleanPages(VAddr addr, u64 size);

    enum class WriterState { Current, Busy, Done };
    /// Idea B: stores [addr, addr + size) as written by the command buffer being recorded. `open`: a
    /// binding whose command is not recorded yet, and may land in a later command buffer (CloseGpuWrites)
    void RecordGpuWrite(VAddr addr, u64 size, bool open = false);
    /// Newest writer of the bytes a readback copies (srcOffset relative to `arena_base`)
    /// `wait_tick` receives the tick an ahead copy on another queue must wait for (Busy/Done only): the
    /// writer's when it is tracked, else the tick before the oldest tracked one
    WriterState ClassifyWriters(VAddr arena_base, std::span<const vk::BufferCopy> copies,
                                u64* wait_tick);
    /// SHADPS4_READBACK_AHEAD_VERIFY: copies the bytes again after a Finish and compares them with the
    /// ahead copy
    void VerifyAheadCopy(const Buffer* arena, std::span<const vk::BufferCopy> copies,
                         const u8* ahead_data, u64 ahead_offset);

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    Vulkan::Runtime& runtime;
    Vulkan::StagingBufferPool& staging_pool;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    TextureCache& texture_cache;
    std::unique_ptr<MemoryTracker> memory_tracker;

    StreamBuffer stream_buffer;
    Buffer gds_buffer;
    Buffer loop_cap_buffer;
    RangeSet gpu_modified_ranges;
    struct CleanPage {
        VAddr page_addr = ~0ULL;
        u64 map_generation;
        const u8* backing;
        // Hull of the GPU-written bytes of the page, as offsets; dirty_lo >= dirty_hi when there are none
        u32 dirty_lo;
        u32 dirty_hi;
    };
    static constexpr u64 CleanPageBits = 12;
    static constexpr u64 CleanPageSize = 1ULL << CleanPageBits;
    static constexpr size_t NumCleanPages = 1024;
    std::array<CleanPage, NumCleanPages> clean_pages{};
    bool clean_pages_used = false; // Set by the first FillCleanPage
    std::atomic<u64> clean_map_generation{};
    u32 loop_cap_hits_reported{};
    u32 loop_cap_hits_minute{};
    std::chrono::steady_clock::time_point loop_cap_last_check{};
    std::chrono::steady_clock::time_point loop_cap_last_report{};

    // Idea B: GPU writes to arena memory per command buffer tick, newest last; only the last few ticks
    struct TickWrites {
        u64 tick;
        RangeSet ranges;
    };
    std::deque<TickWrites> writer_ticks;
    boost::container::small_vector<std::pair<VAddr, u64>, 16> open_writes;
    const bool readback_ahead;
    const bool track_writers;
    /// readback_hot_regions: readback windows requested in the last seconds
    struct HotReadback {
        const Buffer* arena;
        VAddr addr;
        u64 size;
        std::chrono::steady_clock::time_point last_request;
    };
    std::vector<HotReadback> hot_readbacks;
    /// readback_hot_regions: windows with copies recorded by
    /// RecordHotDownloads, waiting for the drain
    struct HotDownload {
        const Buffer* arena;
        VAddr addr;
        u64 size;
        u32 first_copy;
        u32 num_copies;
    };
    std::vector<HotDownload> hot_downloads;
    std::vector<vk::BufferCopy> hot_copies;
    /// readback_hot_regions: host buffer the hot copies land in, reused by every drain
    /// and grown in powers of two (a staging request of a different size each drain
    /// allocated a dedicated buffer)
    std::unique_ptr<Buffer> hot_buffer;
    u64 hot_total{};
    // Frente J probe: the readback being downloaded was asked for by a guest thread's fault
    bool guest_readback{};

    std::unique_ptr<FaultManager> fault_manager;
    std::unique_ptr<Buffer> bda_pagetable_buffer;
    bool fault_process_pending{};

    std::array<const Buffer*, NUM_ARENA_PAGES> address_space{};
    std::deque<Buffer> arenas;
    std::vector<ArenaBinds> pending_binds;
    Vulkan::Semaphore memory_semaphore;
    /// Graphics tick that makes migrated arena bindings visible to transfer copies
    bool migration_binds_pending{};
    u64 migration_bind_tick{};

    struct Backing : public Interval {
        vk::DeviceMemory memory;
        u64 offset;
        constexpr bool CanMergeWith(const Backing& other) const noexcept {
            return memory == other.memory && offset + (end - start) == other.offset;
        }
        constexpr Backing SubRange(u64 a, u64 b) const noexcept {
            return {{a, b}, memory, offset + (a - start)};
        }
    };
    IntervalList<Backing> resident_ranges;

    struct SyncRange : Interval {
        bool written;
        constexpr bool CanMergeWith(const SyncRange& o) const noexcept {
            return written == o.written;
        }
        constexpr SyncRange SubRange(u64 a, u64 b) const noexcept {
            return {{a, b}, written};
        }
        constexpr bool Dominant(const SyncRange& o) const noexcept {
            return written && !o.written;
        }
    };
    DomIntervalList<SyncRange> sync_batch{};
    std::atomic<u64> sync_batch_generation{};
    std::atomic<u64> arena_generation{};
    /// Whether the current sync batch already covers every resident range
    bool dma_sync_covered{};
    const bool dma_sync_once;
    const bool dma_sync_once_toggle; ///< Whether dma_sync_once_per_batch is active for this interval of SHADPS4_AB_TOGGLE
    /// With SHADPS4_DMA_SYNC_ONCE_VERIFY, check every skipped sync against the ranges it would have
    /// added to the current batch
    const bool dma_sync_once_verify;
    /// Stack generation used by the last covered DMA sweep
    const bool sweep_skip_stacks;
    const bool sweep_skip_toggle;
    const bool readback_hot_regions;
    const bool readback_hot_toggle;
    u64 dma_sync_stack_generation{};
    /// Cached resident ranges with guest stacks removed
    std::vector<std::pair<VAddr, u64>> dma_sweep_pieces;
    u64 dma_sweep_pieces_generation{};
    bool dma_sweep_pieces_valid{};
    u32 num_flushes_per_frame{};

    u32 arena_memory_type_index{};
    u32 block_size{};
    u32 block_shift{};
    u32 blocks_per_arena_page{};
    u32 blocks_per_arena_page_shift{};
};

} // namespace VideoCore
