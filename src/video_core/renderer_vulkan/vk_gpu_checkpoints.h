// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string_view>
#include <utility>
#include "common/logging/log.h"
#include "common/types.h"

// Device loss diagnostics, enabled by the per-game gpu_checkpoints setting (off by default)
//
// A host ring tracks draws and dispatches, with NVIDIA checkpoints pointing to the command running at device
// loss. A submit ledger identifies the first unfinished submit, even if the host recorded later commands
// SHADPS4_GPU_DIAG also captures resource details for each command
namespace Vulkan::GpuCheckpoints {

/// Set once by the Instance when the gpu_checkpoints setting is on
inline bool g_enabled = false;
/// Set once by the Instance when VK_NV_device_diagnostic_checkpoints is enabled as well
inline bool g_nv_markers = false;

inline bool Enabled() noexcept {
    return g_enabled;
}

enum class Kind : u8 {
    Draw,
    DrawIndexed,
    DrawIndirect,
    DrawIndexedIndirect,
    Dispatch,
    DispatchIndirect,
    Internal, // emulator-side dispatch, hash0 = InternalTag
};

enum class InternalTag : u64 {
    FaultBufferProcess = 1,
    Detile = 2,
    Tile = 3,
};

struct Record {
    u64 seq;
    u64 hash0; // vs or cs program hash
    u64 hash1; // fs program hash (graphics only)
    u64 addr;  // indirect args guest address (indirect only)
    u32 a;     // num_indices / dim_x / max_count
    u32 b;     // num_instances / dim_y
    u32 c;     // dim_z
    Kind kind;
};

constexpr size_t RingSize = 1 << 16;
inline std::array<Record, RingSize> g_ring{};
inline std::atomic<u64> g_seq{0};

inline const char* KindName(Kind kind) {
    switch (kind) {
    case Kind::Draw:
        return "Draw";
    case Kind::DrawIndexed:
        return "DrawIndexed";
    case Kind::DrawIndirect:
        return "DrawIndirect";
    case Kind::DrawIndexedIndirect:
        return "DrawIndexedIndirect";
    case Kind::Dispatch:
        return "Dispatch";
    case Kind::DispatchIndirect:
        return "DispatchIndirect";
    case Kind::Internal:
        return "Internal";
    }
    return "?";
}

inline const Record* Push(Kind kind, u64 hash0, u64 hash1, u32 a, u32 b, u32 c, u64 addr = 0) {
    const u64 seq = g_seq.fetch_add(1, std::memory_order_relaxed) + 1;
    Record& record = g_ring[seq % RingSize];
    record = Record{
        .seq = seq,
        .hash0 = hash0,
        .hash1 = hash1,
        .addr = addr,
        .a = a,
        .b = b,
        .c = c,
        .kind = kind,
    };
    return &record;
}

inline void LogRecord(const char* prefix, const Record& record) {
    LOG_CRITICAL(Render_Vulkan, "{} seq={} {} hash0={:#x} hash1={:#x} a={} b={} c={} addr={:#x}",
                 prefix, record.seq, KindName(record.kind), record.hash0, record.hash1, record.a,
                 record.b, record.c, record.addr);
}

/// Returns the ring record a checkpoint marker points to, or null for foreign markers
inline const Record* FromMarker(const void* marker) {
    const auto* begin = reinterpret_cast<const u8*>(g_ring.data());
    const auto* end = begin + sizeof(g_ring);
    const auto* ptr = reinterpret_cast<const u8*>(marker);
    if (ptr < begin || ptr >= end || ((ptr - begin) % sizeof(Record)) != 0) {
        return nullptr;
    }
    return reinterpret_cast<const Record*>(ptr);
}

inline void DumpRecent(const char* where, u32 count = 32) {
    const u64 last = g_seq.load(std::memory_order_relaxed);
    LOG_CRITICAL(Render_Vulkan, "=== Last {} recorded GPU commands ({}), newest last ===", count,
                 where);
    const u64 first = last > count ? last - count + 1 : 1;
    for (u64 seq = first; seq <= last; ++seq) {
        LogRecord("recorded:", g_ring[seq % RingSize]);
    }
}

struct SubmitRecord {
    u64 id;
    u64 semaphore; // VkSemaphore handle of the submitting scheduler's timeline
    u64 tick;      // value signalled on that timeline when the submit completes
    u64 first_seq; // checkpoint records carried by this submit (empty when first > last)
    u64 last_seq;
    u64 known_tick; // last completed tick the host had observed on this timeline
    u64 time_us;
    // readback_ahead: `semaphore` is a VkFence signalled by this submit alone and `tick` numbers the submits
    // on it (one in flight at a time). It carries no guest command and runs after command last_seq
    bool fence;
};

constexpr size_t SubmitRingSize = 1 << 12;
inline std::array<SubmitRecord, SubmitRingSize> g_submits{};
inline std::atomic<u64> g_submit_id{0};

inline u64 NowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

inline void RecordSubmit(u64 semaphore, u64 tick, u64 first_seq, u64 last_seq, u64 known_tick,
                         bool fence = false) {
    const u64 id = g_submit_id.fetch_add(1, std::memory_order_relaxed) + 1;
    g_submits[id % SubmitRingSize] = SubmitRecord{
        .id = id,
        .semaphore = semaphore,
        .tick = tick,
        .first_seq = first_seq,
        .last_seq = last_seq,
        .known_tick = known_tick,
        .time_us = NowUs(),
        .fence = fence,
    };
}

struct BufferRef {
    u64 addr;
    u32 size;
    bool written;
    const u32* words; // host-visible copy of the first dwords, written by the GPU (diag only)
    bool has_cpu_words;
    bool overlaps_stack; // inside a registered stack range, i.e. excluded from CPU write tracking
    std::array<u32, 8> cpu_words; // guest memory at record time, read through the backing
};

/// Fill value of the host copies: a word still holding it was never written by the GPU
constexpr u32 CaptureSentinel = 0xDEADBEEF;
constexpr u32 CaptureWords = 8;

struct ImageRef {
    u64 addr;
    u32 format; // data_format | num_format << 8
    bool written;
};

struct Detail {
    static constexpr size_t MaxBuffers = 16;
    static constexpr size_t MaxImages = 8;
    u64 seq;
    u32 num_buffers;
    u32 num_images;
    u32 dropped; // bindings that did not fit
    bool uses_dma;
    u32 num_args;
    const u32* args; // host-visible copy of the indirect arguments, written by the GPU
    std::array<BufferRef, MaxBuffers> buffers;
    std::array<ImageRef, MaxImages> images;

    void Reset() {
        num_buffers = 0;
        num_images = 0;
        dropped = 0;
        uses_dma = false;
        num_args = 0;
        args = nullptr;
    }
    void AddBuffer(u64 addr, u64 size, bool written) {
        if (num_buffers == MaxBuffers) {
            ++dropped;
            return;
        }
        buffers[num_buffers++] = {addr, static_cast<u32>(size), written, nullptr, false, false, {}};
    }
    void AddImage(u64 addr, u32 format, bool written) {
        if (num_images == MaxImages) {
            ++dropped;
            return;
        }
        images[num_images++] = {addr, format, written};
    }
};

/// SHADPS4_GPU_DIAG: per-command resource details (only with gpu_checkpoints on)
inline bool DiagEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_GPU_DIAG");
        return value != nullptr && *value != '\0' && std::string_view{value} != "0";
    }();
    return enabled && Enabled();
}

/// Number of bindings per command whose head is copied on the GPU. SHADPS4_GPU_DIAG=cpu copies none and cpu0
/// only the first: every copy adds barriers between commands, which can hide the very hazard being
/// investigated. Host-side values are always recorded
inline u32 DiagGpuCopies() {
    static const u32 count = [] {
        const char* value = std::getenv("SHADPS4_GPU_DIAG");
        if (!value) {
            return 0u;
        }
        const std::string_view mode{value};
        return mode == "cpu" ? 0u : mode == "cpu0" ? 1u : 16u;
    }();
    return count;
}

constexpr size_t DetailRingSize = 1 << 14;
inline std::array<Detail, DetailRingSize> g_details{};

/// Attaches resource details to the most recently pushed record
inline void AttachDetail(const Detail& detail) {
    const u64 seq = g_seq.load(std::memory_order_relaxed);
    Detail& slot = g_details[seq % DetailRingSize];
    slot = detail;
    slot.seq = seq;
}

inline void LogDetail(const Detail& detail) {
    LOG_CRITICAL(Render_Vulkan, "    dma={} buffers={} images={} dropped={}", detail.uses_dma,
                 detail.num_buffers, detail.num_images, detail.dropped);
    for (u32 i = 0; i < detail.num_buffers; ++i) {
        const auto& buffer = detail.buffers[i];
        LOG_CRITICAL(Render_Vulkan, "    buffer[{}] addr={:#x} size={:#x} written={} stack={}", i,
                     buffer.addr, buffer.size, buffer.written, buffer.overlaps_stack);
        if (buffer.has_cpu_words) {
            const auto& w = buffer.cpu_words;
            LOG_CRITICAL(Render_Vulkan,
                         "      cpu:  {:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x}", w[0], w[1],
                         w[2], w[3], w[4], w[5], w[6], w[7]);
        }
        if (buffer.words) {
            const u32* w = buffer.words;
            LOG_CRITICAL(Render_Vulkan,
                         "      head: {:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x}", w[0], w[1],
                         w[2], w[3], w[4], w[5], w[6], w[7]);
        }
    }
    for (u32 i = 0; i < detail.num_images; ++i) {
        const auto& image = detail.images[i];
        LOG_CRITICAL(Render_Vulkan, "    image[{}] addr={:#x} fmt={:#x} written={}", i, image.addr,
                     image.format, image.written);
    }
    if (detail.args) {
        // Indirect arguments copied on the GPU right before the command; the sentinel means the copy never
        // ran, i.e. an earlier command did not finish
        std::array<u32, 5> args{};
        for (u32 i = 0; i < detail.num_args && i < args.size(); ++i) {
            args[i] = detail.args[i];
        }
        LOG_CRITICAL(Render_Vulkan, "    gpu words: {:#x} {:#x} {:#x} {:#x} {:#x}", args[0],
                     args[1], args[2], args[3], args[4]);
    }
}

/// Logs the commands of one submit, with details when they were captured. Bounded so a large submit cannot
/// flood the log: the head and the tail of the range are kept
inline void DumpSubmitCommands(const SubmitRecord& submit, u64 newest_seq, u64 limit = 96) {
    if (submit.first_seq > submit.last_seq) {
        LOG_CRITICAL(Render_Vulkan, "  (no recorded guest commands)");
        return;
    }
    const u64 oldest_kept = newest_seq > RingSize ? newest_seq - RingSize + 1 : 1;
    const u64 count = submit.last_seq - submit.first_seq + 1;
    for (u64 seq = submit.first_seq; seq <= submit.last_seq; ++seq) {
        if (count > limit && seq == submit.first_seq + limit / 2) {
            LOG_CRITICAL(Render_Vulkan, "  ... {} commands skipped ...", count - limit);
            seq = submit.last_seq - limit / 2;
        }
        if (seq < oldest_kept) {
            continue;
        }
        const Record& record = g_ring[seq % RingSize];
        if (record.seq != seq) {
            continue;
        }
        LogRecord("  cmd:", record);
        const Detail& detail = g_details[seq % DetailRingSize];
        if (DiagEnabled() && detail.seq == seq) {
            LogDetail(detail);
        }
    }
}

/// Reports, per timeline, the first submit that the GPU never completed. query_counter returns false when the
/// counter cannot be read after the loss and leaves the last known value. query_fence tells whether a fence
/// (readback_ahead) is signalled, or nothing when it cannot be read
template <typename QueryCounter, typename QueryFence>
inline void DumpSubmits(const char* where, QueryCounter&& query_counter, QueryFence&& query_fence) {
    static std::mutex report_mutex;
    std::scoped_lock lock{report_mutex};
    const u64 last_id = g_submit_id.load(std::memory_order_relaxed);
    const u64 newest_seq = g_seq.load(std::memory_order_relaxed);
    const u64 first_id = last_id >= SubmitRingSize ? last_id - SubmitRingSize + 1 : 1;
    const u64 now = NowUs();
    LOG_CRITICAL(Render_Vulkan, "=== Submit ledger ({}): submits {}..{}, newest command seq {} ===",
                 where, first_id, last_id, newest_seq);

    // Handles of different Vulkan object types may share a value, so a timeline is the pair
    std::array<std::pair<u64, bool>, 8> semaphores{};
    size_t num_semaphores = 0;
    for (u64 id = first_id; id <= last_id; ++id) {
        const auto& record = g_submits[id % SubmitRingSize];
        const std::pair<u64, bool> key{record.semaphore, record.fence};
        bool seen = false;
        for (size_t i = 0; i < num_semaphores; ++i) {
            seen |= semaphores[i] == key;
        }
        if (!seen && num_semaphores < semaphores.size()) {
            semaphores[num_semaphores++] = key;
        }
    }
    for (size_t i = 0; i < num_semaphores; ++i) {
        const auto [sem, fence] = semaphores[i];
        const auto matches = [&](const SubmitRecord& submit, u64 id) {
            return submit.id == id && submit.semaphore == sem && submit.fence == fence;
        };
        u64 counter = 0;
        u64 newest_tick = 0;
        for (u64 id = first_id; id <= last_id; ++id) {
            const auto& submit = g_submits[id % SubmitRingSize];
            if (matches(submit, id)) {
                counter = std::max(counter, submit.known_tick);
                newest_tick = std::max(newest_tick, submit.tick);
            }
        }
        bool queried = false;
        if (fence) {
            // Only the newest fence submit can be pending: each one is waited for before the next
            if (const std::optional<bool> signalled = query_fence(sem)) {
                queried = true;
                counter = *signalled ? newest_tick : newest_tick - 1;
            }
        } else {
            queried = query_counter(sem, counter);
        }
        const SubmitRecord* first_pending = nullptr;
        u32 pending = 0;
        for (u64 id = first_id; id <= last_id; ++id) {
            const auto& submit = g_submits[id % SubmitRingSize];
            if (!matches(submit, id) || submit.tick <= counter) {
                continue;
            }
            ++pending;
            if (!first_pending) {
                first_pending = &submit;
            }
        }
        LOG_CRITICAL(Render_Vulkan, "{} {:#x}: completed {} {} ({}), {} submits pending",
                     fence ? "readback-ahead fence" : "timeline", sem, fence ? "copy" : "tick",
                     counter, queried ? "queried" : "last known", pending);
        if (!first_pending) {
            continue;
        }
        const auto& submit = *first_pending;
        if (fence) {
            LOG_CRITICAL(Render_Vulkan,
                         "first incomplete readback-ahead copy id={} copy={}, queued after command "
                         "seq {}, submitted {} ms before the report",
                         submit.id, submit.tick, submit.last_seq, (now - submit.time_us) / 1000);
            continue;
        }
        LOG_CRITICAL(Render_Vulkan,
                     "first incomplete submit id={} tick={} commands seq {}..{} submitted {} ms "
                     "before the report",
                     submit.id, submit.tick, submit.first_seq, submit.last_seq,
                     (now - submit.time_us) / 1000);
        DumpSubmitCommands(submit, newest_seq);
    }
}

} // namespace Vulkan::GpuCheckpoints
