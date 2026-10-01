// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <new>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

/// With cp_record_thread, record Vulkan commands and submit them on a separate thread from the
/// command processor
/// The scheduler gives the producer fake command buffer handles, and dispatcher handlers copy their
/// arguments into an ordered stream
/// The consumer replays that stream on real command buffers, while presenter, ImGui and ahead
/// copies using real handles call the driver directly
///
/// Only one producer owns the scheduler at a time and one consumer reads the stream of 256 KiB
/// chunks
/// Data blocks come before the commands using them, and a command may still point into the
/// preceding chunk
/// Recycle a chunk only after the consumer leaves the following chunk so those references remain
/// valid
/// Each producer write is published before the consumer reads it, keeping the stream in the order
/// used by the command processor
/// Copied arrays stay inside these chunks until their commands have run, so the producer does not
/// have to keep the original argument memory alive
class CommandRecorder {
public:
    CommandRecorder();
    ~CommandRecorder();

    CommandRecorder(const CommandRecorder&) = delete;
    CommandRecorder& operator=(const CommandRecorder&) = delete;

    /// Install command handlers once after the default dispatcher has loaded the device functions
    /// Copy ordinary arguments with the generic handler and pointer arguments with dedicated deep-
    /// copy handlers
    /// Stop and report the command name if a fake handle reaches a command whose pointer arguments
    /// are unsupported
    static void InstallThunks();

    [[nodiscard]] static bool IsFake(VkCommandBuffer cmdbuf) noexcept;

    /// Give the producer a fake handle for a command buffer that the consumer will allocate later
    /// The queued closure calls Bind to connect it with the real buffer before recording its
    /// commands
    [[nodiscard]] vk::CommandBuffer NewFakeHandle();

    /// Queue func for the recording thread after every operation already in the stream so the
    /// producer can preserve command order
    template <typename F>
    void Run(F&& func) {
        using T = std::decay_t<F>;
        static_assert(alignof(T) <= BlockAlign, "Closure over-aligned for the stream");
        void* payload = Allocate(sizeof(T), BlockKind::Command, &Invoke<T>);
        new (payload) T(std::forward<F>(func));
        Publish();
    }

    /// Reserve a command block of size bytes for invoke(payload) to run on the recording thread
    /// Write the arguments and any referenced arrays into that block, then call Commit before
    /// making another recorder call
    /// Keeping the arguments with the command lets the caller reuse its original memory after
    /// publication
    [[nodiscard]] void* AllocateCommand(size_t size, void (*invoke)(void* payload)) {
        return Allocate(size, BlockKind::Command, invoke);
    }

    /// Publish the block reserved by AllocateCommand so the consumer can run it in the same ordered
    /// stream used by Run
    void Commit() {
        Publish();
    }

    /// Copy count elements into the stream for the next queued command so the producer can release
    /// the original array, preserving null pointers
    template <typename T>
    [[nodiscard]] T* Copy(const T* src, size_t count = 1) {
        if (src == nullptr || count == 0) {
            return nullptr;
        }
        static_assert(std::is_trivially_copyable_v<T>);
        void* dst = Allocate(sizeof(T) * count, BlockKind::Data, nullptr);
        std::memcpy(dst, src, sizeof(T) * count);
        return static_cast<T*>(dst);
    }

    /// Reserve enough room in the current chunk for the next data and command blocks so their bytes
    /// stay together until the consumer finishes
    void Reserve(size_t bytes);

    /// Publish all queued work and wake the recording thread if it is sleeping at this submit point
    /// Between kicks, publish smaller batches for the consumer to pick up while spinning or after
    /// its short timed wait
    /// A Kick makes every command already queued visible before the submit or drain can continue
    void Kick();

    /// Count this deferred Vulkan call for performance statistics and report the accumulated calls
    /// at the next Kick
    void CountCommand() noexcept {
        ++commands_counted;
    }

    /// Wait until the recording thread has run every operation queued so far before the producer
    /// continues with work needing those results
    void Sync();

    /// Make the calling thread the producer when shutdown moves recording to another thread
    /// The previous producer must already have stopped, and any other unexpected producer change
    /// stops the recorder
    void ClaimProducer();

    /// Connect a fake command buffer handle to the real buffer allocated by the consumer before
    /// queued commands use it
    void Bind(VkCommandBuffer fake, VkCommandBuffer real) noexcept;

    /// Look up the real command buffer belonging to this fake handle when the consumer replays its
    /// queued commands
    [[nodiscard]] VkCommandBuffer Real(VkCommandBuffer fake) const noexcept;

    /// Release count fake handle sessions after the consumer has submitted them so the producer can
    /// reuse those slots later
    void Retire(u32 count) noexcept {
        handles_retired.fetch_add(count, std::memory_order_release);
    }

    [[nodiscard]] bool IsRecordingThread() const noexcept {
        return std::this_thread::get_id() == thread.get_id();
    }

    static constexpr size_t BlockAlign = 16;
    static constexpr u32 NumFakeHandles = 1U << 16;

private:
    enum class BlockKind : u32 { Data, Command, End };

    struct BlockHeader {
        u32 size; ///< Total block size including its header, rounded to a multiple of BlockAlign for the next block
        BlockKind kind;
        void (*invoke)(void* payload);
    };
    static_assert(sizeof(BlockHeader) == BlockAlign);

    struct Chunk {
        static constexpr size_t Size = 256_KB;
        std::atomic<u32> committed{0};
        std::atomic<Chunk*> next{nullptr};
        alignas(64) u8 data[Size];
    };

    template <typename T>
    static void Invoke(void* payload) {
        T& func = *static_cast<T*>(payload);
        func();
        func.~T();
    }

    void* Allocate(size_t size, BlockKind kind, void (*invoke)(void*));
    void Publish();
    void PublishAll();
    void NextChunk();
    Chunk* AcquireChunk();
    void RecycleChunk(Chunk* chunk);
    void WakeConsumer();
    void WakeIfSleeping();
    void CheckProducer();
    void ConsumerLoop(std::stop_token stoken);

    static constexpr size_t MaxChunksInFlight = 256; ///< Allow up to 64 MiB of queued commands before the producer waits for the recording thread to
///< catch up
    static constexpr u32 PublishBatch = 32;

    // State written by the producer while it adds commands and publishes batches to the recording
    // thread
    Chunk* write_chunk{};
    u32 write_pos{};
    u64 handles_issued{};
    u64 sync_requested{};
    u64 commands_counted{};
    u32 unpublished{};
    u32 producer_tid{};
    u32 producer_switches{};

    // State written by the consumer while it runs queued commands and advances through the stream
    // chunks
    std::array<VkCommandBuffer, NumFakeHandles> real_handles{};

    // State shared by both threads to track queued work, sleeping and the recorder lifetime
    std::atomic<u64> handles_retired{0};
    std::atomic<u64> sync_done{0};
    std::atomic<u32> chunks_in_flight{0};
    std::atomic<bool> consumer_sleeping{false};
    std::mutex wake_mutex;
    std::condition_variable wake_cv;
    bool wake_flag{};
    std::mutex free_mutex;
    std::vector<Chunk*> free_chunks;
    Chunk* first_chunk{};
    std::jthread thread;
};

} // namespace Vulkan
