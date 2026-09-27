// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <optional>
#include <shared_mutex>
#include <thread>

#include <gtest/gtest.h>

#include "common/recursive_lock.h"
#include "common/shared_first_mutex.h"

namespace {

using ExclusiveLock = Common::RecursiveScopedLock<std::shared_mutex>;
using SharedLock = Common::RecursiveSharedLock<std::shared_mutex>;

/// Whether another thread could take the mutex exclusively right now
template <typename Mutex>
bool FreeForOtherThread(Mutex& mutex) {
    bool acquired = false;
    std::thread([&] {
        acquired = mutex.try_lock();
        if (acquired) {
            mutex.unlock();
        }
    }).join();
    return acquired;
}

/// Whether another thread could take the mutex shared right now
template <typename Mutex>
bool SharedForOtherThread(Mutex& mutex) {
    bool acquired = false;
    std::thread([&] {
        acquired = mutex.try_lock_shared();
        if (acquired) {
            mutex.unlock_shared();
        }
    }).join();
    return acquired;
}

TEST(RecursiveLock, NestedExclusiveLocksOnce) {
    std::shared_mutex mutex;
    {
        ExclusiveLock outer{mutex};
        EXPECT_FALSE(FreeForOtherThread(mutex));
        {
            ExclusiveLock middle{mutex};
            ExclusiveLock inner{mutex};
            EXPECT_FALSE(SharedForOtherThread(mutex));
        }
        // The inner releases must not unlock the mutex
        EXPECT_FALSE(FreeForOtherThread(mutex));
    }
    EXPECT_TRUE(FreeForOtherThread(mutex));
}

TEST(RecursiveLock, NestedSharedLocksOnce) {
    std::shared_mutex mutex;
    {
        SharedLock outer{mutex};
        {
            SharedLock inner{mutex};
            EXPECT_TRUE(SharedForOtherThread(mutex));
            EXPECT_FALSE(FreeForOtherThread(mutex));
        }
        EXPECT_FALSE(FreeForOtherThread(mutex));
    }
    EXPECT_TRUE(FreeForOtherThread(mutex));
}

TEST(RecursiveLock, SharedAndExclusiveOnDifferentMutexes) {
    std::shared_mutex shared_mutex;
    std::shared_mutex exclusive_mutex;
    {
        SharedLock shared{shared_mutex};
        ExclusiveLock exclusive{exclusive_mutex};
        SharedLock shared_again{shared_mutex};
        ExclusiveLock exclusive_again{exclusive_mutex};
        EXPECT_TRUE(SharedForOtherThread(shared_mutex));
        EXPECT_FALSE(FreeForOtherThread(shared_mutex));
        EXPECT_FALSE(SharedForOtherThread(exclusive_mutex));
    }
    EXPECT_TRUE(FreeForOtherThread(shared_mutex));
    EXPECT_TRUE(FreeForOtherThread(exclusive_mutex));
}

TEST(RecursiveLock, WriterWaitsForOutermostSharedRelease) {
    std::shared_mutex mutex;
    std::optional<SharedLock> outer;
    outer.emplace(mutex);
    bool writer_done = false;
    std::thread writer;
    {
        SharedLock inner{mutex};
        writer = std::thread([&] {
            ExclusiveLock lock{mutex};
            writer_done = true;
        });
    }
    // The inner release leaves the mutex shared, so the writer is still waiting
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_FALSE(FreeForOtherThread(mutex));
    outer.reset();
    writer.join();
    EXPECT_TRUE(writer_done);
    EXPECT_TRUE(FreeForOtherThread(mutex));
}

TEST(RecursiveLock, SharedFirstMutex) {
    // The mutex type behind Rasterizer::IsMapped and the memory manager
    Common::SharedFirstMutex mutex;
    {
        Common::RecursiveSharedLock outer{mutex};
        Common::RecursiveSharedLock inner{mutex};
        EXPECT_TRUE(SharedForOtherThread(mutex));
        EXPECT_FALSE(FreeForOtherThread(mutex));
    }
    {
        Common::RecursiveScopedLock outer{mutex};
        Common::RecursiveScopedLock inner{mutex};
        EXPECT_FALSE(SharedForOtherThread(mutex));
    }
    EXPECT_TRUE(FreeForOtherThread(mutex));
}

TEST(RecursiveLock, MoreThanInlineEntries) {
    // The per-thread table holds 8 locks; the rest go to the overflow map
    constexpr size_t NumMutexes = 13;
    std::array<std::shared_mutex, NumMutexes> mutexes;
    std::array<std::optional<ExclusiveLock>, NumMutexes> exclusive;
    std::array<std::optional<SharedLock>, NumMutexes> shared;
    for (size_t i = 0; i < NumMutexes; ++i) {
        if (i % 2 == 0) {
            exclusive[i].emplace(mutexes[i]);
        } else {
            shared[i].emplace(mutexes[i]);
        }
    }
    for (size_t i = 0; i < NumMutexes; ++i) {
        EXPECT_FALSE(FreeForOtherThread(mutexes[i])) << i;
        EXPECT_EQ(SharedForOtherThread(mutexes[i]), i % 2 == 1) << i;
    }
    // Nest again on a lock in the table and on one in the overflow map
    {
        ExclusiveLock nested_inline{mutexes[2]};
        ExclusiveLock nested_overflow{mutexes[10]};
        SharedLock nested_shared_overflow{mutexes[11]};
    }
    EXPECT_FALSE(FreeForOtherThread(mutexes[2]));
    EXPECT_FALSE(FreeForOtherThread(mutexes[10]));
    EXPECT_FALSE(FreeForOtherThread(mutexes[11]));

    // Free a table entry, then nest on a lock that lives in the overflow map: it must be found
    // there, not added again to the freed entry (which would lock the mutex a second time)
    exclusive[0].reset();
    EXPECT_TRUE(FreeForOtherThread(mutexes[0]));
    {
        ExclusiveLock nested_overflow{mutexes[12]};
    }
    EXPECT_FALSE(FreeForOtherThread(mutexes[12]));
    // A new lock takes the freed entry
    {
        ExclusiveLock reused{mutexes[0]};
        EXPECT_FALSE(FreeForOtherThread(mutexes[0]));
    }
    EXPECT_TRUE(FreeForOtherThread(mutexes[0]));

    // Release the rest out of order (entry 0 is already free): 5 * i mod 13 visits 1..12
    for (size_t i = NumMutexes; i-- > 1;) {
        const size_t index = (i * 5) % NumMutexes;
        exclusive[index].reset();
        shared[index].reset();
    }
    for (size_t i = 0; i < NumMutexes; ++i) {
        EXPECT_TRUE(FreeForOtherThread(mutexes[i])) << i;
    }
    // The table is empty again: the locks work as usual
    {
        ExclusiveLock lock{mutexes[5]};
        ExclusiveLock nested{mutexes[5]};
        EXPECT_FALSE(FreeForOtherThread(mutexes[5]));
    }
    EXPECT_TRUE(FreeForOtherThread(mutexes[5]));
}

TEST(RecursiveLock, StateIsPerThread) {
    std::shared_mutex mutex;
    SharedLock outer{mutex};
    // Another thread holding its own shared lock is not a nested lock of this one
    std::thread([&] {
        SharedLock other{mutex};
        EXPECT_FALSE(FreeForOtherThread(mutex));
    }).join();
    EXPECT_FALSE(FreeForOtherThread(mutex));
}

} // namespace
