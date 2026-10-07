// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <chrono>
#include <optional>
#include <random>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "core/stack_ranges.h"

namespace {

using Core::StackRangeSet;
using Pieces = std::vector<std::pair<VAddr, u64>>;

Pieces ToVector(const StackRangeSet::Pieces& pieces) {
    return {pieces.begin(), pieces.end()};
}

constexpr VAddr ThreadStack = 0x2'0000'0000;
constexpr u64 ThreadStackSize = 0x10'0000;
constexpr VAddr Heap = 0x9'0000'0000;

TEST(StackRangeSet, EmptySetAnswersWithoutRanges) {
    StackRangeSet set;
    EXPECT_FALSE(set.HasRanges());
    EXPECT_FALSE(set.Overlaps(Heap, 0x1000));
    EXPECT_EQ(ToVector(set.Subtract(Heap, 0x1000)), (Pieces{{Heap, 0x1000}}));
    EXPECT_TRUE(set.GetIn(Heap, 0x1000).empty());
    EXPECT_TRUE(set.Subtract(Heap, 0).empty());
}

TEST(StackRangeSet, QueriesAroundOneRange) {
    StackRangeSet set;
    set.Add(ThreadStack, ThreadStack + ThreadStackSize);
    EXPECT_TRUE(set.Overlaps(ThreadStack, 1));
    EXPECT_TRUE(set.Overlaps(ThreadStack - 0x1000, 0x1001));
    EXPECT_TRUE(set.Overlaps(ThreadStack + ThreadStackSize - 1, 0x1000));
    EXPECT_FALSE(set.Overlaps(ThreadStack - 0x1000, 0x1000));
    EXPECT_FALSE(set.Overlaps(ThreadStack + ThreadStackSize, 0x1000));
    EXPECT_FALSE(set.Overlaps(ThreadStack, 0));

    const VAddr lo = ThreadStack - 0x2000;
    const u64 size = ThreadStackSize + 0x3000;
    EXPECT_EQ(ToVector(set.Subtract(lo, size)),
              (Pieces{{lo, 0x2000}, {ThreadStack + ThreadStackSize, 0x1000}}));
    EXPECT_EQ(ToVector(set.GetIn(lo, size)), (Pieces{{ThreadStack, ThreadStackSize}}));
}

TEST(StackRangeSet, RepeatedQueriesInAGap) {
    // The second query of each pair is answered by the per-thread gap cache; both must match
    StackRangeSet set;
    set.Add(ThreadStack, ThreadStack + ThreadStackSize);
    set.Add(Heap + 0x10'0000, Heap + 0x11'0000);
    for (int i = 0; i < 2; ++i) {
        EXPECT_FALSE(set.Overlaps(Heap, 0x1000));
        EXPECT_FALSE(set.Overlaps(ThreadStack + ThreadStackSize, 0x1000));
        EXPECT_FALSE(set.Overlaps(0x1000, 0x1000));
        EXPECT_TRUE(set.Overlaps(Heap + 0x10'F000, 0x2000));
        EXPECT_EQ(ToVector(set.Subtract(Heap, 0x1000)), (Pieces{{Heap, 0x1000}}));
        EXPECT_TRUE(set.GetIn(Heap + 0x20'0000, 0x1000).empty());
    }
    // Inside the cached gap but reaching the next range: not a hit
    EXPECT_TRUE(set.Overlaps(Heap, 0x10'0001));
    EXPECT_EQ(ToVector(set.GetIn(Heap, 0x10'1000)), (Pieces{{Heap + 0x10'0000, 0x1000}}));
}

TEST(StackRangeSet, AddInsideACachedGap) {
    // A thread or fiber stack registered where a query already found nothing (e.g. a fiber context
    // carved out of a heap the command processor reads)
    StackRangeSet set;
    set.Add(ThreadStack, ThreadStack + ThreadStackSize);
    EXPECT_FALSE(set.Overlaps(Heap, 0x1000));
    EXPECT_FALSE(set.Overlaps(Heap, 0x1000));
    const u64 generation = set.Generation();
    set.Add(Heap, Heap + 0x4000);
    EXPECT_NE(set.Generation(), generation);
    EXPECT_TRUE(set.Overlaps(Heap, 0x1000));
    EXPECT_TRUE(set.Subtract(Heap, 0x1000).empty());
    EXPECT_EQ(ToVector(set.GetIn(Heap, 0x1000)), (Pieces{{Heap, 0x1000}}));
}

TEST(StackRangeSet, RemoveAfterCachedQueries) {
    // FreeStack and sceFiberFinalize: the memory is no stack any more
    StackRangeSet set;
    set.Add(ThreadStack, ThreadStack + ThreadStackSize);
    set.Add(Heap, Heap + 0x4000);
    EXPECT_FALSE(set.Overlaps(Heap + 0x4000, 0x1000));
    EXPECT_FALSE(set.Overlaps(Heap + 0x4000, 0x1000));
    EXPECT_TRUE(set.Overlaps(Heap, 0x1000));
    set.Remove(Heap, Heap + 0x4000);
    EXPECT_FALSE(set.Overlaps(Heap, 0x1000));
    EXPECT_EQ(ToVector(set.Subtract(Heap, 0x8000)), (Pieces{{Heap, 0x8000}}));
    // The merged gap now reaches back to the thread stack
    EXPECT_FALSE(set.Overlaps(ThreadStack + ThreadStackSize, Heap - ThreadStack - ThreadStackSize));
    EXPECT_TRUE(set.Overlaps(ThreadStack + ThreadStackSize - 1, 0x1000));
}

TEST(StackRangeSet, NestedRegistrationsAreCounted) {
    // A fiber context inside a thread stack: finalizing the fiber keeps the thread stack
    StackRangeSet set;
    set.Add(ThreadStack, ThreadStack + ThreadStackSize);
    const VAddr fiber = ThreadStack + 0x4'0000;
    set.Add(fiber, fiber + 0x4000);
    EXPECT_EQ(ToVector(set.GetIn(ThreadStack, ThreadStackSize)),
              (Pieces{{ThreadStack, ThreadStackSize}}));
    set.Remove(fiber, fiber + 0x4000);
    EXPECT_TRUE(set.Overlaps(fiber, 0x1000));
    EXPECT_TRUE(set.Subtract(fiber, 0x4000).empty());
    set.Remove(ThreadStack, ThreadStack + ThreadStackSize);
    EXPECT_FALSE(set.Overlaps(fiber, 0x1000));
    EXPECT_FALSE(set.Overlaps(ThreadStack, ThreadStackSize));
    // Removing a range that was never added changes nothing
    set.Remove(Heap, Heap + 0x1000);
    EXPECT_FALSE(set.Overlaps(Heap, 0x1000));
}

TEST(StackRangeSet, MoreGapsThanCacheEntries) {
    StackRangeSet set;
    constexpr int NumStacks = 8;
    for (int i = 0; i < NumStacks; ++i) {
        const VAddr base = ThreadStack + i * 0x20'0000;
        set.Add(base, base + ThreadStackSize);
    }
    for (int round = 0; round < 3; ++round) {
        for (int i = 0; i < NumStacks; ++i) {
            const VAddr base = ThreadStack + i * 0x20'0000;
            EXPECT_FALSE(set.Overlaps(base + ThreadStackSize, 0x1000)) << i;
            EXPECT_TRUE(set.Overlaps(base + ThreadStackSize - 0x1000, 0x1000)) << i;
        }
    }
}

TEST(StackRangeSet, TopOfTheAddressSpace) {
    // The last gap reaches ~0; a range that ends right below it is answered the same, cached or not
    StackRangeSet set;
    set.Add(ThreadStack, ThreadStack + ThreadStackSize);
    const VAddr high = ~VAddr{0} - 0x1000;
    for (int i = 0; i < 2; ++i) {
        EXPECT_FALSE(set.Overlaps(high, 0x1000));
        EXPECT_EQ(ToVector(set.Subtract(high, 0x1000)), (Pieces{{high, 0x1000}}));
        EXPECT_TRUE(set.GetIn(high, 0x1000).empty());
    }
}

TEST(StackRangeSet, SetsDoNotShareCachedGaps) {
    StackRangeSet empty_gap;
    empty_gap.Add(ThreadStack, ThreadStack + ThreadStackSize);
    EXPECT_FALSE(empty_gap.Overlaps(Heap, 0x1000));

    StackRangeSet with_stack;
    with_stack.Add(Heap, Heap + 0x1000);
    EXPECT_TRUE(with_stack.Overlaps(Heap, 0x1000));

    // A set built where a destroyed one lived starts at the same generation
    std::optional<StackRangeSet> set;
    set.emplace();
    set->Add(ThreadStack, ThreadStack + ThreadStackSize);
    EXPECT_FALSE(set->Overlaps(Heap, 0x1000));
    set.reset();
    set.emplace();
    set->Add(Heap, Heap + 0x1000);
    EXPECT_TRUE(set->Overlaps(Heap, 0x1000));
}

TEST(StackRangeSet, ChangesFromAnotherThread) {
    // The gap cache is per thread and the ranges are not: a range another thread adds or removes
    // (thread creation, FreeStack, fibers) is seen by this thread's next query
    StackRangeSet set;
    set.Add(ThreadStack, ThreadStack + ThreadStackSize);
    EXPECT_FALSE(set.Overlaps(Heap, 0x1000));
    std::thread([&] { set.Add(Heap, Heap + 0x1000); }).join();
    EXPECT_TRUE(set.Overlaps(Heap, 0x1000));
    std::thread([&] {
        // This thread has no cached gap of its own yet
        EXPECT_TRUE(set.Overlaps(Heap, 0x1000));
        set.Remove(Heap, Heap + 0x1000);
    }).join();
    EXPECT_FALSE(set.Overlaps(Heap, 0x1000));
}

TEST(StackRangeSet, ConcurrentChurn) {
    // Readers query an address that is always a stack and one that never is while a writer keeps
    // adding and removing ranges between them. Neither answer may flip
    StackRangeSet set;
    const VAddr always = ThreadStack;
    const VAddr never = Heap + 0x100'0000;
    set.Add(always, always + ThreadStackSize);
    std::atomic<bool> stop{false};
    std::atomic<u64> wrong{0};
    std::vector<std::thread> readers;
    for (int t = 0; t < 3; ++t) {
        readers.emplace_back([&] {
            while (!stop.load(std::memory_order_relaxed)) {
                wrong += !set.Overlaps(always + 0x1000, 0x1000);
                wrong += set.Overlaps(never, 0x1000);
                wrong += !set.Subtract(always, 0x1000).empty();
                wrong += set.Subtract(never, 0x1000).size() != 1;
                wrong += set.GetIn(never, 0x1000).size() != 0;
            }
        });
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
    for (u64 i = 0; std::chrono::steady_clock::now() < deadline; ++i) {
        const VAddr churn = Heap + (i % 16) * 0x1'0000;
        set.Add(churn, churn + 0x4000);
        set.Remove(churn, churn + 0x4000);
    }
    stop = true;
    for (auto& reader : readers) {
        reader.join();
    }
    EXPECT_EQ(wrong.load(), 0u);
}

TEST(StackRangeSet, HandshakeWithWriter) {
    // A reader with the gap cached learns of a new stack through another synchronization (a thread
    // starting on it, the command processor seeing a submit): its next query must see the stack,
    // and after the removal the gap again
    StackRangeSet set;
    set.Add(ThreadStack, ThreadStack + ThreadStackSize);
    constexpr int Rounds = 2000;
    std::atomic<int> step{0};
    std::atomic<u64> wrong{0};
    const auto wait_for = [&](int value) {
        while (step.load(std::memory_order_acquire) != value) {
            std::this_thread::yield();
        }
    };
    std::thread reader([&] {
        for (int round = 0; round < Rounds; ++round) {
            wait_for(4 * round);
            wrong += set.Overlaps(Heap, 0x1000);
            step.store(4 * round + 1, std::memory_order_release);
            wait_for(4 * round + 2);
            wrong += !set.Overlaps(Heap, 0x1000);
            step.store(4 * round + 3, std::memory_order_release);
        }
    });
    for (int round = 0; round < Rounds; ++round) {
        wait_for(4 * round + 1);
        set.Add(Heap, Heap + 0x1000);
        step.store(4 * round + 2, std::memory_order_release);
        wait_for(4 * round + 3);
        set.Remove(Heap, Heap + 0x1000);
        step.store(4 * round + 4, std::memory_order_release);
    }
    reader.join();
    EXPECT_EQ(wrong.load(), 0u);
}

TEST(StackRangeSet, CopyQueriesMatchTheMap) {
    // range_fast_paths: the per-thread copy answers like the map through registrations,
    // nested and adjacent ranges and removals, and is rebuilt for another set
    std::mt19937_64 rng{7};
    StackRangeSet set;
    StackRangeSet other;
    other.Add(Heap, Heap + 0x1000);
    std::vector<std::pair<VAddr, VAddr>> live;
    for (int iter = 0; iter < 3000; ++iter) {
        if (live.empty() || rng() % 3 != 0) {
            const VAddr start = ThreadStack + (rng() % 64) * 0x1000;
            const VAddr end = start + (1 + rng() % 8) * 0x1000;
            set.Add(start, end);
            live.emplace_back(start, end);
        } else {
            const size_t index = rng() % live.size();
            set.Remove(live[index].first, live[index].second);
            live.erase(live.begin() + index);
        }
        for (int q = 0; q < 8; ++q) {
            const VAddr addr = ThreadStack - 0x2000 + rng() % (80 * 0x1000);
            const u64 size = 1 + rng() % (12 * 0x1000);
            ASSERT_EQ(set.Overlaps(addr, size, true), set.Overlaps(addr, size)) << iter;
            ASSERT_EQ(ToVector(set.Subtract(addr, size, true)), ToVector(set.Subtract(addr, size)))
                << iter;
            ASSERT_EQ(ToVector(set.GetIn(addr, size, true)), ToVector(set.GetIn(addr, size)))
                << iter;
        }
        if (iter % 100 == 0) {
            ASSERT_TRUE(other.Overlaps(Heap, 0x1000, true));
            ASSERT_FALSE(other.Overlaps(ThreadStack, 0x1000, true));
        }
    }
}

TEST(StackRangeSet, CopyQueriesSeeChangesFromAnotherThread) {
    StackRangeSet set;
    set.Add(ThreadStack, ThreadStack + ThreadStackSize);
    EXPECT_FALSE(set.Overlaps(Heap, 0x1000, true));
    std::thread([&] { set.Add(Heap, Heap + 0x1000); }).join();
    EXPECT_TRUE(set.Overlaps(Heap, 0x1000, true));
    EXPECT_TRUE(set.Subtract(Heap, 0x1000, true).empty());
    std::thread([&] { set.Remove(Heap, Heap + 0x1000); }).join();
    EXPECT_FALSE(set.Overlaps(Heap, 0x1000, true));
    EXPECT_EQ(ToVector(set.GetIn(ThreadStack - 0x1000, 0x2000, true)),
              (Pieces{{ThreadStack, 0x1000}}));
}

} // namespace
