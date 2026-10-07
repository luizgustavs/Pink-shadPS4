// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <random>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "common/interval_set.h"

namespace {

// Same rules as the buffer cache's SyncRange
struct Flagged : Interval {
    bool written;
    constexpr bool CanMergeWith(const Flagged& o) const noexcept {
        return written == o.written;
    }
    constexpr Flagged SubRange(u64 a, u64 b) const noexcept {
        return {{a, b}, written};
    }
    constexpr bool Dominant(const Flagged& o) const noexcept {
        return written && !o.written;
    }
};

/// 0 = absent, 1 = read, 2 = written, per byte; fails on unsorted or empty intervals
std::vector<int> Bytes(const DomIntervalList<Flagged>& list, u64 n) {
    std::vector<int> bytes(n, 0);
    u64 prev_end = 0;
    for (const auto& range : list) {
        EXPECT_LT(range.start, range.end);
        EXPECT_GE(range.start, prev_end);
        prev_end = range.end;
        for (u64 x = range.start; x < range.end; ++x) {
            bytes[x] = range.written ? 2 : 1;
        }
    }
    return bytes;
}

TEST(DomIntervalList, AddSortedMatchesSequentialAdds) {
    std::mt19937_64 rng{1};
    constexpr u64 N = 512;
    for (int iter = 0; iter < 50000; ++iter) {
        DomIntervalList<Flagged> sequential;
        DomIntervalList<Flagged> sorted;
        const int n = static_cast<int>(rng() % 12);
        for (int k = 0; k < n; ++k) {
            const u64 start = rng() % N;
            const u64 len = 1 + rng() % 40;
            const bool written = rng() & 1;
            sequential.Add(start, std::min(N, start + len), written);
            sorted.Add(start, std::min(N, start + len), written);
        }
        // Sorted, disjoint values, some of them touching
        std::vector<Flagged> values;
        u64 cur = rng() % 20;
        while (cur < N) {
            const u64 len = 1 + rng() % 50;
            const bool written = (rng() % 4) == 0;
            values.push_back({{cur, std::min(N, cur + len)}, written});
            cur += len + rng() % 30;
        }
        for (const auto& value : values) {
            sequential.Add(value);
        }
        sorted.AddSorted(values);
        ASSERT_EQ(Bytes(sequential, N), Bytes(sorted, N)) << "iteration " << iter;
        // The result is fully coalesced
        const Flagged* prev = nullptr;
        for (const auto& range : sorted) {
            ASSERT_FALSE(prev && prev->end == range.start && prev->written == range.written)
                << "iteration " << iter;
            prev = &range;
        }
    }
}

TEST(DomIntervalList, AddSortedKeepsDominantRanges) {
    DomIntervalList<Flagged> list;
    list.Add(u64{10}, u64{20}, true);
    list.Add(u64{30}, u64{40}, false);
    const std::vector<Flagged> values{{{0, 15}, false}, {{15, 35}, false}, {{50, 60}, true}};
    list.AddSorted(values);
    const std::vector<std::tuple<u64, u64, bool>> expected{
        {0, 10, false}, {10, 20, true}, {20, 40, false}, {50, 60, true}};
    std::vector<std::tuple<u64, u64, bool>> got;
    for (const auto& range : list) {
        got.emplace_back(range.start, range.end, range.written);
    }
    EXPECT_EQ(got, expected);
}

} // Anonymous namespace
