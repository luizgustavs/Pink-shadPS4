// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <random>

#include <gtest/gtest.h>

#include "common/interval_set.h"
#include "video_core/renderer_vulkan/vk_access_bitmap.h"

namespace {

using Vulkan::AccessBitmap;

constexpr u64 KiB = 1024;
constexpr u64 MiB = 1024 * KiB;

/// A range in [0, limit) whose size is drawn from the classes the rasterizer
/// binds: a few bytes, up to a chunk, up to several chunks (large list) and
/// some ending past the buffer
Interval RandomRange(std::mt19937_64& rng, u64 limit) {
    const u64 cls = rng() % 8;
    u64 size;
    if (cls < 3) {
        size = 1 + rng() % 512;
    } else if (cls < 6) {
        size = 1 + rng() % (64 * KiB);
    } else if (cls == 6) {
        size = 64 * KiB + 1 + rng() % (2 * MiB);
    } else {
        size = 1 + rng() % (limit / 2);
    }
    const u64 start = rng() % limit;
    return {start, start + size};
}

void CheckAgainstList(u64 buffer_size, u32 seed) {
    std::mt19937_64 rng{seed};
    AccessBitmap bitmap{buffer_size};
    IntervalList<Interval> list;
    // The queries may run past the buffer, as unbounded V# ranges do
    const u64 limit = buffer_size + buffer_size / 8;
    for (int round = 0; round < 40; ++round) {
        const int adds = 1 + static_cast<int>(rng() % 200);
        for (int i = 0; i < adds; ++i) {
            const Interval range = RandomRange(rng, limit);
            bitmap.Add(range.start, range.end);
            list.Add(range);
            // Repeated ranges go through the recent cache
            if (rng() % 4 == 0) {
                bitmap.Add(range.start, range.end);
            }
            for (int q = 0; q < 8; ++q) {
                const Interval query = RandomRange(rng, limit);
                bool hit;
                const bool expected = list.Overlaps(query.start, query.end);
                ASSERT_EQ(bitmap.Overlaps(query.start, query.end, hit), expected)
                    << "seed " << seed << " round " << round << " query [" << query.start << ", "
                    << query.end << ")";
                if (expected) {
                    EXPECT_TRUE(hit);
                }
            }
        }
        bitmap.Clear();
        list.Clear();
        bool hit;
        ASSERT_FALSE(bitmap.Overlaps(0, limit, hit));
    }
}

TEST(AccessBitmap, MatchesIntervalListOnSmallBuffer) {
    for (u32 seed = 1; seed <= 20; ++seed) {
        CheckAgainstList(200 * KiB + 100, seed);
    }
}

TEST(AccessBitmap, MatchesIntervalListOnLargeBuffer) {
    for (u32 seed = 100; seed <= 105; ++seed) {
        CheckAgainstList(64 * MiB, seed);
    }
}

TEST(AccessBitmap, EdgesOfGranulesAndChunks) {
    AccessBitmap bitmap{1 * MiB};
    bool hit;
    bitmap.Add(64 * KiB - 1, 64 * KiB);
    EXPECT_TRUE(bitmap.Overlaps(64 * KiB - 1, 64 * KiB, hit));
    EXPECT_FALSE(bitmap.Overlaps(64 * KiB, 64 * KiB + 256, hit));
    // Same granule, no byte in common: the bitmap matches, the exact ranges do not
    EXPECT_FALSE(bitmap.Overlaps(64 * KiB - 256, 64 * KiB - 1, hit));
    EXPECT_TRUE(hit);
    bitmap.Add(256 * KiB, 256 * KiB + 128 * KiB);
    EXPECT_TRUE(bitmap.Overlaps(256 * KiB + 128 * KiB - 1, 1 * MiB, hit));
    EXPECT_FALSE(bitmap.Overlaps(256 * KiB + 128 * KiB, 1 * MiB, hit));
    EXPECT_FALSE(bitmap.Overlaps(10, 10, hit));
    bitmap.Clear();
    EXPECT_FALSE(bitmap.Overlaps(0, 1 * MiB, hit));
    EXPECT_FALSE(hit);
}

} // Anonymous namespace
