// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <vector>

#include <gtest/gtest.h>

#include "common/lru_cache.h"

namespace {

using Cache = Common::LeastRecentlyUsedCache<int, u64>;

std::vector<int> ItemsBelow(Cache& cache, u64 tick) {
    std::vector<int> items;
    cache.ForEachItemBelow(tick, [&](int obj) { items.push_back(obj); });
    return items;
}

TEST(LruCache, TouchMovesItemToTheBack) {
    Cache cache;
    const auto a = cache.Insert(1, 0);
    cache.Insert(2, 0);
    EXPECT_TRUE(cache.Touch(a, 1));
    EXPECT_EQ(ItemsBelow(cache, 1), (std::vector<int>{2, 1}));
}

TEST(LruCache, TouchOfFreedItemIsIgnored) {
    Cache cache;
    const auto a = cache.Insert(1, 0);
    cache.Insert(2, 0);
    EXPECT_TRUE(cache.Free(a));
    // Touching this freed item must leave the list unchanged so garbage collection cannot get stuck
    // walking a cycle
    EXPECT_FALSE(cache.Touch(a, 5));
    EXPECT_EQ(ItemsBelow(cache, 10), (std::vector<int>{2}));
}

TEST(LruCache, SecondFreeIsIgnored) {
    Cache cache;
    const auto a = cache.Insert(1, 0);
    EXPECT_TRUE(cache.Free(a));
    EXPECT_FALSE(cache.Free(a));
    // The freed identifier should be reused once, then the next allocation should receive a
    // different identifier
    const auto b = cache.Insert(2, 0);
    const auto c = cache.Insert(3, 0);
    EXPECT_EQ(b, a);
    EXPECT_NE(c, a);
    EXPECT_EQ(ItemsBelow(cache, 10), (std::vector<int>{2, 3}));
}

TEST(LruCache, ReusedIdCanBeTouchedAgain) {
    Cache cache;
    const auto a = cache.Insert(1, 0);
    cache.Free(a);
    const auto b = cache.Insert(2, 0);
    cache.Insert(3, 0);
    EXPECT_TRUE(cache.Touch(b, 1));
    EXPECT_EQ(ItemsBelow(cache, 1), (std::vector<int>{3, 2}));
}

} // namespace
